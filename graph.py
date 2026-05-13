"""
graph.py
--------
Defines and compiles the LangGraph StateGraph for the multi-agent research assistant.

Graph topology:
                        ┌──────────────────┐
                        │   START          │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  clarity_node    │
                        └────────┬─────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │ needs_clarification                  │ clear
              ▼                                      ▼
    ┌──────────────────┐                  ┌──────────────────┐
    │ interrupt_node   │ ◄── user input   │  research_node   │◄─────────────┐
    └────────┬─────────┘                  └────────┬─────────┘              │
             │ (re-enters at clarity)              │                         │
             │                        ┌────────────┴───────────┐            │
             │                 conf<6 │                        │ conf≥6     │
             │                        ▼                        ▼            │
             │               ┌────────────────┐    ┌──────────────────┐    │
             │               │ validator_node │    │ synthesis_node   │    │
             │               └────────┬───────┘    └──────────────────┘    │
             │                        │                                     │
             │           insufficient │ sufficient                          │
             │           (attempts<3) │                                     │
             │                        ├─────────────────────────────────────┘ (retry)
             │                        │ sufficient / max attempts
             │                        ▼
             │               ┌──────────────────┐
             └──────────────►│  synthesis_node  │
                             └────────┬─────────┘
                                      │
                                      ▼
                                    END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from agents import clarity_agent, research_agent, validator_agent, synthesis_agent
from config import CONFIDENCE_THRESHOLD, MAX_RESEARCH_ATTEMPTS


# ── Node names (constants keep things DRY) ─────────────────────────────────────

NODE_CLARITY   = "clarity"
NODE_RESEARCH  = "research"
NODE_VALIDATOR = "validator"
NODE_SYNTHESIS = "synthesis"


# ── Conditional routing functions ──────────────────────────────────────────────

def route_after_clarity(state: ResearchState) -> str:
    """
    After the Clarity Agent runs, decide where to go next.

    Returns:
        "__interrupt__" if the query needs clarification (triggers HITL interrupt),
        NODE_RESEARCH   if the query is clear.
    """
    if state.get("clarity_status") == "needs_clarification":
        return "__interrupt__"
    return NODE_RESEARCH


def route_after_research(state: ResearchState) -> str:
    """
    After the Research Agent runs, decide where to go next.

    High confidence → skip Validator, go straight to Synthesis.
    Low confidence  → pass through Validator first.

    Returns:
        NODE_SYNTHESIS if confidence_score >= CONFIDENCE_THRESHOLD,
        NODE_VALIDATOR otherwise.
    """
    confidence = state.get("confidence_score", 0.0)
    if confidence >= CONFIDENCE_THRESHOLD:
        return NODE_SYNTHESIS
    return NODE_VALIDATOR


def route_after_validator(state: ResearchState) -> str:
    """
    After the Validator Agent runs, decide where to go next.

    Sufficient OR max retries hit → Synthesis.
    Insufficient AND retries remain → back to Research.

    Returns:
        NODE_SYNTHESIS if findings are good enough or retries are exhausted,
        NODE_RESEARCH  to trigger another research attempt.
    """
    validation_result = state.get("validation_result", "sufficient")
    attempts          = state.get("research_attempts", 0)

    if validation_result == "sufficient" or attempts >= MAX_RESEARCH_ATTEMPTS:
        return NODE_SYNTHESIS

    return NODE_RESEARCH  # retry


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Uses MemorySaver as the checkpointer so conversation state persists
    across multiple .invoke() calls in the same thread.

    Returns:
        A compiled CompiledGraph ready to invoke.
    """
    builder = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node(NODE_CLARITY,   clarity_agent)
    builder.add_node(NODE_RESEARCH,  research_agent)
    builder.add_node(NODE_VALIDATOR, validator_agent)
    builder.add_node(NODE_SYNTHESIS, synthesis_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point(NODE_CLARITY)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Clarity → (interrupt | Research)
    builder.add_conditional_edges(
        NODE_CLARITY,
        route_after_clarity,
        {
            "__interrupt__": END,   # graph pauses; main.py handles re-entry
            NODE_RESEARCH:   NODE_RESEARCH,
        },
    )

    # Research → (Validator | Synthesis)
    builder.add_conditional_edges(
        NODE_RESEARCH,
        route_after_research,
        {
            NODE_VALIDATOR: NODE_VALIDATOR,
            NODE_SYNTHESIS: NODE_SYNTHESIS,
        },
    )

    # Validator → (Research retry | Synthesis)
    builder.add_conditional_edges(
        NODE_VALIDATOR,
        route_after_validator,
        {
            NODE_RESEARCH:  NODE_RESEARCH,
            NODE_SYNTHESIS: NODE_SYNTHESIS,
        },
    )

    # Synthesis → END (always)
    builder.add_edge(NODE_SYNTHESIS, END)

    # ── Compile with in-memory checkpointing ──────────────────────────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
