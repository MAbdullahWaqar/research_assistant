"""
graph.py
--------
Defines and compiles the LangGraph StateGraph for the multi-agent research assistant.

Graph topology (simplified):

  START → Clarity → Research → (Validator or Synthesis, by confidence)
                ↑         │
                └─────────┘  Validator may retry Research (insufficient, attempts < cap)
                Validator or cap → Synthesis → END

Clarity uses ``interrupt()`` for human-in-the-loop when the query is ambiguous; the
client resumes with ``Command(resume=...)`` (see ``utils.hitl.invoke_until_complete``).
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END

from .checkpointing import get_checkpointer
from .state import ResearchState
from .agents import clarity_agent, research_agent, validator_agent, synthesis_agent
from .config import CONFIDENCE_THRESHOLD, MAX_RESEARCH_ATTEMPTS


# ── Node names (constants keep things DRY) ─────────────────────────────────────

NODE_CLARITY   = "clarity"
NODE_RESEARCH  = "research"
NODE_VALIDATOR = "validator"
NODE_SYNTHESIS = "synthesis"


# ── Conditional routing functions ──────────────────────────────────────────────

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

def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """
    Construct and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer. Defaults to ``get_checkpointer()``
            (SQLite on disk unless ``LANGGRAPH_CHECKPOINT_BACKEND=memory``). Pass
            ``MemorySaver()`` from tests for hermetic runs.
    """
    if checkpointer is None:
        checkpointer = get_checkpointer()
    builder = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node(NODE_CLARITY,   clarity_agent)
    builder.add_node(NODE_RESEARCH,  research_agent)
    builder.add_node(NODE_VALIDATOR, validator_agent)
    builder.add_node(NODE_SYNTHESIS, synthesis_agent)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point(NODE_CLARITY)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Clarity → Research (Clarity calls ``interrupt()`` internally when the query
    # is ambiguous; the graph pauses until the client invokes ``Command(resume=...)``.)
    builder.add_edge(NODE_CLARITY, NODE_RESEARCH)

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

    # ── Compile with checkpointer (SQLite by default; see checkpointing.py) ─
    return builder.compile(checkpointer=checkpointer)
