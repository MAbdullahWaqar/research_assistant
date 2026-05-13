"""
state.py
--------
Defines the shared state schema passed between all agents in the LangGraph graph.

Using TypedDict + Annotated with operator.add for message accumulation,
which is the LangGraph convention for append-only list fields.
"""

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """
    Central state object shared across all agents.

    Fields are updated by agents as the graph progresses.
    'messages' uses operator.add so each agent appends rather than overwrites.
    """

    # ── Conversation history ──────────────────────────────────────────────────
    # Accumulates all HumanMessage / AIMessage objects across the session.
    # operator.add means each update APPENDS to the list instead of replacing it.
    messages: Annotated[list[BaseMessage], operator.add]

    # ── User input ────────────────────────────────────────────────────────────
    # The raw query submitted by the user in the current turn.
    user_query: str

    # ── Clarity Agent outputs ─────────────────────────────────────────────────
    # "clear" | "needs_clarification" — the LLM's judgment on the user's phrasing.
    # After HITL, ambiguous turns keep "needs_clarification" here (assignment OUTPUT)
    # while clarification_resolved is True and user_query is enriched for Research.
    clarity_status: str

    # True once the user has supplied clarification via interrupt/resume.
    clarification_resolved: bool

    # Question shown to the user when clarification was needed (audit / UX trail).
    clarification_request: str

    # ── Research Agent outputs ────────────────────────────────────────────────
    # Full text of research findings compiled from search results
    research_findings: str

    # Self-assessed quality score (0–10); < 6 routes to Validator for review
    confidence_score: float

    # Number of research attempts this turn (used to cap the retry loop at 3)
    research_attempts: int

    # ── Validator Agent outputs ───────────────────────────────────────────────
    # "sufficient" → proceed to Synthesis
    # "insufficient" → send back to Research Agent (if attempts < 3)
    validation_result: str  # "sufficient" | "insufficient"

    # Human-readable justification from the Validator
    validation_notes: str

    # ── Synthesis Agent output ────────────────────────────────────────────────
    # The final formatted response shown to the user
    final_response: str

    # ── Misc ──────────────────────────────────────────────────────────────────
    # Stores any extra metadata agents want to surface (e.g., sources list)
    metadata: dict[str, Any]
