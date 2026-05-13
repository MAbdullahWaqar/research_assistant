"""
agents/validator.py
-------------------
Validator Agent — quality-checks the Research Agent's findings.

Responsibilities:
  • Assess whether the research findings adequately answer the user's question.
  • Check for completeness, relevance, and factual grounding.
  • Output validation_result = "sufficient" | "insufficient".
  • If insufficient AND attempts < MAX_RESEARCH_ATTEMPTS, send back to Research Agent.
  • If max attempts reached, pass through to Synthesis regardless.

Routing (handled in graph.py):
  sufficient               → Synthesis Agent
  insufficient + attempts < 3 → Research Agent (retry)
  insufficient + attempts ≥ 3 → Synthesis Agent (max retries hit)
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import MAX_RESEARCH_ATTEMPTS
from llm import get_chat_llm
from state import ResearchState
from utils import show_agent_start, show_agent_result


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Validator Agent in a business research pipeline.

Your job is to critically evaluate whether the research findings are sufficient
to answer the user's original question.

Assess the findings on these dimensions:
1. Relevance    — Do the findings actually address what the user asked?
2. Completeness — Are key facts present (or explained as unavailable)?
3. Recency      — Is the data current enough to be useful?
4. Grounding    — Are claims backed by identifiable sources (not just vague assertions)?

Mark findings as SUFFICIENT if:
- The core question is answered (even if not exhaustively).
- At least some specific, credible data points are present.
- The user would get meaningful value from this response.

Mark findings as INSUFFICIENT if:
- The findings are mostly empty, vague, or off-topic.
- No company-specific data is present.
- The findings would mislead or confuse the user.

Be constructive — if insufficient, explain what's missing so the next research
attempt knows where to focus.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{
  "validation_result": "sufficient" | "insufficient",
  "validation_notes": "<brief explanation of your assessment>",
  "missing_elements": ["<what's missing, if insufficient>"]
}
"""


def validator_agent(state: ResearchState) -> dict:
    """
    Validate the quality and completeness of the research findings.

    Args:
        state: The current graph state.

    Returns:
        Partial state dict with validation_result and validation_notes.
    """
    show_agent_start("Validator Agent", "✅")

    attempts = state.get("research_attempts", 1)
    show_agent_result("Validator Agent", "research_attempts_so_far", str(attempts))

    # If we've hit the retry cap, auto-pass to avoid an infinite loop
    if attempts >= MAX_RESEARCH_ATTEMPTS:
        show_agent_result("Validator Agent", "result", "sufficient (max attempts reached)")
        return {
            "validation_result": "sufficient",
            "validation_notes": (
                f"Maximum research attempts ({MAX_RESEARCH_ATTEMPTS}) reached. "
                "Proceeding to synthesis with best available findings."
            ),
            "messages": [
                AIMessage(
                    content=(
                        f"[Validator Agent] Max retries reached ({attempts}). "
                        "Forcing synthesis with current findings."
                    )
                )
            ],
        }

    llm = get_chat_llm()

    history_text = _build_history_text(state.get("messages", []))

    validation_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Conversation history (for follow-up context):\n{history_text}\n\n"
                f"User's original question: {state['user_query']}\n\n"
                f"Research findings to evaluate:\n{state.get('research_findings', '')}\n\n"
                f"Confidence score self-reported by Research Agent: "
                f"{state.get('confidence_score', 'N/A')}/10\n\n"
                "Is this sufficient? Respond with JSON only."
            )
        ),
    ]

    response = llm.invoke(validation_messages)
    parsed = _parse_json_response(response.content.strip())

    validation_result = parsed.get("validation_result", "sufficient")
    validation_notes  = parsed.get("validation_notes", "")
    missing_elements  = parsed.get("missing_elements", [])

    show_agent_result("Validator Agent", "result", validation_result)
    show_agent_result("Validator Agent", "notes", validation_notes)
    if missing_elements:
        show_agent_result("Validator Agent", "missing", str(missing_elements))

    return {
        "validation_result": validation_result,
        "validation_notes": validation_notes,
        "messages": [
            AIMessage(
                content=(
                    f"[Validator Agent] Validation result: {validation_result}. "
                    f"{validation_notes}"
                )
            )
        ],
    }


def _build_history_text(messages: list) -> str:
    """Flatten message history into a readable string for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages[-10:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """Safely parse JSON; fall back to sufficient to avoid infinite loops."""
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {
            "validation_result": "sufficient",
            "validation_notes": "Could not parse validator response; defaulting to sufficient.",
            "missing_elements": [],
        }
