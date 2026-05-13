"""
agents/clarity.py
-----------------
Clarity Agent — the first node in the graph.

Responsibilities:
  • Determine whether the user's query is specific enough to research.
  • Check that a company name (or clear reference via conversation history) exists.
  • If unclear: ``interrupt()`` pauses the graph; on resume, merge the reply into
    ``user_query``, set ``clarity_status`` to ``needs_clarification`` (LLM verdict on
    the original query), ``clarification_resolved`` to True, and keep the prompt in
    ``clarification_request`` for the audit trail.
  • If already clear: ``clarity_status`` is ``clear``, ``clarification_resolved`` False.

Routing (handled in graph.py):
  Clarity Agent → Research Agent (interrupt blocks inside the node until resumed).
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import interrupt

from llm import get_chat_llm
from state import ResearchState
from utils import show_agent_start, show_agent_result


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Clarity Agent in a business research pipeline.

Your job is to decide whether the user's query is clear enough to research.

A query is CLEAR when:
1. A specific company name is mentioned (e.g. "Apple", "Tesla", "OpenAI"), OR
2. The conversation history makes it obvious which company is being discussed
   (e.g. the user previously asked about Apple and now says "What about their CEO?").
3. The question has a researchable intent (financials, news, competitors, products, etc.).

A query NEEDS CLARIFICATION when:
1. No company is named and conversation history doesn't make it clear.
2. The query is so vague that meaningful research is impossible
   (e.g. "tell me about a company" with no prior context).

IMPORTANT: Be generous. If history makes the company obvious, mark it as clear.
Follow-up questions like "What about their competitors?" or "Tell me more" are CLEAR
if the previous conversation established a company.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{
  "clarity_status": "clear" | "needs_clarification",
  "clarification_request": "<question to ask user, or empty string if clear>",
  "reasoning": "<brief internal reasoning>"
}
"""


def clarity_agent(state: ResearchState) -> dict:
    """
    Evaluate the user's current query for clarity.

    Args:
        state: The current graph state.

    Returns:
        Partial state updates including ``clarity_status`` per assignment (``clear`` or
        ``needs_clarification``), ``clarification_resolved``, and appended messages.
    """
    show_agent_start("Clarity Agent", "🔍")

    # Build conversation context so the agent can use history for follow-ups
    history_text = _build_history_text(state.get("messages", []))

    llm = get_chat_llm()

    prompt_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Conversation history so far:\n{history_text}\n\n"
                f"Current user query: {state['user_query']}\n\n"
                "Is this query clear enough to research? Respond with JSON only."
            )
        ),
    ]

    response = llm.invoke(prompt_messages)
    raw = response.content.strip()

    # Parse the structured JSON output from the LLM
    parsed = _parse_json_response(raw)

    clarity_status       = parsed.get("clarity_status", "needs_clarification")
    clarification_request = parsed.get("clarification_request", "Could you please specify which company you're asking about?")
    reasoning            = parsed.get("reasoning", "")

    show_agent_result("Clarity Agent", "status", clarity_status)
    if reasoning:
        show_agent_result("Clarity Agent", "reasoning", reasoning)

    # ── Ambiguous query: native LangGraph HITL (interrupt → Command(resume=...)) ──
    if clarity_status == "needs_clarification":
        payload = {
            "type": "clarity_hitl",
            "clarification_request": clarification_request,
            "reasoning": reasoning,
        }
        user_answer = interrupt(payload)
        answer_str = str(user_answer).strip() if user_answer is not None else ""
        if not answer_str:
            answer_str = "(no additional detail provided)"
        enriched_query = f"{state['user_query']} — Additional context: {answer_str}"
        show_agent_result("Clarity Agent", "resolved with context", answer_str[:80] + ("…" if len(answer_str) > 80 else ""))
        return {
            "user_query": enriched_query,
            "clarity_status": "needs_clarification",
            "clarification_resolved": True,
            "clarification_request": clarification_request,
            "messages": [
                AIMessage(
                    content=(
                        f"[Clarity Agent] Requested clarification: {clarification_request}"
                    )
                ),
                HumanMessage(content=answer_str),
            ],
        }

    return {
        "clarity_status": "clear",
        "clarification_resolved": False,
        "clarification_request": "",
        "messages": [
            AIMessage(
                content=f"[Clarity Agent] Status: clear. {reasoning or 'Proceeding to research.'}"
            )
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history_text(messages: list) -> str:
    """Flatten message history into a readable string for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages[-10:]:   # cap at last 10 messages to stay within token limits
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Safely parse JSON from the LLM response.
    Falls back to a safe default if parsing fails.
    """
    try:
        # Strip accidental markdown fences the model might add
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        # If the LLM misbehaves, default to asking for clarification
        return {
            "clarity_status": "needs_clarification",
            "clarification_request": "Could you clarify which company you're asking about?",
            "reasoning": "Failed to parse LLM response; defaulting to clarification.",
        }
