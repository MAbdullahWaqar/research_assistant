"""
agents/synthesis.py
-------------------
Synthesis Agent — the final node in the graph.

Responsibilities:
  • Consume research_findings (and validation_notes if available).
  • Leverage full conversation history to handle follow-up questions naturally.
  • Generate a clean, well-structured markdown response for the user.
  • Store the response in final_response and as the last AIMessage in history.

Routing: always → END
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from state import ResearchState
from utils import show_agent_start, show_agent_result


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Synthesis Agent — the final step in a business research pipeline.

Your job is to produce a polished, user-friendly response based on the research findings provided.

Guidelines:
1. Write in clear, professional language accessible to a business audience.
2. Use markdown formatting: headers (##), bullet points, bold key terms.
3. Structure the response logically — lead with the most relevant information.
4. If the research covers multiple topics, use sections with ## headers.
5. Acknowledge any gaps honestly (e.g. "Pricing details were not publicly available").
6. Keep it concise but complete — aim for quality over quantity.
7. Reference conversation history to make follow-up answers feel natural and connected.
8. End with a brief note like "Sources: [1] ... [2] ..." if sources are available.
9. Do NOT invent data or speculate beyond what the research provides.

Tone: informative, confident, helpful — like a knowledgeable analyst briefing an executive.
"""


def synthesis_agent(state: ResearchState) -> dict:
    """
    Generate the final user-facing response from compiled research.

    Args:
        state: The current graph state.

    Returns:
        Partial state dict with final_response and the appended AIMessage.
    """
    show_agent_start("Synthesis Agent", "✍️")

    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.2,          # slightly higher temperature for more natural prose
        max_tokens=LLM_MAX_TOKENS,
    )

    history_text = _build_history_text(state.get("messages", []))
    validation_notes = state.get("validation_notes", "")
    metadata = state.get("metadata", {})
    sources = metadata.get("sources", [])

    # Build context-rich prompt
    prompt_parts = [
        f"User's question: {state['user_query']}",
        f"\nConversation history:\n{history_text}",
        f"\nResearch findings:\n{state.get('research_findings', 'No findings available.')}",
    ]

    if validation_notes:
        prompt_parts.append(f"\nValidator notes: {validation_notes}")

    if sources:
        sources_text = "\n".join(f"- {url}" for url in sources[:8])
        prompt_parts.append(f"\nSources used:\n{sources_text}")

    prompt_parts.append(
        "\nWrite a clear, well-structured markdown response for the user. "
        "Do not include raw JSON. Respond with the final answer only."
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(prompt_parts)),
    ])

    final_response = response.content.strip()
    show_agent_result("Synthesis Agent", "response_length", f"{len(final_response)} chars")

    return {
        "final_response": final_response,
        # The AIMessage here becomes part of conversation history for future turns
        "messages": [AIMessage(content=final_response)],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history_text(messages: list) -> str:
    """Flatten message history into a readable string for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages[-12:]:   # last 12 messages for broader context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate very long messages (e.g. previous synthesis outputs) to keep prompt lean
        content = msg.content
        if len(content) > 600:
            content = content[:600] + "... [truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
