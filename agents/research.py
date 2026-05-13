"""
agents/research.py
------------------
Research Agent — searches the web and compiles business intelligence.

Responsibilities:
  • Build targeted search queries from the user's query + conversation history.
  • Call the Tavily search tool to retrieve real-time data.
  • Synthesise raw results into structured research findings.
  • Self-assess a confidence_score (0–10) based on result quality.

Routing (handled in graph.py):
  confidence_score >= CONFIDENCE_THRESHOLD → Synthesis Agent (skip Validator)
  confidence_score <  CONFIDENCE_THRESHOLD → Validator Agent
"""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import CONFIDENCE_THRESHOLD
from llm import get_chat_llm
from state import ResearchState
from tools import tavily_search, format_search_results
from utils import show_agent_start, show_agent_result


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Research Agent in a business intelligence pipeline.

Given a user's question and search results about a company, your job is to:
1. Extract and organise the most relevant facts from the search results.
2. Structure findings into clear categories (Overview, Financials, News, Leadership, etc.)
   — only include categories for which you actually found data.
3. Assign a confidence_score (0–10) reflecting how complete and reliable your findings are:
   - 8-10: Excellent — multiple authoritative sources, rich detail
   - 6-7:  Good — adequate data to answer the question
   - 4-5:  Partial — some data but notable gaps
   - 0-3:  Poor — little to no useful data found

Always cite sources by number (e.g. [1], [2]) when referencing specific facts.

Respond ONLY with a JSON object — no markdown fences, no extra text:
{
  "research_findings": "<detailed markdown research report>",
  "confidence_score": <float 0-10>,
  "search_queries_used": ["<query1>", "<query2>"],
  "sources": ["<url1>", "<url2>"]
}
"""

_QUERY_BUILDER_PROMPT = """You are a search query expert.

Given a user's question and conversation history, generate 2-3 targeted search queries
that will find the best business information to answer the question.

Focus on recency (add "2025" or "2026" where appropriate) and specificity.

Respond ONLY with a JSON array of query strings:
["query 1", "query 2", "query 3"]
"""


def research_agent(state: ResearchState) -> dict:
    """
    Search the web and compile research findings about the queried company.

    Args:
        state: The current graph state.

    Returns:
        Partial state dict with research_findings, confidence_score, and
        an incremented research_attempts counter.
    """
    show_agent_start("Research Agent", "🔎")

    attempts = state.get("research_attempts", 0) + 1
    show_agent_result("Research Agent", "attempt", f"{attempts}/3")

    llm = get_chat_llm()

    # ── Step 1: Build smart search queries ────────────────────────────────────
    history_text = _build_history_text(state.get("messages", []))
    queries = _generate_search_queries(llm, state["user_query"], history_text)
    show_agent_result("Research Agent", "queries", str(queries))

    # ── Step 2: Execute searches via Tavily ───────────────────────────────────
    all_results: list[dict] = []
    for q in queries:
        results = tavily_search(q)
        all_results.extend(results)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    formatted_results = format_search_results(unique_results[:8])  # cap at 8 sources
    show_agent_result("Research Agent", "sources found", str(len(unique_results)))

    # ── Step 3: Ask the LLM to synthesise findings ────────────────────────────
    synthesis_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question: {state['user_query']}\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Search results:\n{formatted_results}\n\n"
                "Compile your research findings and assign a confidence score. "
                "Respond with JSON only."
            )
        ),
    ]

    response = llm.invoke(synthesis_messages)
    parsed = _parse_json_response(response.content.strip())

    research_findings = parsed.get("research_findings", "No findings available.")
    confidence_score  = float(parsed.get("confidence_score", 5.0))
    sources           = parsed.get("sources", [])

    show_agent_result("Research Agent", "confidence_score", f"{confidence_score:.1f}/10")

    return {
        "research_findings": research_findings,
        "confidence_score": confidence_score,
        "research_attempts": attempts,
        "metadata": {"sources": sources},
        "messages": [
            AIMessage(
                content=(
                    f"[Research Agent] Compiled findings with confidence score "
                    f"{confidence_score:.1f}/10 after searching {len(unique_results)} sources."
                )
            )
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_search_queries(llm: BaseChatModel, user_query: str, history_text: str) -> list[str]:
    """
    Ask the LLM to generate focused search queries based on the user's question.
    Falls back to a sensible default if parsing fails.
    """
    response = llm.invoke([
        SystemMessage(content=_QUERY_BUILDER_PROMPT),
        HumanMessage(
            content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Current question: {user_query}\n\n"
                "Generate 2-3 search queries. Respond with a JSON array only."
            )
        ),
    ])

    raw = response.content.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        queries = json.loads(clean)
        if isinstance(queries, list):
            return [str(q) for q in queries[:3]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: use the raw query directly
    return [user_query]


def _build_history_text(messages: list) -> str:
    """Flatten message history into a readable string for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages[-10:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """Safely parse JSON; fall back gracefully."""
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {
            "research_findings": raw,   # preserve whatever the model said
            "confidence_score": 4.0,
            "search_queries_used": [],
            "sources": [],
        }
