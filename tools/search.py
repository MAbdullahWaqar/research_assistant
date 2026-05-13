"""
tools/search.py
---------------
Wraps the Tavily Search API into a simple callable used by the Research Agent.

Tavily is purpose-built for LLM-friendly web search: it returns clean,
summarised snippets rather than raw HTML, making it ideal for RAG pipelines.
"""

from __future__ import annotations

from typing import Optional
from tavily import TavilyClient

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS


# Module-level client — instantiated once and reused across calls.
_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    """Lazily initialise the Tavily client."""
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set. Check your .env file.")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def tavily_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    """
    Run a Tavily search and return a list of result dicts.

    Each result dict contains:
        - 'title'   (str)  : Page title
        - 'url'     (str)  : Source URL
        - 'content' (str)  : LLM-friendly snippet / summary
        - 'score'   (float): Tavily relevance score (0-1)

    Args:
        query:       The search query string.
        max_results: How many results to return (default from config).

    Returns:
        List of result dicts. Empty list on error.
    """
    try:
        client = _get_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",   # "advanced" gives richer snippets
            include_answer=True,        # Tavily generates a quick answer too
        )
        results = response.get("results", [])
        return results
    except Exception as exc:
        # Surface the error without crashing the agent loop.
        print(f"[Search Tool] Error during Tavily search: {exc}")
        return []


def format_search_results(results: list[dict]) -> str:
    """
    Convert raw Tavily results into a clean markdown string for the LLM.

    Args:
        results: List of dicts returned by tavily_search().

    Returns:
        Multi-line string with numbered sources, titles, URLs, and content.
    """
    if not results:
        return "No search results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title   = r.get("title", "Untitled")
        url     = r.get("url", "")
        content = r.get("content", "").strip()
        lines.append(f"**[{i}] {title}**")
        lines.append(f"Source: {url}")
        lines.append(content)
        lines.append("")   # blank line between entries

    return "\n".join(lines)
