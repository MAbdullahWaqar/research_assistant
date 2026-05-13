"""
tools/search.py
---------------
Tavily-backed search for the Research Agent.

- ``TAVILY_TRANSPORT=sdk`` (default): official ``tavily-python`` client.
- ``TAVILY_TRANSPORT=mcp``: Tavily remote MCP (streamable HTTP via ``langchain-mcp-adapters``).
"""

from __future__ import annotations

import logging
from typing import Optional

from tavily import TavilyClient

from ..config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_TRANSPORT

logger = logging.getLogger(__name__)

_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    """Lazily initialise the Tavily SDK client."""
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set. Check your .env file.")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def _tavily_search_sdk(query: str, max_results: int) -> list[dict]:
    try:
        client = _get_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        return response.get("results", [])
    except Exception as exc:
        logger.warning("Tavily SDK search failed for query=%r: %s", query, exc, exc_info=False)
        return []


def tavily_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    """
    Run a Tavily search and return a list of result dicts (title, url, content, score).

    Transport is selected by ``TAVILY_TRANSPORT`` in ``config`` (from env).
    """
    transport = (TAVILY_TRANSPORT or "sdk").lower()
    if transport in {"mcp", "mcp_http", "remote_mcp"}:
        from .mcp_tavily import tavily_search_mcp

        return tavily_search_mcp(query, max_results=max_results)
    return _tavily_search_sdk(query, max_results)


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
        lines.append("")

    return "\n".join(lines)
