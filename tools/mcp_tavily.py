"""
tools/mcp_tavily.py
-------------------
Tavily search via the official remote MCP server (streamable HTTP).

Used when ``TAVILY_TRANSPORT=mcp``. Each call opens a short-lived MCP session
(connection-per-search); for highest throughput in production, prefer ``sdk`` mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

logger = logging.getLogger(__name__)


def _normalize_to_results(raw: Any) -> list[dict]:
    """Map MCP / LangChain tool output into the same shape as ``tavily_search`` SDK."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if "results" in raw and isinstance(raw["results"], list):
            return [x for x in raw["results"] if isinstance(x, dict)]
        if "title" in raw or "url" in raw:
            return [raw]
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("MCP tool returned non-JSON text; first 200 chars: %s", text[:200])
        return []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [x for x in data["results"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


async def _async_tavily_search_mcp(query: str, max_results: int) -> list[dict]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    base_url = os.environ.get("TAVILY_MCP_URL", "").strip()
    if not base_url:
        if not TAVILY_API_KEY:
            logger.error("TAVILY_MCP_URL or TAVILY_API_KEY is required for MCP transport")
            return []
        base_url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"

    client = MultiServerMCPClient(
        {
            "tavily": {
                "transport": "streamable_http",
                "url": base_url,
            }
        }
    )
    tools = await client.get_tools(server_name="tavily")
    search_tool = None
    for t in tools:
        name = t.name.lower()
        if "tavily" in name and "search" in name:
            search_tool = t
            break
    if search_tool is None and tools:
        search_tool = tools[0]
    if search_tool is None:
        logger.error("No MCP tools returned from Tavily server")
        return []

    try:
        raw = await search_tool.ainvoke(
            {
                "query": query,
                "max_results": max_results,
            }
        )
    except Exception as exc:
        logger.warning("MCP tavily_search failed: %s", exc, exc_info=False)
        return []

    return _normalize_to_results(raw)


def tavily_search_mcp(query: str, *, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    """Synchronous entry point used by ``tools.search.tavily_search``."""
    return asyncio.run(_async_tavily_search_mcp(query, max_results))
