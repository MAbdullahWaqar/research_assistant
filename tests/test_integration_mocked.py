"""
End-to-end graph tests with mocked LLM and search (no network, no API keys).
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph
from state import ResearchState


class _StubChatModel:
    """Minimal chat model: returns queued AIMessage strings in order."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self._i = 0

    def invoke(self, messages, config=None, **kwargs):  # noqa: ANN001, ARG002
        if self._i >= len(self._contents):
            body = self._contents[-1]
        else:
            body = self._contents[self._i]
            self._i += 1
        return AIMessage(content=body)


def _patch_agent_llms(stub: _StubChatModel):
    """Each agent binds ``get_chat_llm`` at import time — patch every module."""
    return (
        patch("agents.clarity.get_chat_llm", return_value=stub),
        patch("agents.research.get_chat_llm", return_value=stub),
        patch("agents.validator.get_chat_llm", return_value=stub),
        patch("agents.synthesis.get_chat_llm", return_value=stub),
    )


def _fake_results() -> list[dict]:
    return [
        {
            "title": "Example Corp profile",
            "url": "https://example.com/news/1",
            "content": "Example Corp reported steady growth.",
            "score": 0.91,
        }
    ]


@pytest.fixture
def high_confidence_responses() -> list[str]:
    return [
        '{"clarity_status":"clear","clarification_request":"","reasoning":"ok"}',
        '["Example Corp overview 2025", "Example Corp financial news"]',
        (
            '{"research_findings":"## Overview\\nExample Corp shows steady growth.",'
            '"confidence_score":8.5,'
            '"search_queries_used":["q1"],'
            '"sources":["https://example.com/news/1"]}'
        ),
        "## Executive summary\nExample Corp is in good shape per mocked research.\n",
    ]


def test_graph_clear_query_high_confidence_skips_validator(
    high_confidence_responses: list[str],
) -> None:
    stub = _StubChatModel(high_confidence_responses)
    graph = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial: ResearchState = {
        "messages": [HumanMessage(content="Tell me about Example Corp")],
        "user_query": "Tell me about Example Corp",
        "clarity_status": "",
        "clarification_resolved": False,
        "clarification_request": "",
        "research_findings": "",
        "confidence_score": 0.0,
        "research_attempts": 0,
        "validation_result": "",
        "validation_notes": "",
        "final_response": "",
        "metadata": {},
    }

    with ExitStack() as stack:
        for ctx in _patch_agent_llms(stub):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("agents.research.tavily_search", side_effect=lambda q, **kw: _fake_results())
        )
        out = graph.invoke(initial, config=cfg)

    assert out.get("final_response")
    assert "Example Corp" in out["final_response"]
    assert out.get("confidence_score", 0) >= 6.0


def test_graph_low_confidence_runs_validator_then_synthesis() -> None:
    responses = [
        '{"clarity_status":"clear","clarification_request":"","reasoning":"ok"}',
        '["Example Corp risks"]',
        (
            '{"research_findings":"## Overview\\nThin data only.",'
            '"confidence_score":4.0,'
            '"search_queries_used":["q1"],'
            '"sources":["https://example.com/a"]}'
        ),
        (
            '{"validation_result":"sufficient",'
            '"validation_notes":"Adequate for a high-level answer.",'
            '"missing_elements":[]}'
        ),
        "## Summary\nLimited public data; answer reflects thin sources.\n",
    ]
    stub = _StubChatModel(responses)
    graph = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial: ResearchState = {
        "messages": [HumanMessage(content="Deep dive on Example Corp litigation")],
        "user_query": "Deep dive on Example Corp litigation",
        "clarity_status": "",
        "clarification_resolved": False,
        "clarification_request": "",
        "research_findings": "",
        "confidence_score": 0.0,
        "research_attempts": 0,
        "validation_result": "",
        "validation_notes": "",
        "final_response": "",
        "metadata": {},
    }

    with ExitStack() as stack:
        for ctx in _patch_agent_llms(stub):
            stack.enter_context(ctx)
        stack.enter_context(
            patch("agents.research.tavily_search", side_effect=lambda q, **kw: _fake_results())
        )
        out = graph.invoke(initial, config=cfg)

    assert out.get("final_response")
    assert "Limited" in out["final_response"] or "Summary" in out["final_response"]
