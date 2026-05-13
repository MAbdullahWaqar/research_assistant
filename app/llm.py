"""
llm.py
------
Shared Groq chat model for all agents.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE


def get_chat_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Return a configured ChatGroq instance (same settings as config defaults unless overridden)."""
    kwargs: dict = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if GROQ_API_KEY:
        kwargs["api_key"] = GROQ_API_KEY
    return ChatGroq(**kwargs)
