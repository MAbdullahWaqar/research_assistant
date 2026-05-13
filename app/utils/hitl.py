"""
utils/hitl.py
-------------
Helpers for LangGraph human-in-the-loop flows using interrupt() / Command(resume=).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.types import Command

from ..state import ResearchState

_log = logging.getLogger(__name__)


def invoke_until_complete(
    graph: Any,
    config: dict,
    initial_state: ResearchState,
    resume_handler: Callable[[Any], str | None],
) -> dict | None:
    """
    Run the graph until it finishes without a pending interrupt.

    When the graph returns ``__interrupt__`` (from ``langgraph.types.interrupt`` inside
    a node), ``resume_handler`` is called with the interrupt payload. It must return
    a non-empty string to pass to ``Command(resume=...)``, or ``None`` to abort.

    If ``None`` is returned from ``resume_handler``, this function returns ``None``.
    The caller should not re-use the same ``thread_id`` without resuming or resetting
    the checkpointer, or the next ``invoke`` may see a stuck interrupt.

    Args:
        graph:          Compiled LangGraph.
        config:         Runnable config (must include ``thread_id`` for checkpointing).
        initial_state:  First input state for ``invoke``.
        resume_handler: ``(payload) -> str | None`` — payload is JSON-serializable.

    Returns:
        Final state dict, or ``None`` if the user aborts during an interrupt.
    """
    thread_id = str(config.get("configurable", {}).get("thread_id", ""))
    _log.info("graph.invoke start thread_id=%s", thread_id)
    result: dict = graph.invoke(initial_state, config=config)

    while True:
        interrupt_list = result.get("__interrupt__")
        if not interrupt_list:
            _log.info("graph.invoke complete thread_id=%s", thread_id)
            return result

        payload = interrupt_list[0].value
        _log.info("graph.__interrupt__ thread_id=%s payload_type=%s", thread_id, type(payload).__name__)
        answer = resume_handler(payload)
        if answer is None:
            _log.warning("graph.resume aborted thread_id=%s", thread_id)
            return None

        _log.info("graph.Command.resume thread_id=%s", thread_id)
        result = graph.invoke(Command(resume=answer), config=config)
