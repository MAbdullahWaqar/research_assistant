"""
main.py
-------
Interactive CLI for the multi-agent business research assistant.

Run with:
    python main.py

The session maintains full conversation history across turns. Type 'exit' to quit.

Human-in-the-loop flow:
  When the Clarity Agent flags an ambiguous query, it calls LangGraph's ``interrupt()``.
  The graph pauses; ``invoke_until_complete`` surfaces the payload, collects your reply,
  and resumes with ``Command(resume=...)`` so execution continues inside the same run.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import HumanMessage

from config import validate_config
from graph import build_graph
from observability import log_graph_turn, setup_application_logging
from state import ResearchState
from utils import (
    console,
    show_welcome,
    show_separator,
    show_clarification_request,
    show_final_response,
    show_error,
    show_info,
    invoke_until_complete,
)


def run_session() -> None:
    """
    Main REPL loop. Maintains a persistent ``thread_id`` so the LangGraph checkpointer
    (SQLite by default — see ``checkpointing.py``) keeps conversation state across turns.
    """
    setup_application_logging()
    log = logging.getLogger(__name__)

    try:
        validate_config()
    except RuntimeError as e:
        show_error(str(e))
        return

    graph = build_graph()

    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    show_welcome()
    show_info(f"Session ID: {thread_id[:8]}…")

    while True:
        show_separator()

        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            console.print("[dim]Goodbye! Have a great day.[/dim]")
            break

        initial_state = _build_initial_state(user_input)

        clarify_aborted = [False]

        def resume_handler(payload: object) -> str | None:
            clarification_msg = (
                payload.get("clarification_request", "Could you please clarify your question?")
                if isinstance(payload, dict)
                else str(payload)
            )
            show_clarification_request(clarification_msg)
            while True:
                try:
                    clarification = console.input("[bold yellow]Your answer:[/bold yellow] ").strip()
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Goodbye![/dim]")
                    clarify_aborted[0] = True
                    return None
                if clarification:
                    return clarification
                show_error("Please enter a non-empty answer (or press Ctrl+C to exit).")

        log_graph_turn(log, thread_id=thread_id, event="invoke_start", query_preview=user_input[:120])
        result = invoke_until_complete(graph, config, initial_state, resume_handler)

        if result is None:
            if clarify_aborted[0]:
                break
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            show_info(
                f"New session ID: {thread_id[:8]}… "
                "(checkpoint reset after incomplete clarification)"
            )
            continue

        if result.get("final_response"):
            log_graph_turn(
                log,
                thread_id=thread_id,
                event="invoke_complete",
                response_chars=len(result["final_response"]),
            )
            show_final_response(result["final_response"])
        else:
            log_graph_turn(log, thread_id=thread_id, event="invoke_no_final_response")
            show_error("No final response was produced. Please try again.")


def _build_initial_state(user_query: str) -> ResearchState:
    """
    Build input state for one graph invocation.

    ``messages`` carries the new HumanMessage; MemorySaver merges with prior history
    via the ``operator.add`` annotation on ``ResearchState.messages``.
    """
    return {
        "messages":              [HumanMessage(content=user_query)],
        "user_query":            user_query,
        "clarity_status":           "",
        "clarification_resolved":   False,
        "clarification_request":    "",
        "research_findings":     "",
        "confidence_score":      0.0,
        "research_attempts":     0,
        "validation_result":     "",
        "validation_notes":      "",
        "final_response":        "",
        "metadata":              {},
    }


if __name__ == "__main__":
    run_session()
