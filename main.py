"""
main.py
-------
Interactive CLI for the multi-agent business research assistant.

Run with:
    python main.py

The session maintains full conversation history across turns. Type 'exit' to quit.

Human-in-the-loop flow:
  When the Clarity Agent flags a query as ambiguous, the graph routes to END
  (via __interrupt__ → END edge). main.py detects clarity_status == "needs_clarification",
  displays the clarification prompt, collects the user's answer, prepends it to the
  query, and re-invokes the graph — cleanly simulating a HITL interrupt/resume cycle.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from config import validate_config
from graph import build_graph
from state import ResearchState
from utils import (
    console,
    show_welcome,
    show_separator,
    show_clarification_request,
    show_final_response,
    show_error,
    show_info,
)


def run_session() -> None:
    """
    Main REPL loop. Maintains a persistent thread_id so MemorySaver keeps
    conversation state across invocations.
    """
    # Validate API keys before doing anything
    try:
        validate_config()
    except RuntimeError as e:
        show_error(str(e))
        return

    # Build the compiled LangGraph
    graph = build_graph()

    # Each session gets a unique thread ID for MemorySaver's checkpointing
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    show_welcome()
    show_info(f"Session ID: {thread_id[:8]}…")

    while True:
        show_separator()

        # ── Get user input ────────────────────────────────────────────────────
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

        # ── Invoke the graph ──────────────────────────────────────────────────
        result = _invoke_graph(graph, config, user_input)

        if result is None:
            continue

        # ── Human-in-the-loop: handle clarification requests ─────────────────
        max_clarification_rounds = 3
        clarification_rounds     = 0

        while (
            result.get("clarity_status") == "needs_clarification"
            and clarification_rounds < max_clarification_rounds
        ):
            clarification_rounds += 1
            clarification_msg = result.get(
                "clarification_request",
                "Could you please clarify your question?"
            )

            show_clarification_request(clarification_msg)

            # Collect clarification from the user
            try:
                clarification = console.input("[bold yellow]Your answer:[/bold yellow] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                return

            if not clarification:
                continue

            # Merge the clarification into the original query and re-run
            enriched_query = f"{user_input} — Additional context: {clarification}"
            show_info(f"Re-running with enriched query:{enriched_query}")
            result = _invoke_graph(graph, config, enriched_query)

            if result is None:
                break

        # ── Display the final synthesised response ────────────────────────────
        if result and result.get("final_response"):
            show_final_response(result["final_response"])
        elif result and result.get("clarity_status") == "needs_clarification":
            show_error("Could not resolve ambiguity after multiple attempts. Please try again with a specific company name.")


def _invoke_graph(graph, config: dict, user_query: str) -> dict | None:
    """
    Prepare the initial state and invoke the graph for one turn.

    Args:
        graph:      The compiled LangGraph.
        config:     Thread config (contains thread_id for MemorySaver).
        user_query: The user's input string.

    Returns:
        The final state dict, or None on error.
    """
    # Build the initial state for this invocation.
    # 'messages' contains just the new HumanMessage; MemorySaver merges it with
    # existing history via the operator.add annotation on ResearchState.messages.
    initial_state: ResearchState = {
        "messages":             [HumanMessage(content=user_query)],
        "user_query":           user_query,
        "clarity_status":       "",
        "clarification_request": "",
        "research_findings":    "",
        "confidence_score":     0.0,
        "research_attempts":    0,
        "validation_result":    "",
        "validation_notes":     "",
        "final_response":       "",
        "metadata":             {},
    }

    try:
        result = graph.invoke(initial_state, config=config)
        return result
    except Exception as exc:
        show_error(f"Graph execution error: {exc}")
        console.print_exception(show_locals=False)
        return None


if __name__ == "__main__":
    run_session()
