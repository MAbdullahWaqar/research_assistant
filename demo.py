"""
demo.py
-------
Demonstrates the research assistant with a scripted set of queries.

This script simulates a realistic multi-turn session:
  Turn 1: Clear query → full pipeline
  Turn 2: Follow-up → uses conversation history
  Turn 3: Ambiguous query → triggers clarification flow
  Turn 4: Clear query after clarification → full pipeline

Run with:
    python demo.py
"""

from __future__ import annotations

import time
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


# ── Demo script ────────────────────────────────────────────────────────────────

DEMO_TURNS = [
    {
        "query":         "What are the latest news and financials for Apple Inc?",
        "clarification": None,   # None means no clarification needed
        "label":         "Turn 1 — Clear query (Apple)",
    },
    {
        "query":         "What about their main competitors?",
        "clarification": None,
        "label":         "Turn 2 — Follow-up using history",
    },
    {
        "query":         "Tell me about that company",
        "clarification": "I mean OpenAI",
        "label":         "Turn 3 — Ambiguous query → clarification → OpenAI",
    },
    {
        "query":         "What is Tesla's current stock price and recent news?",
        "clarification": None,
        "label":         "Turn 4 — Clear query (Tesla)",
    },
]


def run_demo() -> None:
    """Execute the scripted demo turns."""
    try:
        validate_config()
    except RuntimeError as e:
        show_error(str(e))
        return

    graph     = build_graph()
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    show_welcome()
    console.print("[bold magenta][ DEMO MODE — Scripted queries ][/bold magenta]\n")
    show_info(f"Session ID: {thread_id[:8]}…")

    for turn in DEMO_TURNS:
        show_separator()
        console.print(f"\n[bold magenta]📌 {turn['label']}[/bold magenta]")
        time.sleep(0.5)

        user_query = turn["query"]
        console.print(f"[bold cyan]You:[/bold cyan] {user_query}")

        result = _invoke_graph(graph, config, user_query)
        if result is None:
            continue

        # Handle clarification if the graph requests it
        if result.get("clarity_status") == "needs_clarification":
            clarification_msg = result.get(
                "clarification_request",
                "Could you please clarify your question?"
            )
            show_clarification_request(clarification_msg)

            # Use the scripted clarification answer
            scripted_answer = turn.get("clarification")
            if scripted_answer:
                console.print(f"[bold yellow]Your answer:[/bold yellow] {scripted_answer}")
                enriched_query = f"{user_query} — Additional context: {scripted_answer}"
                show_info(f"Re-running with: "{enriched_query}"")
                result = _invoke_graph(graph, config, enriched_query)
            else:
                console.print("[yellow]No scripted clarification for this turn — skipping.[/yellow]")
                continue

        if result and result.get("final_response"):
            show_final_response(result["final_response"])

        time.sleep(1)   # brief pause between turns for readability

    show_separator()
    console.print("\n[bold green]✅ Demo complete![/bold green]")
    console.print("[dim]Run 'python main.py' for an interactive session.[/dim]\n")


def _invoke_graph(graph, config: dict, user_query: str) -> dict | None:
    """Invoke the graph for one turn; return final state or None on error."""
    initial_state: ResearchState = {
        "messages":              [HumanMessage(content=user_query)],
        "user_query":            user_query,
        "clarity_status":        "",
        "clarification_request": "",
        "research_findings":     "",
        "confidence_score":      0.0,
        "research_attempts":     0,
        "validation_result":     "",
        "validation_notes":      "",
        "final_response":        "",
        "metadata":              {},
    }

    try:
        return graph.invoke(initial_state, config=config)
    except Exception as exc:
        show_error(f"Graph error: {exc}")
        console.print_exception(show_locals=False)
        return None


if __name__ == "__main__":
    run_demo()
