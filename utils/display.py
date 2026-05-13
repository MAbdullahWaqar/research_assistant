"""
utils/display.py
----------------
Rich-powered terminal helpers that give the CLI a polished, readable look.

All display logic is centralised here so the agent files stay clean.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich import print as rprint

console = Console()


# ── Agent status banners ───────────────────────────────────────────────────────

def show_agent_start(agent_name: str, icon: str = "🤖") -> None:
    """Print a coloured banner when an agent begins its work."""
    colours = {
        "Clarity Agent":    "cyan",
        "Research Agent":   "blue",
        "Validator Agent":  "yellow",
        "Synthesis Agent":  "green",
    }
    colour = colours.get(agent_name, "white")
    console.print(f"\n[{colour} bold]{icon} {agent_name} running...[/{colour} bold]")


def show_agent_result(agent_name: str, result_label: str, result_value: str) -> None:
    """Print a key → value result from an agent."""
    colours = {
        "Clarity Agent":    "cyan",
        "Research Agent":   "blue",
        "Validator Agent":  "yellow",
        "Synthesis Agent":  "green",
    }
    colour = colours.get(agent_name, "white")
    console.print(f"  [{colour}]↳ {result_label}:[/{colour}] {result_value}")


# ── User interaction ───────────────────────────────────────────────────────────

def show_clarification_request(message: str) -> None:
    """Display the clarification prompt inside a styled panel."""
    console.print(
        Panel(
            f"[bold yellow]{message}[/bold yellow]",
            title="[bold]🔍 Clarification Needed[/bold]",
            border_style="yellow",
        )
    )


def show_final_response(response: str) -> None:
    """Render the synthesis agent's markdown response in a green panel."""
    console.print(
        Panel(
            Markdown(response),
            title="[bold green]📋 Research Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def show_welcome() -> None:
    """Print the welcome banner at startup."""
    console.print(
        Panel(
            "[bold cyan]Multi-Agent Business Research Assistant[/bold cyan]\n"
            "[dim]Powered by LangGraph + Claude + Tavily[/dim]\n\n"
            "Ask me about any company — I'll research it for you.\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def show_separator() -> None:
    """Print a horizontal rule between conversation turns."""
    console.print(Rule(style="dim"))


def show_error(message: str) -> None:
    """Print a red error panel."""
    console.print(
        Panel(f"[red]{message}[/red]", title="[bold red]Error[/bold red]", border_style="red")
    )


def show_info(message: str) -> None:
    """Print a dim informational line."""
    console.print(f"[dim]ℹ  {message}[/dim]")
