"""
config.py
---------
Central configuration for the research assistant.
All tunable parameters live here so they're easy to find and change.
"""

import os
from dotenv import load_dotenv

# Load .env file (does nothing if already set in the environment)
load_dotenv()


# ── LLM settings ──────────────────────────────────────────────────────────────

# Model used by all agents (Groq model id — see https://console.groq.com/docs/models).
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature: low = deterministic/factual (good for agents), high = creative
LLM_TEMPERATURE = 0.1

# Max tokens per agent response (synthesis may need more)
LLM_MAX_TOKENS = 2048


# ── Research / retry settings ──────────────────────────────────────────────────

# Maximum number of times the Research Agent can be retried per user query
MAX_RESEARCH_ATTEMPTS = 3

# Confidence threshold: below this → route through Validator before Synthesis
CONFIDENCE_THRESHOLD = 6.0

# Number of Tavily search results to fetch per query
TAVILY_MAX_RESULTS = 5


# ── API keys (pulled from environment) ────────────────────────────────────────

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")


def validate_config() -> None:
    """
    Raises a clear RuntimeError if any required API key is missing.
    Call this once at startup before building the graph.
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your API keys."
        )
