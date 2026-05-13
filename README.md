# Multi-Agent Business Research Assistant

A production-grade LangGraph multi-agent system that helps users research businesses using specialized, collaborating AI agents.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Clarity Agent  │──── needs_clarification ──► Human Interrupt ──► (re-enter)
└────────┬────────┘
         │ clear
         ▼
┌─────────────────┐
│ Research Agent  │◄─────────────────────────────────────────────┐
└────────┬────────┘                                               │
         │                                                        │
   confidence < 6                                          insufficient
         │             confidence ≥ 6                     (attempts < 3)
         ▼                    │                                   │
┌─────────────────┐           ▼                      ┌───────────┴──────────┐
│ Validator Agent │──────────────────────────────────►   Validator Agent    │
└─────────────────┘     sufficient / max attempts    └──────────────────────┘
         │
         ▼
┌─────────────────┐
│ Synthesis Agent │
└────────┬────────┘
         │
         ▼
      Response
```

## Agents

| Agent | Role | Output |
|-------|------|--------|
| **Clarity Agent** | Checks if the query is specific enough and has a company name | `clarity_status`: `clear` or `needs_clarification` |
| **Research Agent** | Searches the web for business data via Tavily | `research_findings` + `confidence_score` (0–10) |
| **Validator Agent** | Evaluates research quality and completeness | `validation_result`: `sufficient` or `insufficient` |
| **Synthesis Agent** | Generates a clean, structured final answer | Final markdown response |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required keys:
- `GROQ_API_KEY` — from [console.groq.com/keys](https://console.groq.com/keys)
- `TAVILY_API_KEY` — from [tavily.com](https://tavily.com) (free tier available)

### 3. Run

```bash
# Interactive CLI (recommended for demo)
python main.py

# Or run the quick demo with preset queries
python demo.py
```

## Project Structure

```
research_assistant/
├── main.py              # Interactive CLI entry point
├── demo.py              # Demo script with preset queries
├── graph.py             # LangGraph graph definition & compilation
├── state.py             # Shared state schema (TypedDict)
├── config.py            # Configuration (model, settings)
├── llm.py               # Shared Groq chat model factory
├── agents/
│   ├── __init__.py
│   ├── clarity.py       # Clarity Agent
│   ├── research.py      # Research Agent
│   ├── validator.py     # Validator Agent
│   └── synthesis.py     # Synthesis Agent
├── tools/
│   ├── __init__.py
│   └── search.py        # Tavily search tool wrapper
├── utils/
│   ├── __init__.py
│   └── display.py       # Rich terminal display helpers
├── requirements.txt
├── .env.example
└── README.md
```

## Features

- **Multi-turn conversation** — full history maintained across queries
- **Human-in-the-loop** — interrupts when queries are ambiguous
- **Retry loop** — Validator can send Research Agent back for better data (max 3 attempts)
- **Confidence scoring** — Research Agent self-scores its findings 0–10
- **Rich terminal UI** — colored output with agent status indicators
- **Graceful fallback** — max retries hit → proceeds to Synthesis anyway

## Example Queries

```
You: Tell me about Apple
You: What about their competitors?          ← follow-up using history
You: Who is the CEO of the company?         ← ambiguous (triggers clarification)
You: What are Tesla's latest financials?
You: exit
```
