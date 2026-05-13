"""
observability.py
----------------
Application-wide logging setup and lightweight graph invocation tracing.

For deeper traces (LLM spans, tool calls), enable LangSmith via environment
variables documented in README.md — LangChain reads them automatically.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_LOG_CONFIGURED = False


def setup_application_logging() -> None:
    """
    Configure root logging once (idempotent).

    ``LOG_LEVEL`` controls verbosity (default ``INFO``). Use ``JSON_LOGS=1`` for
    single-line JSON records suitable for log aggregators.
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    json_logs = os.environ.get("JSON_LOGS", "").strip() in {"1", "true", "yes"}

    if json_logs:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload: dict[str, Any] = {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, default=str)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Reduce noisy third-party loggers unless DEBUG
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    _LOG_CONFIGURED = True


def log_graph_turn(logger: logging.Logger, *, thread_id: str, event: str, **fields: Any) -> None:
    """Emit a structured breadcrumb for graph lifecycle (invoke / resume / complete)."""
    extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info("graph_turn | thread_id=%s | event=%s | %s", thread_id, event, extra)
