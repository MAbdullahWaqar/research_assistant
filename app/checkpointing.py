"""
checkpointing.py
----------------
LangGraph checkpointer factory: durable SQLite by default, in-memory for tests.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Return a checkpointer based on ``LANGGRAPH_CHECKPOINT_BACKEND``:

    - ``sqlite`` (default): ``langgraph.checkpoint.sqlite.SqliteSaver`` on disk.
      Path from ``LANGGRAPH_SQLITE_PATH`` (default ``.checkpoints/langgraph.sqlite``).
    - ``memory``: ephemeral ``MemorySaver`` (unit tests / scratch runs).

    The SQLite file's parent directory is created automatically.
    """
    backend = os.environ.get("LANGGRAPH_CHECKPOINT_BACKEND", "sqlite").strip().lower()

    if backend in {"memory", "mem", "inmemory"}:
        return MemorySaver()

    if backend != "sqlite":
        raise ValueError(
            f"Unknown LANGGRAPH_CHECKPOINT_BACKEND={backend!r}; use 'sqlite' or 'memory'."
        )

    from langgraph.checkpoint.sqlite import SqliteSaver

    raw_path = os.environ.get(
        "LANGGRAPH_SQLITE_PATH",
        ".checkpoints/langgraph.sqlite",
    )
    path = Path(raw_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
