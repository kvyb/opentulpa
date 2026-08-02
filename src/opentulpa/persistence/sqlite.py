from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Literal

SQLITE_CONNECT_TIMEOUT_SECONDS = 10.0
SQLITE_BUSY_TIMEOUT_MS = 10_000


class _ClosingConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
        return False


def connect_sqlite(
    db_path: Path,
    *,
    check_same_thread: bool = False,
    timeout_seconds: float = SQLITE_CONNECT_TIMEOUT_SECONDS,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
    row_factory: type[sqlite3.Row] | None = sqlite3.Row,
    synchronous_normal: bool = True,
    wal: bool = False,
) -> sqlite3.Connection:
    """Create a SQLite connection using OpenTulpa's runtime-safe defaults."""
    conn = sqlite3.connect(
        db_path,
        check_same_thread=check_same_thread,
        timeout=timeout_seconds,
        factory=_ClosingConnection,
    )
    if row_factory is not None:
        conn.row_factory = row_factory
    conn.execute(f"PRAGMA busy_timeout={max(1, int(busy_timeout_ms))}")
    if synchronous_normal:
        conn.execute("PRAGMA synchronous=NORMAL")
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    return conn
