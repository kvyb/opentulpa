from __future__ import annotations

import sqlite3

import pytest

from opentulpa.persistence.sqlite import connect_sqlite


def test_connect_sqlite_context_manager_closes_connection(tmp_path) -> None:
    with connect_sqlite(tmp_path / "store.db", wal=True) as conn:
        conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")
