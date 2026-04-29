from __future__ import annotations

import asyncio
import sqlite3

from opentulpa.tasks.service import TaskService


def test_task_service_connection_pragmas(tmp_path):
    db_path = tmp_path / "tasks.db"
    service = TaskService(db_path)

    with service._conn() as conn:
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])

    assert journal_mode == "wal"
    assert busy_timeout >= 10_000


def test_task_service_concurrent_creates_do_not_lock(tmp_path):
    db_path = tmp_path / "tasks.db"
    service = TaskService(db_path)

    async def _run():
        async def create_one(i: int):
            return await service.create_task(
                customer_id="cust",
                goal=f"goal-{i}",
                payload={"n": i},
                idempotency_key=f"k-{i}",
            )

        results = await asyncio.gather(*(create_one(i) for i in range(40)))
        assert len(results) == 40

    asyncio.run(_run())

    with sqlite3.connect(db_path) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
    assert count == 40
