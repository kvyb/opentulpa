"""Durable task orchestration with event logs and selective wake policy."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.tasks.sandbox import (
    append_task_event_log,
    list_artifacts,
    run_terminal,
    task_artifact_dir,
    write_file,
)

logger = logging.getLogger(__name__)

WakeCallback = Callable[[dict[str, Any]], Awaitable[Any]]
TERMINAL_WAKE_EVENTS = {"done", "failed", "needs_input", "worker_stopped"}
DEBUG_LOG_PATH = Path(__file__).resolve().parents[3] / ".cursor" / "debug.log"


def _debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "tasks",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Debug logging must never break task execution.
        pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TaskService:
    """Background task runner with SQLite persistence."""

    def __init__(
        self,
        db_path: Path,
        wake_callback: WakeCallback | None = None,
    ) -> None:
        self.db_path = db_path
        self._wake_callback = wake_callback
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._heartbeats: dict[str, float] = {}
        self._watchdog: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_level TEXT NOT NULL DEFAULT 'low',
                    payload_json TEXT NOT NULL,
                    requires_user_input INTEGER NOT NULL DEFAULT 0,
                    final_summary TEXT,
                    idempotency_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_customer ON tasks(customer_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency
                    ON tasks(customer_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id, id ASC);

                CREATE TABLE IF NOT EXISTS task_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    result_status TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );
                """
            )

    async def start(self) -> None:
        """Recover interrupted tasks and start watchdog."""
        with self._conn() as conn:
            now = _utc_now()
            conn.execute(
                """
                UPDATE tasks
                SET status='interrupted', updated_at=?, final_summary=COALESCE(final_summary, 'Interrupted during restart')
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status='interrupted' ORDER BY updated_at DESC LIMIT 100"
            ).fetchall()
        for row in rows:
            await self._emit_event(
                task_id=row["id"],
                event_type="worker_stopped",
                payload={"reason": "service_restart"},
            )
        self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def shutdown(self) -> None:
        if self._watchdog:
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog
        for task in list(self._running_tasks.values()):
            task.cancel()
        self._running_tasks.clear()

    async def create_task(
        self,
        customer_id: str,
        goal: str,
        payload: dict[str, Any],
        *,
        risk_level: str = "low",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            existing_task, task_id = self._insert_task_if_absent(
                customer_id=customer_id,
                goal=goal,
                payload=payload,
                risk_level=risk_level,
                idempotency_key=idempotency_key,
            )
            if existing_task is not None:
                return existing_task

        await self._emit_event(task_id, "queued", {"goal": goal})
        # region agent log
        _debug_log(
            hypothesis_id="tasks",
            location="tasks/service.py:create_task",
            message="task_created",
            data={
                "task_id": task_id,
                "risk_level": risk_level,
                "step_count": len(payload.get("steps", []) if isinstance(payload, dict) else []),
                "has_tests": bool((payload or {}).get("tests"))
                if isinstance(payload, dict)
                else False,
            },
        )
        # endregion
        self._spawn_runner(task_id)
        return self.get_task(task_id)

    def _insert_task_if_absent(
        self,
        *,
        customer_id: str,
        goal: str,
        payload: dict[str, Any],
        risk_level: str,
        idempotency_key: str | None,
    ) -> tuple[dict[str, Any] | None, str]:
        with self._conn() as conn:
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM tasks WHERE customer_id=? AND idempotency_key=?",
                    (customer_id, idempotency_key),
                ).fetchone()
                if existing:
                    return self._row_to_task(existing), str(existing["id"])

            task_id = new_short_id("task")
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO tasks (
                    id, customer_id, goal, status, risk_level, payload_json,
                    requires_user_input, final_summary, idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, 0, NULL, ?, ?, ?)
                """,
                (
                    task_id,
                    customer_id,
                    goal,
                    risk_level,
                    json.dumps(payload),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_runs (task_id, attempt_no, trigger_reason, result_status, created_at, finished_at)
                VALUES (?, 1, 'initial', NULL, ?, NULL)
                """,
                (task_id, now),
            )
            conn.commit()
        return None, task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError("task not found")
            return self._row_to_task(row)

    def list_events(self, task_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, event_type, payload_json, created_at
                FROM task_events
                WHERE task_id=?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (task_id, max(1, min(limit, 500)), max(0, offset)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }
            )
        return out

    def list_task_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        return list_artifacts(task_id)

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        task = self._running_tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            await self._emit_event(task_id, "cancel_requested", {})
        with self._conn() as conn:
            now = _utc_now()
            conn.execute(
                "UPDATE tasks SET status='cancelled', updated_at=?, final_summary=? WHERE id=?",
                (now, "Cancelled by user", task_id),
            )
            conn.commit()
        await self._emit_event(task_id, "cancelled", {})
        return self.get_task(task_id)

    async def relaunch_task(
        self,
        task_id: str,
        *,
        trigger_reason: str,
        clarification: str | None = None,
    ) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError("task not found")
            payload = json.loads(row["payload_json"])
            if clarification:
                payload["clarification"] = clarification
            attempt_no = (
                conn.execute(
                    "SELECT COALESCE(MAX(attempt_no), 0) FROM task_runs WHERE task_id=?",
                    (task_id,),
                ).fetchone()[0]
                + 1
            )
            now = _utc_now()
            conn.execute(
                """
                UPDATE tasks
                SET status='queued', payload_json=?, requires_user_input=0, final_summary=NULL, updated_at=?
                WHERE id=?
                """,
                (json.dumps(payload), now, task_id),
            )
            conn.execute(
                """
                INSERT INTO task_runs (task_id, attempt_no, trigger_reason, result_status, created_at, finished_at)
                VALUES (?, ?, ?, NULL, ?, NULL)
                """,
                (task_id, attempt_no, trigger_reason, now),
            )
            conn.commit()
        await self._emit_event(task_id, "relaunch", {"trigger_reason": trigger_reason})
        self._spawn_runner(task_id)
        return self.get_task(task_id)

    def _spawn_runner(self, task_id: str) -> None:
        if task_id in self._running_tasks and not self._running_tasks[task_id].done():
            return
        runner = asyncio.create_task(self._run_task(task_id))
        self._running_tasks[task_id] = runner

    async def _run_task(self, task_id: str) -> None:
        self._heartbeats[task_id] = time.monotonic()
        try:
            task = self.get_task(task_id)
            payload = task["payload"]
            steps = payload.get("steps", [])
            # region agent log
            _debug_log(
                hypothesis_id="tasks",
                location="tasks/service.py:_run_task",
                message="task_runner_started",
                data={"task_id": task_id, "step_count": len(steps), "status": task.get("status")},
            )
            # endregion

            await self._set_status(task_id, "running")
            await self._emit_event(task_id, "running", {"step_count": len(steps)})
            artifact_dir = task_artifact_dir(task_id)
            if not await self._run_task_steps(task_id=task_id, steps=steps, artifact_dir=artifact_dir):
                return
            tests = payload.get("tests", [])
            if not await self._run_task_tests(task_id=task_id, tests=tests, artifact_dir=artifact_dir):
                return

            artifacts = list_artifacts(task_id)
            await self._set_status(task_id, "done", summary="Task finished successfully.")
            # region agent log
            _debug_log(
                hypothesis_id="tasks",
                location="tasks/service.py:_run_task",
                message="task_runner_done",
                data={"task_id": task_id, "artifact_count": len(artifacts)},
            )
            # endregion
            await self._emit_event(
                task_id,
                "done",
                {
                    "artifact_count": len(artifacts),
                    "artifacts": artifacts[:20],
                },
            )
        except asyncio.CancelledError:
            await self._set_status(task_id, "cancelled", summary="Task cancelled.")
            await self._emit_event(task_id, "cancelled", {})
            raise
        except Exception as exc:  # pragma: no cover - safety guard
            logger.exception("Task %s worker crashed: %s", task_id, exc)
            await self._set_status(task_id, "failed", summary=f"Worker crashed: {exc}")
            await self._emit_event(task_id, "worker_stopped", {"error": str(exc)})
        finally:
            self._heartbeats.pop(task_id, None)
            self._running_tasks.pop(task_id, None)

    async def _run_task_steps(self, *, task_id: str, steps: Any, artifact_dir: Any) -> bool:
        if not steps:
            await self._set_status(task_id, "needs_input", requires_user_input=True)
            await self._emit_event(
                task_id,
                "needs_input",
                {"reason": "No executable steps provided in payload."},
            )
            return False
        for idx, step in enumerate(steps):
            if not await self._run_one_task_step(
                task_id=task_id,
                index=idx,
                step=step,
                artifact_dir=artifact_dir,
            ):
                return False
        return True

    async def _run_one_task_step(
        self,
        *,
        task_id: str,
        index: int,
        step: Any,
        artifact_dir: Any,
    ) -> bool:
        self._heartbeats[task_id] = time.monotonic()
        await self._emit_event(task_id, "step_started", {"index": index, "step": step})
        if not isinstance(step, dict):
            await self._handle_invalid_step(task_id=task_id, index=index, step=step)
            return False
        step_type = step.get("type")
        if step_type == "write_file":
            target = write_file(step.get("path", ""), step.get("content", ""))
            await self._emit_event(task_id, "step_done", {"index": index, "type": step_type, "path": str(target)})
            return True
        if step_type == "run_terminal":
            return await self._run_terminal_step(
                task_id=task_id,
                index=index,
                step=step,
                artifact_dir=artifact_dir,
            )
        if step_type == "sleep":
            await asyncio.sleep(float(step.get("seconds", 1)))
            await self._emit_event(task_id, "step_done", {"index": index, "type": step_type})
            return True
        if step_type == "reload_tulpa":
            await self._emit_event(
                task_id,
                "step_done",
                {"index": index, "type": step_type, "note": "Call tulpa_reload tool now."},
            )
            return True
        await self._handle_failure(task_id, "unknown_step_type", {"step": step})
        return False

    async def _handle_invalid_step(self, *, task_id: str, index: int, step: Any) -> None:
        _debug_log(
            hypothesis_id="tasks",
            location="tasks/service.py:_run_task",
            message="invalid_step_schema",
            data={"task_id": task_id, "index": index, "received_type": type(step).__name__},
        )
        await self._set_status(
            task_id,
            "needs_input",
            requires_user_input=True,
            summary="Task steps must be objects with a 'type' field.",
        )
        await self._emit_event(
            task_id,
            "needs_input",
            {
                "reason": "Invalid task step schema.",
                "index": index,
                "expected": "step object like {'type': 'run_terminal', ...}",
                "received_type": type(step).__name__,
            },
        )

    async def _run_terminal_step(
        self,
        *,
        task_id: str,
        index: int,
        step: dict[str, Any],
        artifact_dir: Any,
    ) -> bool:
        result = run_terminal(
            command=step.get("command", ""),
            working_dir=step.get("working_dir", "tulpa_stuff"),
            timeout_seconds=int(step.get("timeout_seconds", 90)),
            extra_env={"TASK_ARTIFACT_DIR": str(artifact_dir)},
        )
        await self._emit_event(task_id, "step_done", _terminal_step_payload(index=index, result=result))
        if result["ok"]:
            return True
        await self._handle_failure(task_id, "command_failed", result)
        return False

    async def _run_task_tests(self, *, task_id: str, tests: Any, artifact_dir: Any) -> bool:
        for test in tests:
            result = run_terminal(
                command=test.get("command", ""),
                working_dir=test.get("working_dir", "tulpa_stuff"),
                timeout_seconds=int(test.get("timeout_seconds", 90)),
                extra_env={"TASK_ARTIFACT_DIR": str(artifact_dir)},
            )
            await self._emit_event(task_id, "test_result", _test_result_payload(test=test, result=result))
            if not result["ok"]:
                await self._handle_failure(task_id, "test_failed", result)
                return False
        return True

    async def _handle_failure(self, task_id: str, reason: str, data: dict[str, Any]) -> None:
        task = self.get_task(task_id)
        payload = task["payload"]
        auto_mode = payload.get("auto_mode", "auto_low_risk")
        max_retries = int(payload.get("max_retries", 1))
        retries_used = int(payload.get("retries_used", 0))

        retryable = reason in {"command_failed", "test_failed"} and isinstance(
            data.get("returncode"), int
        )
        # region agent log
        _debug_log(
            hypothesis_id="tasks",
            location="tasks/service.py:_handle_failure",
            message="task_failure_decision",
            data={
                "task_id": task_id,
                "reason": reason,
                "returncode": data.get("returncode"),
                "retryable": retryable,
                "retries_used": retries_used,
                "max_retries": max_retries,
                "auto_mode": auto_mode,
            },
        )
        # endregion
        if auto_mode == "auto_low_risk" and retryable and retries_used < max_retries:
            payload["retries_used"] = retries_used + 1
            with self._conn() as conn:
                conn.execute(
                    "UPDATE tasks SET payload_json=?, updated_at=? WHERE id=?",
                    (json.dumps(payload), _utc_now(), task_id),
                )
                conn.commit()
            await self._emit_event(
                task_id,
                "retry_scheduled",
                {
                    "reason": reason,
                    "retry_no": retries_used + 1,
                    "max_retries": max_retries,
                    "last_returncode": data.get("returncode"),
                },
            )
            # Allow immediate respawn after this run exits.
            self._running_tasks.pop(task_id, None)
            await self.relaunch_task(task_id, trigger_reason=f"auto_retry:{reason}")
            return

        await self._set_status(task_id, "failed", summary=f"Task failed: {reason}")
        await self._emit_event(task_id, "failed", {"reason": reason, "details": data})

    async def _set_status(
        self,
        task_id: str,
        status: str,
        *,
        requires_user_input: bool = False,
        summary: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status=?, requires_user_input=?, final_summary=COALESCE(?, final_summary), updated_at=?
                WHERE id=?
                """,
                (status, 1 if requires_user_input else 0, summary, _utc_now(), task_id),
            )
            conn.execute(
                """
                UPDATE task_runs
                SET result_status=?, finished_at=?
                WHERE task_id=? AND finished_at IS NULL
                """,
                (status, _utc_now(), task_id),
            )
            conn.commit()

    async def _emit_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
            "created_at": _utc_now(),
        }
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO task_events (task_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, event_type, json.dumps(payload), event["created_at"]),
            )
            conn.commit()
        try:
            append_task_event_log(task_id, event)
        except Exception:
            logger.exception("Failed to append task event log for %s", task_id)

        if self._wake_callback and event_type in TERMINAL_WAKE_EVENTS:
            customer_id: str | None = None
            try:
                customer_id = self.get_task(task_id).get("customer_id")
            except Exception:
                customer_id = None
            try:
                await self._wake_callback(
                    {
                        "type": "task_event",
                        "task_id": task_id,
                        "event_type": event_type,
                        "payload": payload,
                        "customer_id": customer_id,
                    }
                )
            except Exception:  # pragma: no cover
                logger.exception("Wake callback failed for task %s event %s", task_id, event_type)

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            for task_id, runner in list(self._running_tasks.items()):
                hb = self._heartbeats.get(task_id, now)
                if runner.done() and self.get_task(task_id)["status"] == "running":
                    await self._set_status(
                        task_id, "failed", summary="Worker stopped unexpectedly."
                    )
                    await self._emit_event(
                        task_id, "worker_stopped", {"reason": "runner_done_while_running"}
                    )
                elif now - hb > 120:
                    await self._set_status(task_id, "failed", summary="Worker heartbeat timeout.")
                    await self._emit_event(
                        task_id, "worker_stopped", {"reason": "heartbeat_timeout"}
                    )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "customer_id": row["customer_id"],
            "goal": row["goal"],
            "status": row["status"],
            "risk_level": row["risk_level"],
            "payload": json.loads(row["payload_json"]),
            "requires_user_input": bool(row["requires_user_input"]),
            "final_summary": row["final_summary"],
            "idempotency_key": row["idempotency_key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _terminal_step_payload(*, index: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "type": "run_terminal",
        "ok": result["ok"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "cwd": result["cwd"],
    }


def _test_result_payload(*, test: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result["ok"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "command": test.get("command", ""),
    }
