"""APScheduler-based scheduler with durable routine storage."""

import json
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.date import DateTrigger  # type: ignore[import-untyped]

from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.scheduler.models import Routine

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEBUG_LOG_PATH = PROJECT_ROOT / ".cursor" / "debug.log"
DEFAULT_DB_PATH = PROJECT_ROOT / ".opentulpa" / "scheduler.db"


def _debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "scheduler",
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
        pass


class SchedulerService:
    """In-process scheduler; jobs can invoke a wake-agent callback."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._scheduler = AsyncIOScheduler()
        self._wake_callback: Callable[[dict[str, Any]], Awaitable[Any]] | None = None
        self._routines: dict[str, Routine] = {}
        self._db_path = (db_path or DEFAULT_DB_PATH).resolve()
        self._started = False
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, wal=True)

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS routines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    is_cron INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _to_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    def _upsert_routine(self, routine: Routine) -> None:
        now = self._utc_now_iso()
        created_at = (
            self._to_aware_datetime(routine.created_at).isoformat()
            if isinstance(routine.created_at, datetime)
            else now
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO routines (id, name, schedule, payload_json, enabled, is_cron, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    schedule=excluded.schedule,
                    payload_json=excluded.payload_json,
                    enabled=excluded.enabled,
                    is_cron=excluded.is_cron,
                    updated_at=excluded.updated_at
                """,
                (
                    routine.id,
                    routine.name,
                    routine.schedule,
                    json.dumps(routine.payload, ensure_ascii=False),
                    1 if routine.enabled else 0,
                    1 if routine.is_cron else 0,
                    created_at,
                    now,
                ),
            )
            conn.commit()

    def _delete_routine(self, routine_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM routines WHERE id=?", (routine_id,))
            conn.commit()

    def _load_routines(self) -> list[Routine]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, name, schedule, payload_json, enabled, is_cron, created_at
                FROM routines
                ORDER BY created_at ASC
                """
            ).fetchall()
        loaded: list[Routine] = []
        for row in rows:
            created_at_raw = str(row["created_at"] or "").strip()
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except Exception:
                created_at = datetime.now(UTC)
            payload_raw = row["payload_json"] or "{}"
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {}
            loaded.append(
                Routine(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    schedule=str(row["schedule"]),
                    payload=payload if isinstance(payload, dict) else {},
                    enabled=bool(row["enabled"]),
                    created_at=created_at,
                    is_cron=bool(row["is_cron"]),
                )
            )
        return loaded

    def _schedule_job(self, routine: Routine) -> None:
        if not routine.enabled:
            return
        if routine.is_cron:
            try:
                parts = routine.schedule.strip().split()
                if len(parts) != 5:
                    logger.warning("Invalid cron '%s' for routine %s", routine.schedule, routine.id)
                    return
                self._scheduler.add_job(
                    self._run_routine,
                    CronTrigger.from_crontab(routine.schedule),
                    id=routine.id,
                    args=[routine.id],
                    replace_existing=True,
                    coalesce=True,
                    # Skip stale executions after downtime/restart.
                    misfire_grace_time=1,
                )
            except Exception as e:
                logger.exception("Failed to add cron job for %s: %s", routine.id, e)
            return

        try:
            run_time = datetime.fromisoformat(routine.schedule.replace("Z", "+00:00"))
            run_time = self._to_aware_datetime(run_time)
            now = datetime.now(UTC)
            if run_time <= now:
                # Persist as disabled so missed one-off jobs are never replayed on restart.
                routine.enabled = False
                self._routines[routine.id] = routine
                self._upsert_routine(routine)
                return
            self._scheduler.add_job(
                self._run_routine,
                DateTrigger(run_date=run_time),
                id=routine.id,
                args=[routine.id],
                replace_existing=True,
                # Do not replay if scheduler comes back after target time.
                misfire_grace_time=1,
            )
        except Exception as e:
            logger.exception("Failed to add one-off job for %s: %s", routine.id, e)

    def set_wake_callback(self, callback: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
        """Set callback to run when a routine fires (e.g. send message to agent)."""
        self._wake_callback = callback

    def add_routine(self, routine: Routine) -> None:
        """Schedule a routine."""
        self._routines[routine.id] = routine
        self._upsert_routine(routine)
        self._schedule_job(routine)

    async def _run_routine(self, routine_id: str) -> None:
        routine = self._routines.get(routine_id)
        # region agent log
        _debug_log(
            hypothesis_id="scheduler",
            location="scheduler/service.py:_run_routine",
            message="routine_triggered",
            data={"routine_id": routine_id, "enabled": bool(routine.enabled) if routine else False},
        )
        # endregion
        if not routine or not routine.enabled:
            return
        wake_payload = {
            "type": "routine_event",
            "event_type": "scheduled",
            "routine_id": routine.id,
            "routine_name": routine.name,
            "customer_id": str(routine.payload.get("customer_id", "")).strip(),
            "notify_user": bool(routine.payload.get("notify_user", False)),
            "payload": routine.payload,
        }
        if self._wake_callback:
            try:
                await self._wake_callback(wake_payload)
            except Exception as e:
                logger.exception("Wake callback failed for %s: %s", routine_id, e)
        else:
            logger.warning(
                "No wake callback set; routine %s payload: %s", routine_id, wake_payload
            )
        if not routine.is_cron:
            routine.enabled = False
            self._routines[routine.id] = routine
            self._upsert_routine(routine)

    def remove_routine(self, routine_id: str) -> bool:
        """Remove a routine and its job."""
        existed = routine_id in self._routines
        if existed:
            del self._routines[routine_id]
        with suppress(Exception):
            self._scheduler.remove_job(routine_id)
        self._delete_routine(routine_id)
        return existed

    def list_routines(self) -> list[Routine]:
        return list(self._routines.values())

    def get_routine(self, routine_id: str) -> Routine | None:
        return self._routines.get(routine_id)

    def start(self) -> None:
        if self._started:
            return
        self._routines = {r.id: r for r in self._load_routines()}
        for routine in self._routines.values():
            self._schedule_job(routine)
        self._scheduler.start()
        self._started = True
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=wait)
        self._started = False
