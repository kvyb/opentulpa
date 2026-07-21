"""Durable dispatcher for immutable TriggerSpecs and pinned AgentSpecs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.date import DateTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.interval import (  # type: ignore[import-untyped]
    IntervalTrigger as APSIntervalTrigger,
)

from opentulpa.deep_agent import (
    AgentApproval,
    AgentRunContext,
    AgentRunRequest,
    AgentRunSnapshot,
)
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.specs.models import (
    AtTrigger,
    CronTriggerSpec,
    EventTrigger,
    IntervalTrigger,
    TriggerSpec,
    TriggerSpecWrite,
)
from opentulpa.specs.protocol import OriginRef
from opentulpa.specs.store import (
    AgentSpecStore,
    SpecConflictError,
    SpecNotFoundError,
    TriggerSpecStore,
)

logger = logging.getLogger(__name__)


class TriggerAgentService(Protocol):
    async def run(self, request: AgentRunRequest) -> AgentRunSnapshot: ...


class TriggerDelivery(Protocol):
    async def __call__(
        self,
        *,
        trigger: TriggerSpec,
        snapshot: AgentRunSnapshot,
        mode: Literal["origin", "owner"],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _ClaimedDelivery:
    id: int
    tenant_id: str
    trigger_id: str
    trigger_revision: int
    fire_key: str
    mode: Literal["origin", "owner"]
    snapshot: AgentRunSnapshot
    lease_token: str


class TriggerExecutionStore:
    """Fence duplicate scheduler workers and external-event retries."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trigger_executions (
                    tenant_id TEXT NOT NULL,
                    trigger_id TEXT NOT NULL,
                    trigger_revision INTEGER NOT NULL,
                    fire_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('running', 'completed', 'interrupted', 'failed')
                    ),
                    run_id TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, trigger_id, trigger_revision, fire_key)
                );

                CREATE TABLE IF NOT EXISTS trigger_delivery_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    trigger_id TEXT NOT NULL,
                    trigger_revision INTEGER NOT NULL,
                    fire_key TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('origin', 'owner')),
                    snapshot_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'delivering', 'delivered')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (tenant_id, trigger_id, trigger_revision, fire_key)
                );

                CREATE INDEX IF NOT EXISTS idx_trigger_delivery_pending
                ON trigger_delivery_outbox (status, lease_expires_at, id);
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(trigger_executions)").fetchall()
            }
            if "attempt_count" not in columns:
                conn.execute(
                    "ALTER TABLE trigger_executions "
                    "ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 1"
                )
            if "lease_expires_at" not in columns:
                conn.execute(
                    "ALTER TABLE trigger_executions ADD COLUMN lease_expires_at TEXT"
                )
            if "lease_token" not in columns:
                conn.execute("ALTER TABLE trigger_executions ADD COLUMN lease_token TEXT")
            conn.commit()

    def claim(
        self,
        trigger: TriggerSpec,
        fire_key: str,
        *,
        lease_seconds: float,
    ) -> str | None:
        if lease_seconds <= 0:
            raise ValueError("trigger execution lease must be positive")
        now = self._clock().astimezone(UTC)
        lease_token = uuid4().hex
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO trigger_executions (
                    tenant_id, trigger_id, trigger_revision, fire_key, status,
                    attempt_count, lease_token, lease_expires_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', 1, ?, ?, ?)
                """,
                (
                    trigger.tenant_id,
                    trigger.id,
                    trigger.revision,
                    fire_key,
                    lease_token,
                    lease_expires_at,
                    now.isoformat(),
                ),
            )
            if inserted.rowcount == 1:
                conn.commit()
                return lease_token
            reclaimed = conn.execute(
                """
                UPDATE trigger_executions
                SET attempt_count = attempt_count + 1,
                    lease_token = ?, lease_expires_at = ?, updated_at = ?
                WHERE tenant_id = ? AND trigger_id = ?
                  AND trigger_revision = ? AND fire_key = ?
                  AND status = 'running'
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (
                    lease_token,
                    lease_expires_at,
                    now.isoformat(),
                    trigger.tenant_id,
                    trigger.id,
                    trigger.revision,
                    fire_key,
                    now.isoformat(),
                ),
            )
            conn.commit()
            return lease_token if reclaimed.rowcount == 1 else None

    def finish(
        self,
        trigger: TriggerSpec,
        fire_key: str,
        *,
        lease_token: str,
        status: Literal["completed", "interrupted", "failed"],
        run_id: str | None,
        delivery: tuple[AgentRunSnapshot, Literal["origin", "owner"]] | None = None,
    ) -> bool:
        now = self._clock().astimezone(UTC).isoformat()
        snapshot_json = _serialize_snapshot(delivery[0]) if delivery is not None else None
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE trigger_executions
                SET status = ?, run_id = ?, lease_token = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE tenant_id = ? AND trigger_id = ?
                  AND trigger_revision = ? AND fire_key = ? AND status = 'running'
                  AND lease_token = ?
                """,
                (
                    status,
                    run_id,
                    now,
                    trigger.tenant_id,
                    trigger.id,
                    trigger.revision,
                    fire_key,
                    lease_token,
                ),
            )
            if updated.rowcount == 1 and delivery is not None:
                conn.execute(
                    """
                    INSERT INTO trigger_delivery_outbox (
                        tenant_id, trigger_id, trigger_revision, fire_key, mode,
                        snapshot_json, status, attempt_count, lease_token,
                        lease_expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, NULL, ?, ?)
                    """,
                    (
                        trigger.tenant_id,
                        trigger.id,
                        trigger.revision,
                        fire_key,
                        delivery[1],
                        snapshot_json,
                        now,
                        now,
                    ),
                )
            conn.commit()
        return updated.rowcount == 1

    def claim_deliveries(
        self,
        *,
        lease_seconds: float,
        limit: int = 100,
    ) -> list[_ClaimedDelivery]:
        if lease_seconds <= 0:
            raise ValueError("trigger delivery lease must be positive")
        if limit <= 0:
            raise ValueError("trigger delivery claim limit must be positive")
        now = self._clock().astimezone(UTC)
        lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
        claimed: list[_ClaimedDelivery] = []
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM trigger_delivery_outbox
                WHERE status = 'pending'
                   OR (
                       status = 'delivering'
                       AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                   )
                ORDER BY id ASC
                LIMIT ?
                """,
                (now.isoformat(), limit),
            ).fetchall()
            for row in rows:
                lease_token = uuid4().hex
                updated = conn.execute(
                    """
                    UPDATE trigger_delivery_outbox
                    SET status = 'delivering', attempt_count = attempt_count + 1,
                        lease_token = ?, lease_expires_at = ?, updated_at = ?
                    WHERE id = ?
                      AND (
                          status = 'pending'
                          OR (
                              status = 'delivering'
                              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                          )
                      )
                    """,
                    (
                        lease_token,
                        lease_expires_at,
                        now.isoformat(),
                        row["id"],
                        now.isoformat(),
                    ),
                )
                if updated.rowcount != 1:
                    continue
                claimed.append(
                    _ClaimedDelivery(
                        id=int(row["id"]),
                        tenant_id=str(row["tenant_id"]),
                        trigger_id=str(row["trigger_id"]),
                        trigger_revision=int(row["trigger_revision"]),
                        fire_key=str(row["fire_key"]),
                        mode=str(row["mode"]),  # type: ignore[arg-type]
                        snapshot=_deserialize_snapshot(str(row["snapshot_json"])),
                        lease_token=lease_token,
                    )
                )
            conn.commit()
        return claimed

    def next_delivery_lease_delay(self) -> float | None:
        now = self._clock().astimezone(UTC)
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            row = conn.execute(
                """
                SELECT lease_expires_at
                FROM trigger_delivery_outbox
                WHERE status = 'delivering'
                ORDER BY lease_expires_at IS NOT NULL ASC, lease_expires_at ASC
                LIMIT 1
                """
            ).fetchone()
        if row is None or row["lease_expires_at"] is None:
            return 0.0 if row is not None else None
        expires_at = datetime.fromisoformat(str(row["lease_expires_at"])).astimezone(UTC)
        return max(0.0, (expires_at - now).total_seconds())

    def complete_delivery(self, delivery: _ClaimedDelivery) -> bool:
        now = self._clock().astimezone(UTC).isoformat()
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            updated = conn.execute(
                """
                UPDATE trigger_delivery_outbox
                SET status = 'delivered', lease_token = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND status = 'delivering' AND lease_token = ?
                """,
                (now, delivery.id, delivery.lease_token),
            )
            conn.commit()
        return updated.rowcount == 1

    def retry_delivery(self, delivery: _ClaimedDelivery) -> bool:
        now = self._clock().astimezone(UTC).isoformat()
        with closing(connect_sqlite(self.db_path, wal=True)) as conn:
            updated = conn.execute(
                """
                UPDATE trigger_delivery_outbox
                SET status = 'pending', lease_token = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND status = 'delivering' AND lease_token = ?
                """,
                (now, delivery.id, delivery.lease_token),
            )
            conn.commit()
        return updated.rowcount == 1


class TriggerDispatcher:
    """Translate deterministic triggers into the universal Agent Run protocol."""

    def __init__(
        self,
        *,
        triggers: TriggerSpecStore,
        agent_specs: AgentSpecStore,
        agent_service: TriggerAgentService,
        executions: TriggerExecutionStore,
        deliver: TriggerDelivery | None = None,
        scheduler: AsyncIOScheduler | None = None,
        clock: Callable[[], datetime] | None = None,
        delivery_lease_seconds: float = 60.0,
        delivery_retry_base_seconds: float = 1.0,
        delivery_retry_max_seconds: float = 60.0,
    ) -> None:
        if delivery_lease_seconds <= 0:
            raise ValueError("trigger delivery lease must be positive")
        if delivery_retry_base_seconds <= 0:
            raise ValueError("trigger delivery retry base must be positive")
        if delivery_retry_max_seconds < delivery_retry_base_seconds:
            raise ValueError("trigger delivery retry max must not be below its base")
        self._triggers = triggers
        self._agent_specs = agent_specs
        self._agent = agent_service
        self._executions = executions
        self._deliver = deliver
        self._scheduler = scheduler or AsyncIOScheduler(timezone="UTC")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._delivery_lease_seconds = delivery_lease_seconds
        self._delivery_retry_base_seconds = delivery_retry_base_seconds
        self._delivery_retry_max_seconds = delivery_retry_max_seconds
        self._delivery_tasks: set[asyncio.Task[None]] = set()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        self._schedule_delivery_drain()
        for tenant_id in self._triggers.list_tenant_ids():
            self.sync_tenant(tenant_id)

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._started:
            return
        for task in tuple(self._delivery_tasks):
            task.cancel()
        self._scheduler.shutdown(wait=wait)
        self._started = False

    def sync_tenant(self, tenant_id: str) -> None:
        current = {
            trigger.id: trigger for trigger in self._triggers.list_active(tenant_id=tenant_id)
        }
        prefix = self._job_prefix(tenant_id)
        for job in self._scheduler.get_jobs():
            if job.id.startswith(prefix) and job.id.removeprefix(prefix) not in current:
                self._scheduler.remove_job(job.id)
        for trigger in current.values():
            self.upsert(trigger)

    def upsert(self, trigger: TriggerSpec) -> None:
        job_id = self._job_id(trigger.tenant_id, trigger.id)
        source = self._aps_trigger(trigger)
        if not trigger.enabled or source is None:
            self.remove(tenant_id=trigger.tenant_id, trigger_id=trigger.id)
            return
        self._scheduler.add_job(
            self._run_scheduled,
            trigger=source,
            id=job_id,
            replace_existing=True,
            kwargs={
                "tenant_id": trigger.tenant_id,
                "trigger_id": trigger.id,
                "trigger_revision": trigger.revision,
            },
            max_instances=1,
            coalesce=True,
            misfire_grace_time=1,
        )

    def remove(self, *, tenant_id: str, trigger_id: str) -> None:
        job_id = self._job_id(tenant_id, trigger_id)
        if self._scheduler.get_job(job_id) is not None:
            self._scheduler.remove_job(job_id)

    async def dispatch_event(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        source_event_id: str,
        event_type: str,
        source: str,
        authenticated: bool,
        payload: Mapping[str, Any] | None = None,
    ) -> AgentRunSnapshot | None:
        trigger = self._triggers.get_active(tenant_id=tenant_id, trigger_id=trigger_id)
        if trigger is None or not trigger.enabled or not isinstance(trigger.source, EventTrigger):
            return None
        if trigger.source.event_type != event_type or trigger.source.source != source:
            return None
        if trigger.source.authentication == "required" and not authenticated:
            raise PermissionError("authenticated trigger event is required")
        rendered_payload = json.dumps(
            dict(payload or {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        if len(rendered_payload.encode("utf-8")) > 1_000_000:
            raise ValueError("trigger event payload exceeds the size limit")
        instruction = trigger.instruction
        if rendered_payload != "{}":
            instruction = f"{instruction}\n\nAuthenticated event payload:\n{rendered_payload}"
        return await self._execute(
            trigger,
            fire_key=source_event_id,
            instruction=instruction,
            actor_id=f"event:{source}",
            origin=OriginRef(
                interface=source,
                source_id=trigger.id,
                message_id=source_event_id[:200],
            ),
        )

    async def _run_scheduled(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        trigger_revision: int,
    ) -> None:
        trigger = self._triggers.get_active(tenant_id=tenant_id, trigger_id=trigger_id)
        if (
            trigger is None
            or not trigger.enabled
            or trigger.revision != trigger_revision
            or isinstance(trigger.source, EventTrigger)
        ):
            return
        try:
            await self._execute(
                trigger,
                fire_key=self._scheduled_fire_key(trigger),
                instruction=trigger.instruction,
                actor_id="scheduler",
                origin=OriginRef(interface="trigger", source_id=trigger.id),
            )
        finally:
            if isinstance(trigger.source, AtTrigger):
                self._disable_one_off(trigger)

    async def _execute(
        self,
        trigger: TriggerSpec,
        *,
        fire_key: str,
        instruction: str,
        actor_id: str,
        origin: OriginRef,
    ) -> AgentRunSnapshot | None:
        await self._drain_deliveries()
        spec = self._agent_specs.get_revision(trigger.agent_spec)
        if spec is None or spec.isolation != trigger.exposure:
            return None
        lease_token = self._executions.claim(
            trigger,
            fire_key,
            lease_seconds=float(spec.max_runtime_seconds) + 60.0,
        )
        if lease_token is None:
            return None
        trust_class: Literal["background", "external"] = (
            "external" if trigger.exposure == "external" else "background"
        )
        channel = "intake" if trust_class == "external" else "routine"
        event_digest = hashlib.sha256(fire_key.encode()).hexdigest()[:24]
        idempotency_digest = hashlib.sha256(
            (f"{trigger.tenant_id}\0{trigger.id}\0{trigger.revision}\0{fire_key}").encode()
        ).hexdigest()
        context = AgentRunContext(
            tenant_id=trigger.tenant_id,
            actor_id=actor_id,
            thread_id=f"trigger:{trigger.id}:{event_digest}",
            channel=channel,
            run_kind=spec.runtime_profile,
            correlation_id=f"trigger:{trigger.id}:r{trigger.revision}:{event_digest}",
            origin=origin,
            agent_spec=trigger.agent_spec,
            trust_class=trust_class,
        )
        try:
            snapshot = await self._agent.run(
                AgentRunRequest(
                    context=context,
                    text=instruction,
                    idempotency_key=f"trigger:{trigger.id}:{idempotency_digest}",
                )
            )
        except Exception as exc:
            logger.error(
                "TriggerSpec execution failed: trigger=%s error_type=%s",
                trigger.id,
                type(exc).__name__,
            )
            snapshot = AgentRunSnapshot(
                run_id=f"trigger-failure-{idempotency_digest[:24]}",
                context=context,
                status="failed",
                error="The trigger run failed before producing a result.",
            )
        execution_status: Literal["completed", "interrupted", "failed"] = (
            "completed"
            if snapshot.status == "completed"
            else "interrupted"
            if snapshot.status == "interrupted"
            else "failed"
        )
        delivery: tuple[AgentRunSnapshot, Literal["origin", "owner"]] | None = None
        if trigger.delivery.mode != "none" or snapshot.status in {"interrupted", "failed"}:
            mode: Literal["origin", "owner"] = (
                "owner"
                if snapshot.status in {"interrupted", "failed"}
                else "origin"
                if trigger.delivery.mode == "origin"
                else "owner"
            )
            delivery = (snapshot, mode)
        self._executions.finish(
            trigger,
            fire_key,
            lease_token=lease_token,
            status=execution_status,
            run_id=snapshot.run_id,
            delivery=delivery,
        )
        retry_pending = await self._drain_deliveries()
        if retry_pending and self._started:
            self._schedule_delivery_drain(
                initial_delay=self._delivery_retry_base_seconds,
            )
        return snapshot

    def _schedule_delivery_drain(self, *, initial_delay: float = 0.0) -> None:
        if self._deliver is None:
            return
        if any(not task.done() for task in self._delivery_tasks):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Trigger delivery drain deferred because no event loop is running"
            )
            return
        task = loop.create_task(
            self._drain_deliveries_on_start(initial_delay=initial_delay)
        )
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_task_done)

    def _delivery_task_done(self, task: asyncio.Task[None]) -> None:
        self._delivery_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Trigger delivery drain stopped unexpectedly: error_type=%s",
                type(error).__name__,
            )

    async def _drain_deliveries_on_start(self, *, initial_delay: float = 0.0) -> None:
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        retry_delay = self._delivery_retry_base_seconds
        while self._started:
            retry_pending = await self._drain_deliveries()
            lease_delay = self._executions.next_delivery_lease_delay()
            delays: list[float] = []
            if retry_pending:
                delays.append(retry_delay)
                retry_delay = min(
                    retry_delay * 2,
                    self._delivery_retry_max_seconds,
                )
            else:
                retry_delay = self._delivery_retry_base_seconds
            if lease_delay is not None:
                delays.append(max(0.01, lease_delay + 0.01))
            if not delays:
                return
            await asyncio.sleep(min(delays))

    async def _drain_deliveries(self) -> bool:
        if self._deliver is None:
            return False
        while True:
            claimed = self._executions.claim_deliveries(
                lease_seconds=self._delivery_lease_seconds,
            )
            if not claimed:
                return False
            retry_pending = False
            for delivery in claimed:
                trigger = self._triggers.get_revision(
                    tenant_id=delivery.tenant_id,
                    trigger_id=delivery.trigger_id,
                    revision=delivery.trigger_revision,
                )
                if trigger is None:
                    self._executions.retry_delivery(delivery)
                    retry_pending = True
                    logger.error(
                        "Trigger delivery deferred because its immutable TriggerSpec is missing: "
                        "trigger=%s revision=%s",
                        delivery.trigger_id,
                        delivery.trigger_revision,
                    )
                    continue
                try:
                    await self._deliver(
                        trigger=trigger,
                        snapshot=delivery.snapshot,
                        mode=delivery.mode,
                    )
                except asyncio.CancelledError:
                    self._executions.retry_delivery(delivery)
                    raise
                except Exception as exc:
                    self._executions.retry_delivery(delivery)
                    retry_pending = True
                    logger.error(
                        "Trigger delivery failed and remains pending: "
                        "trigger=%s error_type=%s",
                        delivery.trigger_id,
                        type(exc).__name__,
                    )
                else:
                    self._executions.complete_delivery(delivery)
            if retry_pending:
                return True

    def _aps_trigger(self, trigger: TriggerSpec) -> Any | None:
        source = trigger.source
        if isinstance(source, EventTrigger):
            return None
        if isinstance(source, AtTrigger):
            if source.run_at.astimezone(UTC) <= self._clock().astimezone(UTC):
                return None
            return DateTrigger(run_date=source.run_at)
        if isinstance(source, CronTriggerSpec):
            return CronTrigger.from_crontab(source.expression, timezone=source.timezone)
        if isinstance(source, IntervalTrigger):
            return APSIntervalTrigger(seconds=source.every_seconds, timezone="UTC")
        return None

    def _disable_one_off(self, trigger: TriggerSpec) -> None:
        """Persist that an absolute trigger has fired, even if its run failed."""

        active = self._triggers.get_active(
            tenant_id=trigger.tenant_id,
            trigger_id=trigger.id,
        )
        if active is None or active.revision != trigger.revision or not active.enabled:
            return
        write = TriggerSpecWrite.model_validate(
            trigger.model_dump(
                include={
                    "name",
                    "source",
                    "exposure",
                    "agent_spec",
                    "instruction",
                    "delivery",
                    "enabled",
                    "source_key",
                    "source_revision",
                    "labels",
                }
            )
        ).model_copy(
            update={
                "enabled": False,
                "source_revision": (
                    trigger.source_revision + 1
                    if trigger.source_revision is not None
                    else None
                ),
            }
        )
        try:
            disabled = self._triggers.create_revision(
                tenant_id=trigger.tenant_id,
                trigger_id=trigger.id,
                write=write,
                expected_revision=trigger.revision,
                created_by="scheduler",
            )
            self._triggers.activate(
                tenant_id=trigger.tenant_id,
                trigger_id=trigger.id,
                revision=disabled.revision,
                expected_active_revision=trigger.revision,
                updated_by="scheduler",
            )
        except (SpecConflictError, SpecNotFoundError):
            # Another scheduler worker or an owner edit already advanced the trigger.
            return

    def _scheduled_fire_key(self, trigger: TriggerSpec) -> str:
        source = trigger.source
        if isinstance(source, AtTrigger):
            return source.run_at.astimezone(UTC).isoformat()
        now = self._clock().astimezone(UTC)
        if isinstance(source, CronTriggerSpec):
            return now.replace(second=0, microsecond=0).isoformat()
        if isinstance(source, IntervalTrigger):
            bucket = int(now.timestamp()) // source.every_seconds
            return f"interval:{bucket}"
        raise ValueError("event triggers do not have scheduled fire keys")

    @staticmethod
    def _job_prefix(tenant_id: str) -> str:
        digest = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]
        return f"trigger:{digest}:"

    @classmethod
    def _job_id(cls, tenant_id: str, trigger_id: str) -> str:
        return f"{cls._job_prefix(tenant_id)}{trigger_id}"


def _serialize_snapshot(snapshot: AgentRunSnapshot) -> str:
    return json.dumps(
        {
            "run_id": snapshot.run_id,
            "context": snapshot.context.model_dump(mode="json"),
            "status": snapshot.status,
            "final_text": snapshot.final_text,
            "error": snapshot.error,
            "approvals": [
                {
                    "id": approval.id,
                    "tool_name": approval.tool_name,
                    "description": approval.description,
                    "arguments": approval.arguments,
                    "allowed_decisions": list(approval.allowed_decisions),
                    "status": approval.status,
                    "edited_arguments": approval.edited_arguments,
                    "message": approval.message,
                }
                for approval in snapshot.approvals
            ],
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize_snapshot(payload: str) -> AgentRunSnapshot:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("trigger delivery snapshot must be an object")
    approvals = value.get("approvals", [])
    if not isinstance(approvals, list):
        raise ValueError("trigger delivery approvals must be a list")
    return AgentRunSnapshot(
        run_id=str(value["run_id"]),
        context=AgentRunContext.model_validate(value["context"]),
        status=str(value["status"]),  # type: ignore[arg-type]
        final_text=str(value.get("final_text") or ""),
        error=str(value.get("error") or ""),
        approvals=tuple(
            AgentApproval(
                id=str(item["id"]),
                tool_name=str(item["tool_name"]),
                description=str(item["description"]),
                arguments=dict(item.get("arguments") or {}),
                allowed_decisions=tuple(
                    str(decision) for decision in item.get("allowed_decisions", [])
                ),
                status=str(item.get("status") or "pending"),  # type: ignore[arg-type]
                edited_arguments=(
                    dict(item["edited_arguments"])
                    if isinstance(item.get("edited_arguments"), dict)
                    else None
                ),
                message=(str(item["message"]) if item.get("message") is not None else None),
            )
            for item in approvals
            if isinstance(item, dict)
        ),
        created_at=str(value.get("created_at") or ""),
        updated_at=str(value.get("updated_at") or ""),
    )


__all__ = [
    "TriggerAgentService",
    "TriggerDelivery",
    "TriggerDispatcher",
    "TriggerExecutionStore",
]
