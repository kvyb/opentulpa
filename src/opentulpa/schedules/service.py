"""Typed schedule facade backed by immutable AgentSpec and TriggerSpec revisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from builtins import list as list_type
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.schedules.models import (
    AgentJob,
    At,
    Cron,
    LegacyMigrationIssue,
    LegacyMigrationReport,
    Reminder,
    Schedule,
    ScheduleWrite,
)
from opentulpa.specs.models import (
    AtTrigger,
    CronTriggerSpec,
    DeliverySpec,
    TriggerSpec,
    TriggerSpecWrite,
)
from opentulpa.specs.protocol import AgentSpecRef
from opentulpa.specs.service import TriggerSpecService
from opentulpa.specs.store import SpecConflictError, SpecNotFoundError

_SCHEDULE_SOURCE_PREFIX = "schedule:"
_SCHEDULE_CONTRACT = "v1"
_REMINDER_PREFIX = "Deliver this reminder to the owner exactly as written:\n\n"


class ScheduleConflictError(RuntimeError):
    """The requested optimistic schedule revision no longer matches storage."""


class ScheduleNotFoundError(KeyError):
    """The tenant-owned schedule does not exist or is inactive."""


class _SkipLegacyRoutineError(RuntimeError):
    pass


class ScheduleService:
    """Project the simple schedule contract onto the universal trigger control plane."""

    def __init__(
        self,
        trigger_specs: TriggerSpecService,
        *,
        resolve_agent_spec: Callable[[str], AgentSpecRef],
        clock: Callable[[], datetime] | None = None,
        on_changed: Callable[[TriggerSpec], None] | None = None,
        on_deleted: Callable[[str, str], None] | None = None,
    ) -> None:
        self._trigger_specs = trigger_specs
        self._resolve_agent_spec = resolve_agent_spec
        self._clock = clock or (lambda: datetime.now(UTC))
        self._on_changed = on_changed
        self._on_deleted = on_deleted

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("schedule service clock must return an aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _tenant_id(value: str) -> str:
        tenant_id = str(value or "").strip()
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if len(tenant_id) > 200:
            raise ValueError("tenant_id must be at most 200 characters")
        return tenant_id

    @staticmethod
    def _schedule_id(value: str | None) -> str:
        schedule_id = str(value or "").strip() or new_short_id("sch")
        if len(schedule_id) > 100:
            raise ValueError("schedule_id must be at most 100 characters")
        return schedule_id

    @staticmethod
    def _actor_id(value: str) -> str:
        actor_id = str(value or "").strip()
        if not actor_id:
            raise ValueError("actor_id is required")
        if len(actor_id) > 200:
            raise ValueError("actor_id must be at most 200 characters")
        return actor_id

    @staticmethod
    def _trigger_id(schedule_id: str) -> str:
        digest = hashlib.sha256(schedule_id.encode()).hexdigest()[:24]
        return f"schedule_{digest}"

    @staticmethod
    def _source_key(schedule_id: str) -> str:
        return f"{_SCHEDULE_SOURCE_PREFIX}{schedule_id}"

    def list(self, *, tenant_id: str) -> list_type[Schedule]:
        tenant_id = self._tenant_id(tenant_id)
        schedules = [
            self._project(trigger)
            for trigger in self._trigger_specs.list_active(tenant_id=tenant_id)
            if self._is_schedule(trigger)
        ]
        return sorted(schedules, key=lambda item: (item.created_at, item.id))

    def list_tenant_ids(self) -> list_type[str]:
        return self._trigger_specs.list_tenant_ids()

    def get(self, *, tenant_id: str, schedule_id: str) -> Schedule | None:
        tenant_id = self._tenant_id(tenant_id)
        schedule_id = self._schedule_id(schedule_id)
        trigger = self._trigger_specs.get_active(
            tenant_id=tenant_id,
            trigger_id=self._trigger_id(schedule_id),
        )
        if trigger is None or trigger.source_key != self._source_key(schedule_id):
            return None
        if not self._is_schedule(trigger):
            return None
        return self._project(trigger)

    def save(
        self,
        *,
        tenant_id: str,
        write: ScheduleWrite,
        schedule_id: str | None = None,
        expected_revision: int | None = None,
        actor_id: str = "owner",
    ) -> Schedule:
        tenant_id = self._tenant_id(tenant_id)
        schedule_id = self._schedule_id(schedule_id)
        actor_id = self._actor_id(actor_id)
        if (
            isinstance(write.trigger, At)
            and write.enabled
            and write.trigger.run_at.astimezone(UTC) <= self._now()
        ):
            # Missed one-offs remain visible but must never replay after restart.
            write = write.model_copy(update={"enabled": False})

        trigger_id = self._trigger_id(schedule_id)
        source_key = self._source_key(schedule_id)
        latest = self._trigger_specs.get_latest(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
        )
        active = self._trigger_specs.get_active(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
        )
        if latest is not None and latest.source_key != source_key:
            raise ScheduleConflictError("schedule identifier collides with another trigger")

        if active is None:
            if latest is not None:
                raise ScheduleConflictError(
                    f"schedule {schedule_id!r} is inactive at revision {latest.revision}"
                )
            if expected_revision is not None:
                raise ScheduleConflictError(
                    f"schedule {schedule_id!r} does not exist at revision {expected_revision}"
                )
            next_revision = 1
            expected_active_revision = None
            agent_spec = self._resolve_agent_spec(tenant_id)
        else:
            if expected_revision is None:
                raise ScheduleConflictError("expected_revision is required when updating")
            if active.revision != expected_revision:
                raise ScheduleConflictError(
                    f"expected revision {expected_revision}, found {active.revision}"
                )
            if latest is None or latest.revision != active.revision:
                raise ScheduleConflictError("schedule changed outside the schedule facade")
            next_revision = active.revision + 1
            expected_active_revision = active.revision
            agent_spec = active.agent_spec

        spec_write = self._trigger_write(
            schedule_id=schedule_id,
            revision=next_revision,
            write=write,
            agent_spec=agent_spec,
        )
        try:
            created = self._trigger_specs.save(
                tenant_id=tenant_id,
                actor_id=actor_id,
                write=spec_write,
                trigger_id=trigger_id,
                expected_revision=latest.revision if latest is not None else None,
            )
            activated = self._trigger_specs.activate(
                tenant_id=tenant_id,
                actor_id=actor_id,
                trigger_id=trigger_id,
                revision=created.revision,
                expected_active_revision=expected_active_revision,
            )
        except SpecConflictError as exc:
            raise ScheduleConflictError(str(exc)) from exc
        except SpecNotFoundError as exc:
            raise ScheduleNotFoundError(schedule_id) from exc
        if self._on_changed is not None:
            self._on_changed(activated)
        return self._project(activated)

    def delete(
        self,
        *,
        tenant_id: str,
        schedule_id: str,
        expected_revision: int,
        actor_id: str = "owner",
    ) -> None:
        tenant_id = self._tenant_id(tenant_id)
        schedule_id = self._schedule_id(schedule_id)
        self._actor_id(actor_id)
        trigger_id = self._trigger_id(schedule_id)
        active = self._trigger_specs.get_active(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
        )
        if (
            active is None
            or active.source_key != self._source_key(schedule_id)
            or not self._is_schedule(active)
        ):
            raise ScheduleNotFoundError(schedule_id)
        if active.revision != expected_revision:
            raise ScheduleConflictError(
                f"expected revision {expected_revision}, found {active.revision}"
            )
        try:
            self._trigger_specs.deactivate(
                tenant_id=tenant_id,
                trigger_id=trigger_id,
                expected_active_revision=expected_revision,
            )
        except SpecConflictError as exc:
            raise ScheduleConflictError(str(exc)) from exc
        except SpecNotFoundError as exc:
            raise ScheduleNotFoundError(schedule_id) from exc
        if self._on_deleted is not None:
            self._on_deleted(tenant_id, trigger_id)

    @staticmethod
    def executor_options(schedule: Schedule) -> dict[str, int | bool]:
        """Expose the fixed stale-fire policy used by the unified dispatcher."""

        return {
            "misfire_grace_time": schedule.trigger.misfire_grace_seconds,
            "coalesce": isinstance(schedule.trigger, Cron),
        }

    def _trigger_write(
        self,
        *,
        schedule_id: str,
        revision: int,
        write: ScheduleWrite,
        agent_spec: AgentSpecRef,
    ) -> TriggerSpecWrite:
        if isinstance(write.trigger, At):
            source: AtTrigger | CronTriggerSpec = AtTrigger(
                run_at=write.trigger.run_at,
                timezone=write.trigger.timezone,
            )
        elif isinstance(write.trigger, Cron):
            source = CronTriggerSpec(
                expression=write.trigger.expression,
                timezone=write.trigger.timezone,
            )
        else:  # pragma: no cover - ScheduleWrite forbids other variants
            raise TypeError("unsupported schedule trigger")

        if isinstance(write.action, AgentJob):
            instruction = write.action.instruction
            action_kind = "agent_job"
        elif isinstance(write.action, Reminder):
            instruction = f"{_REMINDER_PREFIX}{write.action.message}"
            action_kind = "reminder"
        else:  # pragma: no cover - ScheduleWrite forbids other variants
            raise TypeError("unsupported schedule action")

        return TriggerSpecWrite(
            name=write.name,
            source=source,
            exposure="private",
            agent_spec=agent_spec,
            instruction=instruction,
            delivery=DeliverySpec(mode="owner" if write.notify_owner else "none"),
            enabled=write.enabled,
            source_key=self._source_key(schedule_id),
            source_revision=revision,
            labels={
                "schedule_action": action_kind,
                "schedule_contract": _SCHEDULE_CONTRACT,
            },
        )

    def _project(self, trigger: TriggerSpec) -> Schedule:
        source_key = str(trigger.source_key or "")
        if not source_key.startswith(_SCHEDULE_SOURCE_PREFIX):
            raise ValueError("TriggerSpec is not schedule-owned")
        schedule_id = source_key.removeprefix(_SCHEDULE_SOURCE_PREFIX)
        if isinstance(trigger.source, AtTrigger):
            source: At | Cron = At(
                run_at=trigger.source.run_at,
                timezone=trigger.source.timezone,
            )
        elif isinstance(trigger.source, CronTriggerSpec):
            source = Cron(
                expression=trigger.source.expression,
                timezone=trigger.source.timezone,
            )
        else:
            raise ValueError("schedule TriggerSpec has an unsupported source")

        action_kind = str(trigger.labels.get("schedule_action") or "")
        if action_kind == "reminder" or (
            not action_kind and trigger.instruction.startswith(_REMINDER_PREFIX)
        ):
            if not trigger.instruction.startswith(_REMINDER_PREFIX):
                raise ValueError("schedule reminder instruction is malformed")
            action: AgentJob | Reminder = Reminder(
                message=trigger.instruction.removeprefix(_REMINDER_PREFIX)
            )
        elif action_kind in {"", "agent_job"}:
            action = AgentJob(instruction=trigger.instruction)
        else:
            raise ValueError("schedule action kind is unsupported")

        revisions = self._trigger_specs.list_revisions(
            tenant_id=trigger.tenant_id,
            trigger_id=trigger.id,
        )
        created_at = revisions[-1].created_at if revisions else trigger.created_at
        return Schedule(
            id=schedule_id,
            tenant_id=trigger.tenant_id,
            revision=trigger.revision,
            name=trigger.name,
            trigger=source,
            action=action,
            notify_owner=trigger.delivery.mode == "owner",
            enabled=trigger.enabled,
            created_at=created_at,
            updated_at=trigger.created_at,
        )

    @staticmethod
    def _is_schedule(trigger: TriggerSpec) -> bool:
        source_key = str(trigger.source_key or "")
        if not source_key.startswith(_SCHEDULE_SOURCE_PREFIX):
            return False
        return (
            trigger.labels.get("schedule_contract") == _SCHEDULE_CONTRACT
            or trigger.labels.get("migrated_from") == "schedule"
        )

    def migrate_legacy_routines(
        self,
        *,
        legacy_db_path: Path,
        default_timezone: str,
        resolve_tenant_id: Callable[[str], str] | None = None,
        dry_run: bool = False,
    ) -> LegacyMigrationReport:
        """Convert deterministic legacy routine rows directly into TriggerSpecs."""

        timezone = Cron(timezone=default_timezone, expression="0 0 * * *").timezone
        source_path = legacy_db_path.expanduser().resolve()
        rows = self._legacy_rows(source_path)
        checksum = hashlib.sha256(
            json.dumps(
                [dict(row) for row in rows],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        eligible = 0
        migrated = 0
        skipped = 0
        invalid = 0
        issues: list_type[LegacyMigrationIssue] = []

        for row in rows:
            routine_id = str(row["id"] or "").strip()
            try:
                schedule = self._legacy_schedule(
                    row,
                    default_timezone=timezone,
                    resolve_tenant_id=resolve_tenant_id,
                )
            except _SkipLegacyRoutineError as exc:
                skipped += 1
                issues.append(
                    LegacyMigrationIssue(
                        routine_id=routine_id,
                        disposition="skipped",
                        error=str(exc),
                    )
                )
                continue
            except Exception as exc:
                invalid += 1
                issues.append(
                    LegacyMigrationIssue(
                        routine_id=routine_id,
                        error=str(exc) or type(exc).__name__,
                    )
                )
                continue

            existing = self.get(tenant_id=schedule.tenant_id, schedule_id=schedule.id)
            if existing is not None:
                if self._same_schedule_content(existing, schedule):
                    skipped += 1
                else:
                    invalid += 1
                    issues.append(
                        LegacyMigrationIssue(
                            routine_id=routine_id,
                            disposition="conflict",
                            error=(
                                "destination schedule exists with different content; "
                                "the legacy routine was not changed"
                            ),
                        )
                    )
                continue
            eligible += 1
            if dry_run:
                continue
            try:
                self.save(
                    tenant_id=schedule.tenant_id,
                    schedule_id=schedule.id,
                    actor_id="migration:legacy-routines",
                    write=ScheduleWrite.model_validate(
                        schedule.model_dump(
                            include={"name", "trigger", "action", "notify_owner", "enabled"}
                        )
                    ),
                )
            except ScheduleConflictError as exc:
                invalid += 1
                eligible -= 1
                issues.append(
                    LegacyMigrationIssue(
                        routine_id=routine_id,
                        disposition="conflict",
                        error=str(exc) or type(exc).__name__,
                    )
                )
            except Exception as exc:
                invalid += 1
                eligible -= 1
                issues.append(
                    LegacyMigrationIssue(
                        routine_id=routine_id,
                        error=str(exc) or type(exc).__name__,
                    )
                )
            else:
                migrated += 1

        return LegacyMigrationReport(
            dry_run=dry_run,
            scanned=len(rows),
            eligible=eligible,
            migrated=migrated,
            skipped=skipped,
            invalid=invalid,
            source_checksum=checksum,
            issues=issues,
        )

    @staticmethod
    def _same_schedule_content(existing: Schedule, legacy: Schedule) -> bool:
        fields = {"name", "trigger", "action", "notify_owner", "enabled"}
        return existing.model_dump(include=fields, mode="json") == legacy.model_dump(
            include=fields,
            mode="json",
        )

    @staticmethod
    def _legacy_rows(db_path: Path) -> list_type[sqlite3.Row]:
        if not db_path.exists():
            return []
        with closing(connect_sqlite(db_path)) as conn:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'routines'"
            ).fetchone()
            if table is None:
                return []
            columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(routines)").fetchall()
            }
            required = {
                "id",
                "name",
                "schedule",
                "payload_json",
                "enabled",
                "is_cron",
                "created_at",
                "updated_at",
            }
            missing = required - columns
            if missing:
                raise ValueError(f"legacy routines table is missing columns: {sorted(missing)}")
            return conn.execute(
                """
                SELECT id, name, schedule, payload_json, enabled, is_cron,
                       created_at, updated_at
                FROM routines
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()

    def _legacy_schedule(
        self,
        row: sqlite3.Row,
        *,
        default_timezone: str,
        resolve_tenant_id: Callable[[str], str] | None,
    ) -> Schedule:
        routine_id = str(row["id"] or "").strip()
        if not routine_id:
            raise ValueError("legacy routine id is required")
        try:
            payload = json.loads(str(row["payload_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json is invalid") from exc
        if not isinstance(payload, dict):
            raise ValueError("payload_json must contain an object")
        if str(payload.get("workflow_type") or "").strip() == "intake_workflow":
            raise _SkipLegacyRoutineError(
                "intake workflow triggers must be migrated by the intake subsystem"
            )

        tenant_id = str(payload.get("customer_id") or "").strip()
        if resolve_tenant_id is not None and tenant_id:
            tenant_id = str(resolve_tenant_id(tenant_id) or "").strip()
        tenant_id = self._tenant_id(tenant_id)
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("payload.instruction is required")

        timezone = str(payload.get("timezone") or default_timezone).strip()
        schedule_value = str(row["schedule"] or "").strip()
        if bool(row["is_cron"]):
            trigger: At | Cron = Cron(timezone=timezone, expression=schedule_value)
        else:
            try:
                run_at = datetime.fromisoformat(schedule_value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("one-off schedule is not an ISO datetime") from exc
            if run_at.tzinfo is None:
                run_at = run_at.replace(tzinfo=ZoneInfo(timezone))
            trigger = At(timezone=timezone, run_at=run_at)

        if str(payload.get("action") or "").strip().lower() == "reminder":
            action: Reminder | AgentJob = Reminder(message=instruction)
        else:
            action = AgentJob(instruction=instruction)

        created_at = self._legacy_datetime(row["created_at"])
        updated_at = self._legacy_datetime(row["updated_at"])
        enabled = bool(row["enabled"])
        if isinstance(trigger, At) and trigger.run_at.astimezone(UTC) <= self._now():
            enabled = False
        return Schedule(
            id=routine_id,
            tenant_id=tenant_id,
            revision=1,
            name=str(row["name"] or "Unnamed").strip() or "Unnamed",
            trigger=trigger,
            action=action,
            notify_owner=bool(payload.get("notify_user", False)),
            enabled=enabled,
            created_at=created_at,
            updated_at=updated_at,
        )

    @staticmethod
    def _legacy_datetime(value: Any) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("legacy timestamp must be an ISO datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("legacy timestamp must include a UTC offset")
        return parsed.astimezone(UTC)


__all__ = [
    "ScheduleConflictError",
    "ScheduleNotFoundError",
    "ScheduleService",
]
