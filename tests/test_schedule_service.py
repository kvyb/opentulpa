from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from opentulpa.schedules.models import (
    AgentJob,
    At,
    Cron,
    Reminder,
    ScheduleWrite,
)
from opentulpa.schedules.service import (
    ScheduleConflictError,
    ScheduleNotFoundError,
    ScheduleService,
)
from opentulpa.specs import (
    AgentSpecStore,
    TriggerSpecService,
    TriggerSpecStore,
    seed_default_agent_spec_refs,
)

NOW = datetime(2026, 7, 19, 12, tzinfo=UTC)


def _service(
    tmp_path: Path,
    *,
    clock: Callable[[], datetime] | None = None,
) -> ScheduleService:
    effective_clock = clock or (lambda: NOW)
    agent_specs = AgentSpecStore(tmp_path / "agent_specs.db", clock=effective_clock)
    triggers = TriggerSpecStore(
        tmp_path / "trigger_specs.db",
        agent_specs=agent_specs,
        clock=effective_clock,
    )

    def resolve(tenant_id: str):
        ref = agent_specs.get_active_ref(tenant_id=tenant_id, spec_id="routine")
        if ref is None:
            ref = seed_default_agent_spec_refs(
                agent_specs,
                tenant_id=tenant_id,
                actor_id="test",
            )["routine"]
        return ref

    return ScheduleService(
        TriggerSpecService(triggers),
        resolve_agent_spec=resolve,
        clock=effective_clock,
    )


def _cron_write(*, name: str = "Daily brief") -> ScheduleWrite:
    return ScheduleWrite(
        name=name,
        trigger=Cron(timezone="Europe/Moscow", expression="0 9 * * *"),
        action=AgentJob(instruction="Prepare a concise daily brief"),
        notify_owner=True,
    )


def test_schedule_save_is_tenant_scoped_and_revisioned(tmp_path: Path) -> None:
    service = _service(tmp_path)

    created = service.save(tenant_id="tenant-a", schedule_id="sch_shared", write=_cron_write())
    other = service.save(tenant_id="tenant-b", schedule_id="sch_shared", write=_cron_write())

    assert created.revision == 1
    assert other.revision == 1
    assert service.list(tenant_id="tenant-a") == [created]
    assert service.get(tenant_id="tenant-a", schedule_id="sch_shared") == created

    with pytest.raises(ScheduleConflictError, match="expected_revision is required"):
        service.save(tenant_id="tenant-a", schedule_id="sch_shared", write=_cron_write())
    with pytest.raises(ScheduleConflictError, match="expected revision 9, found 1"):
        service.save(
            tenant_id="tenant-a",
            schedule_id="sch_shared",
            write=_cron_write(),
            expected_revision=9,
        )

    updated = service.save(
        tenant_id="tenant-a",
        schedule_id="sch_shared",
        write=_cron_write(name="Updated brief"),
        expected_revision=1,
    )
    assert updated.revision == 2
    assert updated.name == "Updated brief"
    assert updated.created_at == created.created_at

    with pytest.raises(ScheduleConflictError, match="expected revision 1, found 2"):
        service.delete(tenant_id="tenant-a", schedule_id="sch_shared", expected_revision=1)
    with pytest.raises(ScheduleNotFoundError):
        service.delete(tenant_id="tenant-c", schedule_id="sch_shared", expected_revision=1)

    service.delete(tenant_id="tenant-a", schedule_id="sch_shared", expected_revision=2)
    assert service.list(tenant_id="tenant-a") == []
    assert len(service.list(tenant_id="tenant-b")) == 1


def test_schedule_update_preserves_original_created_at(tmp_path: Path) -> None:
    current = NOW
    service = _service(tmp_path, clock=lambda: current)
    created = service.save(
        tenant_id="tenant-a",
        schedule_id="sch_history",
        write=_cron_write(),
    )
    current = NOW + timedelta(days=1)

    updated = service.save(
        tenant_id="tenant-a",
        schedule_id="sch_history",
        write=_cron_write(name="Updated"),
        expected_revision=created.revision,
    )

    assert updated.created_at == NOW
    assert updated.updated_at == NOW + timedelta(days=1)


def test_missed_one_off_is_persisted_disabled_and_never_replayed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    write = ScheduleWrite(
        name="Already missed",
        trigger=At(timezone="UTC", run_at=datetime(2026, 7, 19, 11, tzinfo=UTC)),
        action=Reminder(message="This should not replay"),
        enabled=True,
    )

    schedule = service.save(tenant_id="tenant-a", write=write)

    assert schedule.enabled is False
    assert ScheduleService.executor_options(schedule) == {
        "misfire_grace_time": 1,
        "coalesce": False,
    }
    reloaded = _service(tmp_path).get(
        tenant_id="tenant-a",
        schedule_id=schedule.id,
    )
    assert reloaded == schedule


def _create_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE routines (
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
    rows = [
        (
            "rtn_cron",
            "Daily brief",
            "0 9 * * *",
            json.dumps(
                {
                    "customer_id": "telegram_1",
                    "instruction": "Prepare a daily brief",
                    "notify_user": True,
                }
            ),
            1,
            1,
            "2026-07-01T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
        ),
        (
            "rtn_missed",
            "Missed reminder",
            "2026-07-18T09:00:00",
            json.dumps(
                {
                    "customer_id": "telegram_1",
                    "instruction": "Call the customer",
                    "action": "reminder",
                }
            ),
            1,
            0,
            "2026-07-01T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
        ),
        (
            "rtn_invalid",
            "Unknown owner",
            "0 8 * * *",
            json.dumps({"instruction": "Cannot safely migrate this"}),
            1,
            1,
            "2026-07-01T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
        ),
        (
            "rtn_intake",
            "Legacy intake polling",
            "*/5 * * * *",
            json.dumps(
                {
                    "customer_id": "telegram_1",
                    "instruction": "Poll the intake workflow",
                    "workflow_type": "intake_workflow",
                    "workflow_id": "iwf_1",
                }
            ),
            1,
            1,
            "2026-07-01T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
        ),
    ]
    conn.executemany("INSERT INTO routines VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_legacy_migration_is_explicit_dry_runnable_and_idempotent(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.db"
    _create_legacy_db(legacy_path)
    service = _service(tmp_path)

    dry_run = service.migrate_legacy_routines(
        legacy_db_path=legacy_path,
        default_timezone="Europe/Moscow",
        resolve_tenant_id=lambda value: {"telegram_1": "tenant-a"}[value],
        dry_run=True,
    )
    assert dry_run.model_dump(exclude={"source_checksum", "issues"}) == {
        "dry_run": True,
        "scanned": 4,
        "eligible": 2,
        "migrated": 0,
        "skipped": 1,
        "invalid": 1,
    }
    assert len(dry_run.source_checksum) == 64
    assert {(issue.routine_id, issue.disposition) for issue in dry_run.issues} == {
        ("rtn_invalid", "invalid"),
        ("rtn_intake", "skipped"),
    }
    assert service.list(tenant_id="tenant-a") == []

    migrated = service.migrate_legacy_routines(
        legacy_db_path=legacy_path,
        default_timezone="Europe/Moscow",
        resolve_tenant_id=lambda value: {"telegram_1": "tenant-a"}[value],
    )
    assert migrated.migrated == 2
    assert migrated.source_checksum == dry_run.source_checksum
    schedules = {schedule.id: schedule for schedule in service.list(tenant_id="tenant-a")}
    assert isinstance(schedules["rtn_cron"].trigger, Cron)
    assert isinstance(schedules["rtn_cron"].action, AgentJob)
    assert schedules["rtn_cron"].notify_owner is True
    assert isinstance(schedules["rtn_missed"].trigger, At)
    assert isinstance(schedules["rtn_missed"].action, Reminder)
    assert schedules["rtn_missed"].enabled is False

    rerun = service.migrate_legacy_routines(
        legacy_db_path=legacy_path,
        default_timezone="Europe/Moscow",
        resolve_tenant_id=lambda value: {"telegram_1": "tenant-a"}[value],
    )
    assert rerun.migrated == 0
    assert rerun.skipped == 3
    assert rerun.invalid == 1
