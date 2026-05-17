from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.scheduler.models import Routine
from opentulpa.scheduler.service import SchedulerService


def _mk_client(
    tmp_path: Path,
    *,
    customer_profiles: CustomerProfileService | None = None,
) -> TestClient:
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    scheduler.add_routine(
        Routine(
            id="rtn_user1",
            name="User1 Routine",
            schedule="0 9 * * *",
            payload={"customer_id": "telegram_1"},
            is_cron=True,
        )
    )
    scheduler.add_routine(
        Routine(
            id="rtn_user2",
            name="User2 Routine",
            schedule="0 10 * * *",
            payload={"customer_id": "telegram_2"},
            is_cron=True,
        )
    )
    app = create_app(scheduler=scheduler, customer_profile_service=customer_profiles)
    return TestClient(app)


def test_scheduler_routine_filter_and_owner_delete(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        listed = client.get("/internal/scheduler/routines", params={"customer_id": "telegram_1"})
        assert listed.status_code == 200
        routines = listed.json()["routines"]
        assert {r["id"] for r in routines} == {"rtn_user1"}

        denied = client.request(
            "DELETE",
            "/internal/scheduler/routine/rtn_user2",
            params={"customer_id": "telegram_1"},
        )
        assert denied.status_code == 403

        deleted = client.request(
            "DELETE",
            "/internal/scheduler/routine/rtn_user1",
            params={"customer_id": "telegram_1"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["ok"] is True


def test_scheduler_routine_filter_resolves_customer_alias(tmp_path: Path) -> None:
    profiles = CustomerProfileService(tmp_path / "profiles.db")
    profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="1")

    with _mk_client(tmp_path, customer_profiles=profiles) as client:
        listed = client.get("/internal/scheduler/routines", params={"customer_id": "usr_default"})
        assert listed.status_code == 200
        assert {r["id"] for r in listed.json()["routines"]} == {"rtn_user1"}


def test_scheduler_route_requires_instruction_payload(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        created = client.post(
            "/internal/scheduler/routine",
            json={
                "name": "Missing Instruction",
                "schedule": "0 * * * *",
                "is_cron": True,
                "enabled": True,
                "payload": {"customer_id": "telegram_1", "notify_user": True},
            },
        )
        assert created.status_code == 400
        assert created.json()["detail"] == "payload.instruction is required"
