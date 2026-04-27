from __future__ import annotations

import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

apscheduler_module = types.ModuleType("apscheduler")
schedulers_module = types.ModuleType("apscheduler.schedulers")
asyncio_module = types.ModuleType("apscheduler.schedulers.asyncio")
triggers_module = types.ModuleType("apscheduler.triggers")
cron_module = types.ModuleType("apscheduler.triggers.cron")
date_module = types.ModuleType("apscheduler.triggers.date")
mem0_module = types.ModuleType("mem0")


class _DummyAsyncIOScheduler:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs
        self.started = False
        self.jobs: dict[str, dict[str, object]] = {}

    def add_job(self, func: object, trigger: object, *, id: str, args: list[object], **kwargs: object) -> None:
        self.jobs[str(id)] = {
            "func": func,
            "trigger": trigger,
            "args": list(args),
            **kwargs,
        }

    def remove_job(self, job_id: str) -> None:
        self.jobs.pop(str(job_id), None)

    def start(self) -> None:
        self.started = True

    def shutdown(self, wait: bool = True) -> None:
        _ = wait
        self.started = False


class _DummyCronTrigger:
    @classmethod
    def from_crontab(cls, value: str) -> _DummyCronTrigger:
        _ = value
        return cls()


class _DummyDateTrigger:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class _DummyMemory:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


asyncio_module.AsyncIOScheduler = _DummyAsyncIOScheduler
cron_module.CronTrigger = _DummyCronTrigger
date_module.DateTrigger = _DummyDateTrigger
apscheduler_module.schedulers = schedulers_module
apscheduler_module.triggers = triggers_module
schedulers_module.asyncio = asyncio_module
triggers_module.cron = cron_module
triggers_module.date = date_module
mem0_module.Memory = _DummyMemory
sys.modules.setdefault("apscheduler", apscheduler_module)
sys.modules.setdefault("apscheduler.schedulers", schedulers_module)
sys.modules.setdefault("apscheduler.schedulers.asyncio", asyncio_module)
sys.modules.setdefault("apscheduler.triggers", triggers_module)
sys.modules.setdefault("apscheduler.triggers.cron", cron_module)
sys.modules.setdefault("apscheduler.triggers.date", date_module)
sys.modules.setdefault("mem0", mem0_module)

from opentulpa.api.app import create_app  # noqa: E402
from opentulpa.skills.service import SkillStoreService  # noqa: E402


def _mk_client(tmp_path: Path) -> TestClient:
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=skills)
    return TestClient(app)


def test_public_base_url_route_returns_empty_when_unconfigured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    client = _mk_client(tmp_path)

    with client:
        response = client.get("/internal/system/public_base_url")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["available"] is False
    assert payload["public_base_url"] == ""
