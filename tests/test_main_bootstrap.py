from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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

from opentulpa import __main__ as entry  # noqa: E402


def test_resolve_public_base_url_prefers_explicit_public_base(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com/")
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "ignored.up.railway.app")
    assert entry._resolve_public_base_url() == "https://example.com"


def test_resolve_public_base_url_falls_back_to_railway_domain(monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "my-app.up.railway.app")
    assert entry._resolve_public_base_url() == "https://my-app.up.railway.app"


def test_ensure_telegram_webhook_secret_uses_existing(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "env-secret")
    settings = SimpleNamespace(telegram_webhook_secret="settings-secret")
    assert entry._ensure_telegram_webhook_secret(settings) == "settings-secret"


def test_ensure_telegram_webhook_secret_generates_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    settings = SimpleNamespace(telegram_webhook_secret=None)
    generated = entry._ensure_telegram_webhook_secret(settings)
    assert generated
    assert os.environ.get("TELEGRAM_WEBHOOK_SECRET") == generated


def test_shutdown_grace_seconds_defaults_and_parses(monkeypatch) -> None:
    monkeypatch.delenv("OPENTULPA_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", raising=False)
    assert entry._shutdown_grace_seconds() == 300

    monkeypatch.setenv("OPENTULPA_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "45.5")
    assert entry._shutdown_grace_seconds() == 45


def test_auto_configure_telegram_webhook_posts_secret_and_business_updates(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Resp:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": True}

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, data: dict[str, object] | None = None) -> _Resp:
            calls.append({"url": url, "data": data or {}})
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com/")
    settings = SimpleNamespace(
        telegram_bot_token="123:abc",
        telegram_webhook_secret="settings-secret",
    )

    entry._auto_configure_telegram_webhook(settings)

    assert len(calls) == 1
    assert str(calls[0]["url"]).endswith("/setWebhook")
    payload = calls[0]["data"]
    assert isinstance(payload, dict)
    assert payload["url"] == "https://example.com/webhook/telegram"
    assert payload["secret_token"] == "settings-secret"
    allowed_updates = json.loads(str(payload["allowed_updates"]))
    assert "business_connection" in allowed_updates
    assert "business_message" in allowed_updates
    assert "edited_business_message" in allowed_updates
    assert "deleted_business_messages" in allowed_updates


def test_telegram_bot_commands_include_debug_logs() -> None:
    commands = entry._telegram_bot_commands()
    assert [str(item.get("command", "")).strip() for item in commands] == [
        "start",
        "status",
        "fresh",
        "debug_logs",
    ]


def test_bootstrap_persistent_storage_aliases_runtime_dirs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".opentulpa").mkdir()
    (project_root / ".opentulpa" / "state.db").write_text("checkpoint", encoding="utf-8")
    (project_root / "tulpa_stuff").mkdir()
    (project_root / "tulpa_stuff" / "__init__.py").write_text(
        '"""Agent-created integrations and skills."""\n',
        encoding="utf-8",
    )

    data_root = tmp_path / "data"
    entry._bootstrap_persistent_storage(project_root, str(data_root))

    assert (project_root / ".opentulpa").is_symlink()
    assert (project_root / "tulpa_stuff").is_symlink()
    assert (data_root / ".opentulpa" / "state.db").read_text(encoding="utf-8") == "checkpoint"
    assert (data_root / "tulpa_stuff" / "__init__.py").read_text(encoding="utf-8").strip()


def test_bootstrap_persistent_storage_keeps_existing_volume_contents(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "tulpa_stuff").mkdir()
    (project_root / "tulpa_stuff" / "README.md").write_text("image seed", encoding="utf-8")

    data_root = tmp_path / "data"
    (data_root / "tulpa_stuff").mkdir(parents=True)
    (data_root / "tulpa_stuff" / "README.md").write_text("persisted", encoding="utf-8")

    entry._bootstrap_persistent_storage(project_root, str(data_root))

    assert (project_root / "tulpa_stuff").is_symlink()
    assert (data_root / "tulpa_stuff" / "README.md").read_text(encoding="utf-8") == "persisted"


def test_auto_configure_telegram_commands_posts_set_my_commands(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Resp:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": True}

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, object] | None = None) -> _Resp:
            calls.append({"url": url, "json": json or {}})
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = SimpleNamespace(telegram_bot_token="123:abc")
    entry._auto_configure_telegram_commands(settings)

    assert str(calls[0].get("url", "")).endswith("/setMyCommands")
    payload = calls[0].get("json")
    assert isinstance(payload, dict)
    commands = payload.get("commands", [])
    assert isinstance(commands, list)
    assert any(str(item.get("command", "")).strip() == "fresh" for item in commands if isinstance(item, dict))
    assert any(
        str(item.get("command", "")).strip() == "debug_logs"
        for item in commands
        if isinstance(item, dict)
    )
    assert len(calls) == 1


def test_auto_configure_telegram_commands_posts_support_chat_scope(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _Resp:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": True}

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, object] | None = None) -> _Resp:
            calls.append({"url": url, "json": json or {}})
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = SimpleNamespace(
        telegram_bot_token="123:abc",
        telegram_support_user_ids="900,not-a-number",
    )
    entry._auto_configure_telegram_commands(settings)

    assert len(calls) == 2
    support_payload = calls[1]["json"]
    assert isinstance(support_payload, dict)
    assert support_payload["scope"] == {"type": "chat", "chat_id": 900}
    support_commands = support_payload["commands"]
    assert isinstance(support_commands, list)
    assert any(
        str(item.get("command", "")).strip() == "support_bind"
        for item in support_commands
        if isinstance(item, dict)
    )
