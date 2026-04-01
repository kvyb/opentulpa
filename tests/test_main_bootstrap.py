from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from opentulpa import __main__ as entry


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


def test_telegram_bot_commands_only_include_start_status_and_fresh() -> None:
    commands = entry._telegram_bot_commands()
    assert [str(item.get("command", "")).strip() for item in commands] == [
        "start",
        "status",
        "fresh",
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
    called: dict[str, object] = {}

    class _Resp:
        status_code = 200
        content = b'{"ok":true}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": True}

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, object] | None = None) -> _Resp:
            called["url"] = url
            called["json"] = json or {}
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = SimpleNamespace(telegram_bot_token="123:abc")
    entry._auto_configure_telegram_commands(settings)

    assert str(called.get("url", "")).endswith("/setMyCommands")
    payload = called.get("json")
    assert isinstance(payload, dict)
    commands = payload.get("commands", [])
    assert isinstance(commands, list)
    assert any(str(item.get("command", "")).strip() == "fresh" for item in commands if isinstance(item, dict))
