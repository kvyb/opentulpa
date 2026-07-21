from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from opentulpa.client import local_server


class _Process:
    pid = 43210

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None


def test_local_server_starts_detached_and_persists_private_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "pyproject.toml").write_text("[project]\nname='opentulpa'\n", encoding="utf-8")
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(local_server, "_source_root", lambda: source_root)
    monkeypatch.setattr(local_server, "_port_available", lambda port: True)
    monkeypatch.setattr(local_server, "_wait_ready", lambda *args, **kwargs: True)
    launches: list[tuple[list[str], dict[str, object]]] = []

    def launch(command: list[str], **kwargs: object) -> _Process:
        launches.append((command, kwargs))
        return _Process()

    monkeypatch.setattr(local_server.subprocess, "Popen", launch)

    url = local_server.ensure_local_server()

    assert url == "http://127.0.0.1:8000"
    assert launches[0][0][-2:] == ["-m", "opentulpa.host"]
    assert launches[0][1]["start_new_session"] is True
    state_path = tmp_path / "data" / "bootstrap" / "local-server.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 43210
    assert payload["url"] == url
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600


def test_loopback_url_detection() -> None:
    assert local_server.is_loopback_url("http://127.0.0.1:8000")
    assert local_server.is_loopback_url("http://localhost:9000")
    assert not local_server.is_loopback_url("https://tulpa.example")


def test_first_local_server_does_not_adopt_an_unremembered_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(local_server, "_port_available", lambda port: False)
    monkeypatch.setattr(local_server, "_free_port", lambda: 8123)
    monkeypatch.setattr(local_server, "_host_ready", lambda url: url.endswith(":8000"))
    monkeypatch.setattr(local_server, "_wait_ready", lambda url, **kwargs: url.endswith(":8123"))
    monkeypatch.setattr(local_server.subprocess, "Popen", lambda *args, **kwargs: _Process())

    assert local_server.ensure_local_server() == "http://127.0.0.1:8123"
