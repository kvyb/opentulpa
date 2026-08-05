from __future__ import annotations

import json
import stat
import subprocess
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
    monkeypatch.setattr(
        local_server,
        "_capture_process_identity",
        lambda *args, **kwargs: local_server._ProcessIdentity(  # noqa: SLF001
            start_token="test:1",
            executable="/test/python",
            argv=("/test/python", "-m", "opentulpa.host"),
            command="",
        ),
    )

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


def test_port_probe_can_reuse_a_recently_released_local_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    class Listener:
        def __enter__(self) -> Listener:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def setsockopt(self, level: int, option: int, value: int) -> None:
            calls.append((level, option, value))

        def bind(self, address: tuple[str, int]) -> None:
            assert address == ("127.0.0.1", 8000)

    monkeypatch.setattr(local_server.socket, "socket", lambda *args: Listener())

    assert local_server._port_available(8000) is True  # noqa: SLF001
    assert calls == [(local_server.socket.SOL_SOCKET, local_server.socket.SO_REUSEADDR, 1)]


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
    monkeypatch.setattr(
        local_server,
        "_capture_process_identity",
        lambda *args, **kwargs: local_server._ProcessIdentity(  # noqa: SLF001
            start_token="test:1",
            executable="/test/python",
            argv=("/test/python", "-m", "opentulpa.host"),
            command="",
        ),
    )

    assert local_server.ensure_local_server() == "http://127.0.0.1:8123"


def test_local_server_launches_exact_new_controller_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = "a" * 64
    executable = tmp_path / "controller" / "generations" / generation_id / "bin" / "opentulpa-host"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o500)
    source_root = tmp_path / "source"
    source_root.mkdir()
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(local_server, "_source_root", lambda: source_root)
    monkeypatch.setattr(local_server, "_port_available", lambda port: True)
    monkeypatch.setattr(local_server, "_wait_ready", lambda *args, **kwargs: True)
    launches: list[list[str]] = []

    def launch(command: list[str], **kwargs: object) -> _Process:
        del kwargs
        launches.append(command)
        return _Process()

    monkeypatch.setattr(local_server.subprocess, "Popen", launch)
    monkeypatch.setattr(
        local_server,
        "_capture_process_identity",
        lambda *args, **kwargs: local_server._ProcessIdentity(  # noqa: SLF001
            start_token="test:2",
            executable="/test/python",
            argv=(str(executable),),
            command="",
        ),
    )

    local_server.ensure_local_server(
        controller_executable=executable,
        controller_generation_id=generation_id,
    )

    assert launches == [[str(executable)]]
    payload = json.loads(
        (tmp_path / "data" / "bootstrap" / "local-server.json").read_text(encoding="utf-8")
    )
    assert payload["launch_argv"] == [str(executable)]
    assert payload["controller_generation_id"] == generation_id


def test_restart_verifies_old_pid_then_requests_new_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = local_server.LocalServerState(
        pid=43210,
        port=8000,
        url="http://127.0.0.1:8000",
        source_root=str(tmp_path),
        started_at="now",
        executable="/old/generation/bin/python",
        launch_argv=("/old/generation/bin/python", "-m", "opentulpa.host"),
        controller_generation_id="b" * 64,
        process_start_token="test:old",
        process_executable="/old/generation/bin/python",
    )
    new_id = "c" * 64
    new_executable = tmp_path / "controller" / "generations" / new_id / "bin" / "opentulpa-host"
    new_executable.parent.mkdir(parents=True)
    new_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    new_executable.chmod(0o500)
    data = tmp_path / "data"
    bootstrap = data / "bootstrap"
    bootstrap.mkdir(parents=True, mode=0o700)
    data.chmod(0o700)
    bootstrap.chmod(0o700)
    state_path = bootstrap / "local-server.json"
    state_path.write_text("{}\n", encoding="utf-8")
    state_path.chmod(0o600)
    states = iter((old, old))
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(data))
    monkeypatch.setattr(local_server, "_load_state", lambda: next(states))
    matches = iter((True, False, False))
    monkeypatch.setattr(local_server, "_pid_matches_local_host", lambda state: next(matches))
    monkeypatch.setattr(local_server, "_open_verified_pidfd", lambda state: 17)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        local_server.signal,
        "pidfd_send_signal",
        lambda descriptor, sent_signal: signals.append((descriptor, sent_signal)),
        raising=False,
    )
    monkeypatch.setattr(local_server.os, "close", lambda descriptor: None)
    starts: list[dict[str, object]] = []
    monkeypatch.setattr(
        local_server,
        "ensure_local_server",
        lambda **kwargs: starts.append(kwargs) or old.url,
    )

    url = local_server.restart_remembered_local_server(
        controller_executable=new_executable,
        controller_generation_id=new_id,
    )

    assert url == old.url
    assert signals == [(17, 15)]
    assert starts[0]["controller_executable"] == new_executable
    assert starts[0]["controller_generation_id"] == new_id


def test_pid_match_rejects_reused_pid_birth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    state = local_server.LocalServerState(
        pid=43210,
        port=8000,
        url="http://127.0.0.1:8000",
        source_root="/source",
        started_at="now",
        executable="/controller/bin/opentulpa-host",
        launch_argv=("/controller/bin/opentulpa-host",),
        process_start_token="linux:10",
        process_executable="/controller/bin/python",
    )
    monkeypatch.setattr(
        local_server,
        "_read_process_identity",
        lambda pid: local_server._ProcessIdentity(  # noqa: SLF001
            start_token="linux:11",
            executable="/controller/bin/python",
            argv=state.launch_argv,
            command="",
        ),
    )

    assert local_server._pid_matches_local_host(state) is False  # noqa: SLF001


def test_verified_pidfd_is_closed_when_identity_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    state = local_server.LocalServerState(
        pid=43210,
        port=8000,
        url="http://127.0.0.1:8000",
        source_root="/source",
        started_at="now",
        executable="/controller/bin/opentulpa-host",
        launch_argv=("/controller/bin/opentulpa-host",),
        process_start_token="linux:10",
    )
    closed: list[int] = []
    monkeypatch.setattr(local_server.os, "pidfd_open", lambda pid, flags: 17, raising=False)
    monkeypatch.setattr(local_server.signal, "pidfd_send_signal", lambda *args: None, raising=False)
    monkeypatch.setattr(local_server, "_pid_matches_local_host", lambda candidate: False)
    monkeypatch.setattr(local_server.os, "close", lambda descriptor: closed.append(descriptor))

    with pytest.raises(local_server.LocalServerError, match="identity changed"):
        local_server._open_verified_pidfd(state)  # noqa: SLF001

    assert closed == [17]


def test_restart_refuses_racy_pid_signal_without_pidfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = local_server.LocalServerState(
        pid=43210,
        port=8000,
        url="http://127.0.0.1:8000",
        source_root=str(tmp_path),
        started_at="now",
        executable="/old/controller/bin/opentulpa-host",
        launch_argv=("/old/controller/bin/opentulpa-host",),
        process_start_token="test:old",
    )
    generation_id = "d" * 64
    executable = tmp_path / "controller" / "generations" / generation_id / "bin" / "opentulpa-host"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o500)
    monkeypatch.setattr(local_server, "_load_state", lambda: state)
    monkeypatch.setattr(local_server, "_pid_matches_local_host", lambda candidate: True)
    monkeypatch.setattr(local_server, "_open_verified_pidfd", lambda candidate: None)

    with pytest.raises(local_server.LocalServerError, match="requires pidfd"):
        local_server.restart_remembered_local_server(
            controller_executable=executable,
            controller_generation_id=generation_id,
        )


def test_kernel_observed_identity_recognizes_a_shebang_process(tmp_path: Path) -> None:
    executable = tmp_path / "console-script"
    executable.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    executable.chmod(0o700)
    process = subprocess.Popen(  # noqa: S603
        [str(executable)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        identity = local_server._capture_process_identity(process.pid, process=process)  # noqa: SLF001
        state = local_server.LocalServerState(
            pid=process.pid,
            port=8000,
            url="http://127.0.0.1:8000",
            source_root=str(tmp_path),
            started_at="now",
            executable=str(executable),
            launch_argv=identity.argv,
            process_start_token=identity.start_token,
            process_executable=identity.executable,
            process_command=identity.command,
        )

        assert local_server._pid_matches_local_host(state) is True  # noqa: SLF001
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize(
    "kind",
    ["state-symlink", "state-hardlink", "log-symlink", "log-hardlink"],
)
def test_local_server_rejects_linked_state_and_log_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    data = tmp_path / "data"
    bootstrap = data / "bootstrap"
    bootstrap.mkdir(parents=True, mode=0o700)
    data.chmod(0o700)
    bootstrap.chmod(0o700)
    target = tmp_path / "target"
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(data))
    if kind == "state-symlink":
        (bootstrap / "local-server.json").symlink_to(target)
        with pytest.raises(local_server.LocalServerError, match="private and trusted"):
            local_server._load_state()  # noqa: SLF001
    elif kind == "state-hardlink":
        (bootstrap / "local-server.json").hardlink_to(target)
        with pytest.raises(local_server.LocalServerError, match="private and trusted"):
            local_server._load_state()  # noqa: SLF001
    elif kind == "log-symlink":
        (bootstrap / "local-server.log").symlink_to(target)
        with pytest.raises(local_server.LocalServerError, match="private and trusted"):
            local_server._open_private_append(bootstrap / "local-server.log")  # noqa: SLF001
    else:
        (bootstrap / "local-server.log").hardlink_to(target)
        with pytest.raises(local_server.LocalServerError, match="private and trusted"):
            local_server._open_private_append(bootstrap / "local-server.log")  # noqa: SLF001


def test_local_server_rejects_symlink_ancestor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    monkeypatch.setenv("OPENTULPA_DATA_ROOT", str(linked / "data"))

    with pytest.raises(local_server.LocalServerError, match="symbolic-link ancestor"):
        local_server.local_data_root()
