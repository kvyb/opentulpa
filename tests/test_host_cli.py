from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import httpx
import pytest

from opentulpa.client.config import Connection
from opentulpa.host import cli


def _client_factory(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.MockTransport,
) -> None:
    original = httpx.Client

    def client(**kwargs: object) -> httpx.Client:
        return original(transport=handler, **kwargs)

    monkeypatch.setattr(cli.httpx, "Client", client)


def test_connect_accepts_owner_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer owner-token"
        return httpx.Response(200, json={"claimed": True, "authenticated": True})

    _client_factory(monkeypatch, httpx.MockTransport(handler))

    assert cli._connect_credential("https://tulpa.example", "owner-token") == "owner-token"  # noqa: SLF001


def test_connect_exchanges_pairing_code_for_owner_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"claimed": False, "authenticated": False})
        return httpx.Response(200, json={"claimed": True, "owner_token": "issued-owner-token"})

    _client_factory(monkeypatch, httpx.MockTransport(handler))

    assert (
        cli._connect_credential("https://tulpa.example", "one-time-pairing-code")  # noqa: SLF001
        == "issued-owner-token"
    )
    assert requests[1].method == "POST"
    assert requests[1].url.path == "/_host/api/claim"
    assert b"one-time-pairing-code" in requests[1].content


def test_bare_opentulpa_opens_tui_not_server(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(cli, "_open_tui", lambda: opened.append(True))
    monkeypatch.setattr(cli, "serve", lambda: pytest.fail("bare command started the server"))
    monkeypatch.setattr("sys.argv", ["opentulpa"])

    cli.main()

    assert opened == [True]


def test_first_launch_defaults_to_private_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Connection(
        url="http://127.0.0.1:8123",
        token="",
        thread_id="thread-1",
        credential_storage="none",
    )
    configured: list[Connection] = []
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(cli, "ensure_local_server", lambda: connection.url)
    monkeypatch.setattr(cli, "_connect_credential", lambda url, token: "")
    monkeypatch.setattr(cli, "save_connection", lambda url, token: connection)
    monkeypatch.setattr(cli, "_configure_runtime", configured.append)

    assert cli._first_connection() == connection  # noqa: SLF001
    assert configured == [connection]


def test_first_launch_configures_model_without_printing_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"configured": False})
        return httpx.Response(200, json={"runtime": "ready"})

    _client_factory(monkeypatch, httpx.MockTransport(handler))
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    answers = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "model-secret")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    connection = Connection(
        url="http://127.0.0.1:8000",
        token="",
        thread_id="thread-1",
        credential_storage="none",
    )

    cli._configure_runtime(connection)  # noqa: SLF001

    assert requests[-1].method == "PUT"
    assert b"model-secret" in requests[-1].content
    assert b"moonshotai/kimi-k3" in requests[-1].content
    assert "model-secret" not in capsys.readouterr().out


def test_server_origin_and_pairing_code_are_pasteable(tmp_path: Path) -> None:
    assert (
        cli._server_origin(  # noqa: SLF001
            host="0.0.0.0", port=8000, public_url="https://tulpa.example/"
        )
        == "https://tulpa.example"
    )
    code = cli._private_pairing_code(tmp_path / "pairing-code")  # noqa: SLF001
    assert len(code) == 17
    assert code.count("-") == 2


def test_native_tui_receives_credentials_only_through_inherited_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "opentulpa-tui"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o700)
    captured: dict[str, object] = {}

    class Process:
        returncode = 0

        def __init__(
            self,
            args: list[str],
            *,
            env: dict[str, str],
            pass_fds: tuple[int, ...],
        ) -> None:
            captured["args"] = args
            captured["env"] = env
            captured["pass_fds"] = pass_fds
            self.connection_fd = os.dup(int(env["OPENTULPA_CONNECTION_FD"]))
            self.state_fd = os.dup(int(env["OPENTULPA_STATE_FD"]))

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            with os.fdopen(self.connection_fd, "rb") as stream:
                captured["connection"] = json.loads(stream.read())
            with os.fdopen(self.state_fd, "wb") as stream:
                stream.write(b'{"thread_id":"thread-2"}')
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

    connection = Connection(
        url="https://tulpa.example",
        token="owner-secret",
        thread_id="thread-1",
        credential_storage="keyring",
    )
    updated: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "_ensure_tui_binary", lambda: binary)
    monkeypatch.setattr(cli, "_migrate_legacy_session_names", lambda _: None)
    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    monkeypatch.setattr(
        cli,
        "update_connection",
        lambda value, **changes: updated.append((value.thread_id, changes["thread_id"])),
    )

    cli._launch_tui(connection)  # noqa: SLF001

    assert captured["args"] == [str(binary)]
    assert "owner-secret" not in " ".join(captured["args"])
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "owner-secret" not in environment.values()
    assert captured["connection"] == {
        "url": "https://tulpa.example",
        "token": "owner-secret",
        "thread_id": "thread-1",
        "credential_storage": "keyring",
    }
    assert updated == [("thread-1", "thread-2")]


def test_source_checkout_prefers_its_built_tui_over_an_installed_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system = {"Darwin": "darwin", "Linux": "linux"}[platform.system()]
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64"}[
        platform.machine()
    ]
    project = tmp_path / "project"
    module = project / "src" / "opentulpa" / "host" / "cli.py"
    module.parent.mkdir(parents=True)
    local_binary = project / "clients" / "tui" / "dist" / f"opentulpa-tui-{system}-{machine}"
    local_binary.parent.mkdir(parents=True)
    local_binary.write_text("local", encoding="utf-8")
    local_binary.chmod(0o700)
    (local_binary.parent / "manifest.json").write_text(
        '{"protocol_version":2,"source_digest":"source-digest"}',
        encoding="utf-8",
    )
    installed_binary = tmp_path / "bin" / "opentulpa-tui"
    installed_binary.parent.mkdir()
    installed_binary.write_text("installed", encoding="utf-8")
    installed_binary.chmod(0o700)
    monkeypatch.setattr(cli, "__file__", str(module))
    monkeypatch.setattr(cli.shutil, "which", lambda _: str(installed_binary))
    monkeypatch.setattr(cli, "_tui_source_digest", lambda _: "source-digest")
    monkeypatch.setattr(cli, "_tui_protocol", lambda _: "2")

    assert cli._ensure_tui_binary() == local_binary  # noqa: SLF001
