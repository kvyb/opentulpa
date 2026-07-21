from __future__ import annotations

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
