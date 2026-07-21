from __future__ import annotations

import httpx
import pytest

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
