from __future__ import annotations

import json

import httpx

from opentulpa.bootstrap.recovery_cli import run


def _environment() -> dict[str, str]:
    return {
        "OPENTULPA_RECOVERY_URL": "http://bootstrap.local:8000",
        "OPENTULPA_RECOVERY_TOKEN": "recovery-token-" + "r" * 40,
    }


def test_recovery_cli_uses_host_headers(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"].startswith("Bearer recovery-token-")
        assert "origin" not in request.headers
        assert "referer" not in request.headers
        assert not any(name.startswith("sec-fetch-") for name in request.headers)
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"state": {"safe_mode": False}})
        raise AssertionError(f"unexpected recovery request: {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert run(["status"], environ=_environment(), client=client) == 0
    assert json.loads(capsys.readouterr().out)["state"]["safe_mode"] is False
    assert [request.url.path for request in requests] == ["/bootstrap/v1/status"]
    client.close()


def test_recovery_cli_exposes_remaining_stable_operations(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(202, json={"status": "accepted"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    commands = (
        ["rollback", "--reason", "Regression"],
        ["restart"],
        ["safe-mode"],
    )
    for command in commands:
        assert run(command, environ=_environment(), client=client) == 0
        capsys.readouterr()
    assert calls == [
        ("POST", "/bootstrap/v1/rollback"),
        ("POST", "/bootstrap/v1/restart"),
        ("POST", "/bootstrap/v1/safe-mode"),
    ]
    client.close()
