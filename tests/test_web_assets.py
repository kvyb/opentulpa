from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from opentulpa.api.principal import OwnerPrincipalResolver
from opentulpa.api.web_auth import OWNER_SESSION_COOKIE, local_owner_cookie_enabled
from opentulpa.interfaces import web


def test_web_interface_serves_static_assets_with_fixed_security_headers() -> None:
    app = FastAPI()
    web.register_owner_web_interface(app)
    client = TestClient(app)

    page = client.get("/")
    script = client.get("/assets/app.js")
    favicon = client.get("/assets/favicon.svg")

    assert page.status_code == 200
    assert '<script src="/assets/app.js" defer></script>' in page.text
    assert script.status_code == 200
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")
    assert "POST /v2/agent/runs" not in script.text
    assert "What are we building?" in page.text
    assert "Deep Agents · owner" in page.text
    assert 'id="new-thread"' in page.text
    assert "localStorage.otThread = `web-${crypto.randomUUID()}`" in script.text
    assert "const pendingRunText = 'Planning next moves'" in script.text
    assert "event.type === 'tool.started'" in script.text
    assert "event.type === 'tool.started' && data.name" in script.text
    assert "event.type === 'tool.completed'" in script.text
    assert "body.textContent = data.message || 'Run failed.'" in script.text
    assert "Run completed without a response." in script.text
    assert page.headers["cache-control"] == "no-store"
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"


def test_loopback_web_interface_bootstraps_http_only_owner_session() -> None:
    app = FastAPI()
    resolver = OwnerPrincipalResolver(
        token="private-owner-token",
        tenant_id="owner",
        local_cookie_token="ephemeral-browser-session",
    )
    web.register_owner_web_interface(
        app,
        local_owner_token="ephemeral-browser-session",
    )

    @app.post("/protected")
    async def protected(request: Request) -> dict[str, str]:
        return {"tenant_id": resolver(request).tenant_id}

    client = TestClient(
        app,
        base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 50123),
    )
    page = client.get("/", headers={"Sec-Fetch-Site": "none"})

    cookie = page.headers["set-cookie"]
    assert f"{OWNER_SESSION_COOKIE}=" in cookie
    assert "private-owner-token" not in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert client.post(
        "/protected",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Sec-Fetch-Site": "same-origin",
        },
    ).json() == {"tenant_id": "owner"}
    assert client.post("/protected").status_code == 401


@pytest.mark.parametrize(
    ("base_url", "client_host", "headers"),
    [
        ("http://127.0.0.1:8000", "192.0.2.10", {}),
        ("http://example.test", "127.0.0.1", {}),
        (
            "http://127.0.0.1:8000",
            "127.0.0.1",
            {"X-Forwarded-Host": "public.example.test"},
        ),
        (
            "http://127.0.0.1:8000",
            "127.0.0.1",
            {"Sec-Fetch-Site": "cross-site"},
        ),
    ],
)
def test_owner_session_is_never_bootstrapped_outside_direct_loopback(
    base_url: str,
    client_host: str,
    headers: dict[str, str],
) -> None:
    app = FastAPI()
    web.register_owner_web_interface(app, local_owner_token="private-owner-token")
    client = TestClient(app, base_url=base_url, client=(client_host, 50123))

    response = client.get("/", headers=headers)

    assert response.status_code == 200
    assert "set-cookie" not in response.headers


def test_public_mode_does_not_enable_local_cookie_auth() -> None:
    assert local_owner_cookie_enabled(
        bind_host="127.0.0.1",
        public_base_url="",
    ) is True
    assert local_owner_cookie_enabled(
        bind_host="0.0.0.0",
        public_base_url="",
    ) is False
    assert local_owner_cookie_enabled(
        bind_host="127.0.0.1",
        public_base_url="https://public.example.test",
    ) is False


@pytest.mark.parametrize("path", ["../README.md", ".env", "bad.py", "/absolute.js"])
def test_web_asset_path_fails_closed(path: str) -> None:
    with pytest.raises(HTTPException):
        web._asset_path(path)


def test_web_asset_path_rejects_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "outside.js"
    outside.write_text("secret", encoding="utf-8")
    root = tmp_path / "assets"
    root.mkdir()
    (root / "escape.js").symlink_to(outside)
    monkeypatch.setattr(web, "_ASSET_ROOT", root)

    with pytest.raises(HTTPException):
        web._asset_path("escape.js")
