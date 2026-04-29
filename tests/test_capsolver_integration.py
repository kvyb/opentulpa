from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from opentulpa.integrations.browser_use_captcha import (
    _DETECT_CAPTCHA_SCRIPT,
    build_capsolver_controller,
    detect_browser_captcha,
    inject_browser_captcha_token,
)
from opentulpa.integrations.capsolver import CapSolverClient, CapSolverError, CapSolverSolveResult


class _FakePage:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def evaluate(self, script: str, *args: Any) -> Any:
        self.calls.append((script, args))
        if not self.responses:
            return None
        return self.responses.pop(0)


class _FakeBrowserSession:
    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def get_current_page(self) -> _FakePage:
        return self.page


class _FakeCapSolver:
    async def solve_recaptcha_v2(
        self,
        *,
        website_url: str,
        website_key: str,
    ) -> CapSolverSolveResult:
        assert website_url == "https://example.com/login"
        assert website_key == "site-key"
        return CapSolverSolveResult(
            task_id="task-1",
            token="solution-token",
            captcha_type="recaptcha_v2",
        )

    async def solve_recaptcha_v3(
        self,
        *,
        website_url: str,
        website_key: str,
        page_action: str | None = None,
    ) -> CapSolverSolveResult:
        assert website_url == "https://example.com/login"
        assert website_key == "site-key"
        assert page_action == "login"
        return CapSolverSolveResult(
            task_id="task-1",
            token="solution-token",
            captcha_type="recaptcha_v3",
        )

    async def solve_turnstile(
        self,
        *,
        website_url: str,
        website_key: str,
    ) -> CapSolverSolveResult:  # pragma: no cover - not used in this test
        raise AssertionError("unexpected turnstile solve")


def _mock_client(
    responses: list[dict[str, Any]],
) -> tuple[CapSolverClient, list[tuple[str, dict[str, Any]]], httpx.AsyncClient]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls.append((request.url.path, payload))
        if responses:
            return httpx.Response(200, json=responses.pop(0))
        return httpx.Response(200, json={"errorId": 0, "status": "processing"})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = CapSolverClient(
        api_key="cap-key",
        poll_interval_seconds=0.01,
        timeout_seconds=0.03,
        http_client=async_client,
    )
    return client, calls, async_client


@pytest.mark.asyncio
async def test_capsolver_get_balance_success() -> None:
    client, calls, async_client = _mock_client([{"errorId": 0, "balance": 10.5}])
    try:
        payload = await client.get_balance()
    finally:
        await async_client.aclose()

    assert payload["balance"] == 10.5
    assert calls == [("/getBalance", {"clientKey": "cap-key"})]


@pytest.mark.asyncio
async def test_capsolver_solve_recaptcha_success() -> None:
    client, calls, async_client = _mock_client(
        [
            {"errorId": 0, "taskId": "task-1"},
            {
                "errorId": 0,
                "status": "ready",
                "solution": {"gRecaptchaResponse": "token-1"},
            },
        ]
    )
    try:
        result = await client.solve_recaptcha_v2(
            website_url="https://example.com",
            website_key="site-key",
        )
    finally:
        await async_client.aclose()

    assert result.task_id == "task-1"
    assert result.token == "token-1"
    assert result.captcha_type == "recaptcha_v2"
    assert calls[0][0] == "/createTask"
    assert calls[0][1]["task"]["type"] == "ReCaptchaV2TaskProxyLess"
    assert calls[1] == ("/getTaskResult", {"clientKey": "cap-key", "taskId": "task-1"})


@pytest.mark.asyncio
async def test_capsolver_solve_recaptcha_v3_success() -> None:
    client, calls, async_client = _mock_client(
        [
            {"errorId": 0, "taskId": "task-1"},
            {
                "errorId": 0,
                "status": "ready",
                "solution": {"gRecaptchaResponse": "token-1"},
            },
        ]
    )
    try:
        result = await client.solve_recaptcha_v3(
            website_url="https://example.com",
            website_key="site-key",
            page_action="login",
        )
    finally:
        await async_client.aclose()

    assert result.task_id == "task-1"
    assert result.token == "token-1"
    assert result.captcha_type == "recaptcha_v3"
    assert calls[0][0] == "/createTask"
    assert calls[0][1]["task"] == {
        "type": "ReCaptchaV3TaskProxyLess",
        "websiteURL": "https://example.com",
        "websiteKey": "site-key",
        "pageAction": "login",
    }
    assert calls[1] == ("/getTaskResult", {"clientKey": "cap-key", "taskId": "task-1"})


@pytest.mark.asyncio
async def test_capsolver_create_task_failure() -> None:
    client, _, async_client = _mock_client(
        [{"errorId": 1, "errorCode": "ERROR_ZERO_BALANCE", "errorDescription": "balance is empty"}]
    )
    try:
        with pytest.raises(CapSolverError, match="ERROR_ZERO_BALANCE"):
            await client.solve_recaptcha_v2(
                website_url="https://example.com",
                website_key="site-key",
            )
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_capsolver_transport_error_becomes_capsolver_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network unavailable", request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = CapSolverClient(api_key="cap-key", http_client=async_client)
    try:
        with pytest.raises(CapSolverError, match="CapSolver request failed"):
            await client.get_balance()
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_capsolver_invalid_json_becomes_capsolver_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, content=b"not-json")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = CapSolverClient(api_key="cap-key", http_client=async_client)
    try:
        with pytest.raises(CapSolverError, match="invalid JSON"):
            await client.get_balance()
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_capsolver_poll_failure() -> None:
    client, _, async_client = _mock_client(
        [
            {"errorId": 0, "taskId": "task-1"},
            {"errorId": 0, "status": "failed"},
        ]
    )
    try:
        with pytest.raises(CapSolverError, match="task status is failed"):
            await client.solve_recaptcha_v2(
                website_url="https://example.com",
                website_key="site-key",
            )
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_capsolver_poll_timeout() -> None:
    client, _, async_client = _mock_client(
        [
            {"errorId": 0, "taskId": "task-1"},
            {"errorId": 0, "status": "processing"},
        ]
    )
    try:
        with pytest.raises(CapSolverError, match="timed out"):
            await client.solve_recaptcha_v2(
                website_url="https://example.com",
                website_key="site-key",
            )
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_capsolver_missing_site_key_does_not_call_api() -> None:
    client, calls, async_client = _mock_client([])
    try:
        with pytest.raises(CapSolverError, match="website_key is required"):
            await client.solve_recaptcha_v2(
                website_url="https://example.com",
                website_key="",
            )
    finally:
        await async_client.aclose()

    assert calls == []


@pytest.mark.asyncio
async def test_browser_captcha_detection_parses_recaptcha_json() -> None:
    page = _FakePage(
        [
            json.dumps(
                {
                    "captchaType": "recaptcha_v2",
                    "websiteUrl": "https://example.com/login",
                    "websiteKey": "site-key",
                    "marker": "g-recaptcha",
                }
            )
        ]
    )

    challenge = await detect_browser_captcha(page)

    assert challenge is not None
    assert challenge.captcha_type == "recaptcha_v2"
    assert challenge.website_url == "https://example.com/login"
    assert challenge.website_key == "site-key"


@pytest.mark.asyncio
async def test_browser_captcha_detection_parses_recaptcha_v3_json() -> None:
    page = _FakePage(
        [
            {
                "captchaType": "recaptcha_v3",
                "websiteUrl": "https://example.com/login",
                "websiteKey": "site-key",
                "pageAction": "login",
                "marker": "recaptcha v3",
            }
        ]
    )

    challenge = await detect_browser_captcha(page)

    assert challenge is not None
    assert challenge.captcha_type == "recaptcha_v3"
    assert challenge.website_url == "https://example.com/login"
    assert challenge.website_key == "site-key"
    assert challenge.page_action == "login"


@pytest.mark.asyncio
async def test_browser_captcha_detection_script_keeps_turnstile_callback_widget() -> None:
    pytest.importorskip("playwright.async_api")
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            pytest.skip(f"Playwright Chromium is unavailable: {exc}")

        try:
            page = await browser.new_page()
            await page.set_content(
                """
                <html>
                  <body>
                    <div
                      class="cf-turnstile"
                      data-sitekey="site-key"
                      data-callback="onTurnstile"
                    ></div>
                  </body>
                </html>
                """
            )

            detected = await page.evaluate(_DETECT_CAPTCHA_SCRIPT)
        finally:
            await browser.close()

    assert detected["captchaType"] == "turnstile"
    assert detected["websiteKey"] == "site-key"


@pytest.mark.asyncio
async def test_browser_captcha_injection_passes_token_as_argument() -> None:
    page = _FakePage([{"ok": True, "fieldsUpdated": 1, "callbacksCalled": 0}])

    result = await inject_browser_captcha_token(
        page,
        captcha_type="turnstile",
        token="solution-token",
    )

    assert result["ok"] is True
    script, args = page.calls[0]
    assert "solution-token" not in script
    assert args == ("turnstile", "solution-token")


@pytest.mark.asyncio
async def test_capsolver_browser_use_action_returns_non_terminal_action_result() -> None:
    original_env = os.environ.copy()
    page = _FakePage(
        [
            {
                "captchaType": "recaptcha_v2",
                "websiteUrl": "https://example.com/login",
                "websiteKey": "site-key",
                "marker": "g-recaptcha",
            },
            {"ok": True, "fieldsUpdated": 1, "callbacksCalled": 0},
        ]
    )
    try:
        controller = build_capsolver_controller(_FakeCapSolver())  # type: ignore[arg-type]
        action = controller.registry.registry.actions["solve_captcha_with_capsolver"]

        result = await action.function(browser_session=_FakeBrowserSession(page))
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    assert result.success is None
    assert result.is_done is False
    assert result.extracted_content == "CAPTCHA solved with CapSolver (recaptcha_v2)."


@pytest.mark.asyncio
async def test_capsolver_browser_use_action_handles_recaptcha_v3() -> None:
    original_env = os.environ.copy()
    page = _FakePage(
        [
            {
                "captchaType": "recaptcha_v3",
                "websiteUrl": "https://example.com/login",
                "websiteKey": "site-key",
                "pageAction": "login",
                "marker": "recaptcha v3",
            },
            {"ok": True, "fieldsUpdated": 1, "callbacksCalled": 0},
        ]
    )
    try:
        controller = build_capsolver_controller(_FakeCapSolver())  # type: ignore[arg-type]
        action = controller.registry.registry.actions["solve_captcha_with_capsolver"]

        result = await action.function(browser_session=_FakeBrowserSession(page))
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    assert result.success is None
    assert result.is_done is False
    assert result.extracted_content == "CAPTCHA solved with CapSolver (recaptcha_v3)."
