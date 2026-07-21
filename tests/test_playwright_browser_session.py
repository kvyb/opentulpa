from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from opentulpa.integrations.playwright_browser_session import PlaywrightBrowserSession


class _FakeLocator:
    async def inner_text(self, *, timeout: int) -> str:
        assert timeout == 5_000
        return "Visible page text"


class _FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.goto_calls: list[tuple[str, dict[str, Any]]] = []
        self.screenshot_kwargs: dict[str, Any] = {}

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self.goto_calls.append((url, kwargs))

    async def title(self) -> str:
        return "Example"

    def locator(self, selector: str) -> _FakeLocator:
        assert selector == "body"
        return _FakeLocator()

    async def screenshot(self, **kwargs: Any) -> bytes:
        self.screenshot_kwargs = kwargs
        return b"png"


class _FakeContext:
    def __init__(self, *, page: _FakePage | None = None) -> None:
        self.pages = [page] if page else []
        self.page = page
        self.routes: list[tuple[str, Any]] = []
        self.closed = False

    async def new_page(self) -> _FakePage:
        self.page = _FakePage()
        self.pages.append(self.page)
        return self.page

    async def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, *, contexts: list[_FakeContext] | None = None) -> None:
        self.contexts = contexts or []
        self.closed = False

    async def new_context(self) -> _FakeContext:
        context = _FakeContext()
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self) -> None:
        self.browser = _FakeBrowser()
        self.cdp_url = ""

    async def connect_over_cdp(self, url: str) -> _FakeBrowser:
        self.cdp_url = url
        return self.browser


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _FakePlaywrightManager:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self.playwright = playwright

    async def start(self) -> _FakePlaywright:
        return self.playwright


class _HostResolver:
    def __init__(self, *answers: list[str]) -> None:
        self.answers = list(answers) or [["93.184.216.34"]]
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> list[str]:
        self.calls.append((hostname, port))
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]


@pytest.mark.asyncio
async def test_cloud_cdp_session_exposes_browser_protocol() -> None:
    playwright = _FakePlaywright()
    session = PlaywrightBrowserSession(
        cdp_url="wss://cdp.browser-use.example/session",
        allowed_domains=["example.com"],
        playwright_factory=lambda: _FakePlaywrightManager(playwright),
    )

    await session.start()
    await session.navigate_to("https://example.com/page")

    assert await session.get_current_page_url() == "https://example.com/page"
    assert await session.get_current_page_title() == "Example"
    assert await session.get_state_as_text() == "Visible page text"
    assert await session.take_screenshot(path="shot.png", full_page=True) == b"png"
    page = await session.get_current_page()
    assert page.goto_calls == [
        (
            "https://example.com/page",
            {"wait_until": "domcontentloaded", "timeout": 30_000},
        )
    ]
    assert page.screenshot_kwargs == {
        "path": "shot.png",
        "full_page": True,
        "type": "png",
    }
    assert playwright.chromium.cdp_url == "wss://cdp.browser-use.example/session"

    await session.stop()
    assert playwright.chromium.browser.closed is True
    assert playwright.stopped is True


@pytest.mark.asyncio
async def test_cloud_cdp_session_enforces_allowed_domains_on_remote_context() -> None:
    playwright = _FakePlaywright()
    session = PlaywrightBrowserSession(
        cdp_url="wss://cdp.browser-use.example/session",
        allowed_domains=["*.example.com"],
        playwright_factory=lambda: _FakePlaywrightManager(playwright),
        host_resolver=_HostResolver(),
    )

    await session.start()
    await session.navigate_to("https://docs.example.com/start")
    with pytest.raises(ValueError, match="outside allowed_domains"):
        await session.navigate_to("https://example.net/")

    route_handler = playwright.chromium.browser.contexts[0].routes[0][1]
    allowed_route = SimpleNamespace(
        request=SimpleNamespace(url="https://cdn.example.com/app.js"),
        continue_=AsyncMock(),
        abort=AsyncMock(),
    )
    blocked_route = SimpleNamespace(
        request=SimpleNamespace(url="https://tracker.invalid/pixel"),
        continue_=AsyncMock(),
        abort=AsyncMock(),
    )
    await route_handler(allowed_route)
    await route_handler(blocked_route)
    allowed_route.continue_.assert_awaited_once_with()
    blocked_route.abort.assert_awaited_once_with("blockedbyclient")
    await session.stop()


@pytest.mark.asyncio
async def test_playwright_session_blocks_private_and_mixed_dns_answers() -> None:
    for answers in (["127.0.0.1"], ["93.184.216.34", "169.254.169.254"]):
        playwright = _FakePlaywright()
        session = PlaywrightBrowserSession(
            cdp_url="wss://cdp.browser-use.example/session",
            allowed_domains=["browser.example"],
            playwright_factory=lambda playwright=playwright: _FakePlaywrightManager(playwright),
            host_resolver=_HostResolver(answers),
        )

        await session.start()
        with pytest.raises(ValueError, match="private/link-local"):
            await session.navigate_to("https://browser.example/start")
        page = await session.get_current_page()
        assert page.goto_calls == []
        await session.stop()


def test_cloud_cdp_session_requires_explicit_non_wildcard_domains() -> None:
    for domains in ([], ["*"]):
        with pytest.raises(ValueError, match="explicit allowed_domains"):
            PlaywrightBrowserSession(
                cdp_url="wss://cdp.browser-use.example/session",
                allowed_domains=domains,
            )


@pytest.mark.asyncio
async def test_cloud_cdp_session_blocks_direct_private_and_link_local_targets() -> None:
    for url in (
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/admin",
    ):
        playwright = _FakePlaywright()
        session = PlaywrightBrowserSession(
            cdp_url="wss://cdp.browser-use.example/session",
            allowed_domains=[url],
            playwright_factory=lambda playwright=playwright: _FakePlaywrightManager(playwright),
        )

        await session.start()
        with pytest.raises(ValueError, match="private/link-local"):
            await session.navigate_to(url)
        page = await session.get_current_page()
        assert page.goto_calls == []
        await session.stop()


@pytest.mark.asyncio
async def test_cloud_cdp_session_rechecks_host_resolution_without_claiming_dns_pin() -> None:
    playwright = _FakePlaywright()
    resolver = _HostResolver(["93.184.216.34"], ["127.0.0.1"])
    session = PlaywrightBrowserSession(
        cdp_url="wss://cdp.browser-use.example/session",
        allowed_domains=["browser.example"],
        playwright_factory=lambda: _FakePlaywrightManager(playwright),
        host_resolver=resolver,
    )

    await session.start()
    await session.navigate_to("https://browser.example/start")
    route_handler = playwright.chromium.browser.contexts[0].routes[0][1]
    rebound_route = SimpleNamespace(
        request=SimpleNamespace(url="https://browser.example/private"),
        continue_=AsyncMock(),
        abort=AsyncMock(),
    )
    await route_handler(rebound_route)

    rebound_route.continue_.assert_not_awaited()
    rebound_route.abort.assert_awaited_once_with("blockedbyclient")
    assert resolver.calls == [("browser.example", 443), ("browser.example", 443)]
    await session.stop()


@pytest.mark.asyncio
async def test_playwright_session_connects_to_existing_cdp_context() -> None:
    playwright = _FakePlaywright()
    existing_page = _FakePage()
    existing_context = _FakeContext(page=existing_page)
    playwright.chromium.browser = _FakeBrowser(contexts=[existing_context])
    session = PlaywrightBrowserSession(
        cdp_url="wss://browser.example/cdp/secret",
        allowed_domains=["example.com"],
        playwright_factory=lambda: _FakePlaywrightManager(playwright),
    )

    await session.start()

    assert playwright.chromium.cdp_url == "wss://browser.example/cdp/secret"
    assert await session.get_current_page() is existing_page
    await session.stop()
