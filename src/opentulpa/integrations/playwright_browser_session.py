"""Playwright CDP client for Browser Use Cloud sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

from opentulpa.integrations.content_fetch import (
    HostResolver,
    SocketHostResolver,
    is_forbidden_address,
    is_forbidden_hostname,
)


class PlaywrightBrowserSession:
    """Control a Browser Use Cloud Chromium session through its CDP endpoint."""

    def __init__(
        self,
        *,
        cdp_url: str,
        allowed_domains: list[str],
        playwright_factory: Callable[[], Any] | None = None,
        host_resolver: HostResolver | None = None,
    ) -> None:
        self._cdp_url = str(cdp_url or "").strip()
        if not self._cdp_url:
            raise ValueError("Browser Use Cloud CDP URL is required")
        normalized_domains = tuple(
            value for raw in allowed_domains if (value := self._normalize_domain(raw))
        )
        if not normalized_domains or "*" in normalized_domains:
            raise ValueError("Browser Use Cloud requires explicit allowed_domains")
        self._allowed_domains = normalized_domains
        self._playwright_factory = playwright_factory
        self._host_resolver = host_resolver or SocketHostResolver()
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the browser once; repeated calls reuse the same page and context."""
        async with self._start_lock:
            if self._context is not None:
                return
            if self._playwright_factory is None:
                from playwright.async_api import async_playwright  # type: ignore[import-not-found]

                playwright_factory = async_playwright
            else:
                playwright_factory = self._playwright_factory

            try:
                self._playwright = await playwright_factory().start()
                chromium = self._playwright.chromium
                self._browser = await chromium.connect_over_cdp(self._cdp_url)
                contexts = list(self._browser.contexts)
                self._context = contexts[0] if contexts else await self._browser.new_context()

                await self._context.route("**/*", self._route_request)
                pages = list(self._context.pages)
                self._page = pages[-1] if pages else await self._context.new_page()
            except Exception:
                await self._stop_unlocked()
                raise

    async def navigate_to(self, url: str) -> None:
        target = str(url or "").strip()
        if not self._is_http_url(target):
            raise ValueError("Browser navigation requires an http or https URL")
        if not await self._request_is_allowed(target):
            raise ValueError(
                "Browser navigation target is outside allowed_domains or is a direct "
                "private/link-local target"
            )
        page = await self.get_current_page()
        await page.goto(target, wait_until="domcontentloaded", timeout=30_000)

    async def get_current_page(self) -> Any:
        await self.start()
        if self._page is None:
            raise RuntimeError("Playwright browser page is unavailable")
        return self._page

    async def get_current_page_url(self) -> str:
        page = await self.get_current_page()
        return str(page.url or "")

    async def get_current_page_title(self) -> str:
        page = await self.get_current_page()
        return str(await page.title())

    async def get_state_as_text(self) -> str:
        page = await self.get_current_page()
        return str(await page.locator("body").inner_text(timeout=5_000))[:20_000]

    async def take_screenshot(
        self,
        *,
        path: str | None = None,
        full_page: bool = False,
        format: str = "png",
        quality: int | None = None,
        clip: dict[str, float] | None = None,
    ) -> bytes:
        page = await self.get_current_page()
        image_type = "jpeg" if str(format).lower() in {"jpg", "jpeg"} else "png"
        kwargs: dict[str, Any] = {
            "full_page": bool(full_page),
            "type": image_type,
        }
        if path:
            kwargs["path"] = path
        if quality is not None and image_type == "jpeg":
            kwargs["quality"] = max(0, min(int(quality), 100))
        if clip is not None:
            kwargs["clip"] = clip
        return bytes(await page.screenshot(**kwargs))

    async def stop(self) -> None:
        async with self._start_lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        context, browser, playwright = self._context, self._browser, self._playwright
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        if context is not None:
            with suppress(Exception):
                await context.close()
        if browser is not None:
            with suppress(Exception):
                await browser.close()
        if playwright is not None:
            with suppress(Exception):
                await playwright.stop()

    async def _route_request(self, route: Any) -> None:
        if await self._request_is_allowed(str(route.request.url or "")):
            await route.continue_()
        else:
            await route.abort("blockedbyclient")

    async def _request_is_allowed(self, url: str) -> bool:
        if not self._url_is_allowed(url):
            return False
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme in {"about", "blob", "data"}:
            return True
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            return False
        hostname = str(parsed.hostname or "").lower().rstrip(".")
        if is_forbidden_hostname(hostname):
            return False
        try:
            port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
        except ValueError:
            return False
        try:
            addresses: Sequence[str] = await self._host_resolver.resolve(hostname, port)
        except Exception:
            return False
        normalized = [str(address or "").strip() for address in addresses]
        return bool(normalized) and not any(
            is_forbidden_address(address) for address in normalized
        )

    def _url_is_allowed(self, url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme in {"about", "blob", "data"}:
            return True
        hostname = str(parsed.hostname or "").lower().rstrip(".")
        return bool(hostname) and (
            any(
                hostname == domain or hostname.endswith(f".{domain}")
                for domain in self._allowed_domains
            )
        )

    @staticmethod
    def _is_http_url(url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)

    @staticmethod
    def _normalize_domain(raw: str) -> str:
        value = str(raw or "").strip().lower()
        if not value:
            return ""
        if value == "*":
            return value
        parsed = urlparse(value if "://" in value else f"//{value}")
        hostname = str(parsed.hostname or "").lower().rstrip(".")
        return hostname.removeprefix("*.")
