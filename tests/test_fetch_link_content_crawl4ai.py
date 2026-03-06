from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from opentulpa.agent import tools_registry


@dataclass
class _Result:
    success: bool = True
    error_message: str = ""
    fit_markdown: str | None = None
    markdown: str | None = None
    extracted_content: Any = None
    cleaned_html: str | None = None
    html: str | None = None
    text: str | None = None
    metadata: dict[str, Any] | None = None


class _Crawler:
    def __init__(self, result: _Result) -> None:
        self._result = result

    async def __aenter__(self) -> _Crawler:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    async def arun(self, *, url: str) -> _Result:
        _ = url
        return self._result


def _install_fake_crawl4ai(monkeypatch: pytest.MonkeyPatch, result: _Result) -> None:
    module = types.SimpleNamespace(AsyncWebCrawler=lambda: _Crawler(result))
    monkeypatch.setitem(sys.modules, "crawl4ai", module)


@pytest.mark.asyncio
async def test_crawl4ai_extract_prefers_markdown_and_title(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_crawl4ai(
        monkeypatch,
        _Result(
            success=True,
            markdown="## Hello\n\nworld",
            metadata={"title": "Test Page"},
        ),
    )
    text, title, error = await tools_registry._crawl4ai_extract("https://example.com")
    assert error is None
    assert "Hello" in text
    assert title == "Test Page"


@pytest.mark.asyncio
async def test_crawl4ai_extract_handles_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_crawl4ai(
        monkeypatch,
        _Result(success=False, error_message="blocked"),
    )
    text, title, error = await tools_registry._crawl4ai_extract("https://example.com")
    assert text == ""
    assert title is None
    assert isinstance(error, str)
    assert "blocked" in error

