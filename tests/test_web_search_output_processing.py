from __future__ import annotations

from typing import Any

import httpx
import pytest

from opentulpa.integrations import web_search as web_search_module
from opentulpa.integrations.web_search import _extract_sources, _sanitize_answer_text


def test_sanitize_answer_text_removes_favicon_noise() -> None:
    raw = (
        "Key points here.\n"
        "Favicon for https://example.com/article\n"
        "Another useful line.\n"
        "Previous slideNext slide\n"
    )
    cleaned = _sanitize_answer_text(raw)
    assert "Favicon for" not in cleaned
    assert "Previous slideNext slide" not in cleaned
    assert "Key points here." in cleaned
    assert "Another useful line." in cleaned


def test_extract_sources_collects_from_payload_and_answer() -> None:
    data = {
        "citations": [
            {"url": "https://one.example/a"},
            "https://two.example/b",
        ],
        "choices": [
            {
                "message": {
                    "sources": [{"link": "https://three.example/c"}],
                    "content": "See https://four.example/d for more.",
                }
            }
        ],
    }
    answer = "Summary with link https://four.example/d and https://two.example/b"
    sources = _extract_sources(data, answer)
    urls = [item["url"] for item in sources]
    assert "https://one.example/a" in urls
    assert "https://two.example/b" in urls
    assert "https://three.example/c" in urls
    assert "https://four.example/d" in urls
    # de-dup
    assert urls.count("https://two.example/b") == 1


def test_exa_result_list_answer_formats_twenty_results() -> None:
    items = [
        {"title": f"Result {index}", "url": f"https://example.com/{index}"}
        for index in range(1, 22)
    ]

    answer = web_search_module._format_result_list_answer(items)

    assert "20. Result 20" in answer
    assert "21. Result 21" not in answer


@pytest.mark.asyncio
async def test_pplx_web_search_requests_medium_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = self
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            request=request,
            json={"choices": [{"message": {"content": "Answer"}}]},
        )

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    result = await web_search_module.web_search("current news")

    assert isinstance(result, dict)
    assert result["provider"] == "pplx"
    assert captured["json"]["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_web_search_auto_selects_exa_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = self
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            request=request,
            json={
                "results": [
                    {
                        "title": "Exa result",
                        "url": "https://example.com/source",
                    }
                ],
            },
        )

    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "openrouter-key")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    result = await web_search_module.web_search("current news")

    assert isinstance(result, dict)
    assert captured["url"] == "https://api.exa.ai/search"
    assert captured["json"] == {"query": "current news", "numResults": 20}
    assert captured["headers"]["x-api-key"] == "exa-key"
    assert captured["headers"]["x-exa-integration"] == "opentulpa"
    assert result["provider"] == "exa"
    assert result["answer"] == "1. Exa result (https://example.com/source)"
    assert result["source_count"] == 1


@pytest.mark.asyncio
async def test_exa_web_search_uses_search_endpoint_for_optional_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = self, headers
        captured["url"] = url
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            request=request,
            json={
                "results": [
                    {
                        "title": "Fresh item",
                        "url": "https://news.example/fresh",
                        "publishedDate": "2026-05-28T00:00:00.000Z",
                        "highlights": ["Fresh highlight"],
                    }
                ],
            },
        )

    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    result = await web_search_module.web_search(
        "latest OpenTulpa news",
        search_type="auto",
        category="news",
    )

    assert isinstance(result, dict)
    assert captured["url"] == "https://api.exa.ai/search"
    assert captured["json"]["query"] == "latest OpenTulpa news"
    assert captured["json"]["type"] == "auto"
    assert captured["json"]["category"] == "news"
    assert captured["json"]["numResults"] == 20
    assert "contents" not in captured["json"]
    assert "outputSchema" not in captured["json"]
    assert result["provider"] == "exa"
    assert result["model"] == "exa-search"
    assert result["answer"] == "1. Fresh item (https://news.example/fresh)"
    assert result["sources"] == [{"url": "https://news.example/fresh", "domain": "news.example"}]


@pytest.mark.asyncio
async def test_exa_web_search_rejects_invalid_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    result = await web_search_module.web_search("current news", search_type="slow")

    assert isinstance(result, str)
    assert "invalid Exa search_type" in result


@pytest.mark.asyncio
async def test_pplx_web_search_rejects_provider_specific_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")

    result = await web_search_module.web_search("current news", category="news")

    assert result == "Web search provider 'pplx' supports only query."


@pytest.mark.asyncio
async def test_pplx_web_search_retries_transient_openrouter_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    async def _fake_sleep(delay: float) -> None:
        captured_delays.append(delay)

    async def _fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        _ = self, json, headers
        calls["count"] += 1
        request = httpx.Request("POST", url)
        if calls["count"] == 1:
            return httpx.Response(status_code=502, request=request, json={"error": "bad gateway"})
        return httpx.Response(
            status_code=200,
            request=request,
            json={"choices": [{"message": {"content": "Recovered answer"}}]},
        )

    captured_delays: list[float] = []
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    monkeypatch.setattr(web_search_module.asyncio, "sleep", _fake_sleep)

    result = await web_search_module.web_search("current news")

    assert calls["count"] == 2
    assert captured_delays == [0.75]
    assert isinstance(result, dict)
    assert result["answer"] == "Recovered answer"
