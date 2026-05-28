"""
Pluggable web search providers.

The agent's general chat model remains separate. This integration is only used
when the web_search tool is explicitly invoked.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from opentulpa.core.config import get_openai_compatible_api_key_from_env

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EXA_BASE = "https://api.exa.ai"
DEFAULT_WEB_SEARCH_MODEL = "perplexity/sonar-pro-search"
DEFAULT_EXA_SEARCH_RESULTS = 20
RETRYABLE_WEB_SEARCH_STATUSES = {408, 429, 500, 502, 503, 504}
EXA_SEARCH_TYPES = {"neural", "fast", "auto", "deep"}
EXA_CATEGORIES = {
    "company",
    "research paper",
    "news",
    "pdf",
    "github",
    "tweet",
    "personal site",
    "financial report",
    "people",
}


def _default_search_model() -> str:
    """Default OpenRouter search model for web-search tool calls."""
    configured = str(os.environ.get("OPENROUTER_WEB_SEARCH_MODEL", "")).strip()
    selected = configured or DEFAULT_WEB_SEARCH_MODEL
    if ":online" in selected.lower():
        logger.warning("Ignoring legacy :online model override for web_search")
        return DEFAULT_WEB_SEARCH_MODEL
    return selected


def _extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _sanitize_answer_text(raw: str) -> str:
    lines = [line.rstrip() for line in str(raw or "").splitlines()]
    cleaned: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if re.match(r"^Favicon for https?://", text, flags=re.IGNORECASE):
            continue
        if text.lower() in {"previous slidenext slide", "next slide"}:
            continue
        cleaned.append(text)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _extract_url_from_item(item: object) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value if value.startswith(("http://", "https://")) else None
    if isinstance(item, Mapping):
        for key in ("url", "link", "uri", "source", "href"):
            candidate = item.get(key)
            if isinstance(candidate, str):
                clean = candidate.strip()
                if clean.startswith(("http://", "https://")):
                    return clean
    return None


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    if value.endswith(")."):
        value = value[:-2]
    elif value.endswith((")", ".", ",")):
        value = value[:-1]
    return value


def _extract_sources(data: dict, answer: str) -> list[dict[str, str]]:
    candidates: list[str] = []
    for key in ("citations", "sources", "references"):
        raw = data.get(key)
        if isinstance(raw, list):
            for item in raw:
                url = _extract_url_from_item(item)
                if url:
                    candidates.append(url)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            for key in ("citations", "sources", "references"):
                raw = message.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        url = _extract_url_from_item(item)
                        if url:
                            candidates.append(url)

    for match in re.findall(r"https?://[^\s<>\]\)\"']+", answer):
        candidates.append(match)

    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for raw_url in candidates:
        normalized = _normalize_url(raw_url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        host = urlparse(normalized).netloc.lower()
        out.append({"url": normalized, "domain": host})
    return out


def _exa_api_key() -> str | None:
    value = str(os.environ.get("EXA_API_KEY", "")).strip()
    return value or None


def _clean_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass(frozen=True)
class WebSearchResult:
    answer: str
    sources: list[dict[str, str]]
    provider: str
    model: str

    def to_payload(self) -> dict[str, object]:
        assert self.provider
        assert self.model
        return {
            "answer": self.answer,
            "sources": self.sources,
            "source_count": len(self.sources),
            "provider": self.provider,
            "model": self.model,
        }


def _safe_exa_search_type(value: object) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    lowered = text.lower()
    return lowered if lowered in EXA_SEARCH_TYPES else None


def _safe_exa_category(value: object) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    lowered = text.lower()
    return lowered if lowered in EXA_CATEGORIES else None


def _exa_search_options(kwargs: dict[str, object]) -> dict[str, object]:
    options: dict[str, object] = {}
    unexpected = sorted(
        key
        for key, value in kwargs.items()
        if value is not None and key not in {"search_type", "category"}
    )
    if unexpected:
        raise ValueError(f"unsupported Exa web_search args: {', '.join(unexpected)}")
    search_type = _safe_exa_search_type(kwargs.get("search_type"))
    category = _safe_exa_category(kwargs.get("category"))
    raw_search_type = _clean_optional_text(kwargs.get("search_type"))
    raw_category = _clean_optional_text(kwargs.get("category"))
    if raw_search_type and search_type is None:
        allowed = ", ".join(sorted(EXA_SEARCH_TYPES))
        raise ValueError(f"invalid Exa search_type '{raw_search_type}' (allowed: {allowed})")
    if raw_category and category is None:
        allowed = ", ".join(sorted(EXA_CATEGORIES))
        raise ValueError(f"invalid Exa category '{raw_category}' (allowed: {allowed})")
    if search_type:
        options["type"] = search_type
    if category:
        options["category"] = category
    return options


class WebSearchProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable provider name used in result metadata."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when provider has enough local config to run."""

    @abc.abstractmethod
    async def search(self, query: str, **kwargs: object) -> WebSearchResult | str:
        """Run search and return OpenTulpa's existing web_search result shape."""


class ExaSearchProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "exa"

    def is_available(self) -> bool:
        return _exa_api_key() is not None

    async def search(self, query: str, **kwargs: object) -> WebSearchResult | str:
        api_key = _exa_api_key()
        assert api_key is not None
        try:
            options = _exa_search_options(kwargs)
        except ValueError as exc:
            return f"Web search invalid argument: {exc!s}."
        payload: dict[str, object] = {
            "query": query,
            "numResults": DEFAULT_EXA_SEARCH_RESULTS,
        }
        payload.update(options)
        data = await _post_json_with_retries(
            f"{EXA_BASE}/search",
            payload=payload,
            headers={
                "x-api-key": api_key,
                "x-exa-integration": "opentulpa",
                "Content-Type": "application/json",
            },
            provider_name=self.name,
        )
        if isinstance(data, str):
            return data
        items = _extract_result_items(data)
        answer = _format_result_list_answer(items)
        return WebSearchResult(
            answer=answer or "No response from web search.",
            sources=_sources_from_result_items(items),
            provider=self.name,
            model="exa-search",
        )


class PplxSearchProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "pplx"

    def is_available(self) -> bool:
        return get_openai_compatible_api_key_from_env() is not None

    async def search(self, query: str, **kwargs: object) -> WebSearchResult | str:
        if any(value is not None for value in kwargs.values()):
            return "Web search provider 'pplx' supports only query."
        api_key = get_openai_compatible_api_key_from_env()
        assert api_key is not None
        use_model = _default_search_model()
        payload = {
            "model": use_model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 2048,
            "reasoning": {"effort": "medium"},
        }
        data = await _post_json_with_retries(
            f"{OPENROUTER_BASE}/chat/completions",
            payload=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            provider_name=self.name,
        )
        if isinstance(data, str):
            return data
        return _pplx_response_payload(data, use_model)


def _providers() -> dict[str, WebSearchProvider]:
    return {
        "exa": ExaSearchProvider(),
        "pplx": PplxSearchProvider(),
    }


def get_web_search_provider() -> WebSearchProvider | None:
    available = _providers()
    for name in ("exa", "pplx"):
        provider = available[name]
        if provider.is_available():
            return provider
    return None


def get_web_search_backend_name() -> str:
    provider = get_web_search_provider()
    if provider is None:
        return "none"
    return provider.name


async def _post_json_with_retries(
    url: str,
    *,
    payload: dict[str, object],
    headers: dict[str, str],
    provider_name: str,
) -> dict[str, object] | str:
    assert url.startswith(("http://", "https://"))
    assert provider_name
    max_attempts = 3
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_attempts):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in RETRYABLE_WEB_SEARCH_STATUSES and attempt < max_attempts - 1:
                    await _sleep_before_retry(provider_name, status_code, attempt, None)
                    continue
                logger.exception("%s web search HTTP error: %s", provider_name, exc)
                return f"Web search request failed: {status_code}."
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < max_attempts - 1:
                    await _sleep_before_retry(provider_name, None, attempt, type(exc).__name__)
                    continue
                logger.exception("%s web search error: %s", provider_name, exc)
                return f"Web search failed: {exc!s}."
            except Exception as exc:
                logger.exception("%s web search error: %s", provider_name, exc)
                return f"Web search failed: {exc!s}."
    return "Web search failed after retries."


async def _sleep_before_retry(
    provider_name: str,
    status_code: int | None,
    attempt: int,
    error_name: str | None,
) -> None:
    delay = 0.75 * (2**attempt)
    if status_code is not None:
        logger.warning(
            "%s web search HTTP error; retrying status=%s attempt=%s delay=%.2fs",
            provider_name,
            status_code,
            attempt + 1,
            delay,
        )
    else:
        logger.warning(
            "%s web search transport error; retrying error=%s attempt=%s delay=%.2fs",
            provider_name,
            error_name,
            attempt + 1,
            delay,
        )
    await asyncio.sleep(delay)


def _pplx_response_payload(data: dict[str, object], use_model: str) -> WebSearchResult | str:
    raw_choices = data.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    if not choices:
        return "No response from web search."
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    answer = _sanitize_answer_text(_extract_text_content(content))
    if not answer:
        answer = "No content in response."
    return WebSearchResult(
        answer=answer,
        sources=_extract_sources(data, answer),
        provider="pplx",
        model=use_model,
    )


def _extract_result_items(data: dict[str, object]) -> list[dict[str, object]]:
    raw_data = data.get("data")
    if isinstance(raw_data, dict):
        for key in ("web", "results"):
            raw_items = raw_data.get(key)
            if isinstance(raw_items, list):
                return [item for item in raw_items if isinstance(item, dict)]
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    for key in ("results", "web", "citations"):
        raw_items = data.get(key)
        if isinstance(raw_items, list):
            return [item for item in raw_items if isinstance(item, dict)]
    return []


def _format_result_list_answer(items: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(items[:DEFAULT_EXA_SEARCH_RESULTS], start=1):
        title = str(item.get("title") or item.get("url") or "Untitled").strip()
        description = str(item.get("description") or item.get("text") or "").strip()
        url = str(item.get("url") or "").strip()
        line = f"{index}. {title}"
        if description:
            line += f" - {description}"
        if url:
            line += f" ({url})"
        lines.append(line)
    return "\n".join(lines).strip()


def _missing_provider_message() -> str:
    return (
        "Web search is not configured "
        "(set EXA_API_KEY or OPENAI_COMPATIBLE_API_KEY; "
        "OPENROUTER_API_KEY also accepted for pplx)."
    )


def _sources_from_result_items(items: list[dict[str, object]]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        url = _extract_url_from_item(item)
        normalized = _normalize_url(url or "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sources.append({"url": normalized, "domain": urlparse(normalized).netloc.lower()})
    return sources


async def web_search(query: str, **kwargs: object) -> dict[str, object] | str:
    """
    Run a web-backed search and return cleaned answer + extracted sources.
    """
    provider = get_web_search_provider()
    if provider is None:
        return _missing_provider_message()
    result = await provider.search(query, **kwargs)
    if isinstance(result, WebSearchResult):
        return result.to_payload()
    return result
