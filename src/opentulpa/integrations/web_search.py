"""
Web search via OpenRouter using Perplexity Sonar Pro Search.

The agent's general chat model remains separate. This integration is only used
when the web_search tool is explicitly invoked.
"""

import logging
import os
import re
from urllib.parse import urlparse

import httpx

from opentulpa.core.config import get_openai_compatible_api_key_from_env

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_WEB_SEARCH_MODEL = "perplexity/sonar-pro-search"


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
    if isinstance(item, dict):
        for key in ("url", "link", "uri", "source", "href"):
            value = item.get(key)
            if isinstance(value, str):
                clean = value.strip()
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


async def web_search(query: str) -> dict[str, object] | str:
    """
    Run a web-backed completion and return cleaned answer + extracted sources.
    """
    api_key = get_openai_compatible_api_key_from_env()
    if not api_key:
        return (
            "Web search is not configured "
            "(OPENAI_COMPATIBLE_API_KEY missing; OPENROUTER_API_KEY also accepted)."
        )

    use_model = _default_search_model()
    url = f"{OPENROUTER_BASE}/chat/completions"

    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 2048,
        "reasoning": {"effort": "high"},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            logger.exception("OpenRouter web search HTTP error: %s", e)
            return f"Web search request failed: {e.response.status_code}."
        except Exception as e:
            logger.exception("OpenRouter web search error: %s", e)
            return f"Web search failed: {e!s}."

    choices = data.get("choices") or []
    if not choices:
        return "No response from web search."
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    answer = _sanitize_answer_text(_extract_text_content(content))
    if not answer:
        answer = "No content in response."
    sources = _extract_sources(data if isinstance(data, dict) else {}, answer)
    return {
        "answer": answer,
        "sources": sources,
        "source_count": len(sources),
        "model": use_model,
    }
