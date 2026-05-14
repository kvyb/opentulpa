from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.scheduler.service import SchedulerService

LIVE_FLAG = "OPENTULPA_ENABLE_LIVE_BROWSER_USE_E2E"
TEST_CUSTOMER_ID = "cust_live_browser_use_google"
pytestmark = [pytest.mark.e2e]

if str(os.getenv(LIVE_FLAG, "")).strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        f"set {LIVE_FLAG}=1 to run live Browser Use Google e2e test",
        allow_module_level=True,
    )

_settings = get_settings()
if not str(_settings.openrouter_api_key or "").strip():
    pytest.skip("OPENROUTER_API_KEY is required for live Browser Use e2e", allow_module_level=True)

try:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        _chromium_path = str(getattr(playwright.chromium, "executable_path", "") or "").strip()
    if not _chromium_path or not Path(_chromium_path).exists():
        pytest.skip(
            "Playwright Chromium is not installed. Run `uv run playwright install chromium`.",
            allow_module_level=True,
        )
except Exception as exc:
    pytest.skip(
        f"Playwright Chromium check failed ({exc}). Run `uv run playwright install chromium`.",
        allow_module_level=True,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _has_title_and_url(text: str) -> bool:
    value = str(text or "").strip()
    if "title:" not in value.lower():
        return False
    return bool(re.search(r"https?://", value))


def _invoke_internal_chat(
    *,
    client: TestClient,
    customer_id: str,
    thread_id: str,
    prompt: str,
) -> str:
    response = client.post(
        "/internal/chat",
        json={
            "customer_id": customer_id,
            "thread_id": thread_id,
            "text": prompt,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("ok") is True
    return str(payload.get("text", "") or "").strip()


def test_live_llm_uses_browser_use_run_for_google_search(tmp_path: Path) -> None:
    behavior_log = tmp_path / "agent_behavior.browser_use_live.jsonl"
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key=str(_settings.openrouter_api_key or "").strip(),
        openrouter_base_url=str(_settings.openrouter_base_url or "").strip(),
        model_name=str(_settings.llm_model or "").strip(),
        wake_classifier_model_name=str(_settings.llm_model or "").strip(),
        checkpoint_db_path=str(tmp_path / "live_browser_use_checkpoints.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log),
        browser_use_headless=True,
        browser_use_model_override=str(
            _settings.browser_use_model or "google/gemini-3-flash-preview"
        ).strip(),
    )
    app = create_app(
        scheduler=SchedulerService(),
        agent_runtime=runtime,
    )

    prompt_primary = (
        "You must use browser_use_run. "
        "Open https://www.google.com/search?q=OpenAI+API and extract one organic result. "
        "Do not use web_search or fetch_url_content. "
        "Return exactly two lines: 'Title: ...' and 'URL: ...'."
    )
    prompt_fallback = (
        "You must use browser_use_run. "
        "Start at https://www.google.com/search?q=OpenAI+API. "
        "Use allowed_domains exactly: ['google.com','www.google.com','bing.com','www.bing.com',"
        "'duckduckgo.com','www.duckduckgo.com','openai.com','www.openai.com']. "
        "If Google is blocked by anti-bot, immediately try Bing or DuckDuckGo; "
        "if both are blocked, open https://openai.com/api/ and extract title and URL from the page. "
        "Do not use web_search or fetch_url_content. "
        "Return exactly two lines: 'Title: ...' and 'URL: ...'."
    )

    answers: list[str] = []
    with TestClient(app) as client:
        answers.append(
            _invoke_internal_chat(
                client=client,
                customer_id=TEST_CUSTOMER_ID,
                thread_id="live-browser-use-google-001",
                prompt=prompt_primary,
            )
        )
        if not _has_title_and_url(answers[-1]):
            answers.append(
                _invoke_internal_chat(
                    client=client,
                    customer_id=TEST_CUSTOMER_ID,
                    thread_id="live-browser-use-google-002",
                    prompt=prompt_fallback,
                )
            )

    answer = ""
    for candidate in answers:
        if _has_title_and_url(candidate):
            answer = candidate
            break
    if not answer and answers:
        answer = answers[-1]

    events = _read_jsonl(behavior_log)
    browser_tool_success = [
        event
        for event in events
        if str(event.get("event", "")).strip() == "graph.tools.success"
        and str(event.get("tool_name", "")).strip() in {"browser_use_run", "tool_group_exec"}
    ]
    assert browser_tool_success, "Expected at least one successful browser_use_run gateway tool call"

    if _has_title_and_url(answer):
        return

    anti_bot_markers = (
        "captcha",
        "anti-bot",
        "unusual traffic",
        "verify you are human",
        "cloudflare",
        "blocked",
    )
    anti_bot_evidence = any(
        any(marker in json.dumps(event, ensure_ascii=False).lower() for marker in anti_bot_markers)
        for event in events
    )
    if anti_bot_evidence:
        pytest.skip(
            "Live Browser Use run verified browser_use_run gateway calls, but anti-bot checks "
            "blocked deterministic extraction in this environment."
        )
    assert _has_title_and_url(answer), answer
