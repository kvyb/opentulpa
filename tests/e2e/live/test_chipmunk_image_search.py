from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from harness.runner import E2EHarness

LIVE_FLAG = "OPENTULPA_ENABLE_LIVE_CHIPMUNK_IMAGE_E2E"

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]

if str(os.getenv(LIVE_FLAG, "")).strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        f"set {LIVE_FLAG}=1 to run live chipmunk image search e2e test",
        allow_module_level=True,
    )


def _wait_until(predicate: Any, timeout_seconds: float = 120.0) -> bool:
    deadline = time.time() + max(1.0, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


def _telegram_message(*, chat_id: int, user_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": 1,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
            "text": text,
        },
    }


def _internal_calls(harness: E2EHarness, path: str) -> list[dict[str, Any]]:
    return [
        item
        for item in harness.internal_api_calls_since()
        if str(item.get("path", "")).strip() == path
    ]


def test_live_telegram_chipmunk_image_search_sends_web_image(
    e2e_harness: E2EHarness,
) -> None:
    owner_chat_id = 931_001
    owner_user_id = 931_002
    sent_count = len(e2e_harness.telegram_client.sent_files)

    prompt = (
        "Go on the internet, find me a real chipmunk image, and send it here please. "
        "Live E2E instruction: first call tool_group_exec with group='web' and "
        "command='web_search' for 'Eastern_Chipmunk_1745.jpg Wikimedia Commons'. "
        "Once search confirms that Wikimedia Commons file name, construct this stable image URL "
        "exactly: https://commons.wikimedia.org/wiki/Special:FilePath/Eastern_Chipmunk_1745.jpg. "
        "Do not fetch pages, do not call an API, do not use browser_use_run, and do not guess "
        "upload.wikimedia.org, Unsplash, Pexels, or CDN image URLs. Your second tool call must be "
        "tool_group_exec group='web' command='web_image_send' with that Special:FilePath URL. "
        "In the final reply, include the exact URL you sent."
    )

    status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text=prompt)
    )
    assert status == 200

    assert _wait_until(
        lambda: len(e2e_harness.telegram_client.sent_files) > sent_count,
        timeout_seconds=180.0,
    )

    web_search_calls = _internal_calls(e2e_harness, "/internal/web_search")
    image_send_calls = _internal_calls(e2e_harness, "/internal/files/send_web_image")
    assert web_search_calls, "Expected OpenTulpa to call web_search before image send"
    assert image_send_calls, "Expected OpenTulpa to call web_image_send"

    send_call = image_send_calls[-1]
    assert int(send_call.get("status_code") or 0) == 200, send_call
    request_body = send_call.get("json_body")
    assert isinstance(request_body, dict)
    sent_url = str(request_body.get("url") or "").strip()
    assert sent_url.startswith(("http://", "https://"))

    sent_file = e2e_harness.telegram_client.sent_files[-1]
    assert sent_file["kind"] in {"photo", "animation"}
    assert int(sent_file["size_bytes"]) > 0
    assert str(sent_file.get("mime_type") or "").startswith("image/")

    report = e2e_harness.write_status_report(
        scenario="live_telegram_chipmunk_image_search_sends_web_image",
        ok=True,
        details={
            "sent_url": sent_url,
            "sent_file": sent_file,
            "web_search_calls": len(web_search_calls),
            "image_send_calls": len(image_send_calls),
        },
    )
    assert report.exists()
