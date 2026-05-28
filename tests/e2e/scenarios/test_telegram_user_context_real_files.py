from __future__ import annotations

import shutil
import subprocess
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from evaluation.judge import assert_e2e_objective_satisfied
from harness.runner import E2EHarness
from openpyxl import Workbook

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]


def _wait_until(predicate: Any, timeout_seconds: float = 180.0) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.5)
    return bool(predicate())


def _assert_user_context_answer_objective(
    harness: E2EHarness,
    *,
    scenario: str,
    objective: str,
    query_calls: list[dict[str, Any]],
    final_text: str,
    sources: list[dict[str, Any]],
) -> None:
    assert_e2e_objective_satisfied(
        scenario=scenario,
        objective=objective,
        evidence={
            "query_calls": query_calls[-3:],
            "final_answer": final_text,
            "sources": sources,
        },
        system_log_path=harness.system_log_path,
        behavior_log_path=harness.behavior_log_path,
        llm_trace_path=harness.llm_trace_path,
    )


def _telegram_text_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    text: str,
    message_id: int,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + message_id,
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "text": text,
        },
    }


def _telegram_document_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    file_name: str,
    mime_type: str,
    file_size: int,
    message_id: int,
    caption: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(datetime.now(UTC).timestamp()),
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": user_id, "is_bot": False, "username": username},
        "document": {
            "file_id": file_id,
            "file_unique_id": f"unique_{file_id}",
            "file_name": file_name,
            "mime_type": mime_type,
            "file_size": int(file_size),
        },
    }
    if caption is not None:
        message["caption"] = caption
    return {"update_id": int(time.time() * 1000) + message_id, "message": message}


def _telegram_photo_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    file_unique_id: str,
    file_size: int,
    message_id: int,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + message_id,
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "photo": [
                {
                    "file_id": file_id,
                    "file_unique_id": file_unique_id,
                    "file_size": int(file_size),
                    "width": 1280,
                    "height": 720,
                }
            ],
        },
    }


def _telegram_video_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    file_unique_id: str,
    file_name: str,
    mime_type: str,
    file_size: int,
    message_id: int,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + message_id,
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "video": {
                "file_id": file_id,
                "file_unique_id": file_unique_id,
                "file_name": file_name,
                "mime_type": mime_type,
                "file_size": int(file_size),
                "duration": 4,
                "width": 1280,
                "height": 720,
            },
        },
    }


def _telegram_audio_message(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    file_unique_id: str,
    file_name: str,
    mime_type: str,
    file_size: int,
    message_id: int,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + message_id,
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": username},
            "audio": {
                "file_id": file_id,
                "file_unique_id": file_unique_id,
                "file_name": file_name,
                "mime_type": mime_type,
                "file_size": int(file_size),
                "duration": 3,
            },
        },
    }


def _xlsx_bytes() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Offers"
    ws.append(["Offer", "Price", "Notes"])
    ws.append(["Launch package", "$900", "Includes blog ideas and scripts"])
    out = BytesIO()
    wb.save(out)
    return out.getvalue()


def _docx_bytes() -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Retainer offer: weekly scenario writing and idea bank refresh.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    out = BytesIO()
    with ZipFile(out, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return out.getvalue()


def _write_fixture(path: Path, raw_bytes: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    return path


def _run_ffmpeg(args: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required to generate real media fixtures")
    result = subprocess.run(
        [ffmpeg, "-y", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def _write_image_fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=1280x720",
            "-frames:v",
            "1",
            "-vf",
            "drawtext=text='BLOG SPRINT CTA':fontcolor=black:fontsize=86:x=(w-text_w)/2:y=(h-text_h)/2",
            str(path),
        ]
    )
    return path


def _write_video_fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:d=4",
            "-vf",
            "drawtext=text='RETENTION LOOP':fontcolor=white:fontsize=92:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
    )
    return path


def _write_pdf_fixture(path: Path) -> Path:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        pytest.skip("reportlab is required to generate real PDF fixtures")
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 720, "PDF RETAINER FLOOR: $1200")
    pdf.drawString(72, 700, "Scope: weekly blog scripts, idea bank refresh, and scenario review.")
    pdf.save()
    return path


def _write_audio_fixture(path: Path) -> Path:
    say = shutil.which("say")
    if not say:
        pytest.skip("macOS say is required to generate real audio fixtures")
    path.parent.mkdir(parents=True, exist_ok=True)
    phrase = "Audio retainer code is blue lantern. Weekly script review is included."
    result = subprocess.run(
        [say, "-o", str(path), phrase],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return path


def _messages_for_chat(harness: E2EHarness, *, chat_id: int, start_index: int) -> list[dict[str, Any]]:
    return [
        item
        for item in harness.telegram_client.sent_messages[start_index:]
        if int(item.get("chat_id", 0)) == int(chat_id)
        and not str(item.get("business_connection_id", "") or "").strip()
    ]


def test_live_telegram_chat_uploads_real_files_then_queries_user_context(
    e2e_harness: E2EHarness,
    tmp_path: Path,
) -> None:
    owner_user_id = 8242
    owner_chat_id = 18242
    username = "context_owner"
    customer_id = f"telegram_{owner_user_id}"
    fixture_dir = tmp_path / "uploaded_context_files"
    offers_path = _write_fixture(fixture_dir / "offers.xlsx", _xlsx_bytes())
    positioning_path = _write_fixture(fixture_dir / "positioning.docx", _docx_bytes())
    offers_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_offers_xlsx",
        path=offers_path,
        filename="offers.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    positioning_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_positioning_docx",
        path=positioning_path,
        filename="positioning.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    internal_start = e2e_harness.count_internal_api_calls()
    message_start = len(e2e_harness.telegram_client.sent_messages)

    intro_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=1001,
            text=(
                "I am going to upload source files. Add each uploaded file to my reusable "
                "interactive user context. After that I will ask a question that must be answered "
                "from user_context_query, not from memory."
            ),
        )
    )
    assert intro_status == 200
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=message_start)) >= 1,
        timeout_seconds=180.0,
    )

    upload_start = len(e2e_harness.telegram_client.sent_messages)
    offers_status = e2e_harness.post_telegram(
        body=_telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=1002,
            file_id="tg_offers_xlsx",
            file_name="offers.xlsx",
            mime_type=str(offers_tg["mime_type"]),
            file_size=int(offers_tg["file_size"]),
        )
    )
    positioning_status = e2e_harness.post_telegram(
        body=_telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=1003,
            file_id="tg_positioning_docx",
            file_name="positioning.docx",
            mime_type=str(positioning_tg["mime_type"]),
            file_size=int(positioning_tg["file_size"]),
        )
    )
    assert offers_status == 200
    assert positioning_status == 200

    def add_file_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/add_files"
        ]

    def added_file_ids() -> list[str]:
        return [
            file_id
            for call in add_file_calls()
            for file_id in call.get("json_body", {}).get("file_ids", [])
        ]

    assert _wait_until(lambda: len(set(added_file_ids())) == 2, timeout_seconds=240.0), add_file_calls()
    collected_file_ids = added_file_ids()
    assert len(set(collected_file_ids)) == 2, add_file_calls()
    assert _wait_until(
        lambda: bool(
            _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=upload_start)
        ),
        timeout_seconds=180.0,
    )
    upload_messages = _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=upload_start)
    assert upload_messages

    list_response = e2e_harness.client.post(
        "/internal/user_context/list_sources",
        json={"customer_id": customer_id, "include_archived": False},
    )
    assert list_response.status_code == 200, list_response.text
    sources = list_response.json()["sources"]
    assert {source["source_kind"] for source in sources} == {"structured_table", "local_source"}
    assert {source["filename"] for source in sources} == {"offers.xlsx", "positioning.docx"}

    question_start = len(e2e_harness.telegram_client.sent_messages)
    question_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=1004,
            text=(
                "Use user_context_query now. Based only on my uploaded user context, "
                "what is the Launch package price and what retainer work is offered?"
            ),
        )
    )
    assert question_status == 200

    def query_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/query"
        ]

    assert _wait_until(lambda: bool(query_calls()), timeout_seconds=240.0), query_calls()
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)) >= 1,
        timeout_seconds=180.0,
    )
    final_text = str(
        _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)[-1].get("text", "")
        or ""
    )
    _assert_user_context_answer_objective(
        e2e_harness,
        scenario="live_telegram_chat_recalls_uploaded_user_context",
        objective=(
            "By the end of the conversation, the assistant answers from uploaded user context "
            "that the Launch package is $900 and the retainer work includes weekly blog scripts, "
            "scenario review, and idea bank refresh."
        ),
        query_calls=query_calls(),
        final_text=final_text,
        sources=sources,
    )

    e2e_harness.recorder.add(
        "live_user_context_real_file_chat_e2e",
        customer_id=customer_id,
        add_file_call_count=len(add_file_calls()),
        query_call_count=len(query_calls()),
        sources=sources,
        final_text=final_text,
    )


def test_live_telegram_chat_recalls_pdf_user_context(
    e2e_harness: E2EHarness,
    tmp_path: Path,
) -> None:
    owner_user_id = 8244
    owner_chat_id = 18244
    username = "pdf_context_owner"
    customer_id = f"telegram_{owner_user_id}"
    fixture_dir = tmp_path / "uploaded_pdf_context"
    pdf_path = _write_pdf_fixture(fixture_dir / "retainer.pdf")
    pdf_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_retainer_pdf",
        path=pdf_path,
        filename="retainer.pdf",
        mime_type="application/pdf",
    )
    internal_start = e2e_harness.count_internal_api_calls()
    message_start = len(e2e_harness.telegram_client.sent_messages)

    intro_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=3001,
            text=(
                "I am going to upload a PDF. Add the uploaded PDF to my reusable interactive "
                "user context. Later answer from user_context_query."
            ),
        )
    )
    assert intro_status == 200
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=message_start)) >= 1,
        timeout_seconds=180.0,
    )

    upload_status = e2e_harness.post_telegram(
        body=_telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=3002,
            file_id="tg_retainer_pdf",
            file_name="retainer.pdf",
            mime_type=str(pdf_tg["mime_type"]),
            file_size=int(pdf_tg["file_size"]),
        )
    )
    assert upload_status == 200

    def add_file_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/add_files"
        ]

    assert _wait_until(lambda: bool(add_file_calls()), timeout_seconds=240.0), add_file_calls()

    list_response = e2e_harness.client.post(
        "/internal/user_context/list_sources",
        json={"customer_id": customer_id, "include_archived": False},
    )
    assert list_response.status_code == 200, list_response.text
    sources = list_response.json()["sources"]
    assert {source["filename"] for source in sources} == {"retainer.pdf"}
    assert len(sources) == 1

    question_start = len(e2e_harness.telegram_client.sent_messages)
    question_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=3003,
            text=(
                "Use user_context_query now. Based only on my uploaded PDF context, "
                "what is the PDF retainer floor and what work is in scope?"
            ),
        )
    )
    assert question_status == 200

    def query_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/query"
        ]

    assert _wait_until(lambda: bool(query_calls()), timeout_seconds=240.0), query_calls()
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)) >= 1,
        timeout_seconds=180.0,
    )
    final_text = str(
        _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)[-1].get("text", "")
        or ""
    )
    _assert_user_context_answer_objective(
        e2e_harness,
        scenario="live_telegram_chat_recalls_pdf_user_context",
        objective=(
            "By the end of the conversation, the assistant answers from uploaded PDF context "
            "that the retainer floor is $1,200 and the in-scope work includes weekly blog "
            "scripts, idea bank refresh, and scenario review."
        ),
        query_calls=query_calls(),
        final_text=final_text,
        sources=sources,
    )

    e2e_harness.recorder.add(
        "live_user_context_pdf_recall_e2e",
        customer_id=customer_id,
        add_file_call_count=len(add_file_calls()),
        query_call_count=len(query_calls()),
        sources=sources,
        final_text=final_text,
    )


def test_live_telegram_chat_recalls_audio_user_context(
    e2e_harness: E2EHarness,
    tmp_path: Path,
) -> None:
    owner_user_id = 8245
    owner_chat_id = 18245
    username = "audio_context_owner"
    customer_id = f"telegram_{owner_user_id}"
    fixture_dir = tmp_path / "uploaded_audio_context"
    audio_path = _write_audio_fixture(fixture_dir / "audio_context.aiff")
    audio_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_audio_context",
        path=audio_path,
        filename="audio_context.aiff",
        mime_type="audio/aiff",
    )
    internal_start = e2e_harness.count_internal_api_calls()
    message_start = len(e2e_harness.telegram_client.sent_messages)

    intro_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=4001,
            text=(
                "I am going to upload an audio file. Add the uploaded audio to my reusable "
                "interactive user context. Later answer from user_context_query."
            ),
        )
    )
    assert intro_status == 200
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=message_start)) >= 1,
        timeout_seconds=180.0,
    )

    upload_status = e2e_harness.post_telegram(
        body=_telegram_audio_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=4002,
            file_id="tg_audio_context",
            file_unique_id="audio_context_unique",
            file_name="audio_context.aiff",
            mime_type=str(audio_tg["mime_type"]),
            file_size=int(audio_tg["file_size"]),
        )
    )
    assert upload_status == 200

    def add_file_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/add_files"
        ]

    assert _wait_until(lambda: bool(add_file_calls()), timeout_seconds=300.0), add_file_calls()

    list_response = e2e_harness.client.post(
        "/internal/user_context/list_sources",
        json={"customer_id": customer_id, "include_archived": False},
    )
    assert list_response.status_code == 200, list_response.text
    sources = list_response.json()["sources"]
    assert {source["source_kind"] for source in sources} == {"derived_from_media"}
    assert {source["filename"] for source in sources} == {"audio_context.aiff"}

    question_start = len(e2e_harness.telegram_client.sent_messages)
    question_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=4003,
            text=(
                "Use user_context_query now. Based only on my uploaded audio context, "
                "what is the audio retainer code and what review is included?"
            ),
        )
    )
    assert question_status == 200

    def query_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/query"
        ]

    assert _wait_until(lambda: bool(query_calls()), timeout_seconds=240.0), query_calls()
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)) >= 1,
        timeout_seconds=180.0,
    )
    final_text = str(
        _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)[-1].get("text", "")
        or ""
    )
    _assert_user_context_answer_objective(
        e2e_harness,
        scenario="live_telegram_chat_recalls_audio_user_context",
        objective=(
            "By the end of the conversation, the assistant answers from uploaded audio context "
            "that the audio retainer code is Blue Lantern and the included review is weekly "
            "script review."
        ),
        query_calls=query_calls(),
        final_text=final_text,
        sources=sources,
    )

    e2e_harness.recorder.add(
        "live_user_context_audio_recall_e2e",
        customer_id=customer_id,
        add_file_call_count=len(add_file_calls()),
        query_call_count=len(query_calls()),
        sources=sources,
        final_text=final_text,
    )


def test_live_telegram_workflow_setup_reuses_existing_user_context_source(
    e2e_harness: E2EHarness,
    tmp_path: Path,
) -> None:
    owner_user_id = 8246
    owner_chat_id = 18246
    username = "reuse_context_owner"
    customer_id = f"telegram_{owner_user_id}"
    fixture_dir = tmp_path / "reuse_context"
    offers_path = _write_fixture(fixture_dir / "offers.xlsx", _xlsx_bytes())
    offers_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_reuse_offers_xlsx",
        path=offers_path,
        filename="offers.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    internal_start = e2e_harness.count_internal_api_calls()
    message_start = len(e2e_harness.telegram_client.sent_messages)

    intro_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=5001,
            text="I will upload a source file. Add it to my reusable interactive user context.",
        )
    )
    assert intro_status == 200
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=message_start)) >= 1,
        timeout_seconds=180.0,
    )

    upload_status = e2e_harness.post_telegram(
        body=_telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=5002,
            file_id="tg_reuse_offers_xlsx",
            file_name="offers.xlsx",
            mime_type=str(offers_tg["mime_type"]),
            file_size=int(offers_tg["file_size"]),
        )
    )
    assert upload_status == 200

    def calls(path: str) -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == path
        ]

    assert _wait_until(lambda: bool(calls("/internal/user_context/add_files")), timeout_seconds=240.0)

    setup_start = len(e2e_harness.telegram_client.sent_messages)
    setup_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=5003,
            text=(
                "Start an intake workflow setup for collecting blog consultation leads into a local CSV. "
                "Reuse my existing user context source offers.xlsx for this workflow setup. "
                "First list or find user context sources, then call business_knowledge_index on the selected "
                "file id for the current workflow setup scope. Do not promote to a final workflow yet."
            ),
        )
    )
    assert setup_status == 200

    def workflow_setup_index_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/knowledge/index_sources"
            and item.get("json_body", {}).get("scope_type") == "workflow_setup"
        ]

    assert _wait_until(lambda: bool(calls("/internal/user_context/list_sources")) or bool(calls("/internal/user_context/find_sources")), timeout_seconds=240.0)
    assert _wait_until(lambda: bool(workflow_setup_index_calls()), timeout_seconds=240.0), e2e_harness.internal_api_calls_since(internal_start)
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=setup_start)) >= 1,
        timeout_seconds=180.0,
    )

    setup_calls = workflow_setup_index_calls()
    assert setup_calls[-1]["json_body"]["customer_id"] == customer_id
    assert setup_calls[-1]["json_body"]["file_ids"]
    assert not calls("/internal/user_context/promote_to_intake")

    e2e_harness.recorder.add(
        "live_user_context_reuse_for_workflow_setup_e2e",
        customer_id=customer_id,
        user_context_list_calls=len(calls("/internal/user_context/list_sources")),
        user_context_find_calls=len(calls("/internal/user_context/find_sources")),
        workflow_setup_index_calls=setup_calls,
    )


def test_live_telegram_chat_recalls_image_and_video_user_context(
    e2e_harness: E2EHarness,
    tmp_path: Path,
) -> None:
    owner_user_id = 8243
    owner_chat_id = 18243
    username = "media_context_owner"
    customer_id = f"telegram_{owner_user_id}"
    fixture_dir = tmp_path / "uploaded_media_context"
    image_path = _write_image_fixture(fixture_dir / "blog_sprint_cta.jpg")
    video_path = _write_video_fixture(fixture_dir / "retention_loop.mp4")
    image_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_blog_sprint_cta",
        path=image_path,
        filename="blog_sprint_cta.jpg",
        mime_type="image/jpeg",
    )
    video_tg = e2e_harness.telegram_client.register_file(
        file_id="tg_retention_loop_video",
        path=video_path,
        filename="retention_loop.mp4",
        mime_type="video/mp4",
    )
    internal_start = e2e_harness.count_internal_api_calls()
    message_start = len(e2e_harness.telegram_client.sent_messages)

    intro_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=2001,
            text=(
                "I am going to upload an image and a video. Add each uploaded media file to my "
                "reusable interactive user context. Later answer from user_context_query."
            ),
        )
    )
    assert intro_status == 200
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=message_start)) >= 1,
        timeout_seconds=180.0,
    )

    def add_file_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/add_files"
        ]

    def added_file_ids() -> list[str]:
        return [
            file_id
            for call in add_file_calls()
            for file_id in call.get("json_body", {}).get("file_ids", [])
        ]

    image_status = e2e_harness.post_telegram(
        body=_telegram_photo_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=2002,
            file_id="tg_blog_sprint_cta",
            file_unique_id="blog_sprint_cta_unique",
            file_size=int(image_tg["file_size"]),
        )
    )
    assert image_status == 200
    assert _wait_until(lambda: len(set(added_file_ids())) == 1, timeout_seconds=240.0), add_file_calls()

    video_status = e2e_harness.post_telegram(
        body=_telegram_video_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=2003,
            file_id="tg_retention_loop_video",
            file_unique_id="retention_loop_unique",
            file_name="retention_loop.mp4",
            mime_type=str(video_tg["mime_type"]),
            file_size=int(video_tg["file_size"]),
        )
    )
    assert video_status == 200

    assert _wait_until(lambda: len(set(added_file_ids())) == 2, timeout_seconds=360.0), add_file_calls()
    assert len(set(added_file_ids())) == 2, add_file_calls()

    list_response = e2e_harness.client.post(
        "/internal/user_context/list_sources",
        json={"customer_id": customer_id, "include_archived": False},
    )
    assert list_response.status_code == 200, list_response.text
    sources = list_response.json()["sources"]
    assert {source["source_kind"] for source in sources} == {"derived_from_media"}
    assert len(sources) == 2

    question_start = len(e2e_harness.telegram_client.sent_messages)
    question_status = e2e_harness.post_telegram(
        body=_telegram_text_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            username=username,
            message_id=2004,
            text=(
                "Use user_context_query now. Based only on my uploaded media context, "
                "what phrase appears in the image and what phrase appears in the video?"
            ),
        )
    )
    assert question_status == 200

    def query_calls() -> list[dict[str, Any]]:
        return [
            item
            for item in e2e_harness.internal_api_calls_since(internal_start)
            if item.get("path") == "/internal/user_context/query"
        ]

    assert _wait_until(lambda: bool(query_calls()), timeout_seconds=240.0), query_calls()
    assert _wait_until(
        lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)) >= 1,
        timeout_seconds=180.0,
    )
    final_text = str(
        _messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=question_start)[-1].get("text", "")
        or ""
    )
    _assert_user_context_answer_objective(
        e2e_harness,
        scenario="live_telegram_chat_recalls_image_and_video_user_context",
        objective=(
            "By the end of the conversation, the assistant answers from uploaded media context "
            "that the image phrase is Blog Sprint and the video phrase is Retention Loop."
        ),
        query_calls=query_calls(),
        final_text=final_text,
        sources=sources,
    )

    e2e_harness.recorder.add(
        "live_user_context_media_recall_e2e",
        customer_id=customer_id,
        add_file_call_count=len(add_file_calls()),
        query_call_count=len(query_calls()),
        sources=sources,
        final_text=final_text,
    )
