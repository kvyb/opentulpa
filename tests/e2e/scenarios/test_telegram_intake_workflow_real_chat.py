from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from evaluation.judge import DEFAULT_JUDGE_MODEL, evaluate_e2e_scenario_with_llm_judge
from harness.lead_simulator import LeadProfile
from harness.runner import E2EHarness

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]


def _wait_until(predicate: Any, timeout_seconds: float = 45.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


def _seed_telegram_business_connection(
    harness: E2EHarness,
    *,
    owner_user_id: int,
    owner_chat_id: int,
    business_connection_id: str = "bc_e2e_123",
) -> str:
    telegram_business = harness.client.app.state.telegram_business
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": owner_chat_id,
            "is_enabled": True,
            "user": {
                "id": owner_user_id,
                "is_bot": False,
                "first_name": "Kim",
                "username": "kim",
            },
            "rights": {"can_reply": True},
        }
    )
    return business_connection_id


def _telegram_message(*, chat_id: int, user_id: int, text: str, message_id: int = 1) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
            "text": text,
        },
    }


def _telegram_document_message(
    *,
    chat_id: int,
    user_id: int,
    caption: str,
    file_id: str,
    file_name: str,
    mime_type: str,
    file_size: int,
    message_id: int = 1,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
            "caption": caption,
            "document": {
                "file_id": file_id,
                "file_unique_id": f"unique_{file_id}",
                "file_name": file_name,
                "mime_type": mime_type,
                "file_size": int(file_size),
            },
        },
    }


def _telegram_business_message(
    *,
    business_connection_id: str,
    lead_chat_id: int,
    lead_user_id: int,
    text: str,
    message_id: int = 100,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "business_message": {
            "business_connection_id": business_connection_id,
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": lead_chat_id, "type": "private", "username": f"lead_{lead_user_id}"},
            "from": {"id": lead_user_id, "is_bot": False, "username": f"lead_{lead_user_id}"},
            "text": text,
        },
    }


def _list_workflows(harness: E2EHarness, *, customer_id: str) -> list[dict[str, Any]]:
    response = harness.client.post(
        "/internal/intake/workflows/list",
        json={"customer_id": customer_id, "include_disabled": True},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    workflows = payload.get("workflows") or []
    return workflows if isinstance(workflows, list) else []


def _workflow_setup_session(
    harness: E2EHarness,
    *,
    customer_id: str,
    thread_id: str,
) -> dict[str, Any]:
    response = harness.client.post(
        "/internal/intake/setup/get",
        json={"customer_id": customer_id, "thread_id": thread_id, "include_paused": True},
    )
    if response.status_code != 200:
        return {}
    payload = response.json()
    session = payload.get("session")
    return session if isinstance(session, dict) else {}


def _workflow_setup_has_proposal(
    harness: E2EHarness,
    *,
    customer_id: str,
    thread_id: str,
) -> bool:
    session = _workflow_setup_session(harness, customer_id=customer_id, thread_id=thread_id)
    return bool(str(session.get("last_proposed_draft_hash", "") or "").strip())


def _telegram_owner_thread_id(*, chat_id: int) -> str:
    from opentulpa.interfaces.telegram import chat_service as chat_module

    state = chat_module.STATE_STORE.load()
    sessions = state.get("sessions") if isinstance(state, dict) else {}
    slot = sessions.get(str(chat_id)) if isinstance(sessions, dict) else {}
    if not isinstance(slot, dict):
        return ""
    return str(slot.get("thread_id", "") or "").strip()


def _latest_message_for_chat(
    harness: E2EHarness,
    *,
    chat_id: int,
    start_index: int = 0,
) -> dict[str, Any] | None:
    for item in reversed(harness.telegram_client.sent_messages[start_index:]):
        if int(item.get("chat_id", 0)) == int(chat_id):
            return item
    return None


def _messages_for_chat(
    harness: E2EHarness,
    *,
    chat_id: int,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    return [
        item
        for item in harness.telegram_client.sent_messages[start_index:]
        if int(item.get("chat_id", 0)) == int(chat_id)
    ]


def _behavior_events(harness: E2EHarness) -> list[dict[str, Any]]:
    if not harness.behavior_log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in harness.behavior_log_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _turn_modes_seen(harness: E2EHarness, *, customer_id: str) -> list[str]:
    modes: list[str] = []
    for event in _behavior_events(harness):
        if str(event.get("customer_id", "") or "") != str(customer_id):
            continue
        mode = str(event.get("turn_mode", "") or "").strip()
        if mode and mode not in modes:
            modes.append(mode)
    return modes


def _csv_rows_for_relative_path(
    harness: E2EHarness,
    *,
    relative_path: str,
) -> list[dict[str, str]]:
    intake_service = harness.client.app.state.intake_workflows
    csv_path = intake_service._project_root / relative_path  # noqa: SLF001
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _lead_source_messages(
    harness: E2EHarness,
    *,
    customer_id: str,
    business_connection_id: str,
    lead_chat_id: int,
) -> list[dict[str, Any]]:
    telegram_business = harness.client.app.state.telegram_business
    payload = telegram_business.get_conversation(
        customer_id=customer_id,
        business_connection_id=business_connection_id,
        conversation_id=str(lead_chat_id),
    )
    conversation = payload.get("conversation") if isinstance(payload, dict) else {}
    messages = conversation.get("messages") if isinstance(conversation, dict) else []
    return messages if isinstance(messages, list) else []


def _judge_verdict(report_payload: dict[str, Any]) -> str:
    evaluation = report_payload.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return ""
    parsed = evaluation.get("parsed", {})
    if isinstance(parsed, dict):
        return str(parsed.get("verdict", "")).strip().lower()
    return str(evaluation.get("verdict", "")).strip().lower()


def _addresses_pricing_question(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "price",
            "pricing",
            "cost",
            "starts at",
            "$",
            "usd",
            "ruble",
            "rupiah",
        )
    )


_AUTOSPA_PRICE_ASSET = Path(__file__).resolve().parents[1] / "assets" / "autospa_price.xlsx"


def _live_google_sheets_target(harness: E2EHarness) -> Any | None:
    return getattr(harness.composio_service, "live_google_sheets_target", None)


def _owner_identity_for_autospa(harness: E2EHarness) -> tuple[int, int, str]:
    target = _live_google_sheets_target(harness)
    customer_id = str(getattr(target, "customer_id", "") or "").strip()
    if customer_id.startswith("telegram_"):
        raw_user_id = customer_id.removeprefix("telegram_").strip()
        if raw_user_id.isdigit():
            owner_user_id = int(raw_user_id)
            return owner_user_id, owner_user_id + 1000, customer_id
    return 901, 1901, "telegram_901"


def _write_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _filtered_autospa_internal_calls(harness: E2EHarness) -> list[dict[str, Any]]:
    prefixes = (
        "/internal/files/",
        "/internal/intake/",
        "/internal/composio/",
        "/internal/telegram/business/",
    )
    return [
        item
        for item in harness.internal_api_calls_since(0)
        if str(item.get("path", "")).startswith(prefixes)
    ]


def _workflow_knowledge_markdown(
    harness: E2EHarness,
    *,
    customer_id: str,
    workflow: dict[str, Any],
) -> str:
    file_vault = getattr(harness.client.app.state.intake_workflows, "_file_vault", None)
    if file_vault is None:
        return ""
    chunks: list[str] = []
    for file_id in workflow.get("knowledge_file_ids") or []:
        raw = file_vault.read_file_bytes(customer_id, str(file_id or "").strip())
        if raw:
            chunks.append(raw.decode("utf-8", errors="replace"))
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _current_autospa_artifacts(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    artifact_dir: Path,
    customer_id: str,
) -> dict[str, str]:
    workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
    prepared_markdown = _workflow_knowledge_markdown(
        harness,
        customer_id=customer_id,
        workflow=workflow,
    )
    if prepared_markdown:
        content_hash = hashlib.sha256(prepared_markdown.encode("utf-8")).hexdigest()
        if state.get("prepared_knowledge_hash") != content_hash:
            state["prepared_knowledge_hash"] = content_hash
            harness.recorder.add(
                "prepared_knowledge_snapshot",
                knowledge_file_ids=workflow.get("knowledge_file_ids") or [],
                markdown_chars=len(prepared_markdown),
                sha256=content_hash,
            )

    workflow_hash = hashlib.sha256(
        json.dumps(workflow, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if workflow and state.get("workflow_hash") != workflow_hash:
        state["workflow_hash"] = workflow_hash
        harness.recorder.add("workflow_snapshot", workflow=workflow)

    paths = {
        "owner_transcript": artifact_dir / "owner_transcript.json",
        "lead_transcripts": artifact_dir / "lead_transcripts.json",
        "prepared_knowledge": artifact_dir / "prepared_knowledge.md",
        "workflow_snapshot": artifact_dir / "workflow_snapshot.json",
        "sheet_writes": artifact_dir / "sheet_writes.json",
        "internal_calls_filtered": artifact_dir / "internal_calls_filtered.json",
        "stage_judgements": artifact_dir / "stage_judgements.json",
    }
    _write_json_artifact(paths["owner_transcript"], state.get("owner_transcript") or [])
    _write_json_artifact(paths["lead_transcripts"], state.get("lead_transcripts") or [])
    paths["prepared_knowledge"].write_text(prepared_markdown, encoding="utf-8")
    _write_json_artifact(paths["workflow_snapshot"], workflow)
    _write_json_artifact(
        paths["sheet_writes"],
        getattr(harness.composio_service, "sheet_writes", []),
    )
    _write_json_artifact(paths["internal_calls_filtered"], _filtered_autospa_internal_calls(harness))
    _write_json_artifact(paths["stage_judgements"], state.get("stage_judgements") or [])
    return {key: str(path) for key, path in paths.items()}


def _write_autospa_failure_debug(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    artifact_dir: Path,
    customer_id: str,
    error: BaseException,
) -> str:
    path = artifact_dir / "failure_debug.json"
    workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
    workflow_id = str(workflow.get("workflow_id", "") or "").strip()
    bookings = []
    if workflow_id:
        bookings = harness.list_bookings(customer_id=customer_id, workflow_id=workflow_id)
    _write_json_artifact(
        path,
        {
            "error_type": type(error).__name__,
            "error": str(error),
            "owner_transcript": state.get("owner_transcript") or [],
            "lead_transcripts": state.get("lead_transcripts") or [],
            "workflow": workflow,
            "workflows": _list_workflows(harness, customer_id=customer_id),
            "bookings": bookings,
            "telegram_sent_messages": harness.telegram_client.sent_messages,
            "sheet_writes": getattr(harness.composio_service, "sheet_writes", []),
            "internal_calls_filtered": _filtered_autospa_internal_calls(harness),
        },
    )
    return str(path)


def _stage_judge_details(
    *,
    state: dict[str, Any],
    stage_name: str,
    stage_goal: str,
    stage_result: dict[str, Any],
    artifact_paths: dict[str, str],
    harness: E2EHarness,
) -> dict[str, Any]:
    workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
    return {
        "stage_name": stage_name,
        "stage_goal": stage_goal,
        "stage_result": stage_result,
        "owner_transcript": state.get("owner_transcript") or [],
        "lead_transcripts": state.get("lead_transcripts") or [],
        "workflow": workflow,
        "sheet_writes": getattr(harness.composio_service, "sheet_writes", []),
        "artifact_paths": artifact_paths,
        "internal_call_count": len(_filtered_autospa_internal_calls(harness)),
        "judge_instruction": (
            "Оцени, достиг ли этап своей цели, используя только эти транскрипты и артефакты. "
            "Если поведение продукта плохое, верни verdict='fail' и объясни, но это не должно "
            "считаться ошибкой pytest."
        ),
    }


def _judge_autospa_stage(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    stage_name: str,
    stage_goal: str,
    stage_result: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    result = evaluate_e2e_scenario_with_llm_judge(
        scenario=f"autospa_telegram_intake:{stage_name}",
        details=_stage_judge_details(
            state=state,
            stage_name=stage_name,
            stage_goal=stage_goal,
            stage_result=stage_result,
            artifact_paths=artifact_paths,
            harness=harness,
        ),
        system_log_path=harness.system_log_path,
        behavior_log_path=harness.behavior_log_path,
        llm_trace_path=harness.llm_trace_path,
        model=DEFAULT_JUDGE_MODEL,
        timeout_seconds=40.0,
    )
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    entry = {
        "stage": stage_name,
        "model": result.get("model", DEFAULT_JUDGE_MODEL),
        "input_artifact_paths": artifact_paths,
        "ok": bool(result.get("ok", False)),
        "attempted": bool(result.get("attempted", False)),
        "reason": result.get("reason"),
        "status_code": result.get("status_code"),
        "verdict": str(parsed.get("verdict", "") or ""),
        "summary": str(parsed.get("summary", "") or ""),
        "failures": parsed.get("failures") if isinstance(parsed, dict) else [],
        "confidence": parsed.get("confidence") if isinstance(parsed, dict) else None,
        "raw_response": str(result.get("raw_response", "") or "")[:4000],
    }
    state.setdefault("stage_judgements", []).append(entry)
    harness.recorder.add("stage_judge_eval", **entry)
    judgements_path = Path(artifact_paths["stage_judgements"])
    _write_json_artifact(judgements_path, state.get("stage_judgements") or [])
    if not bool(result.get("ok", False)):
        raise RuntimeError(f"stage judge failed for {stage_name}: {result}")
    return entry


def _run_autospa_stage(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    artifact_dir: Path,
    customer_id: str,
    stage_name: str,
    stage_goal: str,
    run: Any,
) -> dict[str, Any]:
    harness.recorder.add("stage_started", stage=stage_name, goal=stage_goal)
    started = time.monotonic()
    try:
        result = run()
    except Exception as exc:
        harness.recorder.add("stage_failed", stage=stage_name, error=str(exc), error_type=type(exc).__name__)
        paths = _current_autospa_artifacts(
            harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
        )
        _write_autospa_failure_debug(
            harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            error=exc,
        )
        _judge_autospa_stage(
            harness,
            state=state,
            stage_name=stage_name,
            stage_goal=stage_goal,
            stage_result={"ok": False, "error": str(exc), "error_type": type(exc).__name__},
            artifact_paths=paths,
        )
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)
    stage_result = result if isinstance(result, dict) else {"result": result}
    stage_result["elapsed_ms"] = elapsed_ms
    harness.recorder.add("stage_completed", stage=stage_name, elapsed_ms=elapsed_ms, result=stage_result)
    paths = _current_autospa_artifacts(
        harness,
        state=state,
        artifact_dir=artifact_dir,
        customer_id=customer_id,
    )
    _judge_autospa_stage(
        harness,
        state=state,
        stage_name=stage_name,
        stage_goal=stage_goal,
        stage_result=stage_result,
        artifact_paths=paths,
    )
    return stage_result


def _post_owner_autospa_message(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    owner_chat_id: int,
    owner_user_id: int,
    text: str,
    message_id: int,
    document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state.setdefault("owner_transcript", []).append(
        {"role": "owner", "message_id": message_id, "text": text}
    )
    harness.recorder.add("owner_message", message_id=message_id, text=text, has_document=bool(document))
    start_index = len(harness.telegram_client.sent_messages)
    if document:
        body = _telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            caption=text,
            file_id=str(document["file_id"]),
            file_name=str(document.get("file_name") or document.get("filename") or "autospa_price.xlsx"),
            mime_type=str(document["mime_type"]),
            file_size=int(document["file_size"]),
            message_id=message_id,
        )
    else:
        body = _telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            text=text,
            message_id=message_id,
        )
    status = harness.post_telegram(body=body)
    if status != 200:
        raise RuntimeError(f"owner Telegram webhook returned {status}")
    if not _wait_until(
        lambda: len(_messages_for_chat(harness, chat_id=owner_chat_id, start_index=start_index)) >= 1,
        timeout_seconds=110.0,
    ):
        raise RuntimeError("owner stage produced no assistant reply")
    replies = _messages_for_chat(harness, chat_id=owner_chat_id, start_index=start_index)
    for item in replies:
        payload = {
            "role": "assistant",
            "message_id": item.get("message_id"),
            "text": str(item.get("text", "") or ""),
        }
        state.setdefault("owner_transcript", []).append(payload)
        harness.recorder.add("owner_assistant_reply", **payload)
    return replies


def _send_autospa_lead_message(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    customer_id: str,
    workflow_id: str,
    business_connection_id: str,
    lead_label: str,
    lead_chat_id: int,
    lead_user_id: int,
    text: str,
    message_id: int,
    require_assistant_reply: bool = True,
    require_completion_reply: bool = True,
) -> dict[str, Any]:
    transcript = state.setdefault("lead_transcripts", {}).setdefault(lead_label, [])
    transcript.append({"role": "lead", "message_id": message_id, "text": text})
    harness.recorder.add(
        "lead_message",
        lead_label=lead_label,
        lead_chat_id=lead_chat_id,
        message_id=message_id,
        text=text,
    )
    start_index = len(harness.telegram_client.sent_messages)
    previous_bookings = harness.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow_id,
        conversation_id=str(lead_chat_id),
    )
    status = harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=lead_user_id,
            text=text,
            message_id=message_id,
        )
    )
    if status != 200:
        raise RuntimeError(f"lead Telegram webhook returned {status}")

    def _has_progress() -> bool:
        new_messages = [
            item
            for item in harness.telegram_client.sent_messages[start_index:]
            if int(item.get("chat_id", 0)) == lead_chat_id
        ]
        bookings = harness.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow_id,
            conversation_id=str(lead_chat_id),
        )
        return bool(new_messages) or bookings != previous_bookings

    if not _wait_until(_has_progress, timeout_seconds=110.0):
        raise RuntimeError(f"lead stage produced no observable progress for {lead_label}")

    assistant_messages = [
        item
        for item in harness.telegram_client.sent_messages[start_index:]
        if int(item.get("chat_id", 0)) == lead_chat_id
    ]
    completed_before = {
        str(item.get("booking_id", "") or "")
        for item in previous_bookings
        if str(item.get("status", "") or "").strip().lower() == "completed"
    }
    for item in assistant_messages:
        payload = {
            "role": "assistant",
            "message_id": item.get("message_id"),
            "text": str(item.get("text", "") or ""),
        }
        transcript.append(payload)
        harness.recorder.add(
            "lead_assistant_reply",
            lead_label=lead_label,
            lead_chat_id=lead_chat_id,
            **payload,
        )
    bookings = harness.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow_id,
        conversation_id=str(lead_chat_id),
    )
    newly_completed = [
        item
        for item in bookings
        if str(item.get("status", "") or "").strip().lower() == "completed"
        and str(item.get("booking_id", "") or "") not in completed_before
    ]
    if require_assistant_reply and not assistant_messages:
        raise RuntimeError(f"lead turn produced no assistant reply for {lead_label}")
    if require_completion_reply and newly_completed and not assistant_messages:
        raise RuntimeError(
            f"booking completed without customer-facing assistant reply for {lead_label}"
        )
    harness.recorder.add(
        "booking_state",
        lead_label=lead_label,
        lead_chat_id=lead_chat_id,
        bookings=bookings,
    )
    write_event = (
        "real_google_sheets_write"
        if _live_google_sheets_target(harness) is not None
        else "fake_google_sheets_write"
    )
    for write in getattr(harness.composio_service, "sheet_writes", []):
        harness.recorder.add(write_event, **write)
    return {"assistant_messages": assistant_messages, "bookings": bookings}


@pytest.mark.real_composio
def test_live_autospa_xlsx_russian_telegram_intake_with_stage_judging(
    e2e_harness: E2EHarness,
) -> None:
    if not _AUTOSPA_PRICE_ASSET.exists():
        raise RuntimeError(f"AutoSpa E2E asset is missing: {_AUTOSPA_PRICE_ASSET}")

    owner_user_id, owner_chat_id, customer_id = _owner_identity_for_autospa(e2e_harness)
    live_google_sheets_target = _live_google_sheets_target(e2e_harness)
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_autospa_stage_judged",
    )
    artifact_dir = e2e_harness.status_report_path.parent / "autospa_stage_judged_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "owner_transcript": [],
        "lead_transcripts": {},
        "stage_judgements": [],
        "workflow": {},
        "live_google_sheets_target": live_google_sheets_target,
    }

    file_id = "tg_file_autospa_price"
    registered = e2e_harness.telegram_client.register_file(
        file_id=file_id,
        path=_AUTOSPA_PRICE_ASSET,
        filename="autospa_price.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    e2e_harness.recorder.add("owner_document_uploaded", **registered)

    def stage_owner_upload() -> dict[str, Any]:
        fresh_replies = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text="/fresh",
            message_id=1,
        )
        upload_text = (
            "Хочу создать workflow для Telegram Business входящих сообщений. "
            "Вот прайс AutoSpa. Агент должен использовать файл как источник знаний, "
            "но работать только с категориями Мойка и Шиномонтаж. "
            "Нужно отвечать клиентам в Telegram, помогать выбрать услугу, отвечать на вопросы "
            "по цене из файла и записывать бронирования в Google Sheets. "
            "Сначала подготовь workflow и спроси подтверждение перед активацией."
        )
        upload_replies = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text=upload_text,
            message_id=2,
            document=registered,
        )
        return {
            "fresh_replies": len(fresh_replies),
            "upload_replies": len(upload_replies),
            "registered_file": registered,
            "downloaded_files": e2e_harness.telegram_client.downloaded_files,
        }

    def stage_owner_details() -> dict[str, Any]:
        spreadsheet_id = "sheet_autospa_e2e"
        sheet_name = "Bookings"
        if live_google_sheets_target is not None:
            spreadsheet_id = str(live_google_sheets_target.spreadsheet_id)
            sheet_name = str(live_google_sheets_target.sheet_name)
        text = (
            "Дополняю настройки. Workflow назови «AutoSpa Мойка и Шиномонтаж». "
            "Канал: Telegram Business DM. Подключение уже есть, используй его. "
            "Интент входящих: клиент хочет узнать цену, уточнить услугу или записаться "
            "на мойку или шиномонтаж. Вне этих двух категорий не продавай, лучше уточни, "
            "что workflow покрывает только Мойку и Шиномонтаж. "
            "Для бронирования собери: категория услуги, название услуги, автомобиль или класс авто, "
            "дата, время, имя клиента, телефон, цена если найдена. "
            "Прайс может быть большой, поэтому открой структуру файла, выбери только связанные "
            "разделы для Мойка и Шиномонтаж, подготовь Markdown knowledge pack и прикрепи его к workflow. "
            "Запись сохраняй в тестовую Google Sheets таблицу: "
            f"spreadsheetId={spreadsheet_id}, sheetName={sheet_name}. "
            "Используй sink_type google_sheets_composio, toolkit googlesheets, "
            "field_mapping на понятные колонки: Category, Service, Vehicle, Date, Time, Lead Name, "
            "Phone, Quoted Price, Conversation ID. Подготовь предложение и жди моего подтверждения."
        )
        replies = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text=text,
            message_id=3,
        )
        return {"owner_replies": len(replies)}

    def stage_owner_confirm() -> dict[str, Any]:
        replies = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text=(
                "Подтверждаю. Сохрани и активируй этот workflow сейчас. "
                "Потом используй его для входящих Telegram лидов."
            ),
            message_id=4,
        )
        if not _wait_until(
            lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) >= 1,
            timeout_seconds=110.0,
        ):
            raise RuntimeError("workflow was not created after owner confirmation")
        workflows = _list_workflows(e2e_harness, customer_id=customer_id)
        state["workflow"] = workflows[-1]
        return {"owner_replies": len(replies), "workflow": state["workflow"], "workflow_count": len(workflows)}

    def stage_wash_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run wash lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="wash",
            lead_chat_id=2901,
            lead_user_id=3901,
            message_id=10,
            text=(
                "Здравствуйте. Подскажите, сколько стоит 2х-фазная мойка для Toyota RAV4? "
                "Если цена нормальная, хотел бы записаться на завтра."
            ),
        )
        second = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="wash",
            lead_chat_id=2901,
            lead_user_id=3901,
            message_id=11,
            text="Меня зовут Алексей, телефон +79990000001. Завтра в 10:00 удобно.",
        )
        return {"turns": [first, second]}

    def stage_tire_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run tire lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="tire",
            lead_chat_id=2902,
            lead_user_id=3902,
            message_id=20,
            text=(
                "Добрый день. Нужно переобуть BMW X5, 19 радиус, низкий профиль. "
                "Можно записаться на пятницу в 15:00? Я Мария, +79990000002."
            ),
        )
        return {"turns": [first]}

    def stage_out_of_scope_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run out-of-scope lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="out_of_scope",
            lead_chat_id=2903,
            lead_user_id=3903,
            message_id=30,
            text="Здравствуйте. Сколько стоит оклейка PPF передней части машины?",
        )
        return {"turns": [first]}

    def stage_missing_phone_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run missing-phone lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="missing_phone",
            lead_chat_id=2904,
            lead_user_id=3904,
            message_id=40,
            text="Хочу записаться на 2х-фазную мойку Toyota Camry завтра в 12:00. Меня зовут Игорь.",
        )
        return {"turns": [first]}

    def stage_ambiguous_car_class_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run ambiguous-car-class lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="ambiguous_car_class",
            lead_chat_id=2905,
            lead_user_id=3905,
            message_id=50,
            text="Сколько будет стоить 2х-фазная мойка для обычной машины? Модель пока не помню.",
        )
        return {"turns": [first]}

    def stage_unavailable_price_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run unavailable-price lead stage without workflow_id")
        first = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="unavailable_price",
            lead_chat_id=2906,
            lead_user_id=3906,
            message_id=60,
            text="У вас есть цена на мойку мотоцикла или квадроцикла? Хочу понять бюджет.",
        )
        return {"turns": [first]}

    def stage_update_cancel_lead() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not workflow_id:
            raise RuntimeError("cannot run update/cancel lead stage without workflow_id")
        update = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="update_cancel",
            lead_chat_id=2901,
            lead_user_id=3901,
            message_id=70,
            text="А можно мою запись на мойку перенести с 10:00 на 11:00?",
        )
        cancel = _send_autospa_lead_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            workflow_id=workflow_id,
            business_connection_id=business_connection_id,
            lead_label="update_cancel",
            lead_chat_id=2901,
            lead_user_id=3901,
            message_id=71,
            text="Тогда отмените запись, пожалуйста.",
        )
        return {"turns": [update, cancel]}

    def stage_final_review() -> dict[str, Any]:
        workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        bookings = []
        if workflow_id:
            bookings = e2e_harness.list_bookings(customer_id=customer_id, workflow_id=workflow_id)
        return {
            "workflow": workflow,
            "bookings": bookings,
            "sheet_writes": getattr(e2e_harness.composio_service, "sheet_writes", []),
            "telegram_sent_messages": len(e2e_harness.telegram_client.sent_messages),
            "downloaded_files": e2e_harness.telegram_client.downloaded_files,
        }

    try:
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="owner_upload",
            stage_goal="Владелец начинает свежий чат, загружает XLSX и описывает workflow.",
            run=stage_owner_upload,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="owner_details",
            stage_goal="Владелец задает детали workflow, Google Sheets sink и scoped knowledge.",
            run=stage_owner_details,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="owner_confirm",
            stage_goal="Владелец подтверждает предложение, workflow сохраняется и активируется.",
            run=stage_owner_confirm,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="wash_lead",
            stage_goal="Реалистичный русский лид спрашивает цену на мойку и пытается записаться.",
            run=stage_wash_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="tire_lead",
            stage_goal="Реалистичный русский лид пытается записаться на шиномонтаж.",
            run=stage_tire_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="out_of_scope_lead",
            stage_goal="Лид спрашивает услугу вне scoped workflow, агент должен ответить клиенту.",
            run=stage_out_of_scope_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="missing_phone_lead",
            stage_goal="Лид хочет записаться, но не дает телефон; агент должен запросить недостающее.",
            run=stage_missing_phone_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="ambiguous_car_class_lead",
            stage_goal="Лид задает ценовой вопрос без модели или класса авто.",
            run=stage_ambiguous_car_class_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="unavailable_price_lead",
            stage_goal="Лид спрашивает цену в близкой категории, которой может не быть в прайсе.",
            run=stage_unavailable_price_lead,
        )
        _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="update_cancel_lead",
            stage_goal="Завершенный клиент пробует перенести и отменить запись.",
            run=stage_update_cancel_lead,
        )
        final_stage = _run_autospa_stage(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            stage_name="final_review",
            stage_goal="Собрать полный итоговый снимок workflow, диалогов, бронирований и Google Sheets.",
            run=stage_final_review,
        )
    except Exception as exc:
        _write_autospa_failure_debug(
            e2e_harness,
            state=state,
            artifact_dir=artifact_dir,
            customer_id=customer_id,
            error=exc,
        )
        raise

    artifact_paths = _current_autospa_artifacts(
        e2e_harness,
        state=state,
        artifact_dir=artifact_dir,
        customer_id=customer_id,
    )
    report = e2e_harness.write_status_report(
        scenario="live_autospa_xlsx_russian_telegram_intake_with_stage_judging",
        ok=True,
        details={
            "customer_id": customer_id,
            "business_connection_id": business_connection_id,
            "live_google_sheets_target": live_google_sheets_target,
            "artifact_paths": artifact_paths,
            "final_stage": final_stage,
            "stage_judgements": state.get("stage_judgements") or [],
        },
    )
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    evaluation = report_payload.get("evaluation") if isinstance(report_payload, dict) else {}
    if not isinstance(evaluation, dict) or not bool(evaluation.get("ok", False)):
        raise RuntimeError(f"final status report judge failed: {evaluation}")


def test_live_owner_telegram_chat_can_create_telegram_intake_workflow_and_activate_it(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=1)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    initial_owner_message_count = len(e2e_harness.telegram_client.sent_messages)
    create_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=2,
            text=(
                "Create a Telegram Business DM intake workflow for my car wash. "
                "Use the workflow name 'E2E Telegram Car Wash'. "
                "Collect exactly these fields: car_model, car_type, wash_type, date, time. "
                "Goal: answer direct questions first, then collect only missing booking details. "
                "Save results to local CSV tulpa_stuff/e2e_telegram_carwash.csv. "
                "Start the workflow setup wizard, prepare the exact configuration, and wait for my confirmation before saving."
            ),
        )
    )
    assert create_status == 200
    assert _wait_until(lambda: len(e2e_harness.telegram_client.sent_messages) > initial_owner_message_count)
    assert _list_workflows(e2e_harness, customer_id=customer_id) == []

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=3,
            text="Yes, save that workflow now exactly as proposed.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Telegram Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}

    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=initial_owner_message_count,
    )
    assert latest_owner_message is not None
    latest_text = str(latest_owner_message.get("text", "")).lower()
    assert "backend error" not in latest_text
    assert "workflow" in latest_text

    report = e2e_harness.write_status_report(
        scenario="live_owner_telegram_chat_can_create_telegram_intake_workflow_and_activate_it",
        ok=True,
        details={
            "customer_id": customer_id,
            "workflow_id": workflow["workflow_id"],
            "owner_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()


def test_live_owner_telegram_chat_can_delete_existing_telegram_intake_workflow(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )
    create = e2e_harness.client.post(
        "/internal/intake/workflows/upsert",
        json={
            "customer_id": customer_id,
            "name": "Delete Me Telegram Intake",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": business_connection_id},
            "intent_description": "Handle Telegram booking requests.",
            "required_fields": ["name", "time"],
            "assistant_instructions": "Be concise.",
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/e2e_delete_me.csv"},
            "enabled": True,
        },
    )
    assert create.status_code == 200, create.text
    assert len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1

    start_message_count = len(e2e_harness.telegram_client.sent_messages)
    delete_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=10,
            text=(
                "Delete the active Telegram Business intake workflow now. "
                "Do not just explain; perform the deletion and confirm when it is gone."
            ),
        )
    )
    assert delete_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 0, timeout_seconds=60.0)

    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_message_count,
    )
    assert latest_owner_message is not None
    latest_text = str(latest_owner_message.get("text", "")).lower()
    assert "backend error" not in latest_text
    assert "deleted" in latest_text or "removed" in latest_text or "gone" in latest_text

    report = e2e_harness.write_status_report(
        scenario="live_owner_telegram_chat_can_delete_existing_telegram_intake_workflow",
        ok=True,
        details={
            "customer_id": customer_id,
            "owner_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()


def test_live_telegram_business_lead_message_triggers_active_workflow_reply(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )
    create = e2e_harness.client.post(
        "/internal/intake/workflows/upsert",
        json={
            "customer_id": customer_id,
            "name": "Lead Reply Telegram Intake",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": business_connection_id},
            "intent_description": "Reply to Telegram Business leads and collect booking details.",
            "required_fields": ["car_model", "car_type", "wash_type", "date", "time"],
            "assistant_instructions": (
                "Reply directly to the lead, answer what you can, ask only for missing booking fields, "
                "and keep replies concise."
            ),
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/e2e_lead_replies.csv"},
            "enabled": True,
        },
    )
    assert create.status_code == 200, create.text
    workflow = create.json()["workflow"]
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""

    start_message_count = len(e2e_harness.telegram_client.sent_messages)
    lead_chat_id = 555
    webhook_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=999,
            message_id=100,
            text="Hi, I want to book a wash tomorrow at 10am for my BMW sedan.",
        )
    )
    assert webhook_status == 200

    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == lead_chat_id
            and str(item.get("business_connection_id", "")).strip() == business_connection_id
            and str(item.get("text", "")).strip()
            for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        ),
        timeout_seconds=60.0,
    )

    lead_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=lead_chat_id,
        start_index=start_message_count,
    )
    assert lead_reply is not None
    assert str(lead_reply.get("business_connection_id", "")).strip() == business_connection_id
    assert str(lead_reply.get("text", "")).strip()

    owner_errors = [
        item
        for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        if int(item.get("chat_id", 0)) == owner_chat_id
    ]
    assert owner_errors == []

    report = e2e_harness.write_status_report(
        scenario="live_telegram_business_lead_message_triggers_active_workflow_reply",
        ok=True,
        details={
            "customer_id": customer_id,
            "workflow_id": workflow["workflow_id"],
            "lead_chat_id": lead_chat_id,
            "lead_reply_text": str(lead_reply.get("text", ""))[:500],
        },
    )
    assert report.exists()


def test_live_owner_chat_can_create_quality_workflow_over_multiple_turns_and_handle_aligned_lead(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_quality",
    )

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=50)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )
    owner_thread_id = _telegram_owner_thread_id(chat_id=owner_chat_id)
    assert owner_thread_id

    start_index = len(e2e_harness.telegram_client.sent_messages)
    first_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=51,
            text=(
                "I want to set up a Telegram Business DM intake workflow for my car wash. "
                "Please start the workflow setup wizard and help me shape it step by step."
            ),
        )
    )
    assert first_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=start_index)) >= 1)
    first_wizard_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_index,
    )
    assert first_wizard_reply is not None
    first_wizard_text = str(first_wizard_reply.get("text", "")).lower()
    assert "workflow" in first_wizard_text or "setup" in first_wizard_text

    second_turn_start = len(e2e_harness.telegram_client.sent_messages)
    second_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=52,
            text=(
                "Use the workflow name 'E2E Quality Car Wash'. "
                "Collect exactly: car_model, car_type, wash_type, date, time. "
                "If a lead asks for price, answer directly before asking anything else. "
                "As soon as wash_type and car_type are known, give the exact price immediately. "
                "Use these prices: small car full wash 1000 rubles, SUV full wash 2500 rubles. "
                "Do not repeat already known details. "
                "Only offer exact times like 09:00, 10:00, 11:00, not vague parts of day. "
                "Save bookings to local CSV tulpa_stuff/e2e_quality_carwash.csv."
            ),
        )
    )
    assert second_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=second_turn_start)) >= 1)
    assert _wait_until(
        lambda: "workflow_setup" in _turn_modes_seen(e2e_harness, customer_id=customer_id),
        timeout_seconds=15.0,
    )
    proposal_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=second_turn_start,
    )
    assert proposal_message is not None

    if not _wait_until(
        lambda: _workflow_setup_has_proposal(
            e2e_harness,
            customer_id=customer_id,
            thread_id=owner_thread_id,
        ),
        timeout_seconds=10.0,
    ):
        clarification_start = len(e2e_harness.telegram_client.sent_messages)
        clarification_status = e2e_harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                message_id=53,
                text=(
                    "Intent: answer Telegram leads who ask about full car wash pricing and booking, "
                    "then collect enough details to book them. Full wash only for this test. "
                    "Treat the listed small car and SUV full-wash prices as complete. "
                    "Telegram Business DM has no polling or scan schedule; it runs on inbound messages. "
                    "Use the connected Telegram Business account. Please propose the workflow now and wait for my confirmation."
                ),
            )
        )
        assert clarification_status == 200
        assert _wait_until(
            lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=clarification_start)) >= 1,
            timeout_seconds=90.0,
        )
        proposal_message = _latest_message_for_chat(
            e2e_harness,
            chat_id=owner_chat_id,
            start_index=clarification_start,
        )
        assert proposal_message is not None

    assert _wait_until(
        lambda: _workflow_setup_has_proposal(
            e2e_harness,
            customer_id=customer_id,
            thread_id=owner_thread_id,
        ),
        timeout_seconds=60.0,
    )
    proposal_text = str(proposal_message.get("text", "")).lower()
    assert "confirm" in proposal_text or "save" in proposal_text or "workflow" in proposal_text

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=54,
            text="Looks good. Save and activate that workflow now.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Quality Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}
    instructions = str(workflow.get("assistant_instructions", "")).strip()
    assert len(instructions) >= 120
    lowered_instructions = instructions.lower()
    assert "price" in lowered_instructions
    assert "exact" in lowered_instructions
    assert "time" in lowered_instructions
    assert "repeat" in lowered_instructions or "already known" in lowered_instructions

    lead_start_index = len(e2e_harness.telegram_client.sent_messages)
    lead_chat_id = 556
    lead_text = (
        "How much is a full wash for a small car? "
        "If 10:00 tomorrow works, book it for my BMW 3 Series."
    )
    lead_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=1001,
            message_id=150,
            text=lead_text,
        )
    )
    assert lead_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == lead_chat_id
            and str(item.get("business_connection_id", "")).strip() == business_connection_id
            and str(item.get("text", "")).strip()
            for item in e2e_harness.telegram_client.sent_messages[lead_start_index:]
        ),
        timeout_seconds=60.0,
    )

    lead_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=lead_chat_id,
        start_index=lead_start_index,
    )
    assert lead_reply is not None
    lead_reply_text = str(lead_reply.get("text", "")).strip()
    assert lead_reply_text

    intake_service = e2e_harness.client.app.state.intake_workflows
    bookings = intake_service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )

    owner_messages = _messages_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_index,
    )
    owner_transcript = [
        {"chat_id": int(item.get("chat_id", 0)), "text": str(item.get("text", ""))[:800]}
        for item in owner_messages
    ]
    workflow_snapshot = {
        "workflow_id": workflow["workflow_id"],
        "name": workflow["name"],
        "channel": workflow["channel"],
        "provider": workflow["provider"],
        "required_fields": workflow["required_fields"],
        "assistant_instructions": instructions[:2500],
        "sink_type": workflow["sink_type"],
        "sink_config": workflow["sink_config"],
    }
    booking_snapshot = bookings[0] if bookings else {}

    report = e2e_harness.write_status_report(
        scenario="live_owner_chat_can_create_quality_workflow_over_multiple_turns_and_handle_aligned_lead",
        ok=True,
        details={
            "customer_id": customer_id,
            "owner_transcript": owner_transcript,
            "turn_modes_seen": _turn_modes_seen(e2e_harness, customer_id=customer_id),
            "workflow": workflow_snapshot,
            "lead_message": lead_text,
            "lead_reply_text": lead_reply_text[:1200],
            "bookings_count": len(bookings),
            "first_booking": booking_snapshot,
        },
    )
    assert report.exists()
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    verdict = _judge_verdict(report_payload)
    assert verdict != "fail"


def test_live_owner_chat_can_create_multiturn_telegram_booking_workflow_and_persist_booking(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 321
    owner_chat_id = 888
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_multiturn",
    )

    csv_relative_path = "tulpa_stuff/e2e_multiturn_carwash.csv"

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=70)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    owner_start_index = len(e2e_harness.telegram_client.sent_messages)
    first_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=71,
            text=(
                "I want a Telegram Business DM intake workflow for my car wash. "
                "Start the workflow setup wizard and help me configure it."
            ),
        )
    )
    assert first_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=owner_start_index)) >= 1)

    second_turn_start = len(e2e_harness.telegram_client.sent_messages)
    second_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=72,
            text=(
                "Use the workflow name 'E2E Multiturn Car Wash'. "
                "Collect exactly these fields: car_model, car_type, wash_type, date, time. "
                "When a lead shows booking intent, answer direct questions briefly and then ask only for the next missing field. "
                "Do not save anything until all required fields are known. "
                "Do not repeat details the lead already gave you. "
                "Save completed bookings to local CSV tulpa_stuff/e2e_multiturn_carwash.csv. "
                "Prepare the exact configuration and wait for my confirmation before saving."
            ),
        )
    )
    assert second_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=second_turn_start)) >= 1)

    proposal_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=second_turn_start,
    )
    assert proposal_message is not None
    proposal_text = str(proposal_message.get("text", "")).lower()
    assert "workflow" in proposal_text
    assert "confirm" in proposal_text or "save" in proposal_text

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=73,
            text="Looks correct. Save and activate this workflow now.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Multiturn Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}
    assert workflow["sink_type"] == "local_csv"
    assert workflow["sink_config"] == {"file_path": csv_relative_path}
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}

    lead_chat_id = 654
    first_lead_message_start = len(e2e_harness.telegram_client.sent_messages)
    first_lead_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=2001,
            message_id=201,
            text="Hi, I want to book a wash tomorrow. How much is it for an SUV?",
        )
    )
    assert first_lead_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == lead_chat_id
            and str(item.get("business_connection_id", "")).strip() == business_connection_id
            and str(item.get("text", "")).strip()
            for item in e2e_harness.telegram_client.sent_messages[first_lead_message_start:]
        ),
        timeout_seconds=60.0,
    )

    first_lead_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=lead_chat_id,
        start_index=first_lead_message_start,
    )
    assert first_lead_reply is not None
    first_lead_reply_text = str(first_lead_reply.get("text", "")).strip()
    assert first_lead_reply_text
    assert "backend error" not in first_lead_reply_text.lower()

    bookings_after_first_turn = e2e_harness.client.app.state.intake_workflows.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )
    assert not any(str(item.get("status", "")).lower() == "completed" for item in bookings_after_first_turn)

    second_lead_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=2001,
            message_id=202,
            text=(
                "Car model: Toyota RAV4. "
                "Car type: SUV. "
                "Wash type: full wash. "
                "Date: tomorrow. "
                "Time: 10:00."
            ),
        )
    )
    assert second_lead_status == 200
    assert _wait_until(
        lambda: any(
            str(item.get("status", "")).lower() == "completed"
            for item in e2e_harness.client.app.state.intake_workflows.list_bookings(
                customer_id=customer_id,
                workflow_id=workflow["workflow_id"],
                conversation_id=str(lead_chat_id),
            )
        ),
        timeout_seconds=90.0,
    )

    bookings = e2e_harness.client.app.state.intake_workflows.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id=str(lead_chat_id),
    )
    assert len(bookings) == 1
    booking = bookings[0]
    assert booking["status"] == "completed"
    assert booking["sink_write_status"] == "succeeded"
    extracted = booking["extracted_fields"]
    assert "toyota" in str(extracted.get("car_model", "")).lower()
    assert "rav4" in str(extracted.get("car_model", "")).lower()
    assert "suv" in str(extracted.get("car_type", "")).lower()
    assert "full" in str(extracted.get("wash_type", "")).lower()
    assert str(extracted.get("date", "")).strip()
    assert "10" in str(extracted.get("time", "")).lower()

    csv_rows = _csv_rows_for_relative_path(
        e2e_harness,
        relative_path=csv_relative_path,
    )
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["booking_id"] == booking["booking_id"]
    assert row["workflow_id"] == workflow["workflow_id"]
    assert row["workflow_name"] == workflow["name"]
    assert row["conversation_id"] == str(lead_chat_id)
    assert "toyota" in row["car_model"].lower()
    assert "rav4" in row["car_model"].lower()
    assert "suv" in row["car_type"].lower()
    assert "full" in row["wash_type"].lower()
    assert row["date"].strip()
    assert "10" in row["time"].lower()

    owner_messages = _messages_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=owner_start_index,
    )
    owner_transcript = [
        {"chat_id": int(item.get("chat_id", 0)), "text": str(item.get("text", ""))[:800]}
        for item in owner_messages
    ]
    lead_outbound_messages = [
        {
            "chat_id": int(item.get("chat_id", 0)),
            "text": str(item.get("text", ""))[:800],
            "reply_to_message_id": item.get("reply_to_message_id"),
        }
        for item in e2e_harness.telegram_client.sent_messages
        if int(item.get("chat_id", 0)) == lead_chat_id
    ]
    lead_source_messages = _lead_source_messages(
        e2e_harness,
        customer_id=customer_id,
        business_connection_id=business_connection_id,
        lead_chat_id=lead_chat_id,
    )

    report = e2e_harness.write_status_report(
        scenario="live_owner_chat_can_create_multiturn_telegram_booking_workflow_and_persist_booking",
        ok=True,
        details={
            "customer_id": customer_id,
            "workflow": {
                "workflow_id": workflow["workflow_id"],
                "name": workflow["name"],
                "required_fields": workflow["required_fields"],
                "sink_type": workflow["sink_type"],
                "sink_config": workflow["sink_config"],
                "assistant_instructions": str(workflow.get("assistant_instructions", ""))[:2500],
            },
            "owner_transcript": owner_transcript,
            "lead_source_messages": lead_source_messages,
            "lead_outbound_messages": lead_outbound_messages,
            "booking": booking,
            "csv_rows": csv_rows,
        },
    )
    assert report.exists()
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    verdict = _judge_verdict(report_payload)
    assert verdict != "fail"


def test_live_lead_simulator_can_complete_telegram_car_wash_booking(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 456
    owner_chat_id = 889
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_simulated_lead",
    )

    csv_relative_path = "tulpa_stuff/e2e_simulated_lead_carwash.csv"

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=80)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    owner_start_index = len(e2e_harness.telegram_client.sent_messages)
    first_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=81,
            text=(
                "Create a Telegram Business DM intake workflow for my car wash. "
                "Start the setup wizard and help me configure it."
            ),
        )
    )
    assert first_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=owner_start_index)) >= 1)

    second_turn_start = len(e2e_harness.telegram_client.sent_messages)
    second_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=82,
            text=(
                "Use the workflow name 'E2E Simulated Lead Car Wash'. "
                "Collect exactly these fields: car_model, car_type, wash_type, date, time. "
                "If a lead asks for price, answer directly first and then ask only for the next missing booking detail. "
                "Do not repeat already known details. "
                "Do not save until all required fields are known. "
                f"Save completed bookings to local CSV {csv_relative_path}. "
                "Prepare the exact configuration and wait for my confirmation before saving."
            ),
        )
    )
    assert second_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=second_turn_start)) >= 1)

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=83,
            text="Looks good. Save and activate this workflow now.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Simulated Lead Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["sink_config"] == {"file_path": csv_relative_path}

    profile = LeadProfile(
        objective="Book a full car wash and understand the price before confirming.",
        initial_message="Hi, I want to book a full wash for tomorrow. How much is it for an SUV?",
        known_facts={
            "car_model": "Toyota RAV4",
            "car_type": "SUV",
            "wash_type": "full wash",
            "date": "tomorrow",
            "time": "10:00",
        },
        persona="Friendly, brief, and practical. Acts like a normal Telegram DM lead.",
        rules=[
            "Do not volunteer every booking field in the first message.",
            "If the assistant asks for multiple missing details, answer them together.",
            "Stay consistent with the hidden facts.",
        ],
        max_turns=6,
    )

    lead_chat_id = 655
    simulation = e2e_harness.simulate_telegram_business_lead(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        business_connection_id=business_connection_id,
        lead_chat_id=lead_chat_id,
        lead_user_id=2002,
        profile=profile,
        initial_message_id=300,
        idle_timeout_seconds=90.0,
    )

    assert simulation["ok"] is True, simulation
    assert simulation["reason"] == "booking_completed"
    turn_results = simulation.get("turn_results") or []
    assert len(turn_results) >= 2
    first_turn = turn_results[0]
    first_turn_bookings = first_turn.get("bookings") or []
    assert first_turn_bookings
    first_turn_booking = first_turn_bookings[0]
    assert str(first_turn_booking.get("status", "")).strip().lower() == "active"
    assert str(first_turn_booking.get("sink_write_status", "")).strip().lower() == "pending"
    first_turn_messages = first_turn.get("assistant_messages") or []
    assert first_turn_messages
    first_reply_text = " ".join(
        str(item.get("text", "") or "").strip() for item in first_turn_messages
    ).strip()
    lowered_first_reply = first_reply_text.lower()
    assert _addresses_pricing_question(first_reply_text)
    assert "?" in first_reply_text or "could you" in lowered_first_reply or "what " in lowered_first_reply
    final_turn = turn_results[-1]
    assert bool(final_turn.get("booking_completed", False)) is True
    completed_booking = simulation.get("completed_booking") or {}
    assert completed_booking
    assert str(completed_booking.get("status", "")).strip().lower() == "completed"
    assert str(completed_booking.get("sink_write_status", "")).strip().lower() == "succeeded"
    extracted = completed_booking["extracted_fields"]
    assert "toyota" in str(extracted.get("car_model", "")).lower()
    assert "rav4" in str(extracted.get("car_model", "")).lower()
    assert "suv" in str(extracted.get("car_type", "")).lower()
    assert "full" in str(extracted.get("wash_type", "")).lower()
    assert str(extracted.get("date", "")).strip()
    assert "10" in str(extracted.get("time", "")).lower()

    csv_rows = _csv_rows_for_relative_path(
        e2e_harness,
        relative_path=csv_relative_path,
    )
    assert len(csv_rows) == 1
    row = csv_rows[0]
    assert row["booking_id"] == completed_booking["booking_id"]
    assert row["conversation_id"] == str(lead_chat_id)

    lead_source_messages = _lead_source_messages(
        e2e_harness,
        customer_id=customer_id,
        business_connection_id=business_connection_id,
        lead_chat_id=lead_chat_id,
    )
    assert len(lead_source_messages) >= 3
    assert any(str(item.get("sender_role", "")).strip() == "assistant" for item in lead_source_messages)

    owner_errors = [
        item
        for item in e2e_harness.telegram_client.sent_messages
        if int(item.get("chat_id", 0)) == owner_chat_id
        and "issue" in str(item.get("text", "")).lower()
    ]
    assert owner_errors == []

    owner_messages = _messages_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=owner_start_index,
    )
    owner_transcript = [
        {"chat_id": int(item.get("chat_id", 0)), "text": str(item.get("text", ""))[:800]}
        for item in owner_messages
    ]

    report = e2e_harness.write_status_report(
        scenario="live_lead_simulator_can_complete_telegram_car_wash_booking",
        ok=True,
        details={
            "customer_id": customer_id,
            "lead_simulator_model": e2e_harness.lead_simulator.model,
            "workflow": {
                "workflow_id": workflow["workflow_id"],
                "name": workflow["name"],
                "required_fields": workflow["required_fields"],
                "assistant_instructions": str(workflow.get("assistant_instructions", ""))[:2500],
                "sink_config": workflow["sink_config"],
            },
            "owner_transcript": owner_transcript,
            "simulation": simulation,
            "lead_source_messages": lead_source_messages,
            "csv_rows": csv_rows,
        },
    )
    assert report.exists()
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    verdict = _judge_verdict(report_payload)
    assert verdict != "fail"
