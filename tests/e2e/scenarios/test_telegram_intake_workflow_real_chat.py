from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from evaluation.judge import DEFAULT_JUDGE_MODEL, evaluate_e2e_scenario_with_llm_judge
from harness.lead_simulator import DEFAULT_LEAD_SIMULATOR_MODEL, LeadProfile
from harness.owner_simulator import OwnerProfile
from harness.runner import E2EHarness, effective_live_llm_timeout_seconds

from tests.workbook_fixtures import (
    SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
    SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
    write_sample_vehicle_services_xlsx,
)

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]

DEFAULT_ASSERT_JUDGE_MODEL = "google/gemini-3.1-flash-lite-preview"


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


def _looks_like_owner_proposal_message(item: dict[str, Any]) -> bool:
    text = str(item.get("text", "") or "").strip().lower()
    if not text:
        return False
    has_confirmation_request = any(
        marker in text
        for marker in (
            "confirm",
            "save",
            "activate",
            "commit",
            "looks good",
            "подтверж",
            "сохран",
            "активир",
        )
    )
    has_proposal_content = (
        "proposal" in text
        or "workflow" in text
        or "configuration" in text
        or "channel" in text
        or "fields" in text
        or "предлож" in text
        or "воркфлоу" in text
        or "канал" in text
        or "поля" in text
    )
    return has_confirmation_request and has_proposal_content


def _workflow_setup_owner_proposal_message(
    harness: E2EHarness,
    *,
    customer_id: str,
    thread_id: str,
    owner_chat_id: int,
    start_index: int,
) -> dict[str, Any] | None:
    if not _workflow_setup_has_proposal(harness, customer_id=customer_id, thread_id=thread_id):
        return None
    replies = _messages_for_chat(harness, chat_id=owner_chat_id, start_index=start_index)
    for item in reversed(replies):
        if _looks_like_owner_proposal_message(item):
            return item
    return None


def _workflow_setup_proposal_and_owner_reply_seen(
    harness: E2EHarness,
    *,
    customer_id: str,
    thread_id: str,
    owner_chat_id: int,
    start_index: int,
) -> bool:
    return _workflow_setup_owner_proposal_message(
        harness,
        customer_id=customer_id,
        thread_id=thread_id,
        owner_chat_id=owner_chat_id,
        start_index=start_index,
    ) is not None


def test_workflow_setup_owner_proposal_detection_accepts_russian_confirmation() -> None:
    assert _looks_like_owner_proposal_message(
        {
            "text": (
                "Preflight прошёл успешно. Вот предложение:\n\n"
                "**Workflow «AutoSpa Мойка и Шиномонтаж»**\n"
                "**Канал** | Telegram Business DM\n\n"
                "Подтверждаешь? Если нужны правки — скажи, что изменить."
            )
        }
    )


def _assert_llm_semantic_match(
    harness: E2EHarness,
    *,
    scenario: str,
    expectation: str,
    actual: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = (
        os.getenv("OPENTULPA_E2E_ASSERT_JUDGE_MODEL", "").strip()
        or os.getenv("OPENTULPA_E2E_JUDGE_MODEL", "").strip()
        or DEFAULT_ASSERT_JUDGE_MODEL
    )
    result = evaluate_e2e_scenario_with_llm_judge(
        scenario=f"semantic_assert:{scenario}",
        details={
            "assertion_type": "semantic_match",
            "expectation": expectation,
            "actual": actual,
            "context": context or {},
            "judge_instruction": (
                "Decide whether actual system output satisfies expectation. "
                "Accept wording/language variations. Fail only for material mismatch, missing required behavior, "
                "or explicit backend/product error. Return pass only when evidence is enough."
            ),
        },
        system_log_path=harness.system_log_path,
        behavior_log_path=harness.behavior_log_path,
        llm_trace_path=harness.llm_trace_path,
        model=model,
        timeout_seconds=30.0,
    )
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    verdict = str(parsed.get("verdict", "") or "").strip().lower()
    if not bool(result.get("ok", False)) or verdict != "pass":
        raise AssertionError(
            "LLM semantic assertion failed: "
            + json.dumps(
                {
                    "scenario": scenario,
                    "expectation": expectation,
                    "verdict": verdict,
                    "result": result,
                },
                ensure_ascii=False,
                default=str,
            )[:4000]
        )
    harness.recorder.add(
        "semantic_assertion_passed",
        scenario=scenario,
        model=model,
        expectation=expectation,
        summary=str(parsed.get("summary", "") or "")[:1000],
    )
    return result


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


def _owner_progress_internal_calls(
    harness: E2EHarness,
    *,
    start_index: int,
    customer_id: str,
) -> list[dict[str, Any]]:
    relevant_prefixes = (
        "/internal/intake/setup/",
        "/internal/files/",
        "/internal/knowledge/",
        "/internal/telegram/business/",
        "/internal/composio/",
    )
    matches: list[dict[str, Any]] = []
    for item in harness.internal_api_calls_since(start_index):
        path = str(item.get("path", "") or "")
        if not path.startswith(relevant_prefixes):
            continue
        json_body = item.get("json_body") if isinstance(item.get("json_body"), dict) else {}
        request_customer_id = str(json_body.get("customer_id", "") or "").strip()
        if request_customer_id and request_customer_id != customer_id:
            continue
        matches.append(item)
    return matches


def _owner_progress_snapshot(
    harness: E2EHarness,
    *,
    customer_id: str,
    owner_chat_id: int,
    internal_call_start: int,
    workflow_count_before: int,
) -> dict[str, Any]:
    thread_id = _telegram_owner_thread_id(chat_id=owner_chat_id)
    setup_session = (
        _workflow_setup_session(harness, customer_id=customer_id, thread_id=thread_id)
        if thread_id
        else {}
    )
    workflow_count = len(_list_workflows(harness, customer_id=customer_id))
    internal_calls = _owner_progress_internal_calls(
        harness,
        start_index=internal_call_start,
        customer_id=customer_id,
    )
    progress_kinds: list[str] = []
    if internal_calls:
        progress_kinds.append("internal_calls")
    if workflow_count > workflow_count_before:
        progress_kinds.append("workflow_created")
    return {
        "thread_id": thread_id,
        "setup_session": setup_session,
        "workflow_count": workflow_count,
        "internal_calls": internal_calls,
        "progress_kinds": progress_kinds,
        "progress_seen": bool(progress_kinds),
    }


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


def _booking_category(row: dict[str, str]) -> str:
    return str(row.get("service_category") or row.get("category") or "").strip().lower()


def _service_text_matches_two_phase_wash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    compact = text.replace(" ", "").replace("-", "")
    return "2" in compact and "фаз" in compact


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


def _car_wash_owner_profile(
    *,
    workflow_name: str,
    csv_relative_path: str,
    style_rule: str,
) -> OwnerProfile:
    return OwnerProfile(
        objective=(
            "Create and activate a Telegram Business DM intake workflow for a car wash. "
            "The workflow must answer pricing questions, collect booking fields, and save "
            "completed bookings to local CSV."
        ),
        initial_message=(
            "I need a Telegram Business DM intake workflow for my car wash. "
            "Please help me set it up and activate it."
        ),
        known_facts={
            "workflow_name": workflow_name,
            "channel": "Telegram Business DM",
            "provider": "telegram_bot_api",
            "required_fields": "car_model, car_type, wash_type, date, time",
            "intent": (
                "Handle leads asking about full car wash pricing and booking. Full wash only."
            ),
            "prices": "small car full wash 1000 rubles; SUV full wash 2500 rubles",
            "time_slots": "only exact hour slots such as 09:00, 10:00, 11:00",
            "sink": f"local CSV {csv_relative_path}",
            "confirmation_rule": "wait for my confirmation, then save and activate",
            "style_rule": style_rule,
        },
        rules=[
            "If OpenTulpa asks a setup question, answer with the known facts.",
            "If OpenTulpa proposes a workflow, confirm saving and activation.",
            "Do not invent extra required fields.",
        ],
        max_turns=7,
    )


def _judge_verdict(report_payload: dict[str, Any]) -> str:
    evaluation = report_payload.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return ""
    parsed = evaluation.get("parsed", {})
    if isinstance(parsed, dict):
        return str(parsed.get("verdict", "")).strip().lower()
    return str(evaluation.get("verdict", "")).strip().lower()


def _extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _owner_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "done"}


def _compact_owner_setup_state(
    harness: E2EHarness,
    *,
    customer_id: str,
    owner_chat_id: int,
    file_uploaded: bool,
) -> dict[str, Any]:
    thread_id = _telegram_owner_thread_id(chat_id=owner_chat_id)
    session = (
        _workflow_setup_session(harness, customer_id=customer_id, thread_id=thread_id)
        if thread_id
        else {}
    )
    draft = session.get("draft_upsert") if isinstance(session.get("draft_upsert"), dict) else {}
    scratchpad = session.get("scratchpad") if isinstance(session.get("scratchpad"), dict) else {}
    workflows = _list_workflows(harness, customer_id=customer_id)
    return {
        "file_uploaded": bool(file_uploaded),
        "thread_id": thread_id,
        "workflow_count": len(workflows),
        "workflow_names": [str(item.get("name", "") or "") for item in workflows[-3:]],
        "setup_session": {
            "status": str(session.get("status", "") or ""),
            "has_proposal": bool(str(session.get("last_proposed_draft_hash", "") or "").strip()),
            "last_proposed_draft_hash": str(session.get("last_proposed_draft_hash", "") or ""),
            "created_or_updated_workflow_id": str(
                session.get("created_or_updated_workflow_id", "") or ""
            ),
            "draft_upsert": {
                "name": str(draft.get("name", "") or ""),
                "channel": str(draft.get("channel", "") or ""),
                "provider": str(draft.get("provider", "") or ""),
                "required_fields": draft.get("required_fields") or [],
                "knowledge_file_ids": draft.get("knowledge_file_ids") or [],
                "sink_type": str(draft.get("sink_type", "") or ""),
                "sink_config": draft.get("sink_config") if isinstance(draft.get("sink_config"), dict) else {},
            },
            "scratchpad": {
                "source_file_ids": scratchpad.get("source_file_ids") or [],
                "knowledge_source_file_ids": scratchpad.get("knowledge_source_file_ids") or [],
                "knowledge_last_index": scratchpad.get("knowledge_last_index") or {},
                "knowledge_last_preflight": scratchpad.get("knowledge_last_preflight") or {},
                "missing_fields": scratchpad.get("missing_fields") or [],
                "open_questions": scratchpad.get("open_questions") or [],
                "proposal_summary": str(scratchpad.get("proposal_summary", "") or "")[:1200],
            },
        },
    }


def _plan_autospa_owner_turn(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    customer_id: str,
    owner_chat_id: int,
    file_uploaded: bool,
    turn_index: int,
    csv_relative_path: str,
) -> dict[str, Any]:
    api_key = str(getattr(harness.runtime, "openrouter_api_key", "") or "").strip()
    base_url = str(getattr(harness.runtime, "openrouter_base_url", "") or "").strip().rstrip("/")
    if not api_key or not base_url:
        raise RuntimeError("owner simulator requires the live LLM API key and base URL")
    model = os.getenv("OPENTULPA_E2E_OWNER_SIM_MODEL", DEFAULT_LEAD_SIMULATOR_MODEL)
    setup_state = _compact_owner_setup_state(
        harness,
        customer_id=customer_id,
        owner_chat_id=owner_chat_id,
        file_uploaded=file_uploaded,
    )
    payload = {
        "hidden_owner_objective": {
            "business": "AutoSpa detailing center in Murmansk",
            "workflow_name": "E2E AI Owner AutoSpa Business Knowledge",
            "channel": "Telegram Business DM",
            "knowledge_source": "the attached AutoSpa XLSX price list",
            "scope": "Use only the Мойка and Шиномонтаж sections for this workflow.",
            "source_disambiguation": (
                "For tire fitting prices, prefer the worksheet named Шиномонтаж. Ignore the Диски "
                "worksheet unless the customer explicitly asks about wheels, disks, or powder coating."
            ),
            "business_knowledge_contract": (
                "Tell OpenTulpa to prepare the original XLSX with business_knowledge_index, "
                "query it with business_knowledge_query for representative Мойка and Шиномонтаж "
                "facts, and bind the original source file ids to the final workflow."
            ),
            "required_fields": [
                "service_category",
                "service_name",
                "vehicle_type",
                "date",
                "time",
                "lead_name",
                "phone",
                "quoted_price",
            ],
            "field_guidance": {
                "quoted_price": (
                    "The assistant should fill this from the XLSX business knowledge when possible; "
                    "do not ask the lead to provide the price."
                )
            },
            "sink": {"type": "local_csv", "file_path": csv_relative_path},
            "confirmation_rule": "Confirm once OpenTulpa proposes a workflow matching this objective.",
        },
        "turn_index": int(turn_index),
        "current_setup_state": setup_state,
        "owner_assistant_transcript": [
            {
                "role": str(item.get("role", "") or "")[:30],
                "text": str(item.get("text", "") or "")[:1200],
            }
            for item in list(state.get("owner_transcript") or [])[-12:]
            if isinstance(item, dict)
        ],
    }
    harness.recorder.add("owner_simulator_prompt", model=model, payload=payload)
    system_prompt = (
        "You simulate the business owner using OpenTulpa through Telegram for a live e2e test.\n"
        "Stay in character as the owner only; never speak as OpenTulpa.\n"
        "Write concise Russian Telegram messages.\n"
        "Your goal is to create and activate the exact workflow described in the hidden objective.\n"
        "If the file is not uploaded yet, attach_file must be true and the message must explain how "
        "OpenTulpa should use the attached XLSX.\n"
        "If current_setup_state.setup_session.has_proposal is true and no workflow exists yet, confirm "
        "saving and activating the proposed workflow.\n"
        "If OpenTulpa asks a question, answer it with the hidden objective.\n"
        "Set done=true only when current_setup_state.workflow_count is greater than 0.\n"
        "Return strict JSON only with exactly these keys:\n"
        '{"done": boolean, "message": string, "attach_file": boolean, "reason": string}'
    )
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Plan the owner's next Telegram message.\n\n"
                    + json.dumps(payload, ensure_ascii=False, default=str),
                },
            ],
        },
        timeout=45.0,
    )
    response.raise_for_status()
    raw_text = _extract_chat_completion_text(response.json())
    parsed = _parse_json_object(raw_text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"owner simulator returned invalid JSON: {raw_text[:1000]}")
    plan = {
        "done": _owner_bool(parsed.get("done")),
        "message": str(parsed.get("message", "") or "").strip(),
        "attach_file": _owner_bool(parsed.get("attach_file")),
        "reason": str(parsed.get("reason", "") or "").strip()[:500],
        "raw_text": raw_text,
        "setup_state": setup_state,
    }
    if not file_uploaded and not plan["done"]:
        plan["attach_file"] = True
    if plan["done"] and int(setup_state.get("workflow_count") or 0) <= 0:
        has_proposal = bool(
            ((setup_state.get("setup_session") or {}).get("has_proposal"))
            if isinstance(setup_state.get("setup_session"), dict)
            else False
        )
        if has_proposal and plan["message"]:
            plan["done"] = False
        else:
            raise RuntimeError(f"owner simulator stopped before workflow creation: {plan}")
    if not plan["done"] and not plan["message"]:
        raise RuntimeError(f"owner simulator produced an empty message: {plan}")
    harness.recorder.add("owner_simulator_plan", model=model, payload=plan)
    return plan


def _live_google_sheets_target(harness: E2EHarness) -> Any | None:
    return getattr(harness.composio_service, "live_google_sheets_target", None)


def _sample_price_workbook_path(harness: E2EHarness) -> Path:
    return write_sample_vehicle_services_xlsx(
        harness.status_report_path.parent / SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME
    )


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
        "/internal/knowledge/",
        "/internal/intake/",
        "/internal/composio/",
        "/internal/telegram/business/",
    )
    return [
        item
        for item in harness.internal_api_calls_since(0)
        if str(item.get("path", "")).startswith(prefixes)
    ]


def _workflow_business_knowledge_snapshot(
    harness: E2EHarness,
    *,
    customer_id: str,
    workflow: dict[str, Any],
) -> dict[str, Any]:
    knowledge = getattr(harness.client.app.state, "knowledge_service", None)
    workflow_id = str(workflow.get("workflow_id", "") or "").strip()
    if knowledge is None or not workflow_id:
        return {}
    sources: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    try:
        source_rows = knowledge._source_rows(  # noqa: SLF001 - e2e artifact snapshot
            customer_id=customer_id,
            scope_type="intake_workflow",
            scope_id=workflow_id,
        )
        for row in source_rows:
            sources.append(
                {
                    "file_id": str(row["file_id"]),
                    "filename": str(row["filename"]),
                    "mime_type": str(row["mime_type"]),
                    "status": str(row["status"]),
                    "source_kind": str(row["source_kind"]),
                    "section_count": int(row["section_count"] or 0),
                    "char_count": int(row["char_count"] or 0),
                }
            )
        prepared_sections = knowledge._load_sections(  # noqa: SLF001 - e2e artifact snapshot
            customer_id=customer_id,
            scope_type="intake_workflow",
            scope_id=workflow_id,
        )
        for section in prepared_sections:
            sections.append(
                {
                    "source_ref": section.source_ref,
                    "source_kind": section.source_kind,
                    "preview": str(section.content or "")[:2500],
                }
            )
    except Exception as exc:
        return {
            "workflow_id": workflow_id,
            "knowledge_file_ids": workflow.get("knowledge_file_ids") or [],
            "error": str(exc),
        }
    return {
        "workflow_id": workflow_id,
        "knowledge_file_ids": workflow.get("knowledge_file_ids") or [],
        "sources": sources,
        "section_count": len(sections),
        "section_previews": sections,
    }


def _current_autospa_artifacts(
    harness: E2EHarness,
    *,
    state: dict[str, Any],
    artifact_dir: Path,
    customer_id: str,
) -> dict[str, str]:
    workflow = state.get("workflow") if isinstance(state.get("workflow"), dict) else {}
    business_knowledge = _workflow_business_knowledge_snapshot(
        harness,
        customer_id=customer_id,
        workflow=workflow,
    )
    if business_knowledge:
        content_hash = hashlib.sha256(
            json.dumps(business_knowledge, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if state.get("business_knowledge_hash") != content_hash:
            state["business_knowledge_hash"] = content_hash
            harness.recorder.add(
                "business_knowledge_snapshot",
                knowledge_file_ids=workflow.get("knowledge_file_ids") or [],
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
        "business_knowledge": artifact_dir / "business_knowledge.json",
        "workflow_snapshot": artifact_dir / "workflow_snapshot.json",
        "sheet_writes": artifact_dir / "sheet_writes.json",
        "internal_calls_filtered": artifact_dir / "internal_calls_filtered.json",
        "stage_judgements": artifact_dir / "stage_judgements.json",
    }
    _write_json_artifact(paths["owner_transcript"], state.get("owner_transcript") or [])
    _write_json_artifact(paths["lead_transcripts"], state.get("lead_transcripts") or [])
    _write_json_artifact(paths["business_knowledge"], business_knowledge)
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
            "Для ранних этапов настройки workflow своевременный пользовательский ACK или "
            "устойчивый прогресс setup-пайплайна тоже считается успехом, даже если финальное "
            "предложение придёт позже. Fail ставь только если есть реальная тишина, ложное "
            "подтверждение, или заметно неправильное поведение продукта."
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
    customer_id: str,
    owner_chat_id: int,
    owner_user_id: int,
    text: str,
    message_id: int,
    document: dict[str, Any] | None = None,
    wait_for_setup_proposal: bool = False,
) -> dict[str, Any]:
    state.setdefault("owner_transcript", []).append(
        {"role": "owner", "message_id": message_id, "text": text}
    )
    harness.recorder.add("owner_message", message_id=message_id, text=text, has_document=bool(document))
    start_index = len(harness.telegram_client.sent_messages)
    internal_call_start = harness.count_internal_api_calls()
    workflow_count_before = len(_list_workflows(harness, customer_id=customer_id))
    if document:
        body = _telegram_document_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            caption=text,
            file_id=str(document["file_id"]),
            file_name=str(
                document.get("file_name")
                or document.get("filename")
                or SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME
            ),
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

    started = time.monotonic()
    reply_latency_ms: int | None = None
    progress_latency_ms: int | None = None
    progress_snapshot: dict[str, Any] = {}
    wait_timeout_seconds = 210.0 if wait_for_setup_proposal else 110.0
    deadline = started + wait_timeout_seconds
    while time.monotonic() < deadline:
        replies = _messages_for_chat(harness, chat_id=owner_chat_id, start_index=start_index)
        if replies and reply_latency_ms is None:
            reply_latency_ms = int((time.monotonic() - started) * 1000)
        progress_snapshot = _owner_progress_snapshot(
            harness,
            customer_id=customer_id,
            owner_chat_id=owner_chat_id,
            internal_call_start=internal_call_start,
            workflow_count_before=workflow_count_before,
        )
        if progress_snapshot["progress_seen"] and progress_latency_ms is None:
            progress_latency_ms = int((time.monotonic() - started) * 1000)
        thread_id = str(progress_snapshot.get("thread_id", "") or "")
        if wait_for_setup_proposal and thread_id:
            if _workflow_setup_proposal_and_owner_reply_seen(
                harness,
                customer_id=customer_id,
                thread_id=thread_id,
                owner_chat_id=owner_chat_id,
                start_index=start_index,
            ):
                break
            time.sleep(0.2)
            continue
        if replies or progress_snapshot["progress_seen"]:
            break
        time.sleep(0.2)

    replies = _messages_for_chat(harness, chat_id=owner_chat_id, start_index=start_index)
    if wait_for_setup_proposal:
        thread_id = str(progress_snapshot.get("thread_id", "") or "")
        if not thread_id or not _workflow_setup_proposal_and_owner_reply_seen(
            harness,
            customer_id=customer_id,
            thread_id=thread_id,
            owner_chat_id=owner_chat_id,
            start_index=start_index,
        ):
            raise RuntimeError("owner stage did not produce a visible workflow setup proposal")
    if not replies and not progress_snapshot.get("progress_seen"):
        raise RuntimeError("owner stage produced no visible reply or durable progress")
    for item in replies:
        payload = {
            "role": "assistant",
            "message_id": item.get("message_id"),
            "text": str(item.get("text", "") or ""),
        }
        state.setdefault("owner_transcript", []).append(payload)
        harness.recorder.add("owner_assistant_reply", **payload)
    if progress_snapshot.get("progress_seen"):
        harness.recorder.add(
            "owner_stage_progress",
            message_id=message_id,
            reply_seen=bool(replies),
            progress_kinds=progress_snapshot.get("progress_kinds") or [],
            thread_id=str(progress_snapshot.get("thread_id", "") or ""),
            workflow_count=int(progress_snapshot.get("workflow_count") or 0),
            internal_call_count=len(progress_snapshot.get("internal_calls") or []),
        )
    return {
        "assistant_replies": replies,
        "assistant_reply_count": len(replies),
        "reply_seen": bool(replies),
        "reply_latency_ms": reply_latency_ms,
        "progress_seen": bool(progress_snapshot.get("progress_seen")),
        "progress_latency_ms": progress_latency_ms,
        "progress_kinds": progress_snapshot.get("progress_kinds") or [],
        "thread_id": str(progress_snapshot.get("thread_id", "") or ""),
        "setup_session": progress_snapshot.get("setup_session") or {},
        "workflow_count": int(progress_snapshot.get("workflow_count") or 0),
        "internal_call_count": len(progress_snapshot.get("internal_calls") or []),
    }


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

    workbook_path = _sample_price_workbook_path(e2e_harness)
    file_id = "tg_file_autospa_price"
    registered = e2e_harness.telegram_client.register_file(
        file_id=file_id,
        path=workbook_path,
        filename=SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
        mime_type=SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
    )
    e2e_harness.recorder.add("owner_document_uploaded", **registered)

    def stage_owner_upload() -> dict[str, Any]:
        fresh_result = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
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
        upload_result = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text=upload_text,
            message_id=2,
            document=registered,
        )
        return {
            "fresh_replies": int(fresh_result["assistant_reply_count"]),
            "fresh_reply_latency_ms": fresh_result["reply_latency_ms"],
            "upload_replies": int(upload_result["assistant_reply_count"]),
            "upload_reply_latency_ms": upload_result["reply_latency_ms"],
            "upload_progress_seen": bool(upload_result["progress_seen"]),
            "upload_progress_latency_ms": upload_result["progress_latency_ms"],
            "upload_progress_kinds": upload_result["progress_kinds"],
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
            "Прайс может быть большой, поэтому не вставляй его в контекст и не готовь Markdown pack. "
            "Проиндексируй исходный XLSX через business_knowledge_index, проверь разделы Мойка "
            "и Шиномонтаж через business_knowledge_query, и привяжи оригинальные source file ids "
            "к workflow knowledge_file_ids. "
            "Запись сохраняй в тестовую Google Sheets таблицу: "
            f"spreadsheetId={spreadsheet_id}, sheetName={sheet_name}. "
            "Используй sink_type google_sheets_composio, toolkit googlesheets, "
            "field_mapping на понятные колонки: Category, Service, Vehicle, Date, Time, Lead Name, "
            "Phone, Quoted Price, Conversation ID. Подготовь предложение и жди моего подтверждения."
        )
        result = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text=text,
            message_id=3,
            wait_for_setup_proposal=True,
        )
        return {
            "owner_replies": int(result["assistant_reply_count"]),
            "reply_latency_ms": result["reply_latency_ms"],
            "progress_seen": bool(result["progress_seen"]),
            "progress_latency_ms": result["progress_latency_ms"],
            "progress_kinds": result["progress_kinds"],
        }

    def stage_owner_confirm() -> dict[str, Any]:
        result = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
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
        return {
            "owner_replies": int(result["assistant_reply_count"]),
            "reply_latency_ms": result["reply_latency_ms"],
            "progress_seen": bool(result["progress_seen"]),
            "workflow": state["workflow"],
            "workflow_count": len(workflows),
        }

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


def test_live_ai_owner_creates_autospa_business_knowledge_workflow_and_simulated_leads(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 19012
    owner_chat_id = 29012
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_ai_owner_autospa",
    )
    csv_relative_path = "tulpa_stuff/e2e_ai_owner_autospa.csv"
    artifact_dir = e2e_harness.status_report_path.parent / "autospa_ai_owner_knowledge_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "owner_transcript": [],
        "owner_simulator_plans": [],
        "lead_transcripts": {},
        "stage_judgements": [],
        "workflow": {},
    }

    workbook_path = _sample_price_workbook_path(e2e_harness)
    registered = e2e_harness.telegram_client.register_file(
        file_id="tg_file_autospa_ai_owner_price",
        path=workbook_path,
        filename=SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
        mime_type=SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
    )
    e2e_harness.recorder.add("owner_document_uploaded", **registered)

    try:
        fresh_result = _post_owner_autospa_message(
            e2e_harness,
            state=state,
            customer_id=customer_id,
            owner_chat_id=owner_chat_id,
            owner_user_id=owner_user_id,
            text="/fresh",
            message_id=500,
        )
        assert fresh_result["assistant_reply_count"] >= 1
        assert _list_workflows(e2e_harness, customer_id=customer_id) == []

        file_uploaded = False
        for turn_index in range(1, 8):
            workflows = _list_workflows(e2e_harness, customer_id=customer_id)
            if workflows:
                break
            plan = _plan_autospa_owner_turn(
                e2e_harness,
                state=state,
                customer_id=customer_id,
                owner_chat_id=owner_chat_id,
                file_uploaded=file_uploaded,
                turn_index=turn_index,
                csv_relative_path=csv_relative_path,
            )
            state["owner_simulator_plans"].append(plan)
            if plan["done"]:
                break
            attach_file = bool(plan["attach_file"]) and not file_uploaded
            _post_owner_autospa_message(
                e2e_harness,
                state=state,
                customer_id=customer_id,
                owner_chat_id=owner_chat_id,
                owner_user_id=owner_user_id,
                text=str(plan["message"]),
                message_id=500 + turn_index,
                document=registered if attach_file else None,
            )
            file_uploaded = file_uploaded or attach_file
            if _wait_until(
                lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) >= 1,
                timeout_seconds=effective_live_llm_timeout_seconds(
                    8.0,
                    override_env="OPENTULPA_E2E_OWNER_SETUP_WAIT_TIMEOUT_SECONDS",
                ),
            ):
                break

        workflows = _list_workflows(e2e_harness, customer_id=customer_id)
        assert len(workflows) == 1, workflows
        workflow = workflows[0]
        state["workflow"] = workflow
        assert "autospa" in str(workflow.get("name", "")).lower()
        assert workflow["channel"] == "telegram_business_dm"
        assert workflow["provider"] == "telegram_bot_api"
        assert workflow["source_config"] == {"business_connection_id": business_connection_id}
        assert workflow["sink_type"] == "local_csv"
        assert workflow["sink_config"] == {"file_path": csv_relative_path}
        assert workflow.get("knowledge_file_ids"), workflow
        required_fields = {str(item) for item in workflow.get("required_fields") or []}
        assert {"service_name", "vehicle_type", "date", "time", "phone"}.issubset(
            required_fields
        )

        business_knowledge = _workflow_business_knowledge_snapshot(
            e2e_harness,
            customer_id=customer_id,
            workflow=workflow,
        )
        _write_json_artifact(artifact_dir / "business_knowledge_after_setup.json", business_knowledge)
        knowledge_text = json.dumps(business_knowledge, ensure_ascii=False).lower()
        assert "2х-фазная" in knowledge_text
        assert "мойка" in knowledge_text
        assert "шиномонтаж" in knowledge_text
        assert "комплект 19" in knowledge_text or "19`r" in knowledge_text

        wash_profile = LeadProfile(
            objective=(
                "Get the price for a 2х-фазная мойка кузова for a Toyota RAV4, then book it."
            ),
            initial_message=(
                "Здравствуйте! Сколько стоит 2х-фазная мойка кузова для Toyota RAV4? "
                "Если подходит, хочу записаться."
            ),
            known_facts={
                "service_category": "Мойка",
                "service_name": "2х-фазная мойка кузова",
                "vehicle_type": "Toyota RAV4, SUV",
                "date": "завтра",
                "time": "10:00",
                "lead_name": "Алексей",
                "phone": "+79990001001",
            },
            persona="Russian Telegram customer, practical and brief.",
            rules=[
                "Speak Russian.",
                "Do not reveal name, phone, date, or time until asked.",
                "If the assistant gives a plausible AutoSpa price from the price list, accept it.",
                "Stay focused on booking the 2х-фазная мойка кузова.",
            ],
            max_turns=7,
        )
        tire_profile = LeadProfile(
            objective=(
                "Get the price for tire fitting for a BMW X5: 19R, crossover, low profile, "
                "then book it."
            ),
            initial_message=(
                "Добрый день. Нужно узнать цену и записаться именно на шиномонтаж: BMW X5, "
                "комплект 19`R, кросовер с низким профилем."
            ),
            known_facts={
                "service_category": "Шиномонтаж",
                "service_name": "Комплект 19`R",
                "vehicle_type": "Внедорожник / кросовер + низкий профиль",
                "date": "пятница",
                "time": "15:00",
                "lead_name": "Мария",
                "phone": "+79990001002",
            },
            persona="Russian Telegram customer, concise and cooperative.",
            rules=[
                "Speak Russian.",
                "Do not reveal name, phone, date, or time until asked.",
                "If the assistant gives a plausible AutoSpa price from the price list, accept it.",
                "Stay focused on booking шиномонтаж for 19R low-profile crossover tires.",
            ],
            max_turns=7,
        )

        wash_simulation = e2e_harness.simulate_telegram_business_lead(
            customer_id=customer_id,
            workflow_id=workflow["workflow_id"],
            business_connection_id=business_connection_id,
            lead_chat_id=39101,
            lead_user_id=49101,
            profile=wash_profile,
            initial_message_id=900,
            idle_timeout_seconds=100.0,
        )
        tire_simulation = e2e_harness.simulate_telegram_business_lead(
            customer_id=customer_id,
            workflow_id=workflow["workflow_id"],
            business_connection_id=business_connection_id,
            lead_chat_id=39102,
            lead_user_id=49102,
            profile=tire_profile,
            initial_message_id=950,
            idle_timeout_seconds=100.0,
        )
        state["lead_transcripts"] = {
            "wash": wash_simulation.get("transcript") or [],
            "tire": tire_simulation.get("transcript") or [],
        }
        assert wash_simulation["ok"] is True, wash_simulation
        assert wash_simulation["reason"] == "booking_completed"
        assert tire_simulation["ok"] is True, tire_simulation
        assert tire_simulation["reason"] == "booking_completed"
        for simulation in (wash_simulation, tire_simulation):
            completed_booking = simulation.get("completed_booking") or {}
            assert str(completed_booking.get("status", "")).lower() == "completed"
            assert str(completed_booking.get("sink_write_status", "")).lower() == "succeeded"

        csv_rows = _csv_rows_for_relative_path(
            e2e_harness,
            relative_path=csv_relative_path,
        )
        assert len(csv_rows) >= 2
        by_conversation = {
            str(row.get("conversation_id", "")).strip(): row
            for row in csv_rows
        }
        wash_row = by_conversation.get("39101") or {}
        tire_row = by_conversation.get("39102") or {}
        assert str(wash_row.get("status", "")).strip().lower() == "completed"
        assert _booking_category(wash_row) == "мойка"
        assert _service_text_matches_two_phase_wash(wash_row.get("service_name", ""))
        wash_quoted_price_digits = "".join(
            ch for ch in str(wash_row.get("quoted_price", "")) if ch.isdigit()
        )
        assert "1200" in wash_quoted_price_digits
        assert str(wash_row.get("lead_name", "")).strip().lower() == "алексей"
        assert str(wash_row.get("phone", "")).strip() == "+79990001001"
        assert str(tire_row.get("status", "")).strip().lower() == "completed"
        assert _booking_category(tire_row) == "шиномонтаж"
        tire_service_text = (
            f"{tire_row.get('service_name', '')} {tire_row.get('vehicle_type', '')}".lower()
        )
        tire_quoted_price_digits = "".join(
            ch for ch in str(tire_row.get("quoted_price", "")) if ch.isdigit()
        )
        assert "19" in tire_service_text
        assert "4000" in tire_quoted_price_digits
        assert str(tire_row.get("quoted_price", "")).strip()
        assert str(tire_row.get("lead_name", "")).strip().lower() == "мария"
        assert str(tire_row.get("phone", "")).strip() == "+79990001002"

        lead_source_messages = {
            "wash": _lead_source_messages(
                e2e_harness,
                customer_id=customer_id,
                business_connection_id=business_connection_id,
                lead_chat_id=39101,
            ),
            "tire": _lead_source_messages(
                e2e_harness,
                customer_id=customer_id,
                business_connection_id=business_connection_id,
                lead_chat_id=39102,
            ),
        }
        assert any(
            str(item.get("sender_role", "")).strip() == "assistant"
            for messages in lead_source_messages.values()
            for item in messages
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
        scenario="live_ai_owner_creates_autospa_business_knowledge_workflow_and_simulated_leads",
        ok=True,
        details={
            "customer_id": customer_id,
            "business_connection_id": business_connection_id,
            "owner_simulator_model": os.getenv(
                "OPENTULPA_E2E_OWNER_SIM_MODEL",
                DEFAULT_LEAD_SIMULATOR_MODEL,
            ),
            "lead_simulator_model": e2e_harness.lead_simulator.model,
            "registered_file": registered,
            "owner_simulator_plans": state.get("owner_simulator_plans") or [],
            "workflow": {
                "workflow_id": workflow["workflow_id"],
                "name": workflow["name"],
                "required_fields": workflow["required_fields"],
                "knowledge_file_ids": workflow.get("knowledge_file_ids") or [],
                "assistant_instructions": str(workflow.get("assistant_instructions", ""))[:2500],
                "sink_config": workflow["sink_config"],
            },
            "artifact_paths": artifact_paths,
            "lead_source_messages": lead_source_messages,
            "wash_simulation": wash_simulation,
            "tire_simulation": tire_simulation,
            "csv_rows": csv_rows,
        },
    )
    assert report.exists()
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    verdict = _judge_verdict(report_payload)
    assert verdict != "fail"


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
    assert _wait_until(
        lambda: any(
            _looks_like_owner_proposal_message(item)
            for item in _messages_for_chat(
                e2e_harness,
                chat_id=owner_chat_id,
                start_index=initial_owner_message_count,
            )
        ),
        timeout_seconds=180.0,
    )
    assert _list_workflows(e2e_harness, customer_id=customer_id) == []

    confirm_start_index = len(e2e_harness.telegram_client.sent_messages)
    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=3,
            text="Yes, save that workflow now exactly as proposed.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(
        lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1,
        timeout_seconds=180.0,
    )

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Telegram Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""
    source_config = workflow["source_config"]
    assert source_config["business_connection_id"] == business_connection_id
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}

    assert _wait_until(
        lambda: any(
            workflow["workflow_id"] in str(item.get("text", ""))
            or (
                "workflow" in str(item.get("text", "")).lower()
                and any(
                    marker in str(item.get("text", "")).lower()
                    for marker in ("created", "active", "saved", "activated")
                )
            )
            for item in _messages_for_chat(
                e2e_harness,
                chat_id=owner_chat_id,
                start_index=confirm_start_index,
            )
        ),
        timeout_seconds=180.0,
    )
    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=confirm_start_index,
    )
    assert latest_owner_message is not None
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="owner_create_workflow_confirmation",
        expectation="Assistant confirms the Telegram intake workflow was created or activated successfully, with no backend error.",
        actual={"assistant_reply": latest_owner_message, "workflow": workflow},
    )

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
    assert _wait_until(
        lambda: _latest_message_for_chat(
            e2e_harness,
            chat_id=owner_chat_id,
            start_index=start_message_count,
        )
        is not None,
        timeout_seconds=60.0,
    )

    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_message_count,
    )
    assert latest_owner_message is not None
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="owner_delete_workflow_confirmation",
        expectation="Assistant confirms the active Telegram intake workflow was deleted/removed/gone, with no backend error.",
        actual={"assistant_reply": latest_owner_message},
    )

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

    start_index = len(e2e_harness.telegram_client.sent_messages)
    owner_setup = e2e_harness.simulate_telegram_owner_workflow_setup(
        customer_id=customer_id,
        owner_chat_id=owner_chat_id,
        owner_user_id=owner_user_id,
        profile=_car_wash_owner_profile(
            workflow_name="E2E Quality Car Wash",
            csv_relative_path="tulpa_stuff/e2e_quality_carwash.csv",
            style_rule=(
                "If a lead asks for price, answer directly before asking anything else. "
                "As soon as wash_type and car_type are known, give the exact price immediately. "
                "Do not repeat already known details."
            ),
        ),
        initial_message_id=50,
        idle_timeout_seconds=90.0,
    )
    assert owner_setup["ok"] is True, owner_setup

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Quality Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    source_config = workflow["source_config"]
    assert source_config["business_connection_id"] == business_connection_id
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}
    instructions = str(workflow.get("assistant_instructions", "")).strip()
    assert instructions
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="quality_workflow_instructions",
        expectation=(
            "Workflow instructions preserve owner requirements: answer price questions directly, "
            "use exact known prices, offer exact times, and avoid repeating details already provided."
        ),
        actual={"assistant_instructions": instructions},
        context={"workflow": workflow},
    )

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

    owner_start_index = len(e2e_harness.telegram_client.sent_messages)
    owner_setup = e2e_harness.simulate_telegram_owner_workflow_setup(
        customer_id=customer_id,
        owner_chat_id=owner_chat_id,
        owner_user_id=owner_user_id,
        profile=_car_wash_owner_profile(
            workflow_name="E2E Multiturn Car Wash",
            csv_relative_path=csv_relative_path,
            style_rule=(
                "When a lead shows booking intent, answer direct questions briefly and then "
                "ask only for the next missing field. Do not repeat details already provided."
            ),
        ),
        initial_message_id=70,
        idle_timeout_seconds=90.0,
    )
    assert owner_setup["ok"] is True, owner_setup

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
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="multiturn_first_lead_reply",
        expectation=(
            "Assistant responds helpfully to booking intent and asks for missing booking information "
            "without claiming a booking was saved."
        ),
        actual={"lead_message": "Hi, I want to book a wash tomorrow. How much is it for an SUV?", "assistant_reply": first_lead_reply_text},
    )

    bookings_after_first_turn = e2e_harness.client.app.state.intake_workflows.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )
    assert not any(str(item.get("status", "")).lower() == "completed" for item in bookings_after_first_turn)

    second_lead_message_start = len(e2e_harness.telegram_client.sent_messages)
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
        lambda: _latest_message_for_chat(
            e2e_harness,
            chat_id=lead_chat_id,
            start_index=second_lead_message_start,
        )
        is not None,
        timeout_seconds=90.0,
    )
    bookings_after_second_turn = e2e_harness.client.app.state.intake_workflows.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id=str(lead_chat_id),
    )
    completed_after_second_turn = any(
        str(item.get("status", "")).lower() == "completed" for item in bookings_after_second_turn
    )

    if not completed_after_second_turn:
        third_lead_status = e2e_harness.post_telegram(
            body=_telegram_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=2001,
                message_id=203,
                text="Yes, please confirm and save the booking.",
            ),
        )
        assert third_lead_status == 200
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
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="multiturn_completed_booking_fields",
        expectation=(
            "Completed booking captures the lead's intended Toyota RAV4 SUV full wash appointment "
            "for tomorrow at 10:00 or an equivalent normalized date/time."
        ),
        actual={"booking": booking, "extracted_fields": extracted},
    )

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
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="multiturn_csv_row_fields",
        expectation=(
            "CSV row persists the same completed Toyota RAV4 SUV full wash booking for tomorrow at 10:00 "
            "or equivalent normalized date/time."
        ),
        actual={"csv_row": row, "booking": booking},
    )

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

    owner_start_index = len(e2e_harness.telegram_client.sent_messages)
    owner_setup = e2e_harness.simulate_telegram_owner_workflow_setup(
        customer_id=customer_id,
        owner_chat_id=owner_chat_id,
        owner_user_id=owner_user_id,
        profile=_car_wash_owner_profile(
            workflow_name="E2E Simulated Lead Car Wash",
            csv_relative_path=csv_relative_path,
            style_rule=(
                "If a lead asks for price, answer directly first and then ask only "
                "for the next missing booking detail. Do not save until all fields are known."
            ),
        ),
        initial_message_id=80,
        idle_timeout_seconds=90.0,
    )
    assert owner_setup["ok"] is True, owner_setup

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
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="lead_simulator_first_reply",
        expectation=(
            "Assistant addresses the lead's price question and asks for missing booking details "
            "instead of pretending the booking is complete."
        ),
        actual={"assistant_reply": first_reply_text, "lead_profile": profile.__dict__},
    )
    final_turn = turn_results[-1]
    assert bool(final_turn.get("booking_completed", False)) is True
    completed_booking = simulation.get("completed_booking") or {}
    assert completed_booking
    assert str(completed_booking.get("status", "")).strip().lower() == "completed"
    assert str(completed_booking.get("sink_write_status", "")).strip().lower() == "succeeded"
    _assert_llm_semantic_match(
        e2e_harness,
        scenario="lead_simulator_completed_booking_fields",
        expectation=(
            "Completed booking captures the simulated lead's hidden facts: Toyota RAV4, SUV, "
            "full wash, tomorrow, 10:00 or equivalent normalized date/time."
        ),
        actual={"completed_booking": completed_booking, "lead_profile": profile.__dict__},
    )

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
