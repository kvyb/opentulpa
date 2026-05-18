from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from evaluation.judge import evaluate_e2e_scenario_with_llm_judge
from fastapi.testclient import TestClient
from harness.lead_simulator import LeadProfile, LeadSimulator
from harness.logging import JsonlRecorder
from mocks.composio_instagram import FakeComposioInstagramService
from mocks.telegram import FakeTelegramClient
from reports.status_report import write_status_report

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.scheduler.service import SchedulerService


@dataclass
class E2EHarness:
    client: TestClient
    runtime: OpenTulpaLangGraphRuntime
    recorder: JsonlRecorder
    project_root: Path
    system_log_path: Path
    status_report_path: Path
    behavior_log_path: Path
    llm_trace_path: Path
    telegram_client: FakeTelegramClient
    composio_service: Any
    lead_simulator: LeadSimulator

    def count_internal_api_calls(self) -> int:
        return self.recorder.count("internal_api_call")

    def internal_api_calls_since(self, start: int = 0) -> list[dict[str, Any]]:
        return self.recorder.slice("internal_api_call", start)

    def post_chat(self, *, customer_id: str, thread_id: str, text: str) -> dict[str, Any]:
        started = time.monotonic()
        self.recorder.add("user_turn", customer_id=customer_id, thread_id=thread_id, text=text)
        response = self.client.post(
            "/internal/chat",
            json={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "text": text,
            },
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        payload = response.json()
        self.recorder.add(
            "agent_turn",
            status_code=int(response.status_code),
            elapsed_ms=elapsed_ms,
            payload=payload,
        )
        return {
            "status_code": int(response.status_code),
            "payload": payload,
            "elapsed_ms": elapsed_ms,
        }

    def post_telegram(self, *, body: dict[str, Any], secret: str = "test-secret") -> int:
        response = self.client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": secret},
            json=body,
        )
        self.recorder.add(
            "telegram_webhook",
            status_code=int(response.status_code),
            body=body,
        )
        return int(response.status_code)

    def run_workflow(self, *, customer_id: str, workflow_id: str, event_type: str = "manual_e2e") -> dict[str, Any]:
        response = self.client.post(
            "/internal/intake/workflows/run",
            json={
                "customer_id": customer_id,
                "workflow_id": workflow_id,
                "force": True,
                "event_type": event_type,
            },
        )
        payload = response.json()
        self.recorder.add(
            "intake_run",
            workflow_id=workflow_id,
            status_code=int(response.status_code),
            payload=payload,
        )
        return {"status_code": int(response.status_code), "payload": payload}

    def list_bookings(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        intake_service = self.client.app.state.intake_workflows
        return intake_service.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
        )

    def upsert_instagram_workflow(
        self,
        *,
        customer_id: str,
        name: str,
        conversation_id: str,
        connected_account_id: str,
        required_fields: list[str],
        csv_relative_path: str,
        notify_user: bool = False,
    ) -> dict[str, Any]:
        response = self.client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": customer_id,
                "name": name,
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": connected_account_id,
                    "conversation_id": conversation_id,
                },
                "intent_description": "Extract booking fields from Instagram DMs and save them.",
                "required_fields": required_fields,
                "sink_type": "local_csv",
                "sink_config": {"file_path": csv_relative_path},
                "notify_user": notify_user,
                "enabled": True,
            },
        )
        payload = response.json()
        self.recorder.add(
            "intake_upsert",
            status_code=int(response.status_code),
            payload=payload,
        )
        return {"status_code": int(response.status_code), "payload": payload}

    def simulate_telegram_business_lead(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        business_connection_id: str,
        lead_chat_id: int,
        lead_user_id: int,
        profile: LeadProfile,
        initial_message_id: int = 500,
        idle_timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        transcript: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        turn_results: list[dict[str, Any]] = []
        max_turns = max(1, int(profile.max_turns or 6))
        next_message_id = max(1, int(initial_message_id))
        next_lead_text = str(profile.initial_message or "").strip()
        if not next_lead_text:
            raise ValueError("lead profile initial_message is required")

        def _assistant_messages() -> list[dict[str, Any]]:
            return [
                item
                for item in self.telegram_client.sent_messages
                if int(item.get("chat_id", 0)) == int(lead_chat_id)
                and str(item.get("business_connection_id", "")).strip() == business_connection_id
            ]

        for turn_index in range(max_turns):
            assistant_count_before = len(_assistant_messages())
            inbound_message = {
                "update_id": int(time.time() * 1000),
                "business_message": {
                    "business_connection_id": business_connection_id,
                    "message_id": next_message_id,
                    "date": int(time.time()),
                    "chat": {"id": lead_chat_id, "type": "private", "username": f"lead_{lead_user_id}"},
                    "from": {"id": lead_user_id, "is_bot": False, "username": f"lead_{lead_user_id}"},
                    "text": next_lead_text,
                },
            }
            webhook_status = self.post_telegram(body=inbound_message)
            transcript.append(
                {
                    "role": "lead",
                    "text": next_lead_text,
                    "message_id": next_message_id,
                }
            )
            self.recorder.add(
                "lead_simulator_inbound",
                workflow_id=workflow_id,
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                turn_index=turn_index,
                status_code=webhook_status,
                text=next_lead_text,
            )
            if webhook_status != 200:
                return {
                    "ok": False,
                    "reason": "lead_webhook_rejected",
                    "transcript": transcript,
                    "turn_plans": plans,
                    "turn_results": turn_results,
                    "bookings": self.list_bookings(
                        customer_id=customer_id,
                        workflow_id=workflow_id,
                        conversation_id=str(lead_chat_id),
                    ),
                }

            deadline = time.monotonic() + max(1.0, float(idle_timeout_seconds))
            while time.monotonic() < deadline:
                bookings = self.list_bookings(
                    customer_id=customer_id,
                    workflow_id=workflow_id,
                    conversation_id=str(lead_chat_id),
                )
                completed = next(
                    (
                        item
                        for item in bookings
                        if str(item.get("status", "")).strip().lower() == "completed"
                    ),
                    None,
                )
                assistant_messages = _assistant_messages()
                if len(assistant_messages) > assistant_count_before or completed is not None:
                    break
                time.sleep(0.2)

            bookings = self.list_bookings(
                customer_id=customer_id,
                workflow_id=workflow_id,
                conversation_id=str(lead_chat_id),
            )
            completed = next(
                (
                    item
                    for item in bookings
                    if str(item.get("status", "")).strip().lower() == "completed"
                ),
                None,
            )
            assistant_messages = _assistant_messages()
            new_assistant_messages = assistant_messages[assistant_count_before:]
            for item in new_assistant_messages:
                transcript.append(
                    {
                        "role": "assistant",
                        "text": str(item.get("text", "") or "").strip(),
                        "message_id": item.get("message_id"),
                    }
                )
            turn_result = {
                "turn_index": turn_index,
                "lead_text": next_lead_text,
                "assistant_messages": new_assistant_messages,
                "booking_completed": completed is not None,
                "bookings": bookings,
            }
            turn_results.append(turn_result)
            self.recorder.add(
                "lead_simulator_turn_result",
                workflow_id=workflow_id,
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                **turn_result,
            )
            if completed is not None:
                return {
                    "ok": True,
                    "reason": "booking_completed",
                    "transcript": transcript,
                    "turn_plans": plans,
                    "turn_results": turn_results,
                    "bookings": bookings,
                    "completed_booking": completed,
                }
            if not new_assistant_messages:
                return {
                    "ok": False,
                    "reason": "assistant_did_not_reply",
                    "transcript": transcript,
                    "turn_plans": plans,
                    "turn_results": turn_results,
                    "bookings": bookings,
                }

            plan = self.lead_simulator.plan_next_turn(
                profile=profile,
                transcript=transcript,
                booking_state=bookings[0] if bookings else {},
            )
            plans.append(plan.as_dict())
            self.recorder.add(
                "lead_simulator_plan",
                workflow_id=workflow_id,
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                turn_index=turn_index,
                plan=plan.as_dict(),
            )
            if plan.done or not str(plan.message or "").strip():
                return {
                    "ok": False,
                    "reason": "lead_simulator_stopped_before_completion",
                    "transcript": transcript,
                    "turn_plans": plans,
                    "turn_results": turn_results,
                    "bookings": bookings,
                }
            next_lead_text = str(plan.message or "").strip()
            next_message_id += 1

        return {
            "ok": False,
            "reason": "max_turns_exhausted",
            "transcript": transcript,
            "turn_plans": plans,
            "turn_results": turn_results,
            "bookings": self.list_bookings(
                customer_id=customer_id,
                workflow_id=workflow_id,
                conversation_id=str(lead_chat_id),
            ),
        }

    def write_status_report(self, *, scenario: str, ok: bool, details: dict[str, Any]) -> Path:
        payload = {
            "scenario": scenario,
            "ok": bool(ok),
            "system_log_path": str(self.system_log_path),
            "behavior_log_path": str(self.behavior_log_path),
            "llm_trace_path": str(self.llm_trace_path),
            "telegram_sent_messages": len(self.telegram_client.sent_messages),
            "composio_calls": len(self.composio_service.calls),
            "details": details,
        }
        payload["evaluation"] = evaluate_e2e_scenario_with_llm_judge(
            scenario=scenario,
            details=details,
            system_log_path=self.system_log_path,
            behavior_log_path=self.behavior_log_path,
            llm_trace_path=self.llm_trace_path,
        )
        return write_status_report(self.status_report_path, payload)

def _require_openai_compatible_env() -> tuple[str, str]:
    settings = get_settings()
    api_key = str(settings.openai_compatible_api_key or "").strip()
    base_url = str(settings.openrouter_base_url or "").strip() or "https://openrouter.ai/api/v1"
    return api_key, base_url


def patch_runtime_internal_api(
    *,
    runtime: OpenTulpaLangGraphRuntime,
    app: Any,
    recorder: JsonlRecorder,
) -> None:
    async def _request_with_backoff(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        attempts = max(0, int(retries)) + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.request(
                        method=method,
                        url=path,
                        params=params,
                        json=json_body,
                        timeout=timeout,
                    )
                recorder.add(
                    "internal_api_call",
                    method=str(method).upper(),
                    path=path,
                    params=params or {},
                    json_body=json_body or {},
                    status_code=int(response.status_code),
                    response_text=str(response.text or "")[:3000],
                )
                return response
            except Exception as exc:  # pragma: no cover - defensive retry
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"internal request failed: {last_exc}")

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]


def build_harness(
    *,
    tmp_path: Path,
    monkeypatch: Any,
    scenario_name: str,
    composio_service: Any | None = None,
    memory_service: Any | None = None,
) -> E2EHarness:
    from opentulpa.api import app as app_module
    from opentulpa.interfaces.telegram import attachments as attachments_module
    from opentulpa.interfaces.telegram import chat_service as chat_module
    from opentulpa.interfaces.telegram import relay as relay_module
    from opentulpa.tasks import sandbox as sandbox_module

    api_key, base_url = _require_openai_compatible_env()
    if not api_key:
        raise RuntimeError("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) is required")

    system_log_path = tmp_path / f"{scenario_name}_system_events.jsonl"
    status_report_path = tmp_path / f"{scenario_name}_status_report.json"
    behavior_log_path = tmp_path / f"{scenario_name}_agent_behavior.jsonl"
    recorder = JsonlRecorder(system_log_path)

    fake_tg = FakeTelegramClient("fake-token")
    composio = composio_service or FakeComposioInstagramService()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERNAMES", "")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / f"{scenario_name}_links.sqlite"))
    isolated_project_root = tmp_path / "project_root"
    isolated_project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", isolated_project_root)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", isolated_project_root)
    monkeypatch.setattr(
        chat_module,
        "STATE_STORE",
        TelegramStateStore(isolated_project_root / ".opentulpa" / "telegram_state.json"),
    )
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(attachments_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(chat_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda _token: fake_tg)
    get_settings.cache_clear()
    settings = get_settings()

    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        model_name=str(os.getenv("OPENTULPA_E2E_MODEL", settings.llm_model)),
        wake_classifier_model_name=str(
            os.getenv(
                "OPENTULPA_E2E_WAKE_MODEL",
                settings.wake_classifier_model or settings.llm_model,
            )
        ),
        checkpoint_db_path=str(tmp_path / f"{scenario_name}_checkpoints.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log_path),
    )
    scheduler = SchedulerService(db_path=tmp_path / f"{scenario_name}_scheduler.sqlite")
    app = create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        composio_service=composio,
        memory=memory_service,
    )
    patch_runtime_internal_api(runtime=runtime, app=app, recorder=recorder)
    client = TestClient(app)
    client.__enter__()

    llm_trace_path = behavior_log_path.parent / "llm_call_traces.jsonl"
    return E2EHarness(
        client=client,
        runtime=runtime,
        recorder=recorder,
        project_root=isolated_project_root,
        system_log_path=system_log_path,
        status_report_path=status_report_path,
        behavior_log_path=behavior_log_path,
        llm_trace_path=llm_trace_path,
        telegram_client=fake_tg,
        composio_service=composio,
        lead_simulator=LeadSimulator(
            api_key=api_key,
            base_url=base_url,
            recorder=recorder,
        ),
    )


def close_harness(harness: E2EHarness) -> None:
    from opentulpa.core.config import get_settings

    try:
        harness.client.__exit__(None, None, None)
    finally:
        get_settings.cache_clear()


def _decode_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or "").strip())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items
