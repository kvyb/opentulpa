from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.scheduler.service import SchedulerService
from harness.logging import JsonlRecorder
from mocks.composio_instagram import FakeComposioInstagramService
from mocks.telegram import FakeTelegramClient
from reports.status_report import write_status_report
from evaluation.judge import evaluate_e2e_scenario_with_llm_judge


@dataclass
class E2EHarness:
    client: TestClient
    runtime: OpenTulpaLangGraphRuntime
    recorder: JsonlRecorder
    system_log_path: Path
    status_report_path: Path
    behavior_log_path: Path
    llm_trace_path: Path
    telegram_client: FakeTelegramClient
    composio_service: FakeComposioInstagramService

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

    def latest_approval_id_from_calls(
        self,
        *,
        action_name: str,
        calls: list[dict[str, Any]] | None = None,
    ) -> str:
        for item in reversed(calls or self.internal_api_calls_since(0)):
            if str(item.get("path", "")).strip() != "/internal/approvals/evaluate":
                continue
            json_body = item.get("json_body", {})
            if not isinstance(json_body, dict):
                continue
            if str(json_body.get("action_name", "")).strip() != str(action_name or "").strip():
                continue
            payload = _decode_json_object(str(item.get("response_text", "")))
            approval_id = str(payload.get("approval_id", "")).strip()
            if approval_id and approval_id.lower() not in {"none", "null"}:
                return approval_id
        return ""

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        response = self.client.get(f"/internal/approvals/{approval_id}")
        payload = response.json()
        self.recorder.add(
            "approval_get",
            approval_id=approval_id,
            status_code=int(response.status_code),
            payload=payload,
        )
        return {"status_code": int(response.status_code), "payload": payload}


def _require_openai_compatible_env() -> tuple[str, str]:
    api_key = str(os.getenv("OPENAI_COMPATIBLE_API_KEY", "")).strip() or str(
        os.getenv("OPENROUTER_API_KEY", "")
    ).strip()
    base_url = str(os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")).strip() or str(
        os.getenv("OPENROUTER_BASE_URL", "")
    ).strip() or "https://openrouter.ai/api/v1"
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
    composio_service: FakeComposioInstagramService | None = None,
) -> E2EHarness:
    from opentulpa.api import app as app_module
    from opentulpa.core.config import get_settings

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
    monkeypatch.setenv("APPROVALS_DB_PATH", str(tmp_path / f"{scenario_name}_approvals.sqlite"))
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / f"{scenario_name}_links.sqlite"))
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    get_settings.cache_clear()

    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        model_name=str(os.getenv("OPENTULPA_E2E_MODEL", "openai/gpt-4.1-mini")),
        wake_classifier_model_name=str(os.getenv("OPENTULPA_E2E_WAKE_MODEL", "openai/gpt-4.1-mini")),
        guardrail_classifier_model_name=str(
            os.getenv("OPENTULPA_E2E_GUARDRAIL_MODEL", "openai/gpt-4.1-mini")
        ),
        checkpoint_db_path=str(tmp_path / f"{scenario_name}_checkpoints.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log_path),
    )
    scheduler = SchedulerService(db_path=tmp_path / f"{scenario_name}_scheduler.sqlite")
    app = create_app(agent_runtime=runtime, scheduler=scheduler, composio_service=composio)
    patch_runtime_internal_api(runtime=runtime, app=app, recorder=recorder)
    client = TestClient(app)
    client.__enter__()

    llm_trace_path = behavior_log_path.parent / "llm_call_traces.jsonl"
    return E2EHarness(
        client=client,
        runtime=runtime,
        recorder=recorder,
        system_log_path=system_log_path,
        status_report_path=status_report_path,
        behavior_log_path=behavior_log_path,
        llm_trace_path=llm_trace_path,
        telegram_client=fake_tg,
        composio_service=composio,
    )


def close_harness(harness: E2EHarness) -> None:
    from opentulpa.core.config import get_settings

    try:
        harness.client.__exit__(None, None, None)
    finally:
        get_settings.cache_clear()


def extract_approval_id(text: str) -> str:
    import re

    match = re.search(r"\bapr_[a-z0-9_-]{6,40}\b", str(text or ""), flags=re.IGNORECASE)
    return str(match.group(0)).strip() if match else ""


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
