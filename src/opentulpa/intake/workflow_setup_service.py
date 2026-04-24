"""Wizard-style setup service for intake workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.intake.workflow_setup_store import WorkflowSetupSessionStore


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        safe_key = str(key or "").strip()
        if not safe_key:
            continue
        if isinstance(value, dict) and isinstance(merged.get(safe_key), dict):
            merged[safe_key] = _deep_merge(_safe_dict(merged.get(safe_key)), value)
            continue
        merged[safe_key] = value
    return merged


def _merge_sink_config(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        safe_key = str(key or "").strip()
        if not safe_key:
            continue
        if safe_key == "field_mapping" and isinstance(value, dict):
            merged[safe_key] = dict(value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(safe_key), dict):
            merged[safe_key] = _deep_merge(_safe_dict(merged.get(safe_key)), value)
            continue
        merged[safe_key] = value
    return merged


def _merge_draft(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        safe_key = str(key or "").strip()
        if not safe_key:
            continue
        if safe_key == "field_guidance" and isinstance(value, dict):
            merged[safe_key] = dict(value)
            continue
        if safe_key == "sink_config" and isinstance(value, dict):
            merged[safe_key] = _merge_sink_config(_safe_dict(merged.get(safe_key)), value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(safe_key), dict):
            merged[safe_key] = _deep_merge(_safe_dict(merged.get(safe_key)), value)
            continue
        merged[safe_key] = value
    return merged


def _normalize_local_csv_draft_sink_config(draft: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(draft)
    sink_type = str(normalized.get("sink_type", "") or "").strip().lower()
    if sink_type != "local_csv":
        return normalized
    sink_config = _safe_dict(normalized.get("sink_config"))
    file_path = str(
        sink_config.get("file_path", "")
        or sink_config.get("filename", "")
        or ""
    ).strip()
    normalized["sink_config"] = {"file_path": file_path} if file_path else {}
    return normalized


def _normalize_schedule_for_channel(draft: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(draft)
    channel = str(normalized.get("channel", "") or "").strip().lower()
    if channel == "telegram_business_dm":
        normalized["schedule"] = ""
    else:
        normalized["schedule"] = str(normalized.get("schedule", "*/5 * * * *") or "*/5 * * * *")
    return normalized


class WorkflowSetupService:
    """Owns workflow-setup session lifecycle and commit semantics."""

    def __init__(
        self,
        *,
        store: WorkflowSetupSessionStore,
        intake_workflows: IntakeWorkflowService,
    ) -> None:
        self._store = store
        self._intake_workflows = intake_workflows

    @staticmethod
    def _draft_scaffold() -> dict[str, Any]:
        return _normalize_local_csv_draft_sink_config(
            _normalize_schedule_for_channel(
            {
                "name": "",
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {},
                "intent_description": "",
                "required_fields": [],
                "field_guidance": {},
                "assistant_instructions": "",
                "knowledge_file_ids": [],
                "sink_type": "",
                "sink_config": {},
                "schedule": "*/5 * * * *",
                "notify_user": True,
                "enabled": True,
            }
            )
        )

    @staticmethod
    def _scratchpad_scaffold(*, mode: str, workflow_id: str = "") -> dict[str, Any]:
        return {
            "mode": str(mode or "").strip() or "create",
            "target_workflow_id": str(workflow_id or "").strip(),
            "missing_fields": [],
            "open_questions": [],
            "user_constraints": [],
            "assumptions": [],
            "proposal_summary": "",
            "last_user_confirmable_summary": "",
        }

    @staticmethod
    def _draft_hash(draft_upsert: dict[str, Any]) -> str:
        payload = json.dumps(_safe_dict(draft_upsert), ensure_ascii=False, sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def get_thread_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        include_paused: bool = True,
    ) -> dict[str, Any] | None:
        statuses = ("active", "paused") if include_paused else ("active",)
        return self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=statuses,
        )

    def begin_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        mode: str,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        safe_mode = str(mode or "").strip().lower()
        if safe_mode not in {"create", "edit"}:
            raise ValueError("mode must be create|edit")
        existing_thread_session = self.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=True,
        )
        safe_workflow_id = str(workflow_id or "").strip()
        if existing_thread_session is not None:
            existing_target = str(existing_thread_session.get("target_workflow_id", "") or "").strip()
            if existing_thread_session.get("status") == "paused":
                if safe_workflow_id and existing_target and safe_workflow_id != existing_target:
                    raise ValueError("a paused workflow setup session already exists for this thread")
                return self._store.update_session(
                    session_id=str(existing_thread_session["session_id"]),
                    status="active",
                )
            if safe_workflow_id and existing_target and safe_workflow_id != existing_target:
                raise ValueError("an active workflow setup session already exists for this thread")
            return existing_thread_session

        workflow_snapshot: dict[str, Any] = {}
        if safe_mode == "edit":
            if not safe_workflow_id:
                raise ValueError("workflow_id is required for edit mode")
            workflow_snapshot = self._intake_workflows.get_workflow(
                customer_id=customer_id,
                workflow_id=safe_workflow_id,
            ) or {}
            if not workflow_snapshot:
                raise ValueError("workflow not found")
        draft = self._draft_scaffold()
        if workflow_snapshot:
            draft.update(
                {
                    "name": str(workflow_snapshot.get("name", "") or ""),
                    "channel": str(workflow_snapshot.get("channel", "instagram_dm") or "instagram_dm"),
                    "provider": str(workflow_snapshot.get("provider", "composio") or "composio"),
                    "source_config": _safe_dict(workflow_snapshot.get("source_config")),
                    "intent_description": str(workflow_snapshot.get("intent_description", "") or ""),
                    "required_fields": [str(item or "").strip() for item in _safe_list(workflow_snapshot.get("required_fields")) if str(item or "").strip()],
                    "field_guidance": _safe_dict(workflow_snapshot.get("field_guidance")),
                    "assistant_instructions": str(workflow_snapshot.get("assistant_instructions", "") or ""),
                    "knowledge_file_ids": [str(item or "").strip() for item in _safe_list(workflow_snapshot.get("knowledge_file_ids")) if str(item or "").strip()],
                    "sink_type": str(workflow_snapshot.get("sink_type", "") or ""),
                    "sink_config": _safe_dict(workflow_snapshot.get("sink_config")),
                    "schedule": str(workflow_snapshot.get("schedule", "*/5 * * * *") or "*/5 * * * *"),
                    "notify_user": bool(workflow_snapshot.get("notify_user", True)),
                    "enabled": bool(workflow_snapshot.get("enabled", True)),
                }
            )
            draft = _normalize_schedule_for_channel(draft)
        return self._store.create_session(
            customer_id=customer_id,
            thread_id=thread_id,
            mode=safe_mode,
            target_workflow_id=safe_workflow_id or None,
            target_workflow_snapshot=workflow_snapshot,
            draft_upsert=draft,
            scratchpad=self._scratchpad_scaffold(mode=safe_mode, workflow_id=safe_workflow_id),
        )

    def update_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        draft_patch: dict[str, Any] | None = None,
        scratchpad_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        updated_draft = _merge_draft(_safe_dict(session.get("draft_upsert")), _safe_dict(draft_patch))
        updated_draft = _normalize_local_csv_draft_sink_config(
            _normalize_schedule_for_channel(updated_draft)
        )
        updated_scratchpad = _deep_merge(_safe_dict(session.get("scratchpad")), _safe_dict(scratchpad_patch))
        return self._store.update_session(
            session_id=str(session["session_id"]),
            draft_upsert=updated_draft,
            scratchpad=updated_scratchpad,
            confirmed_draft_hash="",
        )

    def mark_proposed(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        draft_hash = self._draft_hash(_safe_dict(session.get("draft_upsert")))
        return self._store.update_session(
            session_id=str(session["session_id"]),
            last_proposed_draft_hash=draft_hash,
            confirmed_draft_hash="",
        )

    def confirm_current(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        current_hash = self._draft_hash(_safe_dict(session.get("draft_upsert")))
        proposed_hash = str(session.get("last_proposed_draft_hash", "") or "").strip()
        if not proposed_hash:
            raise ValueError("workflow draft has not been proposed yet")
        if current_hash != proposed_hash:
            raise ValueError("workflow draft changed after proposal; propose it again before confirming")
        return self._store.update_session(
            session_id=str(session["session_id"]),
            confirmed_draft_hash=current_hash,
        )

    def commit(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        draft = _safe_dict(session.get("draft_upsert"))
        current_hash = self._draft_hash(draft)
        confirmed_hash = str(session.get("confirmed_draft_hash", "") or "").strip()
        if not confirmed_hash or current_hash != confirmed_hash:
            raise ValueError("workflow draft must be explicitly confirmed before commit")

        mode = str(session.get("mode", "") or "").strip().lower()
        safe_target_workflow_id = str(session.get("target_workflow_id", "") or "").strip()
        target_snapshot = _safe_dict(session.get("target_workflow_snapshot"))
        channel = str(draft.get("channel", "") or "").strip().lower()

        workflow_payload = dict(draft)
        if mode == "edit":
            if channel == "telegram_business_dm":
                if not safe_target_workflow_id:
                    raise ValueError("target_workflow_id is required for Telegram edit mode")
                delete_result = self._intake_workflows.delete_workflow(
                    customer_id=customer_id,
                    workflow_id=safe_target_workflow_id,
                )
                if not bool(delete_result.get("deleted", False)):
                    raise ValueError("failed to delete existing Telegram Business workflow")
                workflow_payload.pop("workflow_id", None)
                created = self._intake_workflows.upsert_workflow(
                    customer_id=customer_id,
                    workflow_id=None,
                    **workflow_payload,
                )
            else:
                target_id = safe_target_workflow_id or str(target_snapshot.get("workflow_id", "") or "").strip()
                if not target_id:
                    raise ValueError("target_workflow_id is required for edit mode")
                created = self._intake_workflows.upsert_workflow(
                    customer_id=customer_id,
                    workflow_id=target_id,
                    **workflow_payload,
                )
        else:
            created = self._intake_workflows.upsert_workflow(
                customer_id=customer_id,
                workflow_id=None,
                **workflow_payload,
            )

        created_workflow_id = str(created.get("workflow_id", "") or "").strip()
        completed = self._store.update_session(
            session_id=str(session["session_id"]),
            status="completed",
            created_or_updated_workflow_id=created_workflow_id,
            completed_at=self._utc_now_iso(),
        )
        completed["workflow"] = created
        return completed

    def pause(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        return self._store.update_session(
            session_id=str(session["session_id"]),
            status="paused",
        )

    def cancel(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active", "paused"),
        )
        if session is None:
            raise ValueError("workflow setup session not found")
        return self._store.update_session(
            session_id=str(session["session_id"]),
            status="cancelled",
        )
