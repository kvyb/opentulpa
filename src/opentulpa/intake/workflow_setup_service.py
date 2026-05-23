"""Wizard-style setup service for intake workflows."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, cast

from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.intake.workflow_setup_store import (
    SetupSessionMode,
    SetupSessionStatus,
    WorkflowSetupSessionStore,
)

logger = logging.getLogger(__name__)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unique_setup_file_ids(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


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
        if safe_key in {"field_guidance", "business_facts"} and isinstance(value, dict):
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
        sink_config.get("file_path", "") or sink_config.get("filename", "") or ""
    ).strip()
    normalized["sink_config"] = {"file_path": file_path} if file_path else {}
    return normalized


def _normalize_schedule_for_channel(draft: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(draft)
    channel = str(normalized.get("channel", "") or "").strip().lower()
    if channel == "telegram_business_dm":
        normalized["schedule"] = ""
    else:
        normalized["schedule"] = str(normalized.get("schedule", "*/2 * * * *") or "*/2 * * * *")
    return normalized


def _normalize_channel_provider_pair(draft: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(draft)
    channel = str(normalized.get("channel", "") or "").strip().lower()
    provider = str(normalized.get("provider", "") or "").strip().lower()
    if provider in {"telegram", "telegram_business"}:
        provider = "telegram_bot_api"
    if channel == "telegram_business_dm" and provider != "telegram_bot_api":
        provider = "telegram_bot_api"
    if channel == "instagram_dm" and provider != "composio":
        provider = "composio"
    if not channel and provider == "telegram_bot_api":
        channel = "telegram_business_dm"
    if not channel and provider == "composio":
        channel = "instagram_dm"
    normalized["channel"] = channel
    normalized["provider"] = provider
    return normalized


def _validate_draft_patch_shape(draft_patch: dict[str, Any] | None) -> None:
    if not isinstance(draft_patch, dict):
        return
    for nested_key in ("draft", "draft_upsert", "workflow", "workflow_upsert"):
        if isinstance(draft_patch.get(nested_key), dict):
            raise ValueError(
                "draft_patch must contain workflow fields directly; "
                f"move fields from draft_patch.{nested_key} into draft_patch"
            )


def _source_intent_patch(draft_patch: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(draft_patch, dict):
        return {}
    source_patch: dict[str, Any] = {}
    if "channel" in draft_patch:
        source_patch["current_requested_channel"] = str(
            draft_patch.get("channel", "") or ""
        ).strip().lower()
    if "provider" in draft_patch:
        source_patch["current_requested_provider"] = str(
            draft_patch.get("provider", "") or ""
        ).strip().lower()
    if not source_patch:
        return {}
    normalized = _normalize_channel_provider_pair(
        {
            "channel": source_patch.get("current_requested_channel", ""),
            "provider": source_patch.get("current_requested_provider", ""),
        }
    )
    if "current_requested_channel" in source_patch:
        source_patch["current_requested_channel"] = str(normalized.get("channel", "") or "")
    if "current_requested_provider" in source_patch:
        source_patch["current_requested_provider"] = str(normalized.get("provider", "") or "")
    return source_patch


def _source_intent_mismatches(
    *,
    draft: dict[str, Any],
    scratchpad: dict[str, Any],
) -> list[str]:
    requested_channel = str(scratchpad.get("current_requested_channel", "") or "").strip().lower()
    requested_provider = str(scratchpad.get("current_requested_provider", "") or "").strip().lower()
    if not requested_channel and not requested_provider:
        return []
    normalized_draft = _normalize_channel_provider_pair(draft)
    draft_channel = str(normalized_draft.get("channel", "") or "").strip().lower()
    draft_provider = str(normalized_draft.get("provider", "") or "").strip().lower()
    blockers: list[str] = []
    if requested_channel and draft_channel != requested_channel:
        blockers.append(
            "User currently requested "
            f"channel={requested_channel}, but draft channel is {draft_channel or 'missing'}. "
            "Update draft_patch.channel before proposing or finalizing."
        )
    if requested_provider and draft_provider != requested_provider:
        blockers.append(
            "User currently requested "
            f"provider={requested_provider}, but draft provider is {draft_provider or 'missing'}. "
            "Update draft_patch.provider before proposing or finalizing."
        )
    return blockers


def _normalize_reply_mode_for_origin(draft: dict[str, Any], *, thread_id: str) -> dict[str, Any]:
    _ = thread_id
    normalized = dict(draft)
    normalized["reply_mode"] = "auto"
    return normalized


def _compact_preflight_for_scratchpad(preflight: dict[str, Any]) -> dict[str, Any]:
    sink_preflight = _safe_dict(preflight.get("sink_preflight"))
    dry_run = _safe_dict(sink_preflight.get("dry_run"))
    return {
        "ok": bool(preflight.get("ok", False)),
        "status": str(preflight.get("status", "") or ""),
        "next_action": str(preflight.get("next_action", "") or ""),
        "commit_blockers": _safe_list(preflight.get("commit_blockers")),
        "errors": _safe_list(preflight.get("errors")),
        "warnings": _safe_list(preflight.get("warnings")),
        "follow_up_questions": _safe_list(preflight.get("follow_up_questions")),
        "draft_hash": str(preflight.get("draft_hash", "") or ""),
        "cache_hit": bool(preflight.get("cache_hit", False)),
        "sink_type": str(sink_preflight.get("sink_type", "") or ""),
        "dry_run": {
            "mode": str(dry_run.get("mode", "") or ""),
            "will_execute": bool(dry_run.get("will_execute", False)),
            "tool_slug": str(dry_run.get("tool_slug", "") or ""),
            "target": _safe_dict(dry_run.get("target")),
        },
    }


class WorkflowSetupService:
    """Owns workflow-setup session lifecycle and commit semantics."""

    def __init__(
        self,
        *,
        store: WorkflowSetupSessionStore,
        intake_workflows: IntakeWorkflowService,
        knowledge_service: Any | None = None,
    ) -> None:
        self._store = store
        self._intake_workflows = intake_workflows
        self._knowledge_service = knowledge_service
        self._preflight_lock_guard = threading.Lock()
        self._preflight_lock_keys: set[str] = set()

    @staticmethod
    def _draft_scaffold() -> dict[str, Any]:
        return _normalize_local_csv_draft_sink_config(
            _normalize_schedule_for_channel(
                {
                    "name": "",
                    "channel": "",
                    "provider": "",
                    "source_config": {},
                    "intent_description": "",
                    "required_fields": [],
                    "field_guidance": {},
                    "assistant_instructions": "",
                    "business_facts": {},
                    "knowledge_file_ids": [],
                    "sink_type": "",
                    "sink_config": {},
                    "schedule": "*/2 * * * *",
                    "notify_user": True,
                    "enabled": True,
                    "reply_mode": "",
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
            "source_file_ids": [],
            "knowledge_source_file_ids": [],
            "knowledge_last_index": {},
            "knowledge_last_preflight": {},
            "candidate_files": [],
            "proposal_summary": "",
            "last_user_confirmable_summary": "",
            "current_requested_channel": "",
            "current_requested_provider": "",
        }

    @staticmethod
    def _draft_hash(draft_upsert: dict[str, Any]) -> str:
        payload = json.dumps(_safe_dict(draft_upsert), ensure_ascii=False, sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _cached_ready_preflight(session: dict[str, Any], *, draft_hash: str) -> dict[str, Any]:
        scratchpad = _safe_dict(session.get("scratchpad"))
        last_preflight = _safe_dict(scratchpad.get("last_preflight"))
        if (
            str(last_preflight.get("draft_hash", "") or "").strip() != draft_hash
            or not bool(last_preflight.get("ok", False))
            or str(last_preflight.get("status", "") or "").strip() != "ready"
        ):
            return {}
        cached = dict(last_preflight)
        cached["cache_hit"] = True
        cached["draft_hash"] = draft_hash
        knowledge_preflight = _safe_dict(scratchpad.get("knowledge_last_preflight"))
        if knowledge_preflight:
            cached["business_knowledge_preflight"] = knowledge_preflight
        return cached

    def _claim_preflight_lock(self, *, session_id: str, draft_hash: str) -> bool:
        lock_key = f"{session_id}:{draft_hash}"
        with self._preflight_lock_guard:
            if lock_key in self._preflight_lock_keys:
                return False
            self._preflight_lock_keys.add(lock_key)
        return True

    def _release_preflight_lock(self, *, session_id: str, draft_hash: str) -> None:
        lock_key = f"{session_id}:{draft_hash}"
        with self._preflight_lock_guard:
            self._preflight_lock_keys.discard(lock_key)

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
            statuses=cast(tuple[SetupSessionStatus, ...], statuses),
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
        setup_mode = cast(SetupSessionMode, safe_mode)
        existing_thread_session = self.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=True,
        )
        safe_workflow_id = str(workflow_id or "").strip()
        if existing_thread_session is not None:
            existing_target = str(
                existing_thread_session.get("target_workflow_id", "") or ""
            ).strip()
            if existing_thread_session.get("status") == "paused":
                if safe_workflow_id and existing_target and safe_workflow_id != existing_target:
                    raise ValueError(
                        "a paused workflow setup session already exists for this thread"
                    )
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
            workflow_snapshot = (
                self._intake_workflows.get_workflow(
                    customer_id=customer_id,
                    workflow_id=safe_workflow_id,
                )
                or {}
            )
            if not workflow_snapshot:
                raise ValueError("workflow not found")
        draft = self._draft_scaffold()
        if workflow_snapshot:
            draft.update(
                {
                    "name": str(workflow_snapshot.get("name", "") or ""),
                    "channel": str(
                        workflow_snapshot.get("channel", "instagram_dm") or "instagram_dm"
                    ),
                    "provider": str(workflow_snapshot.get("provider", "composio") or "composio"),
                    "source_config": _safe_dict(workflow_snapshot.get("source_config")),
                    "intent_description": str(
                        workflow_snapshot.get("intent_description", "") or ""
                    ),
                    "required_fields": [
                        str(item or "").strip()
                        for item in _safe_list(workflow_snapshot.get("required_fields"))
                        if str(item or "").strip()
                    ],
                    "field_guidance": _safe_dict(workflow_snapshot.get("field_guidance")),
                    "assistant_instructions": str(
                        workflow_snapshot.get("assistant_instructions", "") or ""
                    ),
                    "business_facts": _safe_dict(workflow_snapshot.get("business_facts")),
                    "knowledge_file_ids": [
                        str(item or "").strip()
                        for item in _safe_list(workflow_snapshot.get("knowledge_file_ids"))
                        if str(item or "").strip()
                    ],
                    "sink_type": str(workflow_snapshot.get("sink_type", "") or ""),
                    "sink_config": _safe_dict(workflow_snapshot.get("sink_config")),
                    "schedule": str(
                        workflow_snapshot.get("schedule", "*/2 * * * *") or "*/2 * * * *"
                    ),
                    "notify_user": bool(workflow_snapshot.get("notify_user", True)),
                    "enabled": bool(workflow_snapshot.get("enabled", True)),
                    "reply_mode": str(workflow_snapshot.get("reply_mode", "auto") or "auto"),
                }
            )
            draft = _normalize_schedule_for_channel(_normalize_channel_provider_pair(draft))
        draft = _normalize_reply_mode_for_origin(draft, thread_id=thread_id)
        return self._store.create_session(
            customer_id=customer_id,
            thread_id=thread_id,
            mode=setup_mode,
            target_workflow_id=safe_workflow_id or None,
            target_workflow_snapshot=workflow_snapshot,
            draft_upsert=draft,
            scratchpad=self._scratchpad_scaffold(mode=setup_mode, workflow_id=safe_workflow_id),
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
        _validate_draft_patch_shape(draft_patch)
        updated_draft = _merge_draft(
            _safe_dict(session.get("draft_upsert")), _safe_dict(draft_patch)
        )
        updated_draft = _normalize_local_csv_draft_sink_config(
            _normalize_schedule_for_channel(_normalize_channel_provider_pair(updated_draft))
        )
        updated_draft = _normalize_reply_mode_for_origin(updated_draft, thread_id=thread_id)
        updated_scratchpad = _deep_merge(
            _deep_merge(
                _safe_dict(session.get("scratchpad")),
                _safe_dict(scratchpad_patch),
            ),
            _source_intent_patch(draft_patch),
        )
        return self._store.update_session(
            session_id=str(session["session_id"]),
            draft_upsert=updated_draft,
            scratchpad=updated_scratchpad,
            confirmed_draft_hash="",
        )

    def preflight_current(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            raise ValueError("active workflow setup session not found")
        draft = _normalize_reply_mode_for_origin(
            _normalize_channel_provider_pair(_safe_dict(session.get("draft_upsert"))),
            thread_id=thread_id,
        )
        intent_blockers = _source_intent_mismatches(
            draft=draft,
            scratchpad=_safe_dict(session.get("scratchpad")),
        )
        if intent_blockers:
            blocked = {
                "ok": False,
                "status": "needs_clarification",
                "next_action": "update_draft_source_to_match_latest_owner_request",
                "commit_blockers": intent_blockers,
                "errors": intent_blockers,
                "warnings": [],
                "follow_up_questions": intent_blockers,
                "draft_hash": self._draft_hash(draft),
                "cache_hit": False,
            }
            blocked_session = dict(session)
            blocked_session["preflight"] = blocked
            return blocked_session
        session_id = str(session.get("session_id", "") or "").strip()
        draft_hash = self._draft_hash(draft)
        cached_preflight = self._cached_ready_preflight(session, draft_hash=draft_hash)
        if cached_preflight:
            cached_session = dict(session)
            cached_session["preflight"] = cached_preflight
            logger.info(
                "workflow_setup.preflight cache_hit customer_id=%s thread_id=%s session_id=%s draft_hash=%s",
                customer_id,
                thread_id,
                session_id,
                draft_hash,
            )
            return cached_session
        if session_id and not self._claim_preflight_lock(
            session_id=session_id, draft_hash=draft_hash
        ):
            running = {
                "ok": False,
                "status": "running",
                "next_action": "wait_for_preflight",
                "commit_blockers": [],
                "errors": [],
                "warnings": ["Workflow setup preflight is already running for the current draft."],
                "follow_up_questions": [],
                "draft_hash": draft_hash,
                "cache_hit": False,
            }
            running_session = dict(session)
            running_session["preflight"] = running
            return running_session
        target_workflow_id = str(session.get("target_workflow_id", "") or "").strip() or None
        try:
            preflight = self._intake_workflows.preflight_workflow_payload(
                customer_id=customer_id,
                workflow_id=target_workflow_id,
                name=str(draft.get("name", "") or ""),
                channel=str(draft.get("channel", "") or ""),
                provider=str(draft.get("provider", "") or ""),
                source_config=_safe_dict(draft.get("source_config")),
                intent_description=str(draft.get("intent_description", "") or ""),
                required_fields=_safe_list(draft.get("required_fields")),
                field_guidance=_safe_dict(draft.get("field_guidance")),
                assistant_instructions=str(draft.get("assistant_instructions", "") or ""),
                business_facts=_safe_dict(draft.get("business_facts")),
                knowledge_file_ids=_safe_list(draft.get("knowledge_file_ids")),
                sink_type=str(draft.get("sink_type", "") or ""),
                sink_config=_safe_dict(draft.get("sink_config")),
                schedule=str(draft.get("schedule", "*/2 * * * *") or "*/2 * * * *"),
                notify_user=bool(draft.get("notify_user", True)),
                enabled=bool(draft.get("enabled", True)),
                reply_mode=str(draft.get("reply_mode", "auto") or "auto"),
            )
            knowledge_preflight = self._preflight_knowledge_scope(
                customer_id=customer_id,
                session=session,
                draft=draft,
            )
            if knowledge_preflight:
                preflight["business_knowledge_preflight"] = knowledge_preflight
                warnings = _safe_list(preflight.get("warnings"))
                preflight["warnings"] = [
                    *warnings,
                    *[
                        str(item).strip()
                        for item in _safe_list(knowledge_preflight.get("warnings"))
                        if str(item).strip()
                    ],
                ]
                if not bool(knowledge_preflight.get("ok", False)):
                    preflight["ok"] = False
                    preflight["status"] = "needs_clarification"
                    questions = _safe_list(preflight.get("follow_up_questions"))
                    questions.append(
                        "The uploaded business knowledge could not be grounded from source files. Please provide a text, spreadsheet, PDF, DOCX, CSV, or clearer source for the workflow."
                    )
                    preflight["follow_up_questions"] = questions
            commit_blockers = [
                str(item).strip()
                for item in [
                    *_safe_list(preflight.get("errors")),
                    *_safe_list(preflight.get("follow_up_questions")),
                ]
                if str(item).strip()
            ]
            if (
                bool(preflight.get("ok", False))
                and str(preflight.get("status", "") or "") == "ready"
            ):
                preflight["commit_blockers"] = []
                preflight["next_action"] = (
                    "finalize_confirmation_if_owner_confirmed_else_mark_proposed"
                )
            else:
                preflight["commit_blockers"] = commit_blockers
                preflight["next_action"] = "ask_preflight_blocker"
            normalized_draft = (
                _safe_dict(preflight.get("normalized_draft"))
                if isinstance(preflight.get("normalized_draft"), dict)
                else draft
            )
            result_draft_hash = self._draft_hash(normalized_draft)
            preflight["draft_hash"] = result_draft_hash
            preflight["cache_hit"] = False
            scratchpad = _deep_merge(
                _safe_dict(session.get("scratchpad")),
                {
                    "last_preflight": _compact_preflight_for_scratchpad(preflight),
                    "last_preflight_draft_hash": result_draft_hash,
                    "knowledge_last_index": _safe_dict(knowledge_preflight.get("index"))
                    if knowledge_preflight
                    else {},
                    "knowledge_last_preflight": knowledge_preflight or {},
                },
            )
            update_kwargs: dict[str, Any] = {"scratchpad": scratchpad}
            if bool(preflight.get("ok", False)) and isinstance(
                preflight.get("normalized_draft"), dict
            ):
                update_kwargs["draft_upsert"] = _safe_dict(preflight.get("normalized_draft"))
                update_kwargs["confirmed_draft_hash"] = ""
            updated = self._store.update_session(
                session_id=str(session["session_id"]),
                **update_kwargs,
            )
            updated["preflight"] = preflight
            return updated
        finally:
            if session_id:
                self._release_preflight_lock(session_id=session_id, draft_hash=draft_hash)

    def finalize_confirmation(
        self,
        *,
        customer_id: str,
        thread_id: str,
        draft_patch: dict[str, Any] | None = None,
        scratchpad_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply final edits, validate, mark, confirm, and commit in one explicit owner-confirmation path."""

        if isinstance(draft_patch, dict) or isinstance(scratchpad_patch, dict):
            _validate_draft_patch_shape(draft_patch)
            self.update_session(
                customer_id=customer_id,
                thread_id=thread_id,
                draft_patch=draft_patch,
                scratchpad_patch=scratchpad_patch,
            )
        session = self._store.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            statuses=("active",),
        )
        if session is None:
            completed = self._store.get_thread_session(
                customer_id=customer_id,
                thread_id=thread_id,
                statuses=("completed",),
            )
            if completed and str(completed.get("created_or_updated_workflow_id", "") or "").strip():
                completed["already_completed"] = True
                return completed
            raise ValueError("active workflow setup session not found")

        preflight_session = self.preflight_current(customer_id=customer_id, thread_id=thread_id)
        preflight = _safe_dict(preflight_session.get("preflight"))
        if (
            not bool(preflight.get("ok", False))
            or str(preflight.get("status", "") or "") != "ready"
        ):
            blockers = (
                _safe_list(preflight.get("commit_blockers"))
                or _safe_list(preflight.get("follow_up_questions"))
                or _safe_list(preflight.get("errors"))
            )
            blocker = "; ".join(str(item).strip() for item in blockers if str(item).strip())
            if not blocker:
                blocker = "workflow draft is not ready to commit"
            raise ValueError(blocker)

        self.mark_proposed(customer_id=customer_id, thread_id=thread_id)
        self.confirm_current(customer_id=customer_id, thread_id=thread_id)
        committed = self.commit(customer_id=customer_id, thread_id=thread_id)
        committed["preflight"] = preflight
        return committed

    def propose_current(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        """Preflight the current draft and mark it proposed when it is ready."""

        preflight_session = self.preflight_current(customer_id=customer_id, thread_id=thread_id)
        preflight = _safe_dict(preflight_session.get("preflight"))
        if (
            not bool(preflight.get("ok", False))
            or str(preflight.get("status", "") or "") != "ready"
        ):
            return preflight_session
        proposed = self.mark_proposed(customer_id=customer_id, thread_id=thread_id)
        proposed["preflight"] = preflight
        return proposed

    def _preflight_knowledge_scope(
        self,
        *,
        customer_id: str,
        session: dict[str, Any],
        draft: dict[str, Any],
    ) -> dict[str, Any]:
        knowledge = self._knowledge_service
        if knowledge is None:
            return {}
        scratchpad = _safe_dict(session.get("scratchpad"))
        file_ids = _unique_setup_file_ids(
            [
                *_safe_list(draft.get("knowledge_file_ids")),
                *_safe_list(scratchpad.get("source_file_ids")),
                *_safe_list(scratchpad.get("knowledge_source_file_ids")),
            ]
        )
        if not file_ids:
            return {}
        session_id = str(session.get("session_id", "") or "").strip()
        if not session_id:
            return {}
        index_result = knowledge.index_sources(
            customer_id=customer_id,
            scope_type="workflow_setup",
            scope_id=session_id,
            file_ids=file_ids,
        )
        goal = " ".join(
            item
            for item in [
                str(draft.get("name", "") or "").strip(),
                str(draft.get("intent_description", "") or "").strip(),
                str(draft.get("assistant_instructions", "") or "").strip(),
                " ".join(
                    str(item or "").strip() for item in _safe_list(draft.get("required_fields"))
                ),
            ]
            if item
        )
        preflight = knowledge.preflight_scope(
            customer_id=customer_id,
            scope_type="workflow_setup",
            scope_id=session_id,
            workflow_goal=goal,
        )
        return {"index": index_result, **preflight}

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
            raise ValueError(
                "workflow draft changed after proposal; propose it again before confirming"
            )
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
        draft = _normalize_reply_mode_for_origin(
            _normalize_channel_provider_pair(_safe_dict(session.get("draft_upsert"))),
            thread_id=thread_id,
        )
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
                target_id = (
                    safe_target_workflow_id
                    or str(target_snapshot.get("workflow_id", "") or "").strip()
                )
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
        self._promote_knowledge_scope(
            customer_id=customer_id,
            session=session,
            workflow=created,
        )
        completed = self._store.update_session(
            session_id=str(session["session_id"]),
            status="completed",
            created_or_updated_workflow_id=created_workflow_id,
            completed_at=self._utc_now_iso(),
        )
        completed["workflow"] = created
        return completed

    def _promote_knowledge_scope(
        self,
        *,
        customer_id: str,
        session: dict[str, Any],
        workflow: dict[str, Any],
    ) -> None:
        knowledge = self._knowledge_service
        if knowledge is None:
            return
        session_id = str(session.get("session_id", "") or "").strip()
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        if not session_id or not workflow_id:
            return
        result = knowledge.promote_scope(
            customer_id=customer_id,
            source_scope_type="workflow_setup",
            source_scope_id=session_id,
            target_scope_type="intake_workflow",
            target_scope_id=workflow_id,
        )
        if int(result.get("source_count") or 0) > 0:
            return
        file_ids = _unique_setup_file_ids(_safe_list(workflow.get("knowledge_file_ids")))
        if not file_ids:
            return
        with suppress(Exception):
            knowledge.index_sources(
                customer_id=customer_id,
                scope_type="intake_workflow",
                scope_id=workflow_id,
                file_ids=file_ids,
            )

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
