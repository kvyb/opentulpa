"""Application boundary for intake workflow setup mode routing."""

from __future__ import annotations

from typing import Any


class WorkflowSetupOrchestrator:
    """Exposes thread-level workflow setup state to chat orchestration."""

    def __init__(self, *, setup_service: Any | None) -> None:
        self._setup_service = setup_service

    def thread_status(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        service = self._setup_service
        if service is None:
            return {"status": "none"}
        session = service.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=True,
        )
        if session is None:
            return {"status": "none"}
        safe_status = str(session.get("status", "") or "").strip().lower()
        if safe_status == "active":
            return {"status": "active", "session": session}
        if safe_status == "paused":
            return {"status": "paused", "session": session}
        return {"status": "none"}

    @staticmethod
    def _looks_like_confirmable_proposal(reply_text: str) -> bool:
        text = " ".join(str(reply_text or "").strip().lower().split())
        if not text:
            return False
        blockers = (
            "before i propose",
            "before i can propose",
            "before proposing",
            "clarifying question",
            "one question",
            "one more question",
        )
        if any(item in text for item in blockers):
            return False
        has_proposal = any(
            item in text
            for item in (
                "workflow proposal",
                "proposed workflow",
                "here's the proposal",
                "here is the proposal",
                "proposed setup",
            )
        )
        has_confirmation_request = any(
            item in text
            for item in (
                "confirm",
                "approve",
                "save",
                "activate",
                "commit",
                "does this look right",
            )
        )
        return has_proposal and has_confirmation_request

    def after_reply(self, *, customer_id: str, thread_id: str, reply_text: str) -> dict[str, Any]:
        service = self._setup_service
        if service is None or not self._looks_like_confirmable_proposal(reply_text):
            return {"marked": False}
        session = service.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=False,
        )
        if session is None:
            return {"marked": False}
        if str(session.get("confirmed_draft_hash", "") or "").strip():
            return {"marked": False}
        updated = service.mark_proposed(customer_id=customer_id, thread_id=thread_id)
        return {"marked": True, "session": updated}
