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
