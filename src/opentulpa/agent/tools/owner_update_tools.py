"""Owner update tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id, require_thread_id


def register_owner_update_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def send_owner_update(message: str, dedupe_key: str = "") -> Any:
        """Send a short interim update to the current owner/support Telegram chat.

        Use only during live owner/support turns when you will continue working
        with tools. This is for long-running interactive or workflow setup work,
        not final answers, inbound lead replies, routine wakes, or background
        event notifications.
        """
        require_customer_id(runtime)
        require_thread_id(runtime)
        safe_message = str(message or "").strip()
        if not safe_message:
            return {"ok": False, "sent": False, "reason": "empty_message"}
        if len(safe_message) > 500:
            safe_message = safe_message[:497].rstrip() + "..."
        emitter = getattr(runtime, "emit_interactive_update", None)
        if not callable(emitter):
            return {"ok": False, "sent": False, "reason": "interactive_update_unavailable"}
        return await emitter(
            text=safe_message,
            dedupe_key=str(dedupe_key or "").strip(),
        )

    return {
        "send_owner_update": send_owner_update,
    }
