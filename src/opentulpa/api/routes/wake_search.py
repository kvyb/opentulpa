"""Wake queue and web-search route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.integrations.web_search import web_search as run_web_search


def _sanitize_signal_thread_segment(value: str) -> str:
    raw = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    cleaned = cleaned.strip("_")
    return cleaned[:80]


def _derive_signal_thread_id(
    *,
    source: str,
    customer_id: str,
    owner_thread_id: str,
    thread_id: str,
    external_subject_id: str,
    external_conversation_id: str,
) -> str:
    if owner_thread_id:
        return owner_thread_id
    if thread_id:
        return thread_id
    safe_source = str(source or "").strip().lower()
    if safe_source == "manychat":
        subject_segment = _sanitize_signal_thread_segment(external_subject_id)
        if subject_segment:
            return f"inbox_manychat_{subject_segment}"
        conversation_segment = _sanitize_signal_thread_segment(external_conversation_id)
        if conversation_segment:
            return f"inbox_manychat_conv_{conversation_segment}"
    return f"chat-{customer_id}" if customer_id else ""


def register_wake_and_search_routes(
    app: FastAPI,
    *,
    get_wake_queue: Callable[[], Any],
    get_signal_inbox: Callable[[], Any],
    get_skill_store: Callable[[], Any],
    llm_model: str | None,
) -> None:
    """Register wake queue APIs and OpenRouter-backed web search endpoint."""
    _ = llm_model

    envelope_keys = {
        "source",
        "owner_customer_id",
        "owner_thread_id",
        "customer_id",
        "thread_id",
        "event_type",
        "text",
        "payload",
        "dispatch",
        "enqueue_wake",
        "batch_window_seconds",
    }

    async def _ingest_signal_body(body: dict[str, Any]) -> dict[str, Any]:
        source = str(body.get("source", "")).strip()
        owner_customer_id = str(body.get("owner_customer_id", "")).strip()
        owner_thread_id = str(body.get("owner_thread_id", "")).strip()
        customer_id = owner_customer_id or str(body.get("customer_id", "")).strip()
        raw_thread_id = str(body.get("thread_id", "")).strip()
        event_type = str(body.get("event_type", "message")).strip() or "message"
        text = str(body.get("text", "")).strip()
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else None
        if payload is None:
            payload = {key: value for key, value in body.items() if key not in envelope_keys}
        dispatch = body.get("dispatch") if isinstance(body.get("dispatch"), dict) else {}
        external_subject_id = str(body.get("external_subject_id", "")).strip()
        external_conversation_id = str(body.get("external_conversation_id", "")).strip()
        thread_id = _derive_signal_thread_id(
            source=source,
            customer_id=customer_id,
            owner_thread_id=owner_thread_id,
            thread_id=raw_thread_id,
            external_subject_id=external_subject_id,
            external_conversation_id=external_conversation_id,
        )
        if external_subject_id and not str(payload.get("external_subject_id", "")).strip():
            payload = {**payload, "external_subject_id": external_subject_id}
        if external_conversation_id and not str(payload.get("external_conversation_id", "")).strip():
            payload = {**payload, "external_conversation_id": external_conversation_id}
        if external_subject_id and not str(dispatch.get("external_subject_id", "")).strip():
            dispatch = {**dispatch, "external_subject_id": external_subject_id}
        if external_conversation_id and not str(dispatch.get("external_conversation_id", "")).strip():
            dispatch = {**dispatch, "external_conversation_id": external_conversation_id}
        enqueue_wake = body.get("enqueue_wake", True)
        batch_window_seconds = body.get("batch_window_seconds")
        if not source or not customer_id:
            raise ValueError("source and customer_id are required")
        rules = get_signal_inbox().resolve_rule(
            source=source,
            customer_id=customer_id,
            thread_id=thread_id,
        )
        if batch_window_seconds is None:
            batch_window_seconds = int(rules.get("batch_window_seconds", 0) or 0)
        signal = get_signal_inbox().ingest_signal(
            source=source,
            customer_id=customer_id,
            thread_id=thread_id,
            event_type=event_type,
            text=text,
            payload=payload,
            dispatch=dispatch,
            batch_window_seconds=int(batch_window_seconds),
        )
        queue_id = None
        notify = not (
            enqueue_wake is False
            or str(enqueue_wake).strip().lower() in {"0", "false", "no", "off"}
        )
        if notify:
            queue_id = await get_wake_queue().enqueue(
                {
                    "type": "signal_event",
                    "source": source,
                    "customer_id": customer_id,
                    "thread_id": thread_id,
                    "signal_id": signal["id"],
                }
            )
        return {"ok": True, "signal": signal, "queue_id": queue_id, "rule": rules}

    app.state.signal_ingest = _ingest_signal_body

    @app.post("/internal/wake")
    async def internal_wake(request: Request) -> Any:
        """Called by scheduler or external trigger to wake the agent with a payload."""
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"detail": "wake payload must be JSON object"}
            )
        queue_id = await get_wake_queue().enqueue(body)
        return {"ok": True, "queued": True, "queue_id": queue_id}

    @app.get("/internal/wake/queue")
    async def internal_wake_queue_stats() -> Any:
        """Inspect wake queue health and recent entries."""
        return {"ok": True, "queue": get_wake_queue().stats()}

    @app.post("/internal/signals/ingest")
    async def internal_signal_ingest(request: Request) -> Any:
        """Store a normalized external signal and enqueue signal_event processing."""
        body = await request.json()
        try:
            return await _ingest_signal_body(body)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/signals/rules/upsert")
    async def internal_signal_rule_upsert(request: Request) -> Any:
        """Create or update signal wake handling rules."""
        body = await request.json()
        source = str(body.get("source", "")).strip()
        wake_mode = str(body.get("wake_mode", "classifier")).strip()
        customer_id = str(body.get("customer_id", "")).strip()
        thread_id = str(body.get("thread_id", "")).strip()
        batch_window_seconds = body.get("batch_window_seconds", 0)
        auto_reply = body.get("auto_reply", True)
        auto_reply_enabled = not (
            auto_reply is False
            or str(auto_reply).strip().lower() in {"0", "false", "no", "off"}
        )
        handler_skill_name = str(body.get("handler_skill_name", "")).strip()
        guidance_text = str(body.get("guidance_text", "")).strip()
        if auto_reply_enabled and not handler_skill_name:
            return JSONResponse(
                status_code=400,
                content={"detail": "handler_skill_name is required when auto_reply is enabled"},
            )
        if handler_skill_name:
            handler_skill = get_skill_store().get_skill(
                customer_id=customer_id,
                name=handler_skill_name,
                include_files=False,
                include_global=True,
            )
            if handler_skill is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "handler_skill_name does not resolve to an existing user/global skill"
                        )
                    },
                )
        try:
            rule = get_signal_inbox().upsert_rule(
                source=source,
                wake_mode=wake_mode,
                customer_id=customer_id,
                thread_id=thread_id,
                batch_window_seconds=int(batch_window_seconds),
                auto_reply=auto_reply_enabled,
                handler_skill_name=handler_skill_name,
                guidance_text=guidance_text,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "rule": rule}

    @app.get("/internal/signals/rules")
    async def internal_signal_rule_list(
        source: str = "",
        customer_id: str = "",
        thread_id: str = "",
        limit: int = 50,
    ) -> Any:
        """List stored signal rules."""
        return {
            "ok": True,
            "rules": get_signal_inbox().list_rules(
                source=source,
                customer_id=customer_id,
                thread_id=thread_id,
                limit=limit,
            ),
        }

    @app.get("/internal/signals/outbox")
    async def internal_signal_outbox(source: str = "", status: str = "pending", limit: int = 50) -> Any:
        """List queued outbound signal replies for channel adapters to send."""
        return {
            "ok": True,
            "outbox": get_signal_inbox().list_outbox(source=source, status=status, limit=limit),
        }

    @app.post("/internal/signals/outbox/{outbox_id}/sent")
    async def internal_signal_outbox_sent(outbox_id: int) -> Any:
        """Mark an outbound signal reply as sent by a channel adapter."""
        payload = get_signal_inbox().mark_outbound_sent(outbox_id)
        if payload is None:
            return JSONResponse(status_code=404, content={"detail": "outbox entry not found"})
        return {"ok": True, "outbox": payload}

    @app.post("/internal/web_search")
    async def internal_web_search(request: Request) -> Any:
        """Run OpenRouter web search (default: Perplexity Sonar Pro Search)."""
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"detail": "query required"})
        result = await run_web_search(query)
        return {"ok": True, "result": result}
