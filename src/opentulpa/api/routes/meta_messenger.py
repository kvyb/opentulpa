"""Authenticated Meta Messenger webhook ingress for Facebook Page messages."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request, Response

logger = logging.getLogger(__name__)
_MAX_BODY_BYTES = 1_000_000
_MAX_EVENTS_PER_REQUEST = 100


class MetaMessengerEventDispatcher(Protocol):
    async def __call__(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        source_event_id: str,
        event_type: str,
        source: str,
        authenticated: bool,
        payload: Mapping[str, Any] | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class MetaMessengerMessageEvent:
    source_event_id: str
    payload: dict[str, Any]


def _verify_signature(*, body: bytes, signature: str, app_secret: str) -> bool:
    algorithm, separator, supplied_digest = str(signature or "").partition("=")
    if separator != "=" or algorithm != "sha256" or len(supplied_digest) != 64:
        return False
    expected_digest = hmac.new(
        app_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_digest.lower(), expected_digest)


def _message_events(body: dict[str, Any]) -> list[MetaMessengerMessageEvent]:
    if body.get("object") != "page":
        return []
    events: list[MetaMessengerMessageEvent] = []
    entries = body.get("entry")
    if not isinstance(entries, list):
        return events
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page_id = str(entry.get("id") or "").strip()
        messaging = entry.get("messaging")
        if not isinstance(messaging, list):
            continue
        for item in messaging:
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if not isinstance(message, dict) or bool(message.get("is_echo")):
                continue
            sender = item.get("sender")
            recipient = item.get("recipient")
            sender_id = str(sender.get("id") or "").strip() if isinstance(sender, dict) else ""
            recipient_id = (
                str(recipient.get("id") or "").strip()
                if isinstance(recipient, dict)
                else ""
            )
            message_id = str(message.get("mid") or "").strip()
            if not message_id or not sender_id:
                continue
            payload: dict[str, Any] = {
                "page_id": page_id or recipient_id,
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_id": message_id,
                "timestamp": item.get("timestamp"),
            }
            text = message.get("text")
            if isinstance(text, str):
                payload["text"] = text
            attachments = message.get("attachments")
            if isinstance(attachments, list):
                payload["attachments"] = attachments
            quick_reply = message.get("quick_reply")
            if isinstance(quick_reply, dict):
                payload["quick_reply"] = quick_reply
            reply_to = message.get("reply_to")
            if isinstance(reply_to, dict):
                payload["reply_to"] = reply_to
            events.append(
                MetaMessengerMessageEvent(
                    source_event_id=message_id,
                    payload=payload,
                )
            )
            if len(events) > _MAX_EVENTS_PER_REQUEST:
                raise ValueError("too many Meta Messenger events")
    return events


def register_meta_messenger_routes(
    app: FastAPI,
    *,
    get_dispatcher: Callable[[], MetaMessengerEventDispatcher | None],
    tenant_id: str,
    trigger_id: str,
    verify_token: str | None,
    app_secret: str | None,
) -> None:
    """Verify Meta requests, acknowledge quickly, and dispatch Page message events."""

    configured_tenant_id = str(tenant_id or "").strip()
    configured_trigger_id = str(trigger_id or "").strip()

    def require_configuration() -> tuple[str, str, str, str]:
        configured_verify_token = str(verify_token or "").strip()
        configured_app_secret = str(app_secret or "").strip()
        if not all(
            (
                configured_tenant_id,
                configured_trigger_id,
                configured_verify_token,
                configured_app_secret,
            )
        ):
            raise HTTPException(
                status_code=503,
                detail="Meta Messenger webhook is not configured",
            )
        return (
            configured_tenant_id,
            configured_trigger_id,
            configured_verify_token,
            configured_app_secret,
        )

    async def dispatch_safely(
        dispatcher: MetaMessengerEventDispatcher,
        event: MetaMessengerMessageEvent,
        resolved_tenant_id: str,
        resolved_trigger_id: str,
    ) -> None:
        try:
            await dispatcher(
                tenant_id=resolved_tenant_id,
                trigger_id=resolved_trigger_id,
                source_event_id=event.source_event_id,
                event_type="message_received",
                source="meta_messenger",
                authenticated=True,
                payload=event.payload,
            )
        except Exception as exc:
            logger.error(
                "Meta Messenger event dispatch failed: error_type=%s",
                type(exc).__name__,
            )

    @app.get("/webhook/meta/messenger", include_in_schema=False)
    async def verify_meta_messenger_webhook(
        hub_mode: str | None = Query(default=None, alias="hub.mode"),
        hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
        hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    ) -> Response:
        _, _, configured_verify_token, _ = require_configuration()
        if (
            hub_mode != "subscribe"
            or hub_verify_token is None
            or not hmac.compare_digest(hub_verify_token, configured_verify_token)
        ):
            raise HTTPException(status_code=403, detail="invalid Meta verify token")
        if hub_challenge is None:
            raise HTTPException(status_code=400, detail="missing Meta challenge")
        return Response(content=hub_challenge, media_type="text/plain", status_code=200)

    @app.post("/webhook/meta/messenger", include_in_schema=False)
    async def receive_meta_messenger_webhook(request: Request) -> Response:
        (
            resolved_tenant_id,
            resolved_trigger_id,
            _,
            configured_app_secret,
        ) = require_configuration()
        body_bytes = await request.body()
        if len(body_bytes) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Meta webhook body is too large")
        signature = str(request.headers.get("x-hub-signature-256") or "").strip()
        if not _verify_signature(
            body=body_bytes,
            signature=signature,
            app_secret=configured_app_secret,
        ):
            raise HTTPException(status_code=401, detail="invalid Meta webhook signature")
        try:
            body = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid Meta webhook payload") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid Meta webhook payload")
        try:
            events = _message_events(body)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        dispatcher = get_dispatcher()
        if events and dispatcher is None:
            raise HTTPException(status_code=503, detail="Meta Messenger dispatcher unavailable")
        if dispatcher is not None:
            for event in events:
                await dispatch_safely(
                    dispatcher,
                    event,
                    resolved_tenant_id,
                    resolved_trigger_id,
                )
        return Response(content="EVENT_RECEIVED", media_type="text/plain", status_code=200)


__all__ = [
    "MetaMessengerEventDispatcher",
    "MetaMessengerMessageEvent",
    "register_meta_messenger_routes",
]
