"""Telegram webhook route registration."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from opentulpa.core.shutdown_drain import ShutdownDrainingError
from opentulpa.interfaces.telegram.client import (
    parse_telegram_callback_query,
    parse_telegram_update,
)
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.tasks.sandbox import PROJECT_ROOT

logger = logging.getLogger(__name__)
_TELEGRAM_BUSINESS_UPDATE_KEYS = (
    "business_connection",
    "business_message",
    "edited_business_message",
    "deleted_business_messages",
)


def _telegram_webhook_log_path(*, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).astimezone(UTC).date().isoformat()
    return (
        PROJECT_ROOT / ".opentulpa" / "logs" / "webhooks" / f"telegram-webhook-{stamp}.jsonl"
    ).resolve()


def _top_level_update_keys(body: dict[str, Any]) -> list[str]:
    return sorted(str(key) for key in body if key != "message")


def _write_telegram_webhook_event(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    try:
        path = _telegram_webhook_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    except Exception:
        logger.debug("failed to write telegram webhook diagnostic event", exc_info=True)


def _track_background_task(app: FastAPI, task: asyncio.Task[Any]) -> None:
    tasks = getattr(app.state, "telegram_webhook_tasks", None)
    if not isinstance(tasks, set):
        tasks = set()
        app.state.telegram_webhook_tasks = tasks
    tasks.add(task)

    def _on_done(done_task: asyncio.Task[Any]) -> None:
        tasks.discard(done_task)
        with suppress(asyncio.CancelledError):
            exc = done_task.exception()
            if exc is not None:
                logger.error(
                    "telegram webhook background handler failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    task.add_done_callback(_on_done)


def register_telegram_webhook_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    get_telegram_business: Callable[[], Any],
    get_intake_workflows: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    get_shutdown_drain: Callable[[], Any] | None = None,
) -> None:
    """Register Telegram webhook routes."""

    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request) -> Response:
        drain = get_shutdown_drain() if get_shutdown_drain is not None else None
        if drain is not None and bool(getattr(drain, "draining", False)):
            return JSONResponse(status_code=503, content={"detail": "instance draining"})
        if not settings.telegram_bot_token:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        expected_secret = str(settings.telegram_webhook_secret or "").strip()
        if not expected_secret:
            return JSONResponse(
                status_code=503,
                content={"detail": "telegram webhook secret not configured"},
            )
        incoming_secret = str(request.headers.get("x-telegram-bot-api-secret-token", "") or "").strip()
        if not hmac.compare_digest(incoming_secret, expected_secret):
            _write_telegram_webhook_event(
                "rejected_invalid_secret",
                has_secret_header=bool(incoming_secret),
                client_host=str(getattr(request.client, "host", "") or ""),
            )
            logger.warning(
                "telegram webhook rejected invalid secret: has_secret_header=%s client_host=%s",
                bool(incoming_secret),
                str(getattr(request.client, "host", "") or ""),
            )
            return JSONResponse(status_code=403, content={"detail": "invalid telegram secret"})
        body = await request.json()
        update_keys = _top_level_update_keys(body) if isinstance(body, dict) else []
        turn_context = None
        if drain is not None and hasattr(drain, "active_turn"):
            turn_context = drain.active_turn()
            try:
                await turn_context.__aenter__()
            except ShutdownDrainingError:
                return JSONResponse(status_code=503, content={"detail": "instance draining"})
        _write_telegram_webhook_event(
            "accepted",
            update_id=body.get("update_id") if isinstance(body, dict) else None,
            update_keys=update_keys,
            has_business_update=any(key in body for key in _TELEGRAM_BUSINESS_UPDATE_KEYS)
            if isinstance(body, dict)
            else False,
        )

        # Return 200 before long-running agent work so Telegram does not retry the same update.
        _track_background_task(
            app,
            asyncio.create_task(
                _telegram_background_handler(body=body, turn_context=turn_context)
            ),
        )
        return Response(status_code=200)

    async def _telegram_background_handler(
        body: dict[str, Any],
        turn_context: Any | None = None,
    ) -> None:
        try:
            await _run_telegram_background_handler(body)
        finally:
            if turn_context is not None:
                await turn_context.__aexit__(None, None, None)

    async def _run_telegram_background_handler(body: dict[str, Any]) -> None:
        business_result = get_telegram_business().ingest_update(body)
        if bool(business_result.get("handled")):
            _write_telegram_webhook_event(
                "business_update_handled",
                update_id=body.get("update_id"),
                kind=str(business_result.get("kind", "") or ""),
                business_connection_id=str(business_result.get("business_connection_id", "") or ""),
                customer_id=str(business_result.get("customer_id", "") or ""),
                chat_id=str(business_result.get("chat_id", "") or ""),
                message_id=str(business_result.get("message_id", "") or ""),
                trigger_workflows=bool(business_result.get("trigger_workflows")),
            )
            logger.info(
                "telegram business update handled: kind=%s business_connection_id=%s "
                "customer_id=%s trigger_workflows=%s",
                str(business_result.get("kind", "") or ""),
                str(business_result.get("business_connection_id", "") or ""),
                str(business_result.get("customer_id", "") or ""),
                bool(business_result.get("trigger_workflows")),
            )
            if bool(business_result.get("trigger_workflows")):
                customer_id = str(business_result.get("customer_id", "") or "").strip()
                business_connection_id = str(
                    business_result.get("business_connection_id", "") or ""
                ).strip()
                conversation_id = str(business_result.get("chat_id", "") or "").strip()
                owner_chat_id = str(business_result.get("user_chat_id", "") or "").strip()
                if customer_id and business_connection_id and conversation_id:
                    workflows = get_intake_workflows().list_workflows(
                        customer_id=customer_id,
                        include_disabled=False,
                    )
                    matched_workflows = 0
                    for workflow in workflows:
                        if str(workflow.get("channel", "")).strip() != "telegram_business_dm":
                            continue
                        if str(workflow.get("provider", "")).strip() != "telegram_bot_api":
                            continue
                        if not get_intake_workflows()._source_matches_workflow(  # noqa: SLF001
                            workflow=workflow,
                            business_connection_id=business_connection_id,
                            conversation_id=conversation_id,
                        ):
                            continue
                        matched_workflows += 1
                        intake_workflows = get_intake_workflows()
                        enqueue_run = getattr(
                            intake_workflows,
                            "enqueue_telegram_business_workflow_run",
                            None,
                        )
                        if callable(enqueue_run):
                            result = await enqueue_run(
                                customer_id=customer_id,
                                workflow_id=str(workflow.get("workflow_id", "") or "").strip(),
                                conversation_id=conversation_id,
                                owner_chat_id=owner_chat_id,
                                event_type="telegram_business_webhook",
                            )
                        else:
                            result = await intake_workflows.run_workflow(
                                customer_id=customer_id,
                                workflow_id=str(workflow.get("workflow_id", "") or "").strip(),
                                event_type="telegram_business_webhook",
                            )
                        summary = str(result.get("summary", "") or "").strip()
                        _write_telegram_webhook_event(
                            "business_workflow_run",
                            update_id=body.get("update_id"),
                            workflow_id=str(workflow.get("workflow_id", "") or "").strip(),
                            customer_id=customer_id,
                            business_connection_id=business_connection_id,
                            conversation_id=conversation_id,
                            ok=bool(result.get("ok", False)),
                            has_summary=bool(summary),
                        )
                        if (
                            not bool(result.get("ok", False))
                            and owner_chat_id
                            and summary
                            and summary != NO_NOTIFY_TOKEN
                        ):
                            with suppress(Exception):
                                await get_telegram_client().send_message(
                                    chat_id=owner_chat_id,
                                    text=f"Telegram Business workflow issue: {summary}",
                                    parse_mode="HTML",
                                )
                    _write_telegram_webhook_event(
                        "business_workflow_match_summary",
                        update_id=body.get("update_id"),
                        customer_id=customer_id,
                        business_connection_id=business_connection_id,
                        conversation_id=conversation_id,
                        workflow_count=len(workflows),
                        matched_workflow_count=matched_workflows,
                    )
            return

        callback_id, _callback_user_id, callback_chat_id, _callback_data, _callback_message_id = (
            parse_telegram_callback_query(body)
        )
        if callback_id and callback_chat_id:
            with suppress(Exception):
                await get_telegram_client().answer_callback_query(
                    callback_query_id=callback_id,
                    text="This control is no longer active.",
                    show_alert=False,
                )
            return

        message = body.get("message") or body.get("edited_message") or {}
        chat_id = message.get("chat", {}).get("id")
        _ = parse_telegram_update(body)

        try:
            reply = await get_telegram_chat().handle_update(
                body=body,
                allowed_user_ids_csv=settings.telegram_allowed_user_ids,
                allowed_usernames_csv=settings.telegram_allowed_usernames,
                support_user_ids_csv=getattr(settings, "telegram_support_user_ids", None),
                support_usernames_csv=getattr(settings, "telegram_support_usernames", None),
                agent_runtime=get_agent_runtime(),
            )
        except Exception as exc:
            logger.exception("Unhandled Telegram background handler failure: %s", exc)
            if chat_id is not None:
                with suppress(Exception):
                    await get_telegram_client().send_message(
                        chat_id=chat_id,
                        text="I hit an internal error while processing your message. Please try again.",
                        parse_mode="HTML",
                    )
            return
        if reply and chat_id is not None:
            with suppress(Exception):
                await get_telegram_client().send_message(
                    chat_id=chat_id,
                    text=reply,
                    parse_mode="HTML",
                )
            with suppress(Exception):
                get_telegram_chat().touch_assistant_message(int(chat_id))
