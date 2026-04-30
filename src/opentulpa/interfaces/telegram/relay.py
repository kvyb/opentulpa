"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentulpa.agent.runtime import (
    STREAM_PROGRESS_PREFIX,
    STREAM_WAIT_SIGNAL,
    MergedInputSuppressedError,
)
from opentulpa.agent.turn_policy import normalize_turn_mode
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES

logger = logging.getLogger(__name__)
NO_NOTIFY_TOKEN = "__NO_NOTIFY__"
DRAFT_INITIAL_PUBLISH_DELAY_SECONDS = 0.35
DRAFT_PUBLISH_MIN_INTERVAL_SECONDS = 0.9
WORKFLOW_SETUP_FINAL_REPLY_TIMEOUT_SECONDS = 180.0
WORKFLOW_SETUP_BUSY_REPLY = (
    "I'm still working on the workflow setup. I'll send the proposal here as soon as it's ready."
)
WORKFLOW_SETUP_QUEUED_REPLY = (
    "I'm still working on the workflow setup. "
    "I got your latest note and will apply it after the current validation finishes."
)


@dataclass
class _WorkflowSetupRun:
    task: asyncio.Task[Any]
    pending_texts: list[str]
    delivery_task: asyncio.Task[None] | None = None


_WORKFLOW_SETUP_RUNS: dict[str, _WorkflowSetupRun] = {}


def _clean_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _workflow_setup_run_key(*, customer_id: str, thread_id: str) -> str:
    return f"{str(customer_id or '').strip()}:{_clean_thread_id(thread_id)}"


def _workflow_setup_run_active(run: _WorkflowSetupRun) -> bool:
    if not run.task.done():
        return True
    delivery_task = run.delivery_task
    return delivery_task is not None and not delivery_task.done()


def normalize_reply_text(text: str) -> str:
    import re

    t = text.strip().lower()
    t = re.sub(r"[.!?]+$", "", t)
    return " ".join(t.split())


def is_low_signal_reply(text: str) -> bool:
    normalized = normalize_reply_text(text)
    if not normalized:
        return True
    return normalized in LOW_SIGNAL_REPLIES


def _is_progress_signal(partial: str) -> bool:
    if partial == STREAM_WAIT_SIGNAL:
        return True
    return partial.startswith(STREAM_PROGRESS_PREFIX)


def debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "runId": "telegram",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def _emit_typing_until_done(
    *,
    client: TelegramClient,
    chat_id: int,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        with suppress(Exception):
            await client.send_chat_action(chat_id=chat_id, action="typing")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)


async def _classify_workflow_setup_interruption(
    *,
    agent_runtime: Any,
    text: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    classifier = getattr(agent_runtime, "classify_workflow_setup_interruption", None)
    if not callable(classifier):
        return {"ok": False, "kind": "setup_input", "error": "classifier_unavailable"}
    try:
        result = await asyncio.wait_for(
            classifier(user_text=str(text or ""), status=status),
            timeout=5.0,
        )
    except Exception as exc:
        logger.warning(
            "telegram.stream workflow_setup_interruption_classifier_failed error=%s",
            f"{type(exc).__name__}: {exc}",
        )
        return {"ok": False, "kind": "setup_input", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        return {"ok": False, "kind": "setup_input", "error": "invalid_classifier_output"}
    kind = str(result.get("kind", "") or "").strip().lower()
    if kind not in {"status_nudge", "setup_input"}:
        kind = "setup_input"
    return {
        **result,
        "kind": kind,
        "status_reply": str(result.get("status_reply", "") or "").strip(),
    }


def _start_workflow_setup_task(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
) -> asyncio.Task[Any]:
    return asyncio.create_task(
        agent_runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            turn_mode=turn_mode,
        )
    )


async def _resolve_workflow_setup_run(
    *,
    run: _WorkflowSetupRun,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    turn_mode: str,
) -> str:
    final_text = ""
    while True:
        result = await run.task
        safe = str(result or "").strip()
        if safe:
            final_text = safe
        if not run.pending_texts:
            return final_text
        pending_text = "\n\n".join(run.pending_texts).strip()
        run.pending_texts.clear()
        if not pending_text:
            continue
        run.task = _start_workflow_setup_task(
            agent_runtime=agent_runtime,
            thread_id=thread_id,
            customer_id=customer_id,
            text=pending_text,
            turn_mode=turn_mode,
        )


async def _deliver_workflow_setup_run_when_ready(
    *,
    run_key: str,
    run: _WorkflowSetupRun,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    turn_mode: str,
    bot_token: str,
    chat_id: int,
    final_reply_callback: Callable[[str], Any] | None,
) -> None:
    try:
        while True:
            final_text = await _resolve_workflow_setup_run(
                run=run,
                agent_runtime=agent_runtime,
                thread_id=thread_id,
                customer_id=customer_id,
                turn_mode=turn_mode,
            )
            if run.pending_texts:
                continue
            safe = str(final_text or "").strip()
            if not safe or is_low_signal_reply(safe):
                return
            client = TelegramClient(bot_token)
            try:
                sent = await client.send_message(
                    chat_id=chat_id,
                    text=safe,
                    parse_mode="HTML",
                )
                if sent and final_reply_callback is not None:
                    with suppress(Exception):
                        final_reply_callback(safe)
                logger.info(
                    "telegram.stream workflow_setup_background_delivered chat_id=%s thread_id=%s customer_id=%s sent=%s final_chars=%s",
                    chat_id,
                    thread_id,
                    customer_id,
                    sent,
                    len(safe),
                )
            finally:
                if hasattr(client, "aclose"):
                    with suppress(Exception):
                        await client.aclose()
            if not run.pending_texts:
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "telegram.stream workflow_setup_background_failed chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
    finally:
        if _WORKFLOW_SETUP_RUNS.get(run_key) is run:
            _WORKFLOW_SETUP_RUNS.pop(run_key, None)


def _ensure_workflow_setup_delivery_task(
    *,
    run_key: str,
    run: _WorkflowSetupRun,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    turn_mode: str,
    bot_token: str,
    chat_id: int,
    final_reply_callback: Callable[[str], Any] | None,
) -> None:
    if run.delivery_task is not None and not run.delivery_task.done():
        return
    run.delivery_task = asyncio.create_task(
        _deliver_workflow_setup_run_when_ready(
            run_key=run_key,
            run=run,
            agent_runtime=agent_runtime,
            thread_id=thread_id,
            customer_id=customer_id,
            turn_mode=turn_mode,
            bot_token=bot_token,
            chat_id=chat_id,
            final_reply_callback=final_reply_callback,
        )
    )


async def stream_langgraph_reply_to_telegram(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
    turn_mode: str = "interactive",
    interactive_session: Any | None = None,
    final_reply_callback: Callable[[str], Any] | None = None,
) -> tuple[str | None, bool]:
    last_streamed = ""
    final_reply = None
    delivered_any = False
    live_delivery_text = ""
    live_delivery_at = 0.0
    client = TelegramClient(bot_token)
    draft_id = (
        zlib.crc32(f"{thread_id}:{customer_id}:{chat_id}:{new_short_id('dft')}".encode()) or 1
    )
    draft_enabled = chat_id > 0
    waiting_for_segment = True
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _emit_typing_until_done(client=client, chat_id=chat_id, stop_event=typing_stop)
    )
    suppressed = False
    first_token_timeout_s = 90.0
    first_token_retry_timeout_s = 180.0
    stream_idle_timeout_s = 180.0
    stream_idle_retry_timeout_s = 240.0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 2
    stream_started_at = time.monotonic()
    final_only = normalize_turn_mode(turn_mode) == "workflow_setup"
    next_chunk_task: asyncio.Task[Any] | None = None
    logger.info(
        "telegram.stream start chat_id=%s thread_id=%s customer_id=%s text_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        len(str(text or "")),
    )

    async def _session_has_pending_items() -> bool:
        if interactive_session is None or not hasattr(interactive_session, "has_pending_items"):
            return False
        try:
            return bool(await interactive_session.has_pending_items())
        except Exception:
            logger.exception(
                "telegram.stream pending_items_check_failed chat_id=%s thread_id=%s customer_id=%s",
                chat_id,
                thread_id,
                customer_id,
            )
            return False

    async def _recover_after_stream_timeout() -> str | None:
        if not hasattr(agent_runtime, "ainvoke_text"):
            return None
        try:
            recovered = await asyncio.wait_for(
                agent_runtime.ainvoke_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=text,
                    turn_mode=turn_mode,
                ),
                timeout=90.0,
            )
        except Exception:
            return None
        safe = str(recovered or "").strip()
        if not safe or is_low_signal_reply(safe):
            return None
        return safe

    async def _send_draft_reply(text: str, *, force: bool = False) -> None:
        nonlocal delivered_any, draft_enabled, final_reply, live_delivery_text, live_delivery_at
        current = str(text or "").strip()
        if not current:
            return
        if await _session_has_pending_items():
            return
        final_reply = current
        if current == live_delivery_text and not force:
            return
        if not draft_enabled:
            return
        now = time.monotonic()
        if not force:
            earliest_publish_at = (
                stream_started_at + DRAFT_INITIAL_PUBLISH_DELAY_SECONDS
                if not delivered_any
                else live_delivery_at + DRAFT_PUBLISH_MIN_INTERVAL_SECONDS
            )
            if now < earliest_publish_at:
                return
        if not await client.send_message_draft(
            chat_id=chat_id,
            draft_id=draft_id,
            text=current,
            parse_mode="HTML",
        ):
            draft_enabled = False
            if not typing_stop.is_set():
                typing_stop.set()
            logger.warning(
                "telegram.stream draft_disabled chat_id=%s thread_id=%s customer_id=%s",
                chat_id,
                thread_id,
                customer_id,
            )
            return
        if not typing_stop.is_set():
            typing_stop.set()
        delivered_any = True
        live_delivery_text = current
        live_delivery_at = now

    try:
        if final_only:
            draft_enabled = False
            run_key = _workflow_setup_run_key(customer_id=customer_id, thread_id=thread_id)
            existing_run = _WORKFLOW_SETUP_RUNS.get(run_key)
            if existing_run is not None and not _workflow_setup_run_active(existing_run):
                _WORKFLOW_SETUP_RUNS.pop(run_key, None)
                existing_run = None
            if existing_run is not None:
                queued = False
                status = {
                    "state": "workflow_setup_running",
                    "current_status": "The workflow setup/preflight/proposal run is still active.",
                    "pending_setup_updates": len(existing_run.pending_texts),
                    "reply_if_status_nudge": WORKFLOW_SETUP_BUSY_REPLY,
                    "reply_if_setup_input": WORKFLOW_SETUP_QUEUED_REPLY,
                }
                decision = await _classify_workflow_setup_interruption(
                    agent_runtime=agent_runtime,
                    text=text,
                    status=status,
                )
                if str(decision.get("kind", "") or "") == "status_nudge":
                    final_reply = str(decision.get("status_reply", "") or "").strip()
                    if not final_reply:
                        final_reply = WORKFLOW_SETUP_BUSY_REPLY
                else:
                    safe_text = str(text or "").strip()
                    if safe_text:
                        existing_run.pending_texts.append(safe_text)
                        queued = True
                    final_reply = WORKFLOW_SETUP_QUEUED_REPLY
                logger.info(
                    "telegram.stream workflow_setup_existing_run chat_id=%s thread_id=%s customer_id=%s kind=%s queued=%s pending_count=%s",
                    chat_id,
                    thread_id,
                    customer_id,
                    str(decision.get("kind", "") or ""),
                    queued,
                    len(existing_run.pending_texts),
                )
            else:
                run = _WorkflowSetupRun(
                    task=_start_workflow_setup_task(
                        agent_runtime=agent_runtime,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        text=text,
                        turn_mode=turn_mode,
                    ),
                    pending_texts=[],
                )
                _WORKFLOW_SETUP_RUNS[run_key] = run
                try:
                    recovered = await asyncio.wait_for(
                        asyncio.shield(run.task),
                        timeout=WORKFLOW_SETUP_FINAL_REPLY_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.warning(
                        "telegram.stream workflow_setup_backgrounded chat_id=%s thread_id=%s customer_id=%s turn_mode=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        turn_mode,
                    )
                    _ensure_workflow_setup_delivery_task(
                        run_key=run_key,
                        run=run,
                        agent_runtime=agent_runtime,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        turn_mode=turn_mode,
                        bot_token=bot_token,
                        chat_id=chat_id,
                        final_reply_callback=final_reply_callback,
                    )
                    final_reply = WORKFLOW_SETUP_BUSY_REPLY
                except asyncio.CancelledError:
                    _ensure_workflow_setup_delivery_task(
                        run_key=run_key,
                        run=run,
                        agent_runtime=agent_runtime,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        turn_mode=turn_mode,
                        bot_token=bot_token,
                        chat_id=chat_id,
                        final_reply_callback=final_reply_callback,
                    )
                    raise
                except Exception:
                    if _WORKFLOW_SETUP_RUNS.get(run_key) is run:
                        _WORKFLOW_SETUP_RUNS.pop(run_key, None)
                    raise
                else:
                    if run.pending_texts:
                        _ensure_workflow_setup_delivery_task(
                            run_key=run_key,
                            run=run,
                            agent_runtime=agent_runtime,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            turn_mode=turn_mode,
                            bot_token=bot_token,
                            chat_id=chat_id,
                            final_reply_callback=final_reply_callback,
                        )
                        final_reply = WORKFLOW_SETUP_QUEUED_REPLY
                    else:
                        if _WORKFLOW_SETUP_RUNS.get(run_key) is run:
                            _WORKFLOW_SETUP_RUNS.pop(run_key, None)
                        safe = str(recovered or "").strip()
                        if safe and not is_low_signal_reply(safe):
                            final_reply = safe
        else:
            stream = agent_runtime.astream_text(
                thread_id=thread_id,
                customer_id=customer_id,
                text=text,
                turn_mode=turn_mode,
            )
            stream_iter = stream.__aiter__()
            while True:
                if not last_streamed:
                    timeout_s = (
                        first_token_timeout_s
                        if consecutive_timeouts == 0
                        else first_token_retry_timeout_s
                    )
                else:
                    timeout_s = (
                        stream_idle_timeout_s
                        if consecutive_timeouts == 0
                        else stream_idle_retry_timeout_s
                    )
                try:
                    if next_chunk_task is None:
                        next_chunk_task = asyncio.create_task(stream_iter.__anext__())
                    partial = await asyncio.wait_for(
                        asyncio.shield(next_chunk_task),
                        timeout=timeout_s,
                    )
                    next_chunk_task = None
                except StopAsyncIteration:
                    next_chunk_task = None
                    break
                except TimeoutError:
                    consecutive_timeouts += 1
                    if consecutive_timeouts < max_consecutive_timeouts:
                        logger.warning(
                            "telegram.stream timeout_retry chat_id=%s thread_id=%s customer_id=%s stage=%s",
                            chat_id,
                            thread_id,
                            customer_id,
                            "first_token" if not last_streamed else "idle",
                        )
                        continue
                    if next_chunk_task is not None and not next_chunk_task.done():
                        next_chunk_task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            async with asyncio.timeout(1.0):
                                await next_chunk_task
                    next_chunk_task = None
                    with suppress(Exception):
                        async with asyncio.timeout(1.0):
                            await stream.aclose()
                    recovered_text = await _recover_after_stream_timeout()
                    if recovered_text:
                        logger.warning(
                            "telegram.stream timeout_recovered chat_id=%s thread_id=%s customer_id=%s stage=%s",
                            chat_id,
                            thread_id,
                            customer_id,
                            "first_token" if not last_streamed else "idle",
                        )
                        final_reply = recovered_text
                        break
                    timeout_text = (
                        "Still working, but the model response timed out. Please retry in a moment."
                    )
                    logger.error(
                        "telegram.stream timeout_fail chat_id=%s thread_id=%s customer_id=%s stage=%s",
                        chat_id,
                        thread_id,
                        customer_id,
                        "first_token" if not last_streamed else "idle",
                    )
                    final_reply = timeout_text
                    break
                progress_text = partial if isinstance(partial, str) else ""
                if progress_text and _is_progress_signal(progress_text):
                    if not waiting_for_segment:
                        waiting_for_segment = True
                        last_streamed = ""
                    continue
                if not isinstance(partial, str):
                    continue
                consecutive_timeouts = 0
                current = partial.strip()
                if not current or is_low_signal_reply(current) or current == last_streamed:
                    continue
                # Defensive boundary handling for streams that reset partial text without explicit signal.
                if last_streamed and not current.startswith(last_streamed):
                    waiting_for_segment = True
                    last_streamed = ""
                if waiting_for_segment:
                    waiting_for_segment = False
                last_streamed = current
                await _send_draft_reply(current)
    except MergedInputSuppressedError:
        logger.info(
            "telegram.stream suppressed_by_merge chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
    except Exception:
        if next_chunk_task is not None and not next_chunk_task.done():
            next_chunk_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                async with asyncio.timeout(1.0):
                    await next_chunk_task
        if not typing_stop.is_set():
            typing_stop.set()
        with suppress(Exception):
            await typing_task
        if hasattr(client, "aclose"):
            with suppress(Exception):
                await client.aclose()
        raise
    if next_chunk_task is not None and not next_chunk_task.done():
        next_chunk_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            async with asyncio.timeout(1.0):
                await next_chunk_task
    if not typing_stop.is_set():
        typing_stop.set()
    with suppress(Exception):
        await typing_task
    if not suppressed and not final_reply:
        logger.error(
            "telegram.stream no_final_reply chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        final_reply = (
            "I couldn't produce a visible user-facing reply for that step "
            "(the model/tool loop ended without displayable output)."
        )
    if not suppressed and await _session_has_pending_items():
        logger.info(
            "telegram.stream suppressed_by_interactive_pending chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
        final_reply = None
    if not suppressed and final_reply:
        sent = await client.send_message(
            chat_id=chat_id,
            text=final_reply,
            parse_mode="HTML",
        )
        if sent:
            delivered_any = True
        else:
            final_reply = None
    logger.info(
        "telegram.stream complete chat_id=%s thread_id=%s customer_id=%s suppressed=%s final_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        suppressed,
        len(str(final_reply or "")),
    )
    if hasattr(client, "aclose"):
        with suppress(Exception):
            await client.aclose()
    return final_reply, suppressed


async def relay_task_event_via_main_agent(
    *,
    customer_id: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    return await relay_event_via_main_agent(
        customer_id=customer_id,
        event_label=f"task/{event_type}",
        payload={
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
        },
        state_store=state_store,
        find_session_slots=find_session_slots,
        agent_runtime=agent_runtime,
    )


async def relay_event_via_main_agent(
    *,
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    state_store: Any,
    find_session_slots: Callable[[str], list[dict[str, Any]]],
    agent_runtime: Any | None = None,
) -> list[dict[str, Any]]:
    slots = find_session_slots(customer_id)
    if not slots:
        return []
    owner_slots = [slot for slot in slots if str(slot.get("role", "")).strip() != "support"]
    slots = owner_slots or slots[:1]
    if agent_runtime is None:
        raise RuntimeError("Agent runtime unavailable for wake relay")
    routine_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    routine_instruction = str(routine_payload.get("instruction", "")).strip()
    routine_name = str(payload.get("routine_name", "")).strip()
    proactive_heartbeat = bool(routine_payload.get("proactive_heartbeat", False))
    now_utc = datetime.now(UTC)
    replies: list[dict[str, Any]] = []
    for slot in slots:
        chat_id = int(slot["chat_id"])
        chat_key = str(chat_id)
        last_user_at = str(slot.get("last_user_message_at", "")).strip()
        last_assistant_at = str(slot.get("last_assistant_message_at", "")).strip()
        user_idle_hours = "unknown"
        assistant_idle_hours = "unknown"
        if last_user_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_user_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                user_idle_hours = f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"
        if last_assistant_at:
            with suppress(Exception):
                parsed = datetime.fromisoformat(last_assistant_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                assistant_idle_hours = (
                    f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"
                )

        if (
            str(event_label).startswith("routine/")
            and proactive_heartbeat
            and hasattr(agent_runtime, "classify_wake_event")
        ):
            precheck_payload = {
                "event_label": event_label,
                "routine_name": routine_name,
                "routine_payload": routine_payload,
                "last_user_message_at_utc": last_user_at or "unknown",
                "last_assistant_message_at_utc": last_assistant_at or "unknown",
                "user_idle_hours": user_idle_hours,
                "assistant_idle_hours": assistant_idle_hours,
            }
            decision = {"notify_user": True}
            with suppress(Exception):
                decision = await agent_runtime.classify_wake_event(
                    customer_id=customer_id,
                    event_label="routine/heartbeat_precheck",
                    payload=precheck_payload,
                )
            if not bool(decision.get("notify_user", False)):
                continue

        if str(event_label).startswith("routine/"):
            instruction = (
                "System update: a scheduled routine woke you.\n"
                "Decide if the user should be messaged right now.\n"
                f"- event: {event_label}\n"
                f"- routine_name: {routine_name or 'unnamed'}\n"
                f"- routine_instruction: {routine_instruction[:3000] or '(none)'}\n"
                f"- last_user_message_at_utc: {last_user_at or 'unknown'}\n"
                f"- user_idle_hours: {user_idle_hours}\n"
                f"- last_assistant_message_at_utc: {last_assistant_at or 'unknown'}\n"
                f"- assistant_idle_hours: {assistant_idle_hours}\n"
                f"- now_utc: {now_utc.isoformat()}\n"
                f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                f"If you decide to skip messaging this run, reply exactly: {NO_NOTIFY_TOKEN}\n"
                "If you decide to message, send one concise, natural message (no rigid status template)."
            )
        else:
            instruction = (
                "System update: a background event occurred.\n"
                "Respond with concise plain-language status, what happened, and next action.\n"
                f"- event: {event_label}\n"
                f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}"
            )

        def _ensure_wake_thread_id(state: dict[str, Any], _chat_key: str = chat_key) -> str:
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            raw_slot = sessions.get(_chat_key)
            if not isinstance(raw_slot, dict):
                raw_slot = {}
            wake_thread_id = _clean_thread_id(raw_slot.get("wake_thread_id"))
            if not wake_thread_id or not wake_thread_id.lower().startswith("wake_"):
                wake_thread_id = new_short_id("wake")
                raw_slot["wake_thread_id"] = wake_thread_id
                sessions[_chat_key] = raw_slot
                state["sessions"] = sessions
            return wake_thread_id

        wake_thread_id = state_store.update(_ensure_wake_thread_id)
        try:
            text = await agent_runtime.ainvoke_text(
                thread_id=wake_thread_id,
                customer_id=customer_id,
                text=instruction,
                turn_mode="event_notification",
                include_pending_context=False,
                recursion_limit_override=36 if proactive_heartbeat else None,
            )
            safe = str(text or "").strip()
            if not safe:
                continue
            if safe == NO_NOTIFY_TOKEN:
                replies.append({"chat_id": chat_id, "text": NO_NOTIFY_TOKEN})
                continue
            replies.append({"chat_id": chat_id, "text": safe})
        except Exception:
            continue
    return replies
