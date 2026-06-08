"""Telegram reply streaming and wake-event relays."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentulpa.agent.runtime import (
    STREAM_PROGRESS_PREFIX,
    STREAM_WAIT_SIGNAL,
    MergedInputSuppressedError,
)
from opentulpa.agent.turn_policy import normalize_turn_mode
from opentulpa.agent.turn_runtime_policy import effective_turn_mode as _effective_turn_mode
from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.constants import DEBUG_LOG_PATH, LOW_SIGNAL_REPLIES
from opentulpa.interfaces.telegram.status_generation import generate_llm_status_message
from opentulpa.web.events import append_web_event

logger = logging.getLogger(__name__)
NO_NOTIFY_TOKEN = "__NO_NOTIFY__"
DRAFT_INITIAL_PUBLISH_DELAY_SECONDS = 0.35
DRAFT_PUBLISH_MIN_INTERVAL_SECONDS = 0.9
TELEGRAM_TYPING_MIN_INTERVAL_SECONDS = 5.0
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


@dataclass
class _LiveStreamState:
    last_streamed: str = ""
    waiting_for_segment: bool = True
    consecutive_timeouts: int = 0
    final_reply: str | None = None
    timeout_failed_without_reply: bool = False
    timeout_failure_stage: str = ""
    next_chunk_task: asyncio.Task[Any] | None = None


@dataclass
class _TelegramDeliveryState:
    final_reply: str | None = None
    delivered_any: bool = False
    draft_enabled: bool = True
    live_delivery_text: str = ""
    live_delivery_at: float = 0.0
    interim_status_sent: bool = False


@dataclass
class _TelegramStreamResources:
    client: TelegramClient
    delivery: _TelegramDeliveryState
    typing_stop: asyncio.Event
    typing_task: asyncio.Task[None]
    draft_id: int
    stream_started_at: float
    observability_context: Any
    observability_closed: bool = False

    def close_observability_context(self) -> None:
        if self.observability_closed:
            return
        self.observability_closed = True
        with suppress(Exception):
            self.observability_context.__exit__(None, None, None)


_WORKFLOW_SETUP_RUNS: dict[str, _WorkflowSetupRun] = {}
_TELEGRAM_TYPING_LAST_SENT_AT: dict[int, float] = {}
_TELEGRAM_TYPING_LOCK = asyncio.Lock()


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


def _telegram_observability_context(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    chat_id: int,
    turn_mode: str,
) -> Any:
    tracer = getattr(agent_runtime, "_langfuse_tracer", None)
    trace_context = getattr(tracer, "trace_context", None)
    if not callable(trace_context):
        return nullcontext()
    return trace_context(
        name="opentulpa.interactive.turn",
        trace_id=None,
        user_id=customer_id,
        session_id=thread_id,
        input={"text": str(text or ""), "chat_id": chat_id, "mode": "telegram"},
        metadata={"turn_mode": normalize_turn_mode(turn_mode), "chat_id": chat_id},
        tags=[normalize_turn_mode(turn_mode), "telegram"],
    )


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
        safe_chat_id = int(chat_id)
        should_send = False
        async with _TELEGRAM_TYPING_LOCK:
            now = time.monotonic()
            last_sent = _TELEGRAM_TYPING_LAST_SENT_AT.get(safe_chat_id, 0.0)
            if now - last_sent >= TELEGRAM_TYPING_MIN_INTERVAL_SECONDS:
                _TELEGRAM_TYPING_LAST_SENT_AT[safe_chat_id] = now
                should_send = True
        if should_send:
            with suppress(Exception):
                await client.send_chat_action(chat_id=safe_chat_id, action="typing")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)


async def _session_has_pending_items(
    *,
    interactive_session: Any | None,
    chat_id: int,
    thread_id: str,
    customer_id: str,
) -> bool:
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


async def _recover_after_stream_timeout(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
) -> str | None:
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


async def _send_llm_status_update(
    *,
    agent_runtime: Any,
    client: TelegramClient,
    customer_id: str,
    thread_id: str,
    chat_id: int,
    text: str,
    turn_mode: str,
    stage: str,
) -> bool:
    status_text = await generate_llm_status_message(
        runtime=agent_runtime,
        customer_id=customer_id,
        thread_id=thread_id,
        context={
            "event": "telegram_owner_stream_waiting",
            "stage": stage,
            "turn_mode": normalize_turn_mode(turn_mode),
            "latest_user_message": str(text or "").strip()[:1000],
            "latest_user_message_usage": (
                "Background context only. Use it to understand why the turn may be slow, "
                "but do not quote, paraphrase, summarize, or mention it in the status text."
            ),
            "status_goal": (
                "Send a generic progress update that says the assistant is still checking "
                "or preparing the answer."
            ),
        },
        language="Russian",
    )
    if not status_text:
        logger.warning(
            "telegram.stream status_generation_skipped chat_id=%s thread_id=%s customer_id=%s stage=%s",
            chat_id,
            thread_id,
            customer_id,
            stage,
        )
        return False
    sent = await client.send_message(chat_id=chat_id, text=status_text, parse_mode="HTML")
    if not sent:
        return False
    append_web_event(
        customer_id=customer_id,
        thread_id=thread_id,
        source="chat",
        kind="status",
        text=status_text,
        metadata_json=json.dumps({"turn_mode": normalize_turn_mode(turn_mode)}),
    )
    logger.info(
        "telegram.stream status_generation_sent chat_id=%s thread_id=%s customer_id=%s stage=%s chars=%s",
        chat_id,
        thread_id,
        customer_id,
        stage,
        len(status_text),
    )
    return True


async def _send_status_once(
    *,
    delivery: _TelegramDeliveryState,
    agent_runtime: Any,
    client: TelegramClient,
    customer_id: str,
    thread_id: str,
    chat_id: int,
    text: str,
    turn_mode: str,
    stage: str,
) -> None:
    if delivery.interim_status_sent:
        return
    sent = await _send_llm_status_update(
        agent_runtime=agent_runtime,
        client=client,
        customer_id=customer_id,
        thread_id=thread_id,
        chat_id=chat_id,
        text=text,
        turn_mode=turn_mode,
        stage=stage,
    )
    if sent:
        delivery.delivered_any = True
        delivery.interim_status_sent = True


async def _send_draft_reply(
    *,
    delivery: _TelegramDeliveryState,
    client: TelegramClient,
    typing_stop: asyncio.Event,
    interactive_session: Any | None,
    chat_id: int,
    thread_id: str,
    customer_id: str,
    draft_id: int,
    stream_started_at: float,
    text: str,
    force: bool = False,
) -> None:
    current = str(text or "").strip()
    if not current:
        return
    if await _session_has_pending_items(
        interactive_session=interactive_session,
        chat_id=chat_id,
        thread_id=thread_id,
        customer_id=customer_id,
    ):
        return
    delivery.final_reply = current
    if current == delivery.live_delivery_text and not force:
        return
    if not delivery.draft_enabled:
        return
    now = time.monotonic()
    earliest_publish_at = _draft_earliest_publish_at(
        delivery=delivery,
        stream_started_at=stream_started_at,
        force=force,
    )
    if now < earliest_publish_at:
        return
    if await client.send_message_draft(
        chat_id=chat_id,
        draft_id=draft_id,
        text=current,
        parse_mode="HTML",
    ):
        _record_draft_delivery(
            delivery=delivery,
            typing_stop=typing_stop,
            text=current,
            delivered_at=now,
        )
        return
    delivery.draft_enabled = False
    if not typing_stop.is_set():
        typing_stop.set()
    logger.warning(
        "telegram.stream draft_disabled chat_id=%s thread_id=%s customer_id=%s",
        chat_id,
        thread_id,
        customer_id,
    )


def _draft_earliest_publish_at(
    *,
    delivery: _TelegramDeliveryState,
    stream_started_at: float,
    force: bool,
) -> float:
    if force:
        return 0.0
    if not delivery.delivered_any:
        return stream_started_at + DRAFT_INITIAL_PUBLISH_DELAY_SECONDS
    return delivery.live_delivery_at + DRAFT_PUBLISH_MIN_INTERVAL_SECONDS


def _record_draft_delivery(
    *,
    delivery: _TelegramDeliveryState,
    typing_stop: asyncio.Event,
    text: str,
    delivered_at: float,
) -> None:
    if not typing_stop.is_set():
        typing_stop.set()
    delivery.delivered_any = True
    delivery.live_delivery_text = text
    delivery.live_delivery_at = delivered_at


async def _cancel_stream_task(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        async with asyncio.timeout(1.0):
            await task


async def _pump_stream_to_queue(
    *,
    stream: Any,
    stream_queue: asyncio.Queue[tuple[str, Any]],
    done_marker: object,
) -> None:
    try:
        async for partial in stream:
            await stream_queue.put(("chunk", partial))
    except BaseException as exc:
        await stream_queue.put(("error", exc))
    else:
        await stream_queue.put(("done", done_marker))


def _stream_timeout_seconds(state: _LiveStreamState) -> float:
    if not state.last_streamed:
        return 90.0 if state.consecutive_timeouts == 0 else 180.0
    return 180.0 if state.consecutive_timeouts == 0 else 240.0


async def _handle_stream_timeout(
    *,
    state: _LiveStreamState,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
    chat_id: int,
    send_status_once: Callable[..., Any],
) -> bool:
    state.consecutive_timeouts += 1
    stage = "first_token" if not state.last_streamed else "idle"
    if state.consecutive_timeouts < 2:
        logger.warning(
            "telegram.stream timeout_retry chat_id=%s thread_id=%s customer_id=%s stage=%s",
            chat_id,
            thread_id,
            customer_id,
            stage,
        )
        if not state.last_streamed:
            await send_status_once(stage=stage)
        return False
    await _cancel_stream_task(state.next_chunk_task)
    state.next_chunk_task = None
    recovered_text = await _recover_after_stream_timeout(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        turn_mode=turn_mode,
    )
    if recovered_text:
        logger.warning(
            "telegram.stream timeout_recovered chat_id=%s thread_id=%s customer_id=%s stage=%s",
            chat_id,
            thread_id,
            customer_id,
            stage,
        )
        state.final_reply = recovered_text
        return True
    state.timeout_failed_without_reply = True
    state.timeout_failure_stage = stage
    logger.error(
        "telegram.stream timeout_fail chat_id=%s thread_id=%s customer_id=%s stage=%s",
        chat_id,
        thread_id,
        customer_id,
        stage,
    )
    return True


async def _apply_stream_payload(
    *,
    state: _LiveStreamState,
    payload: Any,
    send_draft_reply: Callable[..., Any],
) -> None:
    progress_text = payload if isinstance(payload, str) else ""
    if progress_text and _is_progress_signal(progress_text):
        if not state.waiting_for_segment:
            state.waiting_for_segment = True
            state.last_streamed = ""
        return
    if not isinstance(payload, str):
        return
    state.consecutive_timeouts = 0
    current = payload.strip()
    if not current or is_low_signal_reply(current) or current == state.last_streamed:
        return
    if state.last_streamed and not current.startswith(state.last_streamed):
        state.waiting_for_segment = True
        state.last_streamed = ""
    if state.waiting_for_segment:
        state.waiting_for_segment = False
    state.last_streamed = current
    state.final_reply = current
    await send_draft_reply(current)


async def _run_live_stream_loop(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
    chat_id: int,
    send_draft_reply: Callable[..., Any],
    send_status_once: Callable[..., Any],
) -> _LiveStreamState:
    stream = agent_runtime.astream_text(
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        turn_mode=turn_mode,
    )
    stream_done = object()
    stream_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    state = _LiveStreamState(
        next_chunk_task=asyncio.create_task(
            _pump_stream_to_queue(stream=stream, stream_queue=stream_queue, done_marker=stream_done)
        )
    )
    while True:
        try:
            stream_status, stream_payload = await asyncio.wait_for(
                stream_queue.get(),
                timeout=_stream_timeout_seconds(state),
            )
        except TimeoutError:
            if await _handle_stream_timeout(
                state=state,
                agent_runtime=agent_runtime,
                thread_id=thread_id,
                customer_id=customer_id,
                text=text,
                turn_mode=turn_mode,
                chat_id=chat_id,
                send_status_once=send_status_once,
            ):
                break
            continue
        if stream_status == "done":
            state.next_chunk_task = None
            break
        if stream_status == "error":
            state.next_chunk_task = None
            if isinstance(stream_payload, BaseException):
                raise stream_payload
            raise RuntimeError(str(stream_payload))
        await _apply_stream_payload(
            state=state,
            payload=stream_payload,
            send_draft_reply=send_draft_reply,
        )
    return state


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


async def _existing_workflow_setup_reply(
    *,
    run: _WorkflowSetupRun,
    agent_runtime: Any,
    text: str,
    chat_id: int,
    thread_id: str,
    customer_id: str,
) -> str:
    queued = False
    status = {
        "state": "workflow_setup_running",
        "current_status": "The workflow setup/preflight/proposal run is still active.",
        "pending_setup_updates": len(run.pending_texts),
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
            run.pending_texts.append(safe_text)
            queued = True
        final_reply = WORKFLOW_SETUP_QUEUED_REPLY
    logger.info(
        "telegram.stream workflow_setup_existing_run chat_id=%s thread_id=%s customer_id=%s kind=%s queued=%s pending_count=%s",
        chat_id,
        thread_id,
        customer_id,
        str(decision.get("kind", "") or ""),
        queued,
        len(run.pending_texts),
    )
    return final_reply


async def _new_workflow_setup_reply(
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
) -> str | None:
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
        return WORKFLOW_SETUP_BUSY_REPLY
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
        return WORKFLOW_SETUP_QUEUED_REPLY
    if _WORKFLOW_SETUP_RUNS.get(run_key) is run:
        _WORKFLOW_SETUP_RUNS.pop(run_key, None)
    safe = str(recovered or "").strip()
    return safe if safe and not is_low_signal_reply(safe) else None


async def _workflow_setup_reply(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
    bot_token: str,
    chat_id: int,
    final_reply_callback: Callable[[str], Any] | None,
) -> str | None:
    run_key = _workflow_setup_run_key(customer_id=customer_id, thread_id=thread_id)
    existing_run = _WORKFLOW_SETUP_RUNS.get(run_key)
    if existing_run is not None and not _workflow_setup_run_active(existing_run):
        _WORKFLOW_SETUP_RUNS.pop(run_key, None)
        existing_run = None
    if existing_run is not None:
        return await _existing_workflow_setup_reply(
            run=existing_run,
            agent_runtime=agent_runtime,
            text=text,
            chat_id=chat_id,
            thread_id=thread_id,
            customer_id=customer_id,
        )
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
    return await _new_workflow_setup_reply(
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


async def _finalize_telegram_stream_reply(
    *,
    agent_runtime: Any,
    delivery: _TelegramDeliveryState,
    client: TelegramClient,
    interactive_session: Any | None,
    customer_id: str,
    thread_id: str,
    chat_id: int,
    turn_mode: str,
    suppressed: bool,
    timeout_failed_without_reply: bool,
    timeout_failure_stage: str,
) -> tuple[str | None, bool]:
    final_reply = delivery.final_reply
    log_event = getattr(agent_runtime, "log_behavior_event", None)
    if not suppressed and not final_reply and delivery.live_delivery_text:
        final_reply = delivery.live_delivery_text
    if not suppressed and not final_reply and timeout_failed_without_reply:
        logger.error(
            "telegram.stream timeout_without_final_reply chat_id=%s thread_id=%s customer_id=%s stage=%s "
            "interim_status_sent=%s delivered_any=%s",
            chat_id,
            thread_id,
            customer_id,
            timeout_failure_stage,
            delivery.interim_status_sent,
            delivery.delivered_any,
        )
        final_reply = (
            "I couldn't finish that step before the reply timeout. "
            "Please ask me to continue, and I'll resume from the saved context."
        )
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
    effective_mode = _effective_turn_mode(
        agent_runtime,
        customer_id=customer_id,
        thread_id=thread_id,
        requested_turn_mode=turn_mode,
    )
    suppress_for_pending_session = effective_mode != "workflow_setup"
    if suppress_for_pending_session and not suppressed and await _session_has_pending_items(
        interactive_session=interactive_session,
        chat_id=chat_id,
        thread_id=thread_id,
        customer_id=customer_id,
    ):
        logger.info(
            "telegram.stream suppressed_by_interactive_pending chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        if callable(log_event):
            log_event(
                event="telegram.stream.final_suppressed",
                chat_id=chat_id,
                thread_id=thread_id,
                customer_id=customer_id,
                requested_turn_mode=normalize_turn_mode(turn_mode),
                effective_turn_mode=effective_mode,
                reason="interactive_pending",
            )
        return None, True
    if not suppressed and final_reply:
        final_reply = await _send_final_telegram_reply(
            agent_runtime=agent_runtime,
            client=client,
            customer_id=customer_id,
            thread_id=thread_id,
            chat_id=chat_id,
            turn_mode=effective_mode,
            final_reply=final_reply,
        )
    return final_reply, suppressed


async def _send_final_telegram_reply(
    *,
    agent_runtime: Any,
    client: TelegramClient,
    customer_id: str,
    thread_id: str,
    chat_id: int,
    turn_mode: str,
    final_reply: str,
) -> str | None:
    log_event = getattr(agent_runtime, "log_behavior_event", None)
    if callable(log_event):
        log_event(
            event="telegram.stream.final_send_attempt",
            chat_id=chat_id,
            thread_id=thread_id,
            customer_id=customer_id,
            turn_mode=normalize_turn_mode(turn_mode),
            final_chars=len(str(final_reply or "").strip()),
        )
    sent = await client.send_message(chat_id=chat_id, text=final_reply, parse_mode="HTML")
    if not sent:
        if callable(log_event):
            log_event(
                event="telegram.stream.final_send_failed",
                chat_id=chat_id,
                thread_id=thread_id,
                customer_id=customer_id,
                turn_mode=normalize_turn_mode(turn_mode),
            )
        return None
    append_web_event(
        customer_id=customer_id,
        thread_id=thread_id,
        source="chat",
        kind="assistant_message",
        text=final_reply,
        metadata_json=json.dumps({"turn_mode": normalize_turn_mode(turn_mode)}),
    )
    if callable(log_event):
        log_event(
            event="telegram.stream.final_send_succeeded",
            chat_id=chat_id,
            thread_id=thread_id,
            customer_id=customer_id,
            turn_mode=normalize_turn_mode(turn_mode),
            final_chars=len(str(final_reply or "").strip()),
        )
    return final_reply


def _start_telegram_stream_resources(
    *,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
    turn_mode: str,
) -> _TelegramStreamResources:
    client = TelegramClient(bot_token)
    draft_id = (
        zlib.crc32(f"{thread_id}:{customer_id}:{chat_id}:{new_short_id('dft')}".encode()) or 1
    )
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _emit_typing_until_done(client=client, chat_id=chat_id, stop_event=typing_stop)
    )
    observability_context = _telegram_observability_context(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        chat_id=chat_id,
        turn_mode=turn_mode,
    )
    observability_context.__enter__()
    return _TelegramStreamResources(
        client=client,
        delivery=_TelegramDeliveryState(draft_enabled=chat_id > 0),
        typing_stop=typing_stop,
        typing_task=typing_task,
        draft_id=draft_id,
        stream_started_at=time.monotonic(),
        observability_context=observability_context,
    )


async def _stop_telegram_stream_typing(
    *,
    resources: _TelegramStreamResources,
    next_chunk_task: asyncio.Task[Any] | None,
) -> None:
    await _cancel_stream_task(next_chunk_task)
    if not resources.typing_stop.is_set():
        resources.typing_stop.set()
    with suppress(Exception):
        await resources.typing_task


async def _close_telegram_stream_resources(resources: _TelegramStreamResources) -> None:
    if hasattr(resources.client, "aclose"):
        with suppress(Exception):
            await resources.client.aclose()
    resources.close_observability_context()


async def _run_telegram_stream_delivery(
    *,
    resources: _TelegramStreamResources,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
    turn_mode: str,
    interactive_session: Any | None,
    final_reply_callback: Callable[[str], Any] | None,
) -> _LiveStreamState:
    delivery_turn_mode = _effective_turn_mode(
        agent_runtime,
        customer_id=customer_id,
        thread_id=thread_id,
        requested_turn_mode=turn_mode,
    )
    if delivery_turn_mode == "workflow_setup":
        return await _run_workflow_setup_stream_delivery(
            resources=resources,
            agent_runtime=agent_runtime,
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            bot_token=bot_token,
            chat_id=chat_id,
            turn_mode=delivery_turn_mode,
            final_reply_callback=final_reply_callback,
        )

    async def _draft_reply(value: str, *, force: bool = False) -> None:
        await _send_draft_reply(
            delivery=resources.delivery,
            client=resources.client,
            typing_stop=resources.typing_stop,
            interactive_session=interactive_session,
            chat_id=chat_id,
            thread_id=thread_id,
            customer_id=customer_id,
            draft_id=resources.draft_id,
            stream_started_at=resources.stream_started_at,
            text=value,
            force=force,
        )

    async def _status_once(*, stage: str) -> None:
        await _send_status_once(
            delivery=resources.delivery,
            agent_runtime=agent_runtime,
            client=resources.client,
            customer_id=customer_id,
            thread_id=thread_id,
            chat_id=chat_id,
            text=text,
            turn_mode=turn_mode,
            stage=stage,
        )

    return await _run_live_stream_loop(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        turn_mode=turn_mode,
        chat_id=chat_id,
        send_draft_reply=_draft_reply,
        send_status_once=_status_once,
    )


async def _run_workflow_setup_stream_delivery(
    *,
    resources: _TelegramStreamResources,
    agent_runtime: Any,
    thread_id: str,
    customer_id: str,
    text: str,
    bot_token: str,
    chat_id: int,
    turn_mode: str,
    final_reply_callback: Callable[[str], Any] | None,
) -> _LiveStreamState:
    resources.delivery.draft_enabled = False
    resources.delivery.final_reply = await _workflow_setup_reply(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        turn_mode=turn_mode,
        bot_token=bot_token,
        chat_id=chat_id,
        final_reply_callback=final_reply_callback,
    )
    return _LiveStreamState(final_reply=resources.delivery.final_reply)


async def _complete_telegram_stream(
    *,
    agent_runtime: Any,
    resources: _TelegramStreamResources,
    next_chunk_task: asyncio.Task[Any] | None,
    live_state: _LiveStreamState,
    interactive_session: Any | None,
    customer_id: str,
    thread_id: str,
    chat_id: int,
    turn_mode: str,
    suppressed: bool,
) -> tuple[str | None, bool]:
    await _stop_telegram_stream_typing(resources=resources, next_chunk_task=next_chunk_task)
    final_reply, suppressed = await _finalize_telegram_stream_reply(
        agent_runtime=agent_runtime,
        delivery=resources.delivery,
        client=resources.client,
        interactive_session=interactive_session,
        customer_id=customer_id,
        thread_id=thread_id,
        chat_id=chat_id,
        turn_mode=turn_mode,
        suppressed=suppressed,
        timeout_failed_without_reply=live_state.timeout_failed_without_reply,
        timeout_failure_stage=live_state.timeout_failure_stage,
    )
    logger.info(
        "telegram.stream complete chat_id=%s thread_id=%s customer_id=%s suppressed=%s final_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        suppressed,
        len(str(final_reply or "")),
    )
    await _close_telegram_stream_resources(resources)
    return final_reply, suppressed


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
    resources = _start_telegram_stream_resources(
        agent_runtime=agent_runtime,
        thread_id=thread_id,
        customer_id=customer_id,
        text=text,
        bot_token=bot_token,
        chat_id=chat_id,
        turn_mode=turn_mode,
    )
    suppressed = False
    live_state = _LiveStreamState()
    next_chunk_task: asyncio.Task[Any] | None = None
    logger.info(
        "telegram.stream start chat_id=%s thread_id=%s customer_id=%s text_chars=%s",
        chat_id,
        thread_id,
        customer_id,
        len(str(text or "")),
    )

    try:
        live_state = await _run_telegram_stream_delivery(
            resources=resources,
            agent_runtime=agent_runtime,
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            bot_token=bot_token,
            chat_id=chat_id,
            turn_mode=turn_mode,
            interactive_session=interactive_session,
            final_reply_callback=final_reply_callback,
        )
        resources.delivery.final_reply = live_state.final_reply or resources.delivery.final_reply
        next_chunk_task = live_state.next_chunk_task
    except MergedInputSuppressedError:
        logger.info(
            "telegram.stream suppressed_by_merge chat_id=%s thread_id=%s customer_id=%s",
            chat_id,
            thread_id,
            customer_id,
        )
        suppressed = True
    except Exception:
        await _stop_telegram_stream_typing(resources=resources, next_chunk_task=next_chunk_task)
        await _close_telegram_stream_resources(resources)
        raise
    return await _complete_telegram_stream(
        agent_runtime=agent_runtime,
        resources=resources,
        next_chunk_task=next_chunk_task,
        live_state=live_state,
        interactive_session=interactive_session,
        customer_id=customer_id,
        thread_id=thread_id,
        chat_id=chat_id,
        turn_mode=turn_mode,
        suppressed=suppressed,
    )


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


@dataclass(frozen=True)
class _WakeSlotState:
    chat_id: int
    chat_key: str
    last_user_at: str
    last_assistant_at: str
    user_idle_hours: str
    assistant_idle_hours: str


def _wake_slot_state(slot: dict[str, Any], *, now_utc: datetime) -> _WakeSlotState:
    chat_id = int(slot["chat_id"])
    last_user_at = str(slot.get("last_user_message_at", "")).strip()
    last_assistant_at = str(slot.get("last_assistant_message_at", "")).strip()
    return _WakeSlotState(
        chat_id=chat_id,
        chat_key=str(chat_id),
        last_user_at=last_user_at,
        last_assistant_at=last_assistant_at,
        user_idle_hours=_idle_hours_since(last_user_at, now_utc=now_utc),
        assistant_idle_hours=_idle_hours_since(last_assistant_at, now_utc=now_utc),
    )


def _idle_hours_since(value: str, *, now_utc: datetime) -> str:
    if not value:
        return "unknown"
    with suppress(Exception):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return f"{max(0.0, (now_utc - parsed).total_seconds() / 3600.0):.2f}"
    return "unknown"


async def _routine_wake_should_notify(
    *,
    agent_runtime: Any,
    customer_id: str,
    event_label: str,
    routine_name: str,
    routine_payload: dict[str, Any],
    slot_state: _WakeSlotState,
) -> bool:
    if not hasattr(agent_runtime, "classify_wake_event"):
        return True
    precheck_payload = {
        "event_label": event_label,
        "routine_name": routine_name,
        "routine_payload": routine_payload,
        "last_user_message_at_utc": slot_state.last_user_at or "unknown",
        "last_assistant_message_at_utc": slot_state.last_assistant_at or "unknown",
        "user_idle_hours": slot_state.user_idle_hours,
        "assistant_idle_hours": slot_state.assistant_idle_hours,
    }
    decision = {"notify_user": True}
    with suppress(Exception):
        decision = await agent_runtime.classify_wake_event(
            customer_id=customer_id,
            event_label="routine/heartbeat_precheck",
            payload=precheck_payload,
        )
    return bool(decision.get("notify_user", False))


def _build_wake_instruction(
    *,
    event_label: str,
    payload: dict[str, Any],
    routine_name: str,
    routine_instruction: str,
    slot_state: _WakeSlotState,
    now_utc: datetime,
) -> str:
    if str(event_label).startswith("routine/"):
        return (
            "System update: a scheduled routine woke you.\n"
            "Decide if the user should be messaged right now.\n"
            f"- event: {event_label}\n"
            f"- routine_name: {routine_name or 'unnamed'}\n"
            f"- routine_instruction: {routine_instruction[:3000] or '(none)'}\n"
            f"- last_user_message_at_utc: {slot_state.last_user_at or 'unknown'}\n"
            f"- user_idle_hours: {slot_state.user_idle_hours}\n"
            f"- last_assistant_message_at_utc: {slot_state.last_assistant_at or 'unknown'}\n"
            f"- assistant_idle_hours: {slot_state.assistant_idle_hours}\n"
            f"- now_utc: {now_utc.isoformat()}\n"
            f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
            f"If you decide to skip messaging this run, reply exactly: {NO_NOTIFY_TOKEN}\n"
            "If you decide to message, send one concise, natural message (no rigid status template)."
        )
    return (
        "System update: a background event occurred.\n"
        "Respond with concise plain-language status, what happened, and next action.\n"
        f"- event: {event_label}\n"
        f"- payload: {json.dumps(payload, ensure_ascii=False)[:4000]}"
    )


def _ensure_wake_thread_id(state: dict[str, Any], *, chat_key: str) -> str:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    raw_slot = sessions.get(chat_key)
    if not isinstance(raw_slot, dict):
        raw_slot = {}
    wake_thread_id = _clean_thread_id(raw_slot.get("wake_thread_id"))
    if not wake_thread_id or not wake_thread_id.lower().startswith("wake_"):
        wake_thread_id = new_short_id("wake")
        raw_slot["wake_thread_id"] = wake_thread_id
        sessions[chat_key] = raw_slot
        state["sessions"] = sessions
    return wake_thread_id


async def _relay_wake_slot_via_main_agent(
    *,
    slot: dict[str, Any],
    customer_id: str,
    event_label: str,
    payload: dict[str, Any],
    routine_payload: dict[str, Any],
    routine_instruction: str,
    routine_name: str,
    proactive_heartbeat: bool,
    now_utc: datetime,
    state_store: Any,
    agent_runtime: Any,
) -> dict[str, Any] | None:
    slot_state = _wake_slot_state(slot, now_utc=now_utc)
    if (
        str(event_label).startswith("routine/")
        and proactive_heartbeat
        and not await _routine_wake_should_notify(
            agent_runtime=agent_runtime,
            customer_id=customer_id,
            event_label=event_label,
            routine_name=routine_name,
            routine_payload=routine_payload,
            slot_state=slot_state,
        )
    ):
        return None
    instruction = _build_wake_instruction(
        event_label=event_label,
        payload=payload,
        routine_name=routine_name,
        routine_instruction=routine_instruction,
        slot_state=slot_state,
        now_utc=now_utc,
    )
    wake_thread_id = state_store.update(
        lambda state, chat_key=slot_state.chat_key: _ensure_wake_thread_id(
            state, chat_key=chat_key
        )
    )
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
        return None
    return {
        "chat_id": slot_state.chat_id,
        "text": NO_NOTIFY_TOKEN if safe == NO_NOTIFY_TOKEN else safe,
    }


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
    raw_routine_payload = payload.get("payload")
    routine_payload: dict[str, Any] = (
        dict(raw_routine_payload) if isinstance(raw_routine_payload, dict) else {}
    )
    routine_instruction = str(routine_payload.get("instruction", "")).strip()
    routine_name = str(payload.get("routine_name", "")).strip()
    proactive_heartbeat = bool(routine_payload.get("proactive_heartbeat", False))
    now_utc = datetime.now(UTC)
    replies: list[dict[str, Any]] = []
    for slot in slots:
        try:
            reply = await _relay_wake_slot_via_main_agent(
                slot=slot,
                customer_id=customer_id,
                event_label=event_label,
                payload=payload,
                routine_payload=routine_payload,
                routine_instruction=routine_instruction,
                routine_name=routine_name,
                proactive_heartbeat=proactive_heartbeat,
                now_utc=now_utc,
                state_store=state_store,
                agent_runtime=agent_runtime,
            )
        except Exception:
            continue
        if reply is not None:
            replies.append(reply)
    return replies
