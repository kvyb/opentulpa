"""Application orchestration for post-approval execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any


def _extract_execution_error_text(execution_result: Any) -> str:
    if not isinstance(execution_result, dict):
        return ""
    for key in ("error", "detail"):
        value = str(execution_result.get(key, "")).strip()
        if value:
            return value
    nested = execution_result.get("result")
    if isinstance(nested, dict):
        for key in ("error", "detail"):
            value = str(nested.get(key, "")).strip()
            if value:
                return value
        if nested.get("ok") is False:
            stderr = str(nested.get("stderr", "")).strip()
            if stderr:
                return stderr
            return "approved action returned an unsuccessful result"
    return ""


def _recovery_thread_id(*, thread_id: str, approval_id: str) -> str:
    base = str(thread_id or "").strip() or "approval"
    aid = str(approval_id or "").strip() or "unknown"
    return f"{base}::approval-recovery::{aid}"


class ApprovalExecutionOrchestrator:
    """Executes approved actions and produces user-facing summaries."""

    def __init__(
        self,
        *,
        get_agent_runtime: Callable[[], Any],
        get_context_events: Callable[[], Any],
    ) -> None:
        self._get_agent_runtime = get_agent_runtime
        self._get_context_events = get_context_events

    async def execute_approved_action_and_summarize(
        self,
        *,
        approval_id: str,
        decision_payload: dict[str, Any],
        chat_id: int,
    ) -> str:
        customer_id = str(decision_payload.get("customer_id", "")).strip()
        thread_id = str(decision_payload.get("thread_id", "")).strip() or f"chat-{chat_id}"
        if not customer_id:
            return "I approved this action, but couldn't resolve the customer context to execute it."
        runtime = self._get_agent_runtime()
        if runtime is None:
            return "I approved this action, but runtime is unavailable right now."
        if not hasattr(runtime, "execute_tool"):
            return "I approved this action, but runtime cannot execute approved actions."

        try:
            execution_result = await runtime.execute_tool(
                action_name="guardrail_execute_approved_action",
                action_args={"approval_id": approval_id},
                customer_id=customer_id,
                inject_customer_id=True,
            )
        except Exception as exc:
            error_detail = str(exc).strip() or exc.__class__.__name__
            with suppress(Exception):
                self._get_context_events().add_event(
                    customer_id=customer_id,
                    source="approval",
                    event_type="execute_failed",
                    payload={
                        "approval_id": approval_id,
                        "thread_id": thread_id,
                        "error": error_detail,
                    },
                )
            return f"I couldn't execute the approved action: {error_detail}"

        with suppress(Exception):
            self._get_context_events().add_event(
                customer_id=customer_id,
                source="approval",
                event_type="executed",
                payload={
                    "approval_id": approval_id,
                    "thread_id": thread_id,
                    "execution_result": (
                        execution_result
                        if isinstance(execution_result, dict)
                        else {"raw": str(execution_result)}
                    ),
                },
            )

        if isinstance(execution_result, dict) and bool(execution_result.get("already_executed")):
            return "This approved action was already executed successfully earlier."

        payload_preview = json.dumps(execution_result, ensure_ascii=False)[:6000]
        execution_error_text = _extract_execution_error_text(execution_result)
        is_error_result = bool(execution_error_text)
        retry_succeeded = False
        if is_error_result and isinstance(execution_result, dict) and bool(
            execution_result.get("retryable")
        ):
            try:
                retry_result = await runtime.execute_tool(
                    action_name="guardrail_execute_approved_action",
                    action_args={"approval_id": approval_id},
                    customer_id=customer_id,
                    inject_customer_id=True,
                )
                retry_error = _extract_execution_error_text(retry_result)
                if not retry_error:
                    execution_result = retry_result
                    payload_preview = json.dumps(execution_result, ensure_ascii=False)[:6000]
                    execution_error_text = ""
                    is_error_result = False
                    retry_succeeded = True
                    with suppress(Exception):
                        self._get_context_events().add_event(
                            customer_id=customer_id,
                            source="approval",
                            event_type="executed_retry_success",
                            payload={
                                "approval_id": approval_id,
                                "thread_id": thread_id,
                                "retry_result": (
                                    retry_result
                                    if isinstance(retry_result, dict)
                                    else {"raw": str(retry_result)}
                                ),
                            },
                        )
            except Exception:
                pass

        if retry_succeeded:
            return "The approved action failed once, then succeeded automatically on retry."

        recovery_thread_id = _recovery_thread_id(
            thread_id=thread_id,
            approval_id=approval_id,
        )
        if is_error_result:
            try:
                original_action = str(decision_payload.get("action_name", "")).strip()
                original_summary = str(decision_payload.get("summary", "")).strip()
                original_args = decision_payload.get("action_args")
                if not isinstance(original_args, dict):
                    original_args = {}
                recovery_text = await runtime.ainvoke_text(
                    thread_id=recovery_thread_id,
                    customer_id=customer_id,
                    text=(
                        "An approved action failed during execution. Continue autonomously and try to fix it.\n\n"
                        f"original_action={original_action}\n"
                        f"original_summary={original_summary}\n"
                        f"original_action_args={json.dumps(original_args, ensure_ascii=False)[:4000]}\n"
                        f"failure_result={payload_preview}\n\n"
                        "Instructions:\n"
                        "1) Retry/fix on your own using tools.\n"
                        "2) If resolved, report final success + deliverable.\n"
                        "3) If still unresolved after substantial attempts, report what you tried and ask user whether to continue.\n"
                        "Do not leak internal JSON or system internals."
                    ),
                    include_pending_context=False,
                    recursion_limit_override=48,
                )
                recovered = str(recovery_text or "").strip()
                if recovered:
                    return recovered
            except Exception:
                pass

        try:
            if is_error_result:
                summary_prompt = (
                    "A previously approved action execution failed.\n"
                    f"approval_id={approval_id}\n"
                    f"execution_result={payload_preview}\n\n"
                    "Write a concise user-facing failure update in plain text:\n"
                    "1) what failed,\n"
                    "2) likely reason from the error/result payload,\n"
                    "3) exact next step the user should take.\n"
                    "Do not expose internal JSON or system internals."
                )
            else:
                summary_prompt = (
                    "A previously approved action has just been executed.\n"
                    f"approval_id={approval_id}\n"
                    f"execution_result={payload_preview}\n\n"
                    "Write a concise user-facing outcome message in plain text:\n"
                    "1) what was done,\n"
                    "2) whether it succeeded,\n"
                    "3) next step only if needed.\n"
                    "Do not expose internal JSON or system internals."
                )
            summary = await runtime.ainvoke_text(
                thread_id=recovery_thread_id,
                customer_id=customer_id,
                text=summary_prompt,
                include_pending_context=False,
            )
        except Exception:
            summary = ""

        final = str(summary or "").strip()
        if final:
            return final
        if execution_error_text:
            return f"I couldn't execute the approved action. Error: {execution_error_text}"
        return "Task completed."

    async def execute_group_and_merge(
        self,
        *,
        approval_ids: list[str],
        decision_payload: dict[str, Any],
        chat_id: int,
    ) -> str:
        safe_ids = [str(item).strip() for item in approval_ids if str(item).strip()]
        if not safe_ids:
            return "No approved actions were executed."
        outcomes: list[str] = []
        for aid in safe_ids:
            outcome = await self.execute_approved_action_and_summarize(
                approval_id=aid,
                decision_payload=decision_payload,
                chat_id=chat_id,
            )
            if outcome:
                outcomes.append(str(outcome).strip())
        if not outcomes:
            return "No approved actions were executed."
        if len(outcomes) == 1:
            return outcomes[0]
        merged = "\n\n".join(f"{idx}. {text}" for idx, text in enumerate(outcomes, start=1))
        return f"Completed approved actions:\n\n{merged}"
