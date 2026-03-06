"""Approval-related internal API routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_approval_routes(
    app: FastAPI,
    *,
    get_approvals: Callable[[], Any],
    get_wake_queue: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Register approval endpoints and return shared decision helper."""

    async def _decide_approval_and_maybe_wake(
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
        enqueue_wake: bool = True,
    ) -> dict[str, Any]:
        broker = get_approvals()
        resolved = await broker.decide(
            approval_id=approval_id,
            decision=decision,
            actor_interface=actor_interface,
            actor_id=actor_id,
        )
        resolved_status = str(resolved.get("status", "")).strip()
        if enqueue_wake and bool(resolved.get("ok")) and resolved_status in {"approved", "denied"}:
            approval_ref = str(resolved.get("id", approval_id)).strip()
            payload = {
                "type": "approval_event",
                "event_type": resolved_status,
                "customer_id": str(resolved.get("customer_id", "")).strip(),
                "thread_id": str(resolved.get("thread_id", "")).strip(),
                "approval_id": approval_ref,
                "payload": {
                    "approval_id": approval_ref,
                    "action_name": str(resolved.get("action_name", "")).strip(),
                    "summary": str(resolved.get("summary", "")).strip(),
                    "status": resolved_status,
                    "reason": str(resolved.get("reason", "")).strip(),
                },
            }
            try:
                queue_id = await get_wake_queue().enqueue(payload)
                resolved["wake_queued"] = True
                resolved["wake_queue_id"] = queue_id
            except Exception:
                resolved["wake_queued"] = False
        return resolved

    @app.post("/internal/approvals/evaluate")
    async def internal_approvals_evaluate(request: Request) -> Any:
        broker = get_approvals()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        thread_id = str(body.get("thread_id", "")).strip()
        action_name = str(body.get("action_name", "")).strip()
        action_args = body.get("action_args") if isinstance(body.get("action_args"), dict) else {}
        defer_challenge_delivery = bool(body.get("defer_challenge_delivery", False))
        action_note = (
            str(body.get("action_note", "")).strip()
            or str(body.get("guardrail_note", "")).strip()
            or None
        )
        origin_interface = str(body.get("origin_interface", "")).strip() or None
        origin_user_id = str(body.get("origin_user_id", "")).strip() or None
        origin_conversation_id = str(body.get("origin_conversation_id", "")).strip() or None
        if not customer_id or not thread_id or not action_name:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id, thread_id, and action_name are required"},
            )
        decision = await broker.evaluate_action(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            action_note=action_note,
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
            defer_challenge_delivery=defer_challenge_delivery,
        )
        return decision

    @app.post("/internal/approvals/decide")
    async def internal_approvals_decide(request: Request) -> Any:
        body = await request.json()
        approval_id = str(body.get("approval_id", "")).strip()
        decision_raw = str(body.get("decision", "")).strip().lower()
        decision = decision_raw
        actor_interface = str(body.get("actor_interface", "")).strip()
        actor_id = str(body.get("actor_id", "")).strip()
        if not approval_id or decision not in {"approve", "deny"} or not actor_interface or not actor_id:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        "approval_id, decision (approve|deny), actor_interface, and actor_id are required"
                    )
                },
            )
        return await _decide_approval_and_maybe_wake(
            approval_id=approval_id,
            decision=decision,
            actor_interface=actor_interface,
            actor_id=actor_id,
        )

    @app.post("/internal/approvals/execute")
    async def internal_approvals_execute(request: Request) -> Any:
        broker = get_approvals()
        body = await request.json()
        approval_id = str(body.get("approval_id", "")).strip()
        customer_id = str(body.get("customer_id", "")).strip()
        if not approval_id or not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "approval_id and customer_id are required"}
            )
        agent_runtime = get_agent_runtime()
        if agent_runtime is None:
            return JSONResponse(status_code=503, content={"detail": "agent runtime unavailable"})

        async def _approved_executor(action_name: str, action_args: dict[str, Any], cid: str) -> Any:
            safe_args = action_args if isinstance(action_args, dict) else {}
            if str(action_name or "").strip() in {"tulpa_run_terminal", "routine_create"}:
                safe_args = {**safe_args, "preapproved": True}
            if hasattr(agent_runtime, "execute_tool"):
                return await agent_runtime.execute_tool(
                    action_name=action_name,
                    action_args=safe_args,
                    customer_id=cid,
                    inject_customer_id=True,
                )
            # Backward-compatible fallback for lightweight runtimes in tests.
            tools = getattr(agent_runtime, "_tools", {})
            if not isinstance(tools, dict):
                raise RuntimeError("agent runtime does not expose executable tools")
            tool = tools.get(str(action_name or "").strip())
            if tool is None or not hasattr(tool, "ainvoke"):
                raise RuntimeError(f"unknown tool: {action_name}")
            return await tool.ainvoke(safe_args)

        try:
            result = await broker.execute_approved_action(
                approval_id=approval_id,
                customer_id=customer_id,
                executor=_approved_executor,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": f"execute failed: {exc}"})
        if not bool(result.get("ok")):
            error = str(result.get("error", "")).strip()
            status_code = 400
            if error == "approval_not_found":
                status_code = 404
            elif error == "customer_mismatch":
                status_code = 403
            elif error.startswith("approval_not_executable:"):
                status_code = 409
            logger.warning(
                "Approval execute rejected (status=%s approval_id=%s customer_id=%s error=%s)",
                status_code,
                approval_id,
                customer_id,
                error or "unknown_error",
            )
            return JSONResponse(status_code=status_code, content=result)
        return result

    @app.get("/internal/approvals/{approval_id}")
    async def internal_approvals_get(approval_id: str) -> Any:
        broker = get_approvals()
        payload = broker.get(approval_id)
        if payload is None:
            return JSONResponse(status_code=404, content={"detail": "approval not found"})
        return {"ok": True, "approval": payload}

    @app.get("/internal/approvals/pending/status")
    async def internal_approvals_pending_status(request: Request) -> Any:
        broker = get_approvals()
        customer_id = str(request.query_params.get("customer_id", "")).strip()
        thread_id = str(request.query_params.get("thread_id", "")).strip()
        if not customer_id or not thread_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and thread_id are required"},
            )
        pending = broker.has_pending_for_customer_thread(
            customer_id=customer_id,
            thread_id=thread_id,
        )
        return {"ok": True, "pending": bool(pending)}

    return _decide_approval_and_maybe_wake
