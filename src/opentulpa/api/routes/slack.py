"""Slack internal API route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_slack_routes(
    app: FastAPI,
    *,
    get_slack: Callable[[], Any],
    has_write_consent: Callable[[str], bool],
    grant_write_consent: Callable[[str], None],
) -> None:
    """Register internal Slack endpoints used by tools."""

    @app.get("/internal/slack/channels")
    async def internal_slack_channels(limit: int = 100, cursor: str = "") -> Any:
        result = await get_slack().list_channels(limit=limit, cursor=cursor or None)
        return result

    @app.get("/internal/slack/channels/{channel_id}/history")
    async def internal_slack_history(channel_id: str, limit: int = 20, cursor: str = "") -> Any:
        result = await get_slack().channel_history(channel_id, limit=limit, cursor=cursor or None)
        return result

    @app.post("/internal/slack/consent")
    async def internal_slack_consent(request: Request) -> Any:
        """Grant Slack write consent for this customer (called when user confirms in chat)."""
        body = await request.json()
        customer_id = body.get("customer_id")
        scope = body.get("scope", "write")
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id required"})
        if scope != "write":
            return JSONResponse(status_code=400, content={"detail": "scope must be 'write'"})
        grant_write_consent(customer_id)
        return {"ok": True, "message": "Slack write consent granted."}

    @app.post("/internal/slack/post")
    async def internal_slack_post(request: Request) -> Any:
        body = await request.json()
        customer_id = body.get("customer_id")
        channel_id = body.get("channel_id", "")
        text = body.get("text", "")
        thread_ts = body.get("thread_ts")
        if not customer_id:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "customer_id required",
                    "detail": "customer_id required",
                },
            )
        if not channel_id or not text:
            return JSONResponse(status_code=400, content={"detail": "channel_id and text required"})
        if not has_write_consent(customer_id):
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "error": "consent_required",
                    "message": (
                        "The user has not granted permission to post to Slack. "
                        "Ask the user to confirm they allow the agent to post to Slack on their behalf; "
                        "once they agree, use slack_grant_write_consent and then try posting again."
                    ),
                },
            )
        result = await get_slack().post_message(channel_id, text, thread_ts=thread_ts)
        return result
