"""Authenticated usage telemetry API for dashboards."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_customer_id as resolve_customer_id_value
from opentulpa.api.web_auth import web_auth_error
from opentulpa.telemetry.usage import (
    UsageTelemetryQuery,
    UsageTelemetryResponse,
    UsageTelemetryService,
)


def register_usage_telemetry_routes(
    app: FastAPI,
    *,
    web_token: str | None,
    get_usage_telemetry: Callable[[], UsageTelemetryService],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register dashboard-facing token and cost telemetry routes."""

    @app.get("/web/usage", response_model=UsageTelemetryResponse)
    async def web_usage(
        request: Request,
        customer_id: Annotated[str | None, Query()] = None,
        thread_id: Annotated[str | None, Query()] = None,
        workflow_id: Annotated[str | None, Query()] = None,
        since: Annotated[datetime | None, Query()] = None,
        until: Annotated[datetime | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> UsageTelemetryResponse | JSONResponse:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error

        requested_customer_id = _none_if_empty(customer_id)
        if requested_customer_id is None:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id is required"},
            )

        normalized_since = _normalize_datetime(since)
        normalized_until = _normalize_datetime(until)
        if (
            normalized_since is not None
            and normalized_until is not None
            and normalized_since > normalized_until
        ):
            return JSONResponse(
                status_code=400,
                content={"detail": "since must be before or equal to until"},
            )
        resolved_customer_id = resolve_customer_id_value(
            requested_customer_id,
            resolve_customer_id,
        )
        service = get_usage_telemetry()
        return service.query(
            UsageTelemetryQuery(
                customer_id=resolved_customer_id,
                thread_id=_none_if_empty(thread_id),
                workflow_id=_none_if_empty(workflow_id),
                since=normalized_since,
                until=normalized_until,
                limit=limit,
            )
        )


def _none_if_empty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
