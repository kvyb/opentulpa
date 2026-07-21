"""V2-only FastAPI composition root for OpenTulpa product services."""

from __future__ import annotations

import html
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from opentulpa.api.routes.telegram_deep_agent import (
    TelegramUpdateHandler,
    register_telegram_deep_agent_routes,
)
from opentulpa.api.routes.v2_agent import AgentRunService, register_v2_agent_routes
from opentulpa.api.routes.v2_capabilities import register_v2_capability_routes
from opentulpa.api.routes.v2_control_plane import register_v2_control_plane_routes
from opentulpa.api.routes.v2_evolution import register_v2_evolution_routes
from opentulpa.api.routes.v2_files import FileVaultPort, register_v2_file_routes
from opentulpa.api.routes.v2_intake import register_v2_intake_routes
from opentulpa.api.routes.v2_integrations import (
    IntegrationPort,
    register_v2_integration_routes,
)
from opentulpa.api.routes.v2_notifications import register_v2_notification_routes
from opentulpa.api.routes.v2_principal import V2Principal
from opentulpa.api.routes.v2_schedules import register_v2_schedule_routes
from opentulpa.bootstrap.release_control import (
    ReleaseControlService,
    register_release_control_plane,
)
from opentulpa.core.public_urls import build_public_composio_callback_path
from opentulpa.interfaces.web import register_owner_web_interface

if TYPE_CHECKING:
    from opentulpa.api.routes.v2_evolution import EvolutionService
    from opentulpa.capabilities.service import CapabilityControlService
    from opentulpa.deep_agent.service import DeepAgentService
    from opentulpa.intake.drafts.service import IntakeDraftService
    from opentulpa.intake.poller import IntakePollDispatcher
    from opentulpa.intake.service import IntakeWorkflowService
    from opentulpa.integrations.browser_sessions import TenantBrowserService
    from opentulpa.interfaces.telegram.client import TelegramClient
    from opentulpa.jobs.service import JobService
    from opentulpa.notifications import NotificationService
    from opentulpa.persistence.idempotency import IdempotencyStore
    from opentulpa.schedules.service import ScheduleService
    from opentulpa.secrets import SecretIngressHook, SecretVaultService
    from opentulpa.specs import AgentSpecRef, AgentSpecService, TriggerSpecService
    from opentulpa.specs.dispatcher import TriggerDispatcher

logger = logging.getLogger(__name__)
STARTED_AT = datetime.now(UTC).isoformat()

PrincipalResolver = Callable[[Request], V2Principal | Awaitable[V2Principal]]


def _deployment_identity() -> dict[str, str | None]:
    return {
        "commit_sha": _clean_env("RAILWAY_GIT_COMMIT_SHA") or _clean_env("GIT_COMMIT_SHA"),
        "deployment_id": _clean_env("RAILWAY_DEPLOYMENT_ID")
        or _clean_env("OPENTULPA_DEPLOYMENT_ID"),
        "started_at": STARTED_AT,
    }


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


async def _shutdown_async(name: str, shutdown: Callable[[], Awaitable[None]]) -> None:
    try:
        await shutdown()
    except Exception:
        logger.exception("Failed to shut down %s", name)


def _shutdown_sync(name: str, shutdown: Callable[..., None]) -> None:
    try:
        shutdown(wait=True)
    except Exception:
        logger.exception("Failed to shut down %s", name)


def _register_health_routes(app: FastAPI, agent_service: AgentRunService) -> None:
    @app.get("/healthz", response_model=None)
    async def health() -> dict[str, str | None]:
        return {"status": "ok", **_deployment_identity()}

    @app.get("/agent/healthz", response_model=None)
    async def agent_health() -> JSONResponse:
        healthy = bool(getattr(agent_service, "healthy", lambda: False)())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "backend": "deepagents",
                **_deployment_identity(),
            },
        )


def _register_composio_callback(app: FastAPI) -> None:
    @app.get(build_public_composio_callback_path())
    async def composio_callback_landing(request: Request) -> HTMLResponse:
        connection_id = str(
            request.query_params.get("connectedAccountId")
            or request.query_params.get("connection_id")
            or ""
        ).strip()
        toolkit = str(
            request.query_params.get("toolkit")
            or request.query_params.get("toolkit_slug")
            or request.query_params.get("integration")
            or ""
        ).strip()
        details = ["You can close this tab and return to OpenTulpa."]
        if toolkit:
            details.insert(0, f"Composio finished connecting {toolkit}.")
        if connection_id:
            details.append(f"Connection ID: {connection_id}")
        body = "".join(f"<p>{html.escape(item)}</p>" for item in details)
        return HTMLResponse(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Connection complete</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:640px;"
            "margin:48px auto;padding:0 20px;line-height:1.5;color:#111}"
            "h1{font-size:28px;margin-bottom:16px}p{margin:12px 0}</style>"
            "</head><body><h1>Connection complete</h1>"
            f"{body}</body></html>"
        )


def create_app(
    *,
    agent_service: DeepAgentService,
    job_service: JobService,
    file_vault_service: FileVaultPort,
    integration_service: IntegrationPort | None,
    intake_workflow_service: IntakeWorkflowService,
    intake_draft_service: IntakeDraftService,
    schedule_service: ScheduleService,
    resolve_principal: PrincipalResolver,
    resolve_agent_spec: Callable[[str, str], AgentSpecRef] | None = None,
    secret_ingress: SecretIngressHook | None = None,
    agent_spec_service: AgentSpecService | None = None,
    trigger_spec_service: TriggerSpecService | None = None,
    secret_vault_service: SecretVaultService | None = None,
    capability_service: CapabilityControlService | None = None,
    trigger_dispatcher: TriggerDispatcher | None = None,
    intake_poll_dispatcher: IntakePollDispatcher | None = None,
    telegram_relay: TelegramUpdateHandler | None = None,
    telegram_webhook_secret: str | None = None,
    browser_service: TenantBrowserService | None = None,
    telegram_client: TelegramClient | None = None,
    evolution_service: EvolutionService | None = None,
    idempotency_store: IdempotencyStore | None = None,
    release_control_service: ReleaseControlService | None = None,
    notification_service: NotificationService | None = None,
    max_file_upload_bytes: int = 45_000_000,
    local_owner_session_token: str | None = None,
) -> FastAPI:
    """Create the public V2 API without constructing product or runtime services."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        consumers_enabled = str(
            os.environ.get("OPENTULPA_DISABLE_CONSUMERS", "") or ""
        ).strip().casefold() not in {"1", "true", "yes", "on"}
        try:
            if evolution_service is not None:
                await evolution_service.start()
            if capability_service is None:
                await agent_service.start()
            else:
                # Open the durable runtime first, but wait to resume approved dynamic
                # tools until their exact persisted capability generation is restored.
                await agent_service.start(recover_pending_resumes=False)
            if consumers_enabled:
                relay_start = getattr(telegram_relay, "start", None)
                if callable(relay_start):
                    await relay_start()
                await job_service.start()
                await intake_workflow_service.start()
                if capability_service is not None:
                    await capability_service.start()
                    await agent_service.recover_pending_resumes()
                if trigger_dispatcher is not None:
                    trigger_dispatcher.start()
                if intake_poll_dispatcher is not None:
                    intake_poll_dispatcher.start()
            yield
        finally:
            relay_shutdown = getattr(telegram_relay, "shutdown", None)
            if callable(relay_shutdown):
                await _shutdown_async("Telegram relay", relay_shutdown)
            if intake_poll_dispatcher is not None:
                _shutdown_sync("intake poll dispatcher", intake_poll_dispatcher.shutdown)
            if trigger_dispatcher is not None:
                _shutdown_sync("trigger dispatcher", trigger_dispatcher.shutdown)
            if browser_service is not None:
                await _shutdown_async("browser service", browser_service.shutdown)
            if capability_service is not None:
                await _shutdown_async("capability service", capability_service.shutdown)
            await _shutdown_async("intake workflow service", intake_workflow_service.shutdown)
            await _shutdown_async("job service", job_service.shutdown)
            await _shutdown_async("Deep Agent service", agent_service.shutdown)
            if evolution_service is not None:
                await _shutdown_async("evolution supervisor", evolution_service.shutdown)
            if telegram_client is not None:
                await _shutdown_async("Telegram client", telegram_client.aclose)

    app = FastAPI(
        title="OpenTulpa API",
        version="2",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.agent_service = agent_service
    app.state.job_service = job_service
    app.state.file_vault_service = file_vault_service
    app.state.integration_service = integration_service
    app.state.intake_workflow_service = intake_workflow_service
    app.state.intake_draft_service = intake_draft_service
    app.state.schedule_service = schedule_service
    app.state.intake_poll_dispatcher = intake_poll_dispatcher
    app.state.telegram_relay = telegram_relay
    app.state.telegram_client = telegram_client
    app.state.browser_service = browser_service
    app.state.evolution_service = evolution_service
    app.state.idempotency_store = idempotency_store
    app.state.resolve_principal = resolve_principal
    app.state.agent_spec_service = agent_spec_service
    app.state.trigger_spec_service = trigger_spec_service
    app.state.secret_vault_service = secret_vault_service
    app.state.capability_service = capability_service
    app.state.trigger_dispatcher = trigger_dispatcher
    app.state.release_control = release_control_service
    app.state.notification_service = notification_service

    if release_control_service is not None:
        register_release_control_plane(app, release_control_service)

    _register_health_routes(app, agent_service)
    register_owner_web_interface(app, local_owner_token=local_owner_session_token)
    register_v2_agent_routes(
        app,
        get_agent_service=lambda: agent_service,
        resolve_principal=resolve_principal,
        resolve_agent_spec=resolve_agent_spec,
        secret_ingress=secret_ingress,
    )
    register_v2_evolution_routes(
        app,
        get_evolution_service=lambda: evolution_service,
        resolve_principal=resolve_principal,
        get_idempotency_store=lambda: idempotency_store,
    )
    register_v2_capability_routes(
        app,
        get_capabilities=lambda: capability_service,
        resolve_principal=resolve_principal,
    )
    register_v2_control_plane_routes(
        app,
        get_agent_specs=lambda: agent_spec_service,
        get_trigger_specs=lambda: trigger_spec_service,
        get_secret_vault=lambda: secret_vault_service,
        resolve_principal=resolve_principal,
        on_trigger_changed=(trigger_dispatcher.upsert if trigger_dispatcher is not None else None),
        on_trigger_deactivated=(
            (
                lambda tenant_id, trigger_id: trigger_dispatcher.remove(
                    tenant_id=tenant_id,
                    trigger_id=trigger_id,
                )
            )
            if trigger_dispatcher is not None
            else None
        ),
        dispatch_trigger_event=(
            trigger_dispatcher.dispatch_event if trigger_dispatcher is not None else None
        ),
    )
    register_v2_file_routes(
        app,
        get_file_vault=lambda: file_vault_service,
        get_idempotency_store=lambda: idempotency_store,
        resolve_principal=resolve_principal,
        max_upload_bytes=max_file_upload_bytes,
    )
    register_v2_integration_routes(
        app,
        get_integration_service=lambda: integration_service,
        resolve_principal=resolve_principal,
    )
    register_v2_notification_routes(
        app,
        get_notifications=lambda: notification_service,
        resolve_principal=resolve_principal,
    )
    register_v2_intake_routes(
        app,
        get_draft_service=lambda: intake_draft_service,
        get_intake_workflows=lambda: intake_workflow_service,
        resolve_principal=resolve_principal,
        on_workflow_changed=(
            intake_poll_dispatcher.upsert if intake_poll_dispatcher is not None else None
        ),
        on_workflow_deleted=(
            (
                lambda tenant_id, workflow_id: intake_poll_dispatcher.remove(
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                )
            )
            if intake_poll_dispatcher is not None
            else None
        ),
    )
    register_v2_schedule_routes(
        app,
        get_schedule_service=lambda: schedule_service,
        resolve_principal=resolve_principal,
    )
    register_telegram_deep_agent_routes(
        app,
        get_relay=lambda: telegram_relay,
        webhook_secret=telegram_webhook_secret,
    )
    _register_composio_callback(app)
    return app


__all__ = ["PrincipalResolver", "create_app"]
