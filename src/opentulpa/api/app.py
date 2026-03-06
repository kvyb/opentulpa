"""FastAPI application: health, internal API, Telegram webhook, and agent runtime."""

from __future__ import annotations

import ipaddress
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.routes import (
    register_approval_routes,
    register_chat_routes,
    register_file_routes,
    register_health_routes,
    register_memory_routes,
    register_profile_routes,
    register_scheduler_routes,
    register_skill_routes,
    register_slack_routes,
    register_task_routes,
    register_telegram_webhook_routes,
    register_tulpa_routes,
    register_wake_and_search_routes,
)
from opentulpa.api.tulpa_loader import TulpaRouterLoader
from opentulpa.application import (
    ApprovalExecutionOrchestrator,
    TurnOrchestrator,
    WakeOrchestrator,
)
from opentulpa.approvals.adapters.telegram import TelegramApprovalAdapter
from opentulpa.approvals.broker import ApprovalBroker
from opentulpa.approvals.store import PendingApprovalStore
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.core.config import get_settings
from opentulpa.integrations.slack_client import grant_slack_write_consent, has_slack_write_consent
from opentulpa.interfaces.telegram.chat_service import TelegramChatService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.sandbox import PROJECT_ROOT
from opentulpa.tasks.sandbox import delete_file as sandbox_delete_file
from opentulpa.tasks.service import TaskService
from opentulpa.tasks.wake_queue import WakeQueueService

logger = logging.getLogger(__name__)


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"{name} not initialized")
    return value


def _is_trusted_server_client(host: str) -> bool:
    value = str(host or "").strip().lower()
    if not value:
        return False
    if value in {"localhost", "testclient"}:
        return True
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


def create_app(
    memory: MemoryService | None = None,
    scheduler: SchedulerService | None = None,
    slack_client: Any | None = None,
    task_service: TaskService | None = None,
    agent_runtime: Any | None = None,
    context_events: EventContextService | None = None,
    customer_profile_service: CustomerProfileService | None = None,
    file_vault_service: FileVaultService | None = None,
    link_alias_service: LinkAliasService | None = None,
    skill_store_service: SkillStoreService | None = None,
) -> FastAPI:
    """Create FastAPI app with internal API, webhook, and agent runtime."""
    memory_service = memory
    scheduler_service = scheduler
    slack_service = slack_client
    task_runner = task_service
    runtime = agent_runtime
    settings = get_settings()
    context_events_service = context_events or EventContextService(
        db_path=PROJECT_ROOT / ".opentulpa" / "context_events.db"
    )
    profile_service = customer_profile_service or CustomerProfileService(
        db_path=PROJECT_ROOT / ".opentulpa" / "customer_profiles.db"
    )
    vault_service = file_vault_service or FileVaultService(
        root_dir=PROJECT_ROOT / ".opentulpa" / "file_vault",
        db_path=PROJECT_ROOT / ".opentulpa" / "file_vault.db",
    )
    link_alias_db = Path(settings.link_alias_db_path)
    if not link_alias_db.is_absolute():
        link_alias_db = (PROJECT_ROOT / link_alias_db).resolve()
    alias_service = link_alias_service or LinkAliasService(db_path=link_alias_db)
    skill_service = skill_store_service or SkillStoreService(
        db_path=PROJECT_ROOT / ".opentulpa" / "skills.db",
        root_dir=PROJECT_ROOT / ".opentulpa" / "skills",
    )
    skill_service.ensure_default_skill()
    if runtime is not None and getattr(runtime, "_link_alias_service", None) is None:
        runtime._link_alias_service = alias_service  # type: ignore[attr-defined]

    telegram_client = (
        TelegramClient(settings.telegram_bot_token) if settings.telegram_bot_token else None
    )
    telegram_chat = (
        TelegramChatService(
            bot_token=settings.telegram_bot_token,
            file_vault=vault_service,
            memory=memory_service,
        )
        if settings.telegram_bot_token
        else None
    )

    def get_memory() -> MemoryService:
        return _require(memory_service, "MemoryService")

    def get_scheduler() -> SchedulerService:
        return _require(scheduler_service, "SchedulerService")

    def get_slack() -> Any:
        return _require(slack_service, "Slack")

    def get_tasks() -> TaskService:
        return _require(task_runner, "TaskService")

    def get_context_events() -> EventContextService:
        return _require(context_events_service, "EventContextService")

    def get_profiles() -> CustomerProfileService:
        return _require(profile_service, "CustomerProfileService")

    def get_file_vault() -> FileVaultService:
        return _require(vault_service, "FileVaultService")

    def get_skill_store() -> SkillStoreService:
        return _require(skill_service, "SkillStoreService")

    def get_telegram_chat() -> TelegramChatService:
        return _require(telegram_chat, "TelegramChatService")

    def get_telegram_client() -> TelegramClient:
        return _require(telegram_client, "TelegramClient")

    def get_agent_runtime() -> Any:
        return runtime

    turn_orchestrator = TurnOrchestrator(agent_runtime=runtime)

    def get_turn_orchestrator() -> TurnOrchestrator:
        return turn_orchestrator

    def resolve_approval_origin(customer_id: str, thread_id: str) -> dict[str, Any]:
        if telegram_chat is None:
            return {}
        slots = telegram_chat.find_session_slots(customer_id)
        if not slots:
            return {}
        selected = None
        safe_thread = str(thread_id or "").strip()
        for slot in slots:
            if safe_thread and safe_thread in {
                str(slot.get("thread_id", "")).strip(),
                str(slot.get("wake_thread_id", "")).strip(),
            }:
                selected = slot
                break
        if selected is None:
            selected = slots[0]
        chat_id = str(selected.get("chat_id", "")).strip()
        user_id = str(selected.get("user_id", "")).strip()
        if not chat_id:
            return {}
        return {
            "origin_interface": "telegram",
            "origin_user_id": user_id,
            "origin_conversation_id": chat_id,
        }

    approval_db = Path(settings.approvals_db_path)
    if not approval_db.is_absolute():
        approval_db = (PROJECT_ROOT / approval_db).resolve()
    approval_store = PendingApprovalStore(db_path=approval_db)
    telegram_adapter = TelegramApprovalAdapter(client=telegram_client) if telegram_client else None
    approvals = ApprovalBroker(
        store=approval_store,
        runtime=runtime,
        approval_ttl_minutes=settings.approvals_ttl_minutes,
        adapters={"telegram": telegram_adapter} if telegram_adapter is not None else {},
        origin_resolver=resolve_approval_origin,
    )

    def get_approvals() -> ApprovalBroker:
        return approvals

    approval_execution_orchestrator = ApprovalExecutionOrchestrator(
        get_agent_runtime=get_agent_runtime,
        get_context_events=get_context_events,
    )

    def get_approval_execution_orchestrator() -> ApprovalExecutionOrchestrator:
        return approval_execution_orchestrator

    wake_queue_service: WakeQueueService | None = None
    tulpa_loader: TulpaRouterLoader | None = None
    tulpa_router = APIRouter()

    def get_wake_queue() -> WakeQueueService:
        return _require(wake_queue_service, "WakeQueueService")

    def get_tulpa_loader() -> TulpaRouterLoader:
        return _require(tulpa_loader, "TulpaRouterLoader")

    wake_orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=get_context_events,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        get_approvals=get_approvals,
    )

    async def process_wake_event(body: dict[str, Any]) -> None:
        logger.info("Processing wake event: %s", body)
        await wake_orchestrator.handle_event(body)

    wake_queue_service = WakeQueueService(
        db_path=PROJECT_ROOT / ".opentulpa" / "wake_events.db",
        handler=process_wake_event,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = get_tulpa_loader()
        result = loader.reload()
        if result.get("warnings"):
            logger.info("Tulpa router load warnings: %s", result["warnings"])
        if result.get("errors"):
            logger.warning("Tulpa router load had errors: %s", result["errors"])
        if runtime and hasattr(runtime, "start"):
            await runtime.start()
        if scheduler_service:
            scheduler_service.start()
        if task_runner:
            await task_runner.start()
        if wake_queue_service:
            await wake_queue_service.start()
        yield
        if scheduler_service:
            scheduler_service.shutdown(wait=True)
        if task_runner:
            await task_runner.shutdown()
        if wake_queue_service:
            await wake_queue_service.shutdown()
        if runtime and hasattr(runtime, "shutdown"):
            await runtime.shutdown()

    app = FastAPI(
        title="OpenTulpa",
        description="Background-capable agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def enforce_public_route_boundary(
        request: Request,
        call_next: Any,
    ) -> Any:
        path = request.url.path
        client_host = str(getattr(getattr(request, "client", None), "host", "") or "")
        trusted_server_client = _is_trusted_server_client(client_host)
        public_health_paths = {"/healthz", "/agent/healthz"}

        # Public internet may only reach webhook ingress and read-only health checks.
        if (
            not trusted_server_client
            and not path.startswith("/webhook/")
            and path not in public_health_paths
        ):
            return JSONResponse(status_code=403, content={"detail": "forbidden public endpoint"})
        return await call_next(request)

    tulpa_loader = TulpaRouterLoader(project_root=PROJECT_ROOT, mount_router=tulpa_router)
    app.include_router(tulpa_router, prefix="/tulpa")

    register_health_routes(app, get_agent_runtime=get_agent_runtime)
    register_chat_routes(app, get_turn_orchestrator=get_turn_orchestrator)
    register_memory_routes(app, get_memory=get_memory)
    register_file_routes(
        app,
        get_file_vault=get_file_vault,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        telegram_enabled=bool(settings.telegram_bot_token),
    )
    register_profile_routes(
        app,
        get_profiles=get_profiles,
        get_memory=lambda: memory_service,
    )
    register_skill_routes(
        app,
        get_skill_store=get_skill_store,
        get_memory=lambda: memory_service,
    )

    decide_approval_and_maybe_wake = register_approval_routes(
        app,
        get_approvals=get_approvals,
        get_wake_queue=get_wake_queue,
        get_agent_runtime=get_agent_runtime,
    )
    register_scheduler_routes(
        app,
        get_scheduler=get_scheduler,
        delete_file=sandbox_delete_file,
    )
    register_wake_and_search_routes(
        app,
        get_wake_queue=get_wake_queue,
        llm_model=settings.llm_model,
    )
    register_tulpa_routes(app, get_tulpa_loader=get_tulpa_loader)
    register_task_routes(app, get_tasks=get_tasks)

    if slack_service is not None:
        register_slack_routes(
            app,
            get_slack=get_slack,
            has_write_consent=has_slack_write_consent,
            grant_write_consent=grant_slack_write_consent,
        )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=get_telegram_client,
        get_telegram_chat=get_telegram_chat,
        get_approvals=get_approvals,
        get_agent_runtime=get_agent_runtime,
        get_approval_execution_orchestrator=get_approval_execution_orchestrator,
        decide_approval_and_maybe_wake=decide_approval_and_maybe_wake,
    )

    return app
