"""FastAPI application: health, internal API, Telegram webhook, and agent runtime."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.routes import (
    register_chat_routes,
    register_composio_routes,
    register_debug_log_routes,
    register_file_routes,
    register_generic_chat_routes,
    register_health_routes,
    register_intake_workflow_routes,
    register_knowledge_routes,
    register_memory_routes,
    register_profile_routes,
    register_scheduler_routes,
    register_skill_routes,
    register_system_routes,
    register_task_routes,
    register_telegram_business_routes,
    register_telegram_webhook_routes,
    register_tulpa_routes,
    register_user_context_routes,
    register_wake_and_search_routes,
    register_web_event_routes,
)
from opentulpa.api.tulpa_loader import TulpaRouterLoader
from opentulpa.application import (
    TurnOrchestrator,
    WakeOrchestrator,
    WorkflowSetupOrchestrator,
)
from opentulpa.business_knowledge import BusinessKnowledgeService
from opentulpa.business_knowledge.service import OpenAICompatibleKnowledgeOracleClient
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.context.user_context import UserContextService
from opentulpa.core.config import get_openai_compatible_api_key_from_env, get_settings
from opentulpa.core.shutdown_drain import ShutdownDrain
from opentulpa.intake import (
    IntakeWorkflowService,
    WorkflowSetupService,
    WorkflowSetupSessionStore,
)
from opentulpa.interfaces.telegram.business import TelegramBusinessService
from opentulpa.interfaces.telegram.chat_service import TelegramChatService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.sandbox import PROJECT_ROOT
from opentulpa.tasks.sandbox import delete_file as sandbox_delete_file
from opentulpa.tasks.service import TaskService
from opentulpa.tasks.wake_queue import WakeQueueService
from opentulpa.web.events import WebEventStore, set_default_web_event_store

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from opentulpa.integrations.composio import ComposioService


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"{name} not initialized")
    return value


class _DisabledComposioService:
    enabled = False

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "enabled": False,
            "callback_url_configured": False,
            "default_callback_url": None,
            "resolved_callback_url": None,
        }

    def __getattr__(self, name: str) -> Any:
        _ = name
        raise RuntimeError("Composio is not configured")


def _load_composio_service_class() -> type[Any]:
    from opentulpa.integrations.composio import ComposioService

    return ComposioService


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


def _csv_items(value: Any, *, normalize_username: bool = False) -> list[str]:
    out: list[str] = []
    for item in str(value or "").split(","):
        candidate = item.strip()
        if not candidate:
            continue
        if normalize_username:
            candidate = candidate.removeprefix("@").lower()
        out.append(candidate)
    return out


def _owner_customer_id_from_username(*, username: str, state_path: Path) -> str:
    safe_username = str(username or "").strip().removeprefix("@").lower()
    if not safe_username or not state_path.exists():
        return ""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        return ""
    for slot in sessions.values():
        if not isinstance(slot, dict):
            continue
        if str(slot.get("role") or "owner").strip().lower() != "owner":
            continue
        slot_username = str(slot.get("username") or "").strip().removeprefix("@").lower()
        if slot_username != safe_username:
            continue
        customer_id = str(slot.get("customer_id") or "").strip()
        if customer_id:
            return customer_id
        user_id = str(slot.get("user_id") or "").strip()
        if user_id:
            return f"telegram_{user_id}"
    return ""


def _telegram_business_owner_customer_id(
    *,
    allowed_usernames: Any,
    allowed_user_ids: Any,
    state_path: Path,
) -> str:
    for username in _csv_items(allowed_usernames, normalize_username=True):
        customer_id = _owner_customer_id_from_username(username=username, state_path=state_path)
        if customer_id:
            return customer_id
    for candidate in _csv_items(allowed_user_ids):
        try:
            return f"telegram_{int(candidate)}"
        except Exception:
            continue
    return ""


def _business_knowledge_oracle(
    settings: Any,
    *,
    trace_path: Path | None = None,
    langfuse_tracer: Any | None = None,
) -> OpenAICompatibleKnowledgeOracleClient | None:
    api_key = str(
        getattr(settings, "openai_compatible_api_key", None)
        or get_openai_compatible_api_key_from_env()
        or ""
    ).strip()
    if not api_key:
        return None
    return OpenAICompatibleKnowledgeOracleClient(
        api_key=api_key,
        base_url=str(getattr(settings, "openai_compatible_base_url", "") or ""),
        model=str(getattr(settings, "business_knowledge_oracle_model", "") or ""),
        trace_path=trace_path,
        langfuse_tracer=langfuse_tracer,
    )


def _shutdown_drain_timeout_seconds() -> float:
    raw = str(os.environ.get("OPENTULPA_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "") or "").strip()
    if not raw:
        return 300.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Invalid OPENTULPA_SHUTDOWN_DRAIN_TIMEOUT_SECONDS=%r; using 300", raw)
        return 300.0


def create_app(
    memory: MemoryService | None = None,
    scheduler: SchedulerService | None = None,
    task_service: TaskService | None = None,
    agent_runtime: Any | None = None,
    context_events: EventContextService | None = None,
    customer_profile_service: CustomerProfileService | None = None,
    file_vault_service: FileVaultService | None = None,
    link_alias_service: LinkAliasService | None = None,
    skill_store_service: SkillStoreService | None = None,
    composio_service: ComposioService | None = None,
    intake_workflow_service: IntakeWorkflowService | None = None,
    knowledge_service: BusinessKnowledgeService | None = None,
) -> FastAPI:
    """Create FastAPI app with internal API, webhook, and agent runtime."""
    memory_service = memory
    scheduler_service = scheduler
    task_runner = task_service
    runtime = agent_runtime
    settings = get_settings()
    shutdown_drain = ShutdownDrain(timeout_seconds=_shutdown_drain_timeout_seconds())
    context_events_service = context_events or EventContextService(
        db_path=PROJECT_ROOT / ".opentulpa" / "context_events.db"
    )
    web_event_store = WebEventStore(
        db_path=PROJECT_ROOT / ".opentulpa" / "web_events.db"
    )
    set_default_web_event_store(web_event_store)
    profile_service = customer_profile_service or CustomerProfileService(
        db_path=PROJECT_ROOT / ".opentulpa" / "customer_profiles.db"
    )
    configured_owner_customer_id = str(
        getattr(settings, "opentulpa_owner_customer_id", None) or ""
    ).strip()
    vault_service = file_vault_service or FileVaultService(
        root_dir=PROJECT_ROOT / ".opentulpa" / "file_vault",
        db_path=PROJECT_ROOT / ".opentulpa" / "file_vault.db",
    )
    langfuse_tracer = getattr(runtime, "_langfuse_tracer", None)
    knowledge = knowledge_service or BusinessKnowledgeService(
        root_dir=PROJECT_ROOT / ".opentulpa" / "knowledge",
        db_path=PROJECT_ROOT / ".opentulpa" / "knowledge" / "knowledge.db",
        file_vault=vault_service,
        oracle_client=_business_knowledge_oracle(
            settings,
            trace_path=getattr(runtime, "_llm_call_trace_path", None),
            langfuse_tracer=langfuse_tracer,
        ),
        langfuse_tracer=langfuse_tracer,
    )
    user_context_service = UserContextService(
        db_path=PROJECT_ROOT / ".opentulpa" / "user_context.db",
        knowledge_service=knowledge,
        file_vault=vault_service,
    )
    link_alias_db = Path(settings.link_alias_db_path)
    if not link_alias_db.is_absolute():
        link_alias_db = (PROJECT_ROOT / link_alias_db).resolve()
    alias_service = link_alias_service or LinkAliasService(db_path=link_alias_db)
    skill_service = skill_store_service or SkillStoreService(
        db_path=PROJECT_ROOT / ".opentulpa" / "skills.db",
        root_dir=PROJECT_ROOT / ".opentulpa" / "skills",
    )
    composio_api_key = str(settings.composio_api_key or "").strip()
    composio_default_callback_url = str(settings.composio_default_callback_url or "").strip() or None
    if composio_service is not None:
        composio: Any = composio_service
    elif composio_api_key:
        composio_service_class = _load_composio_service_class()
        composio = composio_service_class(
            api_key=composio_api_key,
            default_callback_url=composio_default_callback_url,
        )
    else:
        composio = _DisabledComposioService()
    skill_service.ensure_default_skill()
    if runtime is not None and getattr(runtime, "_link_alias_service", None) is None:
        runtime._link_alias_service = alias_service  # type: ignore[attr-defined]
    if runtime is not None:
        runtime._composio_service = composio  # type: ignore[attr-defined]

    telegram_client = (
        TelegramClient(settings.telegram_bot_token) if settings.telegram_bot_token else None
    )
    telegram_chat = (
        TelegramChatService(
            bot_token=settings.telegram_bot_token,
            file_vault=vault_service,
            memory=memory_service,
            owner_customer_id=configured_owner_customer_id,
            resolve_customer_id=profile_service.resolve_customer_id,
            resolve_telegram_customer_id=profile_service.resolve_telegram_customer_id,
            bind_telegram_customer_id=profile_service.bind_telegram_user_id,
            alias_user_ids=profile_service.alias_user_ids,
        )
        if settings.telegram_bot_token
        else None
    )

    def get_memory() -> MemoryService:
        return _require(memory_service, "MemoryService")

    def get_scheduler() -> SchedulerService:
        return _require(scheduler_service, "SchedulerService")

    def get_tasks() -> TaskService:
        return _require(task_runner, "TaskService")

    def get_context_events() -> EventContextService:
        return _require(context_events_service, "EventContextService")

    def get_web_events() -> WebEventStore:
        return web_event_store

    def get_profiles() -> CustomerProfileService:
        return _require(profile_service, "CustomerProfileService")

    def get_file_vault() -> FileVaultService:
        return _require(vault_service, "FileVaultService")

    def get_knowledge_service() -> BusinessKnowledgeService:
        return _require(knowledge, "BusinessKnowledgeService")

    def get_user_context_service() -> UserContextService:
        return user_context_service

    def get_skill_store() -> SkillStoreService:
        return _require(skill_service, "SkillStoreService")

    def get_composio() -> Any:
        return composio

    def get_intake_workflows() -> IntakeWorkflowService:
        return _require(intake_service, "IntakeWorkflowService")

    def get_workflow_setup_service() -> WorkflowSetupService:
        return _require(workflow_setup_service, "WorkflowSetupService")

    def get_telegram_chat() -> TelegramChatService:
        return _require(telegram_chat, "TelegramChatService")

    def get_telegram_client() -> TelegramClient:
        return _require(telegram_client, "TelegramClient")

    telegram_business = TelegramBusinessService(
        db_path=PROJECT_ROOT / ".opentulpa" / "telegram_business.db",
        owner_customer_id=profile_service.resolve_customer_id(
            configured_owner_customer_id
            or _telegram_business_owner_customer_id(
                allowed_usernames=settings.telegram_allowed_usernames,
                allowed_user_ids=settings.telegram_allowed_user_ids,
                state_path=PROJECT_ROOT / ".opentulpa" / "telegram_state.json",
            )
        ),
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    telegram_business.client = telegram_client

    def get_telegram_business() -> TelegramBusinessService:
        return telegram_business

    def get_agent_runtime() -> Any:
        return runtime

    def get_shutdown_drain() -> ShutdownDrain:
        return shutdown_drain

    intake_service = intake_workflow_service or IntakeWorkflowService(
        db_path=PROJECT_ROOT / ".opentulpa" / "intake_workflows.db",
        project_root=PROJECT_ROOT,
        scheduler=scheduler_service,
        skill_store=skill_service,
        composio=composio,
        telegram_business=telegram_business,
        file_vault=vault_service,
        knowledge_service=knowledge,
        get_agent_runtime=get_agent_runtime if runtime is not None else (lambda: None),
    )

    def support_customer_listing() -> list[dict[str, Any]]:
        by_customer: dict[str, dict[str, Any]] = {}

        def merge(customer_id: Any, **values: Any) -> None:
            cid = str(customer_id or "").strip()
            if not cid:
                return
            item = by_customer.setdefault(cid, {"customer_id": cid})
            for key, value in values.items():
                if value in (None, ""):
                    continue
                if key == "last_activity":
                    item[key] = max(str(item.get(key, "") or ""), str(value))
                elif key.endswith("_count"):
                    item[key] = max(int(item.get(key) or 0), int(value or 0))
                elif isinstance(value, bool):
                    item[key] = bool(item.get(key, False)) or value
                else:
                    item.setdefault(key, value)

        if telegram_chat is not None:
            for item in telegram_chat.list_owner_customer_summaries():
                merge(
                    item.get("customer_id"),
                    owner_chat_id=item.get("owner_chat_id"),
                    owner_user_id=item.get("owner_user_id"),
                    owner_username=item.get("owner_username"),
                    last_activity=item.get("last_activity"),
                )
        for service, method_name in (
            (telegram_business, "list_customer_summaries"),
            (intake_service, "list_customer_summaries"),
            (vault_service, "list_customer_summaries"),
            (profile_service, "list_customer_summaries"),
        ):
            method = getattr(service, method_name, None)
            if not callable(method):
                continue
            with suppress(Exception):
                for item in method():
                    if not isinstance(item, dict):
                        continue
                    last_activity = max(
                        str(item.get("last_business_at", "") or ""),
                        str(item.get("last_workflow_at", "") or ""),
                        str(item.get("last_file_at", "") or ""),
                        str(item.get("last_profile_at", "") or ""),
                    )
                    values = dict(item)
                    values.pop("customer_id", None)
                    merge(item.get("customer_id"), last_activity=last_activity, **values)
        if bool(getattr(composio, "enabled", False)):
            for cid in list(by_customer):
                with suppress(Exception):
                    accounts = composio.list_connected_accounts(
                        customer_id=cid,
                        statuses=["ACTIVE"],
                        limit=1,
                    )
                    merge(
                        cid,
                        composio_connected=bool((accounts or {}).get("items")),
                    )
        return sorted(
            by_customer.values(),
            key=lambda item: (str(item.get("last_activity", "") or ""), str(item.get("customer_id", "") or "")),
            reverse=True,
        )

    if telegram_chat is not None:
        telegram_chat.support_customer_listing = support_customer_listing

    workflow_setup_store = WorkflowSetupSessionStore(
        db_path=PROJECT_ROOT / ".opentulpa" / "intake_workflow_setup.db",
    )
    workflow_setup_service = WorkflowSetupService(
        store=workflow_setup_store,
        intake_workflows=intake_service,
        knowledge_service=knowledge,
    )
    if runtime is not None:
        runtime._workflow_setup_service = workflow_setup_service  # type: ignore[attr-defined]
    workflow_setup_orchestrator = WorkflowSetupOrchestrator(
        setup_service=workflow_setup_service,
    )
    if telegram_chat is not None:
        telegram_chat.workflow_setup_status = workflow_setup_orchestrator.thread_status
        telegram_chat.workflow_setup_after_reply = workflow_setup_orchestrator.after_reply

    turn_orchestrator = TurnOrchestrator(
        agent_runtime=runtime,
        workflow_setup_orchestrator=workflow_setup_orchestrator,
    )

    def get_turn_orchestrator() -> TurnOrchestrator:
        return turn_orchestrator

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
        get_intake_workflows=get_intake_workflows,
        resolve_customer_id=profile_service.resolve_customer_id,
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
        if runtime and hasattr(runtime, "start"):
            await runtime.start()
        if scheduler_service:
            scheduler_service.start()
        if task_runner:
            await task_runner.start()
        if wake_queue_service:
            await wake_queue_service.start()
        if intake_service and hasattr(intake_service, "start"):
            await intake_service.start()
        yield
        drained = await shutdown_drain.drain()
        if not drained:
            logger.warning(
                "Shutdown drain timed out with active_turns=%s",
                shutdown_drain.status().active_turns,
            )
        if intake_service and hasattr(intake_service, "shutdown"):
            await intake_service.shutdown()
        if scheduler_service:
            scheduler_service.shutdown(wait=True)
        if task_runner:
            await task_runner.shutdown()
        if wake_queue_service:
            await wake_queue_service.shutdown()
        if telegram_client and hasattr(telegram_client, "aclose"):
            await telegram_client.aclose()
        if runtime and hasattr(runtime, "shutdown"):
            await runtime.shutdown()

    app = FastAPI(
        title="OpenTulpa",
        description="Persistent agent runtime API with durable context and direct execution",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.wake_queue = wake_queue_service
    app.state.turn_orchestrator = turn_orchestrator
    app.state.composio = composio
    app.state.intake_workflows = intake_service
    app.state.intake_workflow_setup = workflow_setup_service
    app.state.knowledge_service = knowledge
    app.state.user_context_service = user_context_service
    app.state.telegram_business = telegram_business
    app.state.shutdown_drain = shutdown_drain

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
            and path != "/web/events"
            and not path.startswith("/web/chat/")
            and not path.startswith("/web/intake/drafts/")
            and not path.startswith("/web/files/")
            and not path.startswith("/web/local-files/")
            and path not in public_health_paths
        ):
            return JSONResponse(status_code=403, content={"detail": "forbidden public endpoint"})
        return await call_next(request)

    def refresh_tulpa_mounts() -> None:
        kept_routes: list[Any] = []
        for route in app.router.routes:
            path = str(getattr(route, "path", "") or "")
            if path.startswith("/tulpa/"):
                continue
            kept_routes.append(route)
        app.router.routes[:] = kept_routes
        app.include_router(tulpa_router, prefix="/tulpa")

    tulpa_loader = TulpaRouterLoader(
        project_root=PROJECT_ROOT,
        mount_router=tulpa_router,
    )
    tulpa_loader.reload()

    register_health_routes(
        app,
        get_agent_runtime=get_agent_runtime,
        get_shutdown_drain=get_shutdown_drain,
    )
    register_debug_log_routes(app)
    register_chat_routes(
        app,
        get_turn_orchestrator=get_turn_orchestrator,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_generic_chat_routes(
        app,
        web_token=settings.opentulpa_web_token,
        get_agent_runtime=get_agent_runtime,
        get_file_vault=get_file_vault,
        get_workflow_setup_service=lambda: workflow_setup_orchestrator,
        resolve_customer_id=profile_service.resolve_customer_id,
        get_shutdown_drain=get_shutdown_drain,
    )
    register_memory_routes(
        app,
        get_memory=get_memory,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_file_routes(
        app,
        get_file_vault=get_file_vault,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        telegram_enabled=bool(settings.telegram_bot_token),
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_knowledge_routes(
        app,
        get_knowledge_service=get_knowledge_service,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_user_context_routes(
        app,
        get_user_context_service=get_user_context_service,
        get_file_vault=get_file_vault,
        get_agent_runtime=get_agent_runtime,
        resolve_customer_id=profile_service.resolve_customer_id,
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
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_intake_workflow_routes(
        app,
        get_intake_workflows=get_intake_workflows,
        get_workflow_setup_service=get_workflow_setup_service,
        resolve_customer_id=profile_service.resolve_customer_id,
        web_token=settings.opentulpa_web_token,
    )
    register_telegram_business_routes(
        app,
        get_telegram_business=get_telegram_business,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_system_routes(app)
    register_composio_routes(
        app,
        get_composio=get_composio,
        resolve_customer_id=profile_service.resolve_customer_id,
    )

    register_scheduler_routes(
        app,
        get_scheduler=get_scheduler,
        delete_file=sandbox_delete_file,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_wake_and_search_routes(
        app,
        get_wake_queue=get_wake_queue,
        llm_model=settings.llm_model,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_web_event_routes(
        app,
        settings=settings,
        get_web_events=get_web_events,
        resolve_customer_id=profile_service.resolve_customer_id,
    )
    register_tulpa_routes(
        app,
        get_tulpa_loader=get_tulpa_loader,
        refresh_tulpa_mounts=refresh_tulpa_mounts,
    )
    register_task_routes(
        app,
        get_tasks=get_tasks,
        resolve_customer_id=profile_service.resolve_customer_id,
    )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=get_telegram_client,
        get_telegram_business=get_telegram_business,
        get_intake_workflows=get_intake_workflows,
        get_telegram_chat=get_telegram_chat,
        get_agent_runtime=get_agent_runtime,
        get_shutdown_drain=get_shutdown_drain,
    )

    refresh_tulpa_mounts()

    return app
