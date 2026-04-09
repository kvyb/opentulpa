"""OpenTulpa entry point."""

import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

import uvicorn

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService
from opentulpa.core.config import get_openai_compatible_api_key_from_env, get_settings
from opentulpa.core.public_urls import resolve_public_base_url
from opentulpa.memory.service import MemoryService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from opentulpa.tasks.service import TaskService


async def _wake_callback(payload: dict) -> None:
    """Called when a scheduled routine fires; posts to /internal/wake."""
    import httpx

    settings = get_settings()
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://127.0.0.1:{settings.port}/internal/wake",
                json=payload,
                timeout=5.0,
            )
    except Exception:
        pass


def _mem0_config_openrouter(
    llm_model: str,
    embedding_model: str,
    openrouter_api_key: str | None,
    openrouter_base_url: str,
    qdrant_path: str,
    qdrant_on_disk: bool,
) -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": openrouter_api_key,
                "openai_base_url": openrouter_base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "openai_base_url": openrouter_base_url,
                "api_key": openrouter_api_key,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": qdrant_path,
                "on_disk": qdrant_on_disk,
            },
        },
    }


def _resolve_public_base_url() -> str:
    return resolve_public_base_url()


def _seed_missing_directory_entries(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.rglob("*"):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _alias_directory_into_data_root(project_root: Path, data_root: Path, name: str) -> None:
    target_path = (data_root / name).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    link_path = project_root / name
    if link_path.is_symlink():
        try:
            if link_path.resolve() == target_path:
                return
        except FileNotFoundError:
            pass
        link_path.unlink()
    elif link_path.exists():
        _seed_missing_directory_entries(link_path, target_path)
        shutil.rmtree(link_path)

    link_path.symlink_to(target_path, target_is_directory=True)


def _bootstrap_persistent_storage(project_root: Path, data_root: str | None) -> None:
    raw_root = str(data_root or "").strip()
    if not raw_root:
        return
    resolved_root = Path(raw_root).expanduser()
    if not resolved_root.is_absolute():
        resolved_root = (project_root / resolved_root).resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    _alias_directory_into_data_root(project_root, resolved_root, ".opentulpa")
    _alias_directory_into_data_root(project_root, resolved_root, "tulpa_stuff")


def _ensure_telegram_webhook_secret(settings: Any) -> str:
    secret = str(settings.telegram_webhook_secret or "").strip()
    if secret:
        return secret
    generated = secrets.token_urlsafe(24)
    os.environ["TELEGRAM_WEBHOOK_SECRET"] = generated
    print("TELEGRAM_WEBHOOK_SECRET missing; generated ephemeral secret for this run.")
    return generated


def _telegram_bot_commands() -> list[dict[str, str]]:
    return [
        {"command": "start", "description": "Show quick help and onboarding"},
        {"command": "status", "description": "Check bot and agent status"},
        {"command": "fresh", "description": "Start a fresh chat context"},
        {"command": "debug_logs", "description": "Send the current app log file"},
    ]


def _auto_configure_telegram_webhook(settings: Any) -> None:
    bot_token = str(settings.telegram_bot_token or "").strip()
    if not bot_token:
        return
    public_base_url = _resolve_public_base_url()
    if not public_base_url:
        print(
            "PUBLIC_BASE_URL/RAILWAY_PUBLIC_DOMAIN not set; skipping Telegram webhook auto-config."
        )
        return
    webhook_secret = _ensure_telegram_webhook_secret(settings)
    webhook_url = f"{public_base_url}/webhook/telegram"
    payload = {"url": webhook_url, "secret_token": webhook_secret}
    try:
        import httpx

        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"https://api.telegram.org/bot{bot_token}/setWebhook",
                data=payload,
            )
        if response.status_code != 200:
            print(
                f"Telegram webhook auto-config failed: HTTP {response.status_code} {response.text[:160]}",
                file=sys.stderr,
            )
            return
        data = response.json() if response.content else {}
        if bool(data.get("ok")):
            print(f"Telegram webhook configured: {webhook_url}")
        else:
            print(
                f"Telegram webhook auto-config failed: {data.get('description', 'unknown error')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Telegram webhook auto-config failed: {exc}", file=sys.stderr)


def _auto_configure_telegram_commands(settings: Any) -> None:
    bot_token = str(settings.telegram_bot_token or "").strip()
    if not bot_token:
        return
    payload = {"commands": _telegram_bot_commands()}
    try:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"https://api.telegram.org/bot{bot_token}/setMyCommands",
                json=payload,
            )
        if response.status_code != 200:
            print(
                f"Telegram commands auto-config failed: HTTP {response.status_code} {response.text[:160]}",
                file=sys.stderr,
            )
            return
        data = response.json() if response.content else {}
        if bool(data.get("ok")):
            print("Telegram bot commands configured.")
        else:
            print(
                f"Telegram commands auto-config failed: {data.get('description', 'unknown error')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Telegram commands auto-config failed: {exc}", file=sys.stderr)


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    _bootstrap_persistent_storage(project_root, os.environ.get("OPENTULPA_DATA_ROOT"))
    openrouter_api_key = (
        settings.openai_compatible_api_key or get_openai_compatible_api_key_from_env()
    )
    qdrant_path = Path(settings.mem0_qdrant_path)
    if not qdrant_path.is_absolute():
        qdrant_path = project_root / qdrant_path

    memory = MemoryService(
        user_id=settings.mem0_user_id,
        config=_mem0_config_openrouter(
            settings.llm_model,
            settings.openrouter_embedding_model,
            openrouter_api_key,
            settings.openrouter_base_url,
            str(qdrant_path),
            settings.mem0_qdrant_on_disk,
        ),
    )
    scheduler = SchedulerService()
    scheduler.set_wake_callback(_wake_callback)
    context_events = EventContextService(db_path=project_root / ".opentulpa" / "context_events.db")
    customer_profiles = CustomerProfileService(
        db_path=project_root / ".opentulpa" / "customer_profiles.db"
    )
    customer_profiles.import_legacy(
        directives_db_path=project_root / ".opentulpa" / "directives.db",
        time_profiles_db_path=project_root / ".opentulpa" / "time_profiles.db",
    )
    file_vault = FileVaultService(
        root_dir=project_root / ".opentulpa" / "file_vault",
        db_path=project_root / ".opentulpa" / "file_vault.db",
    )
    thread_rollups = ThreadRollupService(db_path=project_root / ".opentulpa" / "thread_rollups.db")
    link_alias_db = Path(settings.link_alias_db_path)
    if not link_alias_db.is_absolute():
        link_alias_db = project_root / link_alias_db
    link_aliases = LinkAliasService(
        db_path=link_alias_db,
    )
    skill_store = SkillStoreService(
        db_path=project_root / ".opentulpa" / "skills.db",
        root_dir=project_root / ".opentulpa" / "skills",
    )
    skill_store.ensure_default_skill()
    task_service = TaskService(
        db_path=project_root / ".opentulpa" / "tasks.db",
        wake_callback=_wake_callback,
    )
    agent_runtime: OpenTulpaLangGraphRuntime | None = None
    if openrouter_api_key:
        agent_runtime = OpenTulpaLangGraphRuntime(
            app_url=f"http://127.0.0.1:{settings.port}",
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=settings.openrouter_base_url,
            model_name=settings.llm_model,
            reasoning_effort=settings.llm_reasoning_effort,
            wake_classifier_model_name=settings.wake_classifier_model,
            wake_execution_model_name=settings.wake_execution_model,
            telegram_media_model_name=settings.telegram_media_model,
            guardrail_classifier_model_name=settings.guardrail_classifier_model,
            checkpoint_db_path=settings.agent_checkpoint_db_path,
            recursion_limit=settings.agent_recursion_limit,
            max_completion_tokens=settings.agent_max_completion_tokens,
            max_user_reply_chars=settings.agent_max_user_reply_chars,
            context_events=context_events,
            customer_profile_service=customer_profiles,
            thread_rollup_service=thread_rollups,
            link_alias_service=link_aliases,
            context_token_limit=settings.agent_context_token_limit,
            context_recent_tokens=settings.agent_context_recent_tokens,
            context_rollup_tokens=settings.agent_context_rollup_tokens,
            context_compaction_source_tokens=settings.agent_context_compaction_source_tokens,
            proactive_heartbeat_default_hours=settings.proactive_heartbeat_default_hours,
            behavior_log_enabled=settings.agent_behavior_log_enabled,
            behavior_log_path=settings.agent_behavior_log_path,
            browser_use_headless=settings.browser_use_headless,
            browser_use_model_override=settings.browser_use_model,
            browser_use_max_concurrent_tasks=settings.browser_use_max_concurrent_tasks,
            browser_use_task_retention_seconds=settings.browser_use_task_retention_seconds,
            prompt_caching_enabled=settings.agent_prompt_caching_enabled,
            prompt_cache_ttl_1h=settings.agent_prompt_cache_ttl_1h,
        )
    else:
        print(
            "OPENAI_COMPATIBLE_API_KEY is not set; starting FastAPI without AI chat backend. "
            "OPENROUTER_API_KEY is still accepted as a legacy alias. Set key and restart to enable full chat.",
            file=sys.stderr,
        )

    app = create_app(
        memory=memory,
        scheduler=scheduler,
        task_service=task_service,
        agent_runtime=agent_runtime,
        context_events=context_events,
        customer_profile_service=customer_profiles,
        file_vault_service=file_vault,
        link_alias_service=link_aliases,
        skill_store_service=skill_store,
    )
    _auto_configure_telegram_webhook(settings)
    _auto_configure_telegram_commands(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
