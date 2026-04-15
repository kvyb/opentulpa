"""API route registrars."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ROUTE_IMPORTS = {
    "register_approval_routes": "opentulpa.api.routes.approvals",
    "register_chat_routes": "opentulpa.api.routes.chat",
    "register_composio_routes": "opentulpa.api.routes.composio",
    "register_debug_log_routes": "opentulpa.api.routes.debug_logs",
    "register_file_routes": "opentulpa.api.routes.files",
    "register_health_routes": "opentulpa.api.routes.health",
    "register_intake_workflow_routes": "opentulpa.api.routes.intake",
    "register_memory_routes": "opentulpa.api.routes.memory",
    "register_profile_routes": "opentulpa.api.routes.profiles",
    "register_scheduler_routes": "opentulpa.api.routes.scheduler",
    "register_skill_routes": "opentulpa.api.routes.skills",
    "register_system_routes": "opentulpa.api.routes.system",
    "register_task_routes": "opentulpa.api.routes.tasks",
    "register_telegram_business_routes": "opentulpa.api.routes.telegram_business",
    "register_telegram_webhook_routes": "opentulpa.api.routes.telegram_webhook",
    "register_tulpa_routes": "opentulpa.api.routes.tulpa",
    "register_wake_and_search_routes": "opentulpa.api.routes.wake_search",
}

__all__ = list(_ROUTE_IMPORTS)


def __getattr__(name: str) -> Any:
    module_name = _ROUTE_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
