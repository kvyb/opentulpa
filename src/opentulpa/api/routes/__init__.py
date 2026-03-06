"""API route registrars."""

from opentulpa.api.routes.approvals import register_approval_routes
from opentulpa.api.routes.chat import register_chat_routes
from opentulpa.api.routes.files import register_file_routes
from opentulpa.api.routes.health import register_health_routes
from opentulpa.api.routes.memory import register_memory_routes
from opentulpa.api.routes.profiles import register_profile_routes
from opentulpa.api.routes.scheduler import register_scheduler_routes
from opentulpa.api.routes.skills import register_skill_routes
from opentulpa.api.routes.slack import register_slack_routes
from opentulpa.api.routes.tasks import register_task_routes
from opentulpa.api.routes.telegram_webhook import register_telegram_webhook_routes
from opentulpa.api.routes.tulpa import register_tulpa_routes
from opentulpa.api.routes.wake_search import register_wake_and_search_routes

__all__ = [
    "register_approval_routes",
    "register_chat_routes",
    "register_file_routes",
    "register_health_routes",
    "register_memory_routes",
    "register_profile_routes",
    "register_scheduler_routes",
    "register_skill_routes",
    "register_slack_routes",
    "register_task_routes",
    "register_telegram_webhook_routes",
    "register_tulpa_routes",
    "register_wake_and_search_routes",
]
