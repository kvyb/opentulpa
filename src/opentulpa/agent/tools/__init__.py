"""Domain-specific tool registration helpers."""

from opentulpa.agent.tools.browser_tools import register_browser_tools
from opentulpa.agent.tools.composio_tools import register_composio_tools
from opentulpa.agent.tools.core_tools import register_core_tools
from opentulpa.agent.tools.intake_tools import register_intake_tools
from opentulpa.agent.tools.routine_tools import register_routine_tools
from opentulpa.agent.tools.skill_tools import register_skill_tools

__all__ = [
    "register_browser_tools",
    "register_composio_tools",
    "register_core_tools",
    "register_intake_tools",
    "register_routine_tools",
    "register_skill_tools",
]
