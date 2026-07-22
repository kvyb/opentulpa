"""Seed AgentSpecs shared by migration and the authenticated control plane."""

from opentulpa.specs.models import AgentSpecWrite

DEFAULT_OWNER_SPEC_ID = "owner"
DEFAULT_ROUTINE_SPEC_ID = "routine"
DEFAULT_INTAKE_SPEC_ID = "intake"


def default_agent_spec_writes() -> dict[str, AgentSpecWrite]:
    """Return fresh default writes so callers cannot mutate shared state."""

    return {
        DEFAULT_OWNER_SPEC_ID: AgentSpecWrite(
            name="Owner",
            description="Default private owner agent.",
            runtime_profile="owner",
            model_alias="default",
            instructions="Help the owner accomplish their request safely and directly.",
            isolation="private",
            tool_policy="profile_default",
            memory_scope="owner",
            workspace_scope="read_write",
            allow_delegation=True,
        ),
        DEFAULT_ROUTINE_SPEC_ID: AgentSpecWrite(
            name="Routine",
            description="Default private agent for scheduled owner work.",
            runtime_profile="routine",
            model_alias="default",
            instructions="Execute the scheduled instruction and report the result to the owner.",
            isolation="private",
            tool_policy="profile_default",
            memory_scope="owner",
            workspace_scope="read_only",
        ),
        DEFAULT_INTAKE_SPEC_ID: AgentSpecWrite(
            name="Intake",
            description="Isolated externally triggered intake decision agent.",
            runtime_profile="intake",
            model_alias="default",
            instructions="Return a grounded intake decision using only published knowledge.",
            isolation="external",
            tool_policy="allowlist",
            tools=("knowledge_list", "knowledge_find", "knowledge_query"),
            memory_scope="none",
            workspace_scope="none",
        ),
    }


__all__ = [
    "DEFAULT_INTAKE_SPEC_ID",
    "DEFAULT_OWNER_SPEC_ID",
    "DEFAULT_ROUTINE_SPEC_ID",
    "default_agent_spec_writes",
]
