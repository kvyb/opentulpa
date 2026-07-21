"""Versioned capability templates shipped with the OpenTulpa seed distribution."""

from __future__ import annotations

from opentulpa.capabilities.declarative import load_declarative_capabilities
from opentulpa.capabilities.models import (
    AgentInterfaceBinding,
    CapabilityManifest,
    EvalCommand,
    HealthCheck,
    NetworkPolicy,
    SecretRequirement,
    SecretSource,
    WorkerKind,
    WorkerSpec,
)
from opentulpa.capabilities.registry import CapabilityRegistry

WEB_CAPABILITY = CapabilityManifest(
    name="web",
    version="1.0.0",
    module="opentulpa.api.app",
    entrypoint="create_app",
    dependencies=("fastapi>=0.109", "uvicorn[standard]>=0.27"),
    services=("http_api",),
    permissions=("network.listen",),
    network=NetworkPolicy(inbound=True),
    eval_commands=(EvalCommand(argv=("pytest", "-q", "tests/test_v2_app.py")),),
    seed=True,
)

BROWSER_CAPABILITY = CapabilityManifest(
    name="browser",
    version="2.0.0",
    module="opentulpa.integrations.browser_use_cloud",
    entrypoint="BrowserUseCloudSessionProvider",
    dependencies=("browser-use-sdk>=1.0.0", "playwright>=1.50"),
    tools=("browser_start", "browser_get", "browser_act", "browser_stop"),
    services=("browser_sessions",),
    permissions=(
        "browser.control",
        "filesystem.profile_storage",
        "network.tenant_allowlist",
        "secrets.read",
    ),
    network=NetworkPolicy(outbound="tenant_allowlist"),
    secrets=(
        SecretRequirement(
            name="BROWSER_USE_API_KEY",
            scopes=("browser.control",),
            source=SecretSource.HOST,
        ),
    ),
    eval_commands=(
        EvalCommand(
            argv=(
                "pytest",
                "-q",
                "tests/test_browser_use_session_provider.py",
                "tests/test_playwright_browser_session.py",
            ),
            timeout_seconds=600,
        ),
    ),
    seed=True,
)

TELEGRAM_CAPABILITY = CapabilityManifest(
    name="telegram",
    version="1.0.0",
    dependencies=("httpx>=0.26",),
    services=("owner_interface",),
    permissions=(
        "agent.runs.submit",
        "agent.runs.replay",
        "agent.runs.resume",
        "files.upload",
        "notifications.read",
        "notifications.ack",
        "secrets.read",
    ),
    network=NetworkPolicy(outbound="allowlist", allowed_hosts=("api.telegram.org:443",)),
    secrets=(
        SecretRequirement(
            name="TELEGRAM_BOT_TOKEN",
            scopes=("telegram.receive", "telegram.send"),
        ),
        SecretRequirement(
            name="OPENTULPA_AGENT_API_TOKEN",
            scopes=(
                "agent.runs.submit",
                "agent.runs.replay",
                "agent.runs.resume",
                "files.upload",
                "notifications.read",
                "notifications.ack",
            ),
            source=SecretSource.ISSUED,
            required=False,
        ),
        SecretRequirement(
            name="OPENTULPA_TELEGRAM_PAIRING_CODE",
            scopes=("telegram.pair",),
            source=SecretSource.HOST,
            required=False,
        ),
    ),
    config_schema={
        "type": "object",
        "properties": {
            "agent_api_url": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "state_path": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "poll_timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 50},
            "max_attachment_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100_000_000,
            },
        },
        "additionalProperties": False,
    },
    workers=(
        WorkerSpec(
            name="telegram_interface",
            kind=WorkerKind.INTERFACE,
            protocol="agent-interface-v1",
            command=("python", "-m", "opentulpa.capability_workers"),
            agent_binding=AgentInterfaceBinding(
                agent_spec_id="owner",
                run_kind="owner",
                trust_class="owner",
            ),
            permissions=(
                "agent.runs.submit",
                "agent.runs.replay",
                "agent.runs.resume",
                "files.upload",
                "notifications.read",
                "notifications.ack",
                "secrets.read",
            ),
            network=NetworkPolicy(
                outbound="allowlist",
                allowed_hosts=("api.telegram.org:443",),
            ),
            secrets=(
                SecretRequirement(
                    name="TELEGRAM_BOT_TOKEN",
                    scopes=("telegram.receive", "telegram.send"),
                ),
                SecretRequirement(
                    name="OPENTULPA_AGENT_API_TOKEN",
                    scopes=(
                        "agent.runs.submit",
                        "agent.runs.replay",
                        "agent.runs.resume",
                        "files.upload",
                        "notifications.read",
                        "notifications.ack",
                    ),
                    source=SecretSource.ISSUED,
                    required=False,
                ),
                SecretRequirement(
                    name="OPENTULPA_TELEGRAM_PAIRING_CODE",
                    scopes=("telegram.pair",),
                    source=SecretSource.HOST,
                    required=False,
                ),
            ),
            healthcheck=HealthCheck(kind="ready_file", timeout_seconds=30),
        ),
    ),
    eval_commands=(
        EvalCommand(
            argv=(
                "pytest",
                "-q",
                "tests/test_capability_worker_telegram.py",
                "tests/test_capability_worker_agent_api.py",
            )
        ),
    ),
    seed=True,
)

# Web and browser describe source-bundled modules. Telegram is the only bundled
# out-of-process capability and therefore the only installable seed template.
BUNDLED_CAPABILITIES: tuple[CapabilityManifest, ...] = (
    WEB_CAPABILITY,
    BROWSER_CAPABILITY,
)
BUNDLED_CAPABILITY_TEMPLATES: tuple[CapabilityManifest, ...] = (
    TELEGRAM_CAPABILITY,
    *load_declarative_capabilities(),
)
DEFAULT_BUNDLED_REGISTRY = CapabilityRegistry(BUNDLED_CAPABILITIES)
DEFAULT_TEMPLATE_REGISTRY = CapabilityRegistry(BUNDLED_CAPABILITY_TEMPLATES)


def create_bundled_registry() -> CapabilityRegistry:
    """Return the currently composed seed registry."""

    return CapabilityRegistry(BUNDLED_CAPABILITIES)


def create_template_registry() -> CapabilityRegistry:
    """Return all versioned capability templates without loading their code."""

    return CapabilityRegistry(BUNDLED_CAPABILITY_TEMPLATES)


__all__ = [
    "BROWSER_CAPABILITY",
    "BUNDLED_CAPABILITIES",
    "BUNDLED_CAPABILITY_TEMPLATES",
    "DEFAULT_BUNDLED_REGISTRY",
    "DEFAULT_TEMPLATE_REGISTRY",
    "TELEGRAM_CAPABILITY",
    "WEB_CAPABILITY",
    "create_bundled_registry",
    "create_template_registry",
]
