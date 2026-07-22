import importlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from opentulpa.capabilities import (
    BROWSER_CAPABILITY,
    DEFAULT_BUNDLED_REGISTRY,
    WEB_CAPABILITY,
    CapabilityLoadError,
    CapabilityManifest,
    CapabilityNotFoundError,
    CapabilityRegistry,
    CapabilityRegistryError,
    EvalCommand,
    NetworkPolicy,
    SecretRequirement,
    create_bundled_registry,
)


def _manifest(
    name: str,
    *,
    tools: tuple[str, ...] = (),
    services: tuple[str, ...] = (),
) -> CapabilityManifest:
    return CapabilityManifest(
        name=name,
        version="1.0.0",
        module=f"example.{name}",
        entrypoint="create",
        tools=tools,
        services=services,
        eval_commands=(EvalCommand(argv=("pytest", "-q", f"tests/test_{name}.py")),),
    )


def test_default_registry_contains_declarative_web_and_browser_seed_bundles() -> None:
    assert DEFAULT_BUNDLED_REGISTRY.names() == ("web", "browser")
    assert all(manifest.seed for manifest in DEFAULT_BUNDLED_REGISTRY.manifests)

    assert WEB_CAPABILITY.module_entrypoint == "opentulpa.api.app:create_app"
    assert WEB_CAPABILITY.dependencies == (
        "fastapi>=0.109",
        "uvicorn[standard]>=0.27",
    )
    assert WEB_CAPABILITY.services == ("http_api",)
    assert WEB_CAPABILITY.network.inbound is True
    assert WEB_CAPABILITY.network.outbound == "deny"

    assert BROWSER_CAPABILITY.module_entrypoint == (
        "opentulpa.integrations.browser_use_cloud:BrowserUseCloudSessionProvider"
    )
    assert BROWSER_CAPABILITY.tools == (
        "browser_start",
        "browser_get",
        "browser_act",
        "browser_stop",
    )
    assert BROWSER_CAPABILITY.network.outbound == "tenant_allowlist"
    assert {secret.name for secret in BROWSER_CAPABILITY.secrets} == {"BROWSER_USE_API_KEY"}
    assert BROWSER_CAPABILITY.secrets[0].required is True
    assert all(manifest.eval_commands for manifest in DEFAULT_BUNDLED_REGISTRY.manifests)


def test_registry_does_not_import_implementations_until_entrypoint_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    marker = object()

    def create_app() -> object:
        return marker

    def fake_import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(create_app=create_app)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    registry = create_bundled_registry()

    assert imported == []
    assert registry.get("web") is WEB_CAPABILITY
    assert imported == []
    assert registry.load_entrypoint("web")() is marker
    assert imported == ["opentulpa.api.app"]


@pytest.mark.parametrize("export_type", ["tool", "service"])
def test_registry_rejects_ambiguous_export_ownership(export_type: str) -> None:
    kwargs = {f"{export_type}s": ("shared",)}

    with pytest.raises(CapabilityRegistryError, match=f"{export_type} 'shared'"):
        CapabilityRegistry((_manifest("first", **kwargs), _manifest("second", **kwargs)))


def test_registry_rejects_duplicate_capability_names() -> None:
    with pytest.raises(CapabilityRegistryError, match="already registered"):
        CapabilityRegistry((_manifest("duplicate"), _manifest("duplicate")))


def test_manifest_rejects_duplicates_unknown_fields_and_unsafe_network() -> None:
    with pytest.raises(ValidationError, match="tools must be unique"):
        CapabilityManifest(
            name="duplicate_tools",
            version="1.0.0",
            module="example.duplicate_tools",
            entrypoint="create",
            tools=("shared", "shared"),
            eval_commands=(EvalCommand(argv=("pytest", "-q")),),
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CapabilityManifest.model_validate(
            {
                "name": "unexpected",
                "version": "1.0.0",
                "module": "example.unexpected",
                "entrypoint": "create",
                "eval_commands": [{"argv": ["pytest", "-q"]}],
                "shell_command": "pytest -q",
            }
        )

    with pytest.raises(ValidationError, match="requires allowed_hosts"):
        NetworkPolicy(outbound="allowlist")
    with pytest.raises(ValidationError):
        NetworkPolicy(outbound="unrestricted")
    with pytest.raises(ValidationError, match="unsupported characters"):
        EvalCommand(argv=("pytest", "tests/test_safe.py\nrm -rf /"))


def test_manifest_rejects_duplicate_secret_and_eval_declarations() -> None:
    with pytest.raises(ValidationError, match="secrets must be unique"):
        CapabilityManifest(
            name="duplicate_secrets",
            version="1.0.0",
            module="example.duplicate_secrets",
            entrypoint="create",
            secrets=(
                SecretRequirement(name="TOKEN", scopes=("example.invoke",)),
                SecretRequirement(
                    name="TOKEN",
                    scopes=("example.invoke",),
                    required=False,
                ),
            ),
            eval_commands=(EvalCommand(argv=("pytest", "-q")),),
        )

    command = EvalCommand(argv=("pytest", "-q"))
    with pytest.raises(ValidationError, match="eval_commands must be unique"):
        CapabilityManifest(
            name="duplicate_evals",
            version="1.0.0",
            module="example.duplicate_evals",
            entrypoint="create",
            eval_commands=(command, command),
        )


def test_registry_fails_closed_for_unknown_or_invalid_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def fake_import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(create="not callable")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    registry = CapabilityRegistry((_manifest("known", tools=("known_tool",)),))

    with pytest.raises(CapabilityNotFoundError):
        registry.get("unknown")
    with pytest.raises(CapabilityNotFoundError):
        registry.load_entrypoint("unknown")
    assert imported == []

    assert registry.owner_for_tool("known_tool") == "known"
    with pytest.raises(CapabilityNotFoundError):
        registry.owner_for_tool("unknown_tool")
    with pytest.raises(CapabilityLoadError, match="is not callable"):
        registry.load_entrypoint("known")
    assert imported == ["example.known"]


def test_registry_wraps_import_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(_: str) -> SimpleNamespace:
        raise ModuleNotFoundError("dependency missing")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(CapabilityLoadError, match="failed to import capability"):
        CapabilityRegistry((_manifest("broken"),)).load_entrypoint("broken")
