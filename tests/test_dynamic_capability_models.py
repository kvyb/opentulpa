import pytest
from pydantic import ValidationError

from opentulpa.capabilities import (
    DEFAULT_TEMPLATE_REGISTRY,
    AgentInterfaceBinding,
    CapabilityManifest,
    EvalCommand,
    SecretRequirement,
    SecretSource,
    ToolExport,
    WorkerKind,
    WorkerRuntime,
    WorkerSpec,
    canonical_json_digest,
)
from opentulpa.tooling import ApprovalMode, IdempotencyMode, ToolEffect


def _digest() -> str:
    return canonical_json_digest({"type": "object", "properties": {}})


def test_template_registry_contains_dynamic_seed_capabilities() -> None:
    # Web and browser are source-bundled modules. Telegram is the only seed
    # capability with a real out-of-process worker lifecycle.
    assert DEFAULT_TEMPLATE_REGISTRY.names() == ("telegram",)
    assert all(manifest.workers for manifest in DEFAULT_TEMPLATE_REGISTRY.manifests)
    assert DEFAULT_TEMPLATE_REGISTRY.get("telegram").workers[0].protocol == ("agent-interface-v1")
    assert DEFAULT_TEMPLATE_REGISTRY.get("telegram").workers[0].healthcheck.kind == ("ready_file")
    assert DEFAULT_TEMPLATE_REGISTRY.get("telegram").workers[0].agent_binding == (
        AgentInterfaceBinding(
            agent_spec_id="owner",
            run_kind="owner",
            trust_class="owner",
        )
    )


def test_interface_agent_api_authority_must_be_explicit_and_restricted() -> None:
    token = SecretRequirement(
        name="OPENTULPA_AGENT_API_TOKEN",
        scopes=("agent.runs.submit",),
        source=SecretSource.ISSUED,
    )

    with pytest.raises(ValidationError, match="require an agent_binding"):
        WorkerSpec(
            name="public_chat",
            kind=WorkerKind.INTERFACE,
            protocol="agent-interface-v1",
            command=("worker",),
            permissions=("agent.runs.submit",),
            secrets=(token,),
        )

    with pytest.raises(ValidationError, match="cannot bind the owner AgentSpec"):
        AgentInterfaceBinding(
            agent_spec_id="owner",
            run_kind="intake",
            trust_class="external",
        )

    external = WorkerSpec(
        name="public_chat",
        kind=WorkerKind.INTERFACE,
        protocol="agent-interface-v1",
        command=("worker",),
        agent_binding=AgentInterfaceBinding(
            agent_spec_id="public-intake",
            run_kind="intake",
            trust_class="external",
        ),
        permissions=("agent.runs.submit",),
        secrets=(token,),
    )
    assert external.agent_binding is not None
    assert external.agent_binding.trust_class == "external"

    with pytest.raises(ValidationError, match="must use the issued secret source"):
        WorkerSpec(
            name="injected_chat",
            kind=WorkerKind.INTERFACE,
            protocol="agent-interface-v1",
            command=("worker",),
            agent_binding=AgentInterfaceBinding(
                agent_spec_id="public-intake",
                run_kind="intake",
                trust_class="external",
            ),
            permissions=("agent.runs.submit",),
            secrets=(token.model_copy(update={"source": SecretSource.TENANT_HANDLE}),),
        )


@pytest.mark.parametrize(
    "name",
    (
        "PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "OPENTULPA_CAPABILITY_CONFIG",
        "OPENTULPA_WORKER_READY_FILE",
        "OPENTULPA_INTERNAL_AGENT_API_URL",
        "NODE_OPTIONS",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        "GCONV_PATH",
        "LOCPATH",
        "NLSPATH",
        "JAVA_TOOL_OPTIONS",
        "DOTNET_STARTUP_HOOKS",
    ),
)
def test_secret_requirements_cannot_override_worker_runtime_environment(name: str) -> None:
    with pytest.raises(ValidationError, match="reserved for worker runtime control"):
        SecretRequirement(name=name, scopes=("capability.invoke",))


def test_worker_contract_is_executable_strict_and_fail_closed() -> None:
    tool = ToolExport(
        name="lookup",
        description="Read a value.",
        schema_digest=_digest(),
        effect=ToolEffect.READ,
        approval=ApprovalMode.AUTO,
        idempotency=IdempotencyMode.NONE,
    )
    worker = WorkerSpec(
        name="lookup_mcp",
        kind=WorkerKind.MCP,
        protocol="mcp-v1",
        command=("python", "-m", "example.lookup"),
        tools=(tool,),
    )
    manifest = CapabilityManifest(
        name="lookup",
        version="1.0.0",
        workers=(worker,),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )

    assert manifest.module_entrypoint == ""
    assert manifest.exported_tools == ("lookup",)
    assert manifest.content_digest.startswith("sha256:")

    with pytest.raises(ValidationError, match="side-effecting MCP tools require idempotency"):
        ToolExport(
            name="send",
            description="Send a value.",
            schema_digest=_digest(),
            effect=ToolEffect.SEND,
            idempotency=IdempotencyMode.NONE,
        )
    with pytest.raises(ValidationError, match="require protocol"):
        WorkerSpec(
            name="wrong",
            kind=WorkerKind.MCP,
            protocol="agent-interface-v1",
            command=("worker",),
        )
    with pytest.raises(ValidationError, match="digest-pinned image"):
        WorkerSpec(
            name="unsafe_image",
            kind=WorkerKind.MCP,
            protocol="mcp-v1",
            runtime=WorkerRuntime.OCI,
            command=("worker",),
            image="example/worker:latest",
        )


def test_manifest_digest_binds_revision_and_worker_policy() -> None:
    base = CapabilityManifest(
        name="versioned",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="trigger",
                kind=WorkerKind.TRIGGER,
                protocol="agent-trigger-v1",
                command=("worker",),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    next_revision = base.model_copy(update={"revision": 2})

    assert base.content_digest != next_revision.content_digest
