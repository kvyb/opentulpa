from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from opentulpa.api.principal import CapabilityPrincipalResolver
from opentulpa.capabilities import (
    AgentInterfaceBinding,
    CapabilityAPICredentialService,
    CapabilityAPIScope,
    CapabilityControlService,
    CapabilityCredentialStore,
    CapabilityManifest,
    CapabilityRevisionStore,
    CapabilityTestCheck,
    CapabilityTestStatus,
    CapabilityWorkerManager,
    EvalCommand,
    SecretRequirement,
    SecretSource,
    WorkerHandle,
    WorkerKind,
    WorkerLaunch,
    WorkerRuntime,
    WorkerSpec,
)
from opentulpa.secrets import AesGcmHostKeyCipher, SecretVault, VaultCapabilitySecretResolver
from opentulpa.specs import AgentRunBinding, AgentSpecRef


class _Evaluator:
    async def evaluate(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> Sequence[CapabilityTestCheck]:
        del tenant_id, manifest
        return (
            CapabilityTestCheck(
                name="tests",
                status=CapabilityTestStatus.PASSED,
                message="passed",
            ),
        )


class _Host:
    def __init__(self) -> None:
        self.launches: list[WorkerLaunch] = []
        self.fenced: list[tuple[str, str]] = []

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        self.launches.append(launch)
        return WorkerHandle(
            id=f"worker-{len(self.launches)}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        del handle
        return True

    async def stop(self, handle: WorkerHandle) -> None:
        del handle

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        self.fenced.append((tenant_id, capability_name))


def _manifest(revision: int, *, seed: bool = False) -> CapabilityManifest:
    digest = f"sha256:{revision:064x}"
    scopes = ("agent.runs.submit", "agent.runs.replay", "files.upload")
    token = SecretRequirement(
        name="OPENTULPA_AGENT_API_TOKEN",
        scopes=scopes,
        source=SecretSource.ISSUED,
        required=False,
    )
    return CapabilityManifest(
        name="durable_interface",
        version=f"1.{revision - 1}.0",
        revision=revision,
        artifact_digest=digest,
        workers=(
            WorkerSpec(
                name="durable_interface",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                command=("worker",),
                runtime=WorkerRuntime.OCI,
                image=f"example@{digest}",
                permissions=scopes,
                secrets=(token,),
                agent_binding=AgentInterfaceBinding(
                    agent_spec_id="owner",
                    run_kind="owner",
                    trust_class="owner",
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
        seed=seed,
    )


def _resolver(
    tmp_path: Path,
    *,
    credentials: CapabilityCredentialStore,
    revision: list[int],
) -> VaultCapabilitySecretResolver:
    vault = SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"x" * 32),
    )
    return VaultCapabilitySecretResolver(
        vault,
        capability_credentials=CapabilityAPICredentialService(
            credentials,
            resolve_agent_spec=lambda tenant_id, spec_id: AgentSpecRef(
                tenant_id=tenant_id,
                spec_id=spec_id,
                revision=revision[0],
            ),
        ),
    )


async def _save_test_activate(
    service: CapabilityControlService,
    *,
    manifest: CapabilityManifest,
    expected_latest_revision: int | None,
    expected_generation: int | None,
) -> object:
    service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        manifest=manifest,
        expected_latest_revision=expected_latest_revision,
    )
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name=manifest.name,
        revision=manifest.revision,
    )
    return await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name=manifest.name,
        revision=manifest.revision,
        expected_generation=expected_generation,
    )


@pytest.mark.asyncio
async def test_restart_keeps_exact_agent_spec_binding_and_replay_authority(
    tmp_path: Path,
) -> None:
    revision = [1]
    revisions = CapabilityRevisionStore(tmp_path / "capabilities.db")
    credentials = CapabilityCredentialStore(tmp_path / "credentials.db")
    initial_host = _Host()
    initial = CapabilityControlService(
        revisions=revisions,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(initial_host),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=revision),
        bundled=(),
    )
    active = await _save_test_activate(
        initial,
        manifest=_manifest(1),
        expected_latest_revision=None,
        expected_generation=None,
    )
    assert active.agent_binding is not None
    assert active.agent_binding.agent_spec.revision == 1
    await initial.shutdown()

    revision[0] = 2
    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=revisions,
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=revision),
        bundled=(),
    )
    await restored.start()

    token = restored_host.launches[0].secret_environment["OPENTULPA_AGENT_API_TOKEN"]
    credential = credentials.authenticate(token)
    assert credential is not None
    assert credential.agent_binding == active.agent_binding
    assert CapabilityAPIScope.AGENT_RUN_REPLAY.value in credential.scopes
    assert credential.agent_spec.revision == 1
    assert (
        revisions.active(
            namespace="tenant-a",
            capability_name="durable_interface",
        )
        == active
    )

    refreshed = await restored.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="durable_interface",
        revision=1,
        expected_generation=active.generation,
        refresh_agent_binding=True,
    )
    assert refreshed.generation == active.generation + 1
    assert refreshed.agent_binding is not None
    assert refreshed.agent_binding.agent_spec.revision == 2


@pytest.mark.asyncio
async def test_capability_rollback_preserves_current_generation_binding(
    tmp_path: Path,
) -> None:
    revision = [1]
    revisions = CapabilityRevisionStore(tmp_path / "capabilities.db")
    credentials = CapabilityCredentialStore(tmp_path / "credentials.db")
    service = CapabilityControlService(
        revisions=revisions,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=revision),
        bundled=(),
    )
    first = await _save_test_activate(
        service,
        manifest=_manifest(1),
        expected_latest_revision=None,
        expected_generation=None,
    )
    revision[0] = 2
    second = await _save_test_activate(
        service,
        manifest=_manifest(2),
        expected_latest_revision=1,
        expected_generation=first.generation,
    )
    assert second.agent_binding is not None
    assert second.agent_binding.agent_spec.revision == 2

    revision[0] = 3
    rolled_back = await service.rollback(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name="durable_interface",
        expected_generation=second.generation,
    )
    assert rolled_back.revision == 1
    assert rolled_back.agent_binding == second.agent_binding


@pytest.mark.asyncio
async def test_release_state_rollback_restores_its_exact_agent_binding(tmp_path: Path) -> None:
    revision = [1]
    manifest = _manifest(1, seed=True)
    revisions = CapabilityRevisionStore(tmp_path / "capabilities.db")
    credentials = CapabilityCredentialStore(tmp_path / "credentials.db")
    release_state = tmp_path / "release" / "seed_activations.json"
    service = CapabilityControlService(
        revisions=revisions,
        evaluator=_Evaluator(),
        workers=CapabilityWorkerManager(_Host()),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=revision),
        bundled=(manifest,),
        release_state_path=release_state,
    )
    seeded = service.seed_bundled(tenant_id="tenant-a", actor_id="owner-a")[0]
    await service.test(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name=seeded.name,
        revision=seeded.revision,
    )
    first = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name=seeded.name,
        revision=seeded.revision,
        expected_generation=None,
    )
    release_one = release_state.read_bytes()

    revision[0] = 2
    second = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        capability_name=seeded.name,
        revision=seeded.revision,
        expected_generation=first.generation,
        refresh_agent_binding=True,
    )
    assert second.agent_binding is not None
    assert second.agent_binding.agent_spec.revision == 2
    await service.shutdown()
    release_state.write_bytes(release_one)

    restored_host = _Host()
    restored = CapabilityControlService(
        revisions=revisions,
        workers=CapabilityWorkerManager(restored_host),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=revision),
        bundled=(manifest,),
        release_state_path=release_state,
    )
    await restored.start()
    active = revisions.active(namespace="tenant-a", capability_name=manifest.name)
    assert active is not None
    assert active.generation == second.generation + 1
    assert active.agent_binding == first.agent_binding
    token = restored_host.launches[0].secret_environment["OPENTULPA_AGENT_API_TOKEN"]
    credential = credentials.authenticate(token)
    assert credential is not None
    assert credential.agent_binding == first.agent_binding


@pytest.mark.asyncio
async def test_legacy_unbound_interface_generation_is_fenced_and_revoked(
    tmp_path: Path,
) -> None:
    manifest = _manifest(1)
    revisions_path = tmp_path / "capabilities.db"
    revisions = CapabilityRevisionStore(revisions_path)
    revisions.append(
        namespace="tenant-a",
        manifest=manifest,
        expected_latest_revision=None,
    )
    legacy = revisions.activate(
        namespace="tenant-a",
        capability_name=manifest.name,
        revision=1,
        expected_generation=None,
    )
    with sqlite3.connect(revisions_path) as connection:
        connection.execute("ALTER TABLE capability_activations DROP COLUMN agent_binding_json")
        connection.commit()
    # Upgrading adds a nullable binding column. Existing authority is intentionally
    # not guessed; startup below fences the legacy generation.
    revisions = CapabilityRevisionStore(revisions_path)
    credentials = CapabilityCredentialStore(tmp_path / "credentials.db")
    tenant_hash = hashlib.sha256(b"tenant-a").hexdigest()[:16]
    name_hash = hashlib.sha256(manifest.name.encode()).hexdigest()[:8]
    instance_id = f"cap_{tenant_hash}_{manifest.name[:20]}_{name_hash}_g1"
    issued = credentials.issue(
        tenant_id="tenant-a",
        actor_id="legacy",
        capability_name=manifest.name,
        capability_instance_id=instance_id,
        interface=manifest.name,
        source_id=instance_id,
        channel=manifest.name,
        agent_binding=AgentRunBinding(
            agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=1),
            run_kind="owner",
            trust_class="owner",
        ),
        scopes=("agent.runs.submit",),
    )
    host = _Host()
    service = CapabilityControlService(
        revisions=revisions,
        workers=CapabilityWorkerManager(host),
        secret_resolver=_resolver(tmp_path, credentials=credentials, revision=[2]),
        bundled=(),
    )

    await service.start()

    assert legacy.agent_binding is None
    assert revisions.active(namespace="tenant-a", capability_name=manifest.name) is None
    assert revisions.inactive(namespace="tenant-a", capability_name=manifest.name) == legacy
    assert host.launches == []
    assert host.fenced == [("tenant-a", manifest.name)]
    assert credentials.authenticate(issued.token.get_secret_value()) is None


def test_restricted_binding_cannot_receive_owner_control_scopes(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "credentials.db")
    external = AgentRunBinding(
        agent_spec=AgentSpecRef(
            tenant_id="tenant-a",
            spec_id="public-intake",
            revision=1,
        ),
        run_kind="intake",
        trust_class="external",
    )
    for scope in (
        CapabilityAPIScope.AGENT_RUN_RESUME.value,
        CapabilityAPIScope.NOTIFICATIONS_READ.value,
        CapabilityAPIScope.NOTIFICATIONS_ACK.value,
    ):
        with pytest.raises(ValueError, match="restricted interfaces"):
            store.issue(
                tenant_id="tenant-a",
                actor_id="external",
                capability_name="public_chat",
                capability_instance_id=f"public-chat-{scope}",
                interface="public_chat",
                source_id="public-chat",
                channel="public_chat",
                agent_binding=external,
                scopes=(scope,),
            )

    with pytest.raises(ValueError, match="different tenant"):
        store.issue(
            tenant_id="tenant-b",
            actor_id="external",
            capability_name="public_chat",
            capability_instance_id="public-chat-cross-tenant",
            interface="public_chat",
            source_id="public-chat",
            channel="public_chat",
            agent_binding=external,
            scopes=(CapabilityAPIScope.AGENT_RUN_SUBMIT.value,),
        )

    with pytest.raises(ValueError, match="granted together"):
        AgentRunBinding(
            agent_spec=AgentSpecRef(
                tenant_id="tenant-a",
                spec_id="public-intake",
                revision=1,
            ),
            run_kind="owner",
            trust_class="external",
        )
    with pytest.raises(ValueError, match="owner AgentSpec"):
        AgentRunBinding(
            agent_spec=AgentSpecRef(
                tenant_id="tenant-a",
                spec_id="owner",
                revision=1,
            ),
            run_kind="intake",
            trust_class="external",
        )


def test_restarted_credential_authenticator_exposes_persisted_binding(tmp_path: Path) -> None:
    store = CapabilityCredentialStore(tmp_path / "credentials.db")
    binding = AgentRunBinding(
        agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=4),
        run_kind="owner",
        trust_class="owner",
    )
    issued = store.issue(
        tenant_id="tenant-a",
        actor_id="interface",
        capability_name="chat",
        capability_instance_id="chat-g1",
        interface="chat",
        source_id="chat-g1",
        channel="chat",
        agent_binding=binding,
        scopes=(CapabilityAPIScope.AGENT_RUN_REPLAY.value,),
    )
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v2/agent/runs/pending-r1",
        "raw_path": b"/v2/agent/runs/pending-r1",
        "query_string": b"",
        "headers": [
            (
                b"authorization",
                f"Bearer {issued.token.get_secret_value()}".encode(),
            )
        ],
        "server": ("test", 80),
        "client": ("test", 1),
    }
    from starlette.requests import Request

    principal = CapabilityPrincipalResolver(CapabilityCredentialStore(store.db_path))(
        Request(scope)
    )
    assert principal.agent_binding == binding
