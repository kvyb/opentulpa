from __future__ import annotations

import os
import shlex
import shutil
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.core.config import Settings
from opentulpa.evolution.evaluator import (
    EvaluationCommand,
    IsolatedProcessEvaluationRunner,
)
from opentulpa.evolution.generation import StateContract
from opentulpa.evolution.generation_store import GenerationStore
from opentulpa.evolution.release_builder import (
    ReleaseBuildError,
    WheelReleaseBuildPolicy,
)
from opentulpa.evolution.sandbox import (
    CandidateContainerBackend,
    CandidateProcessBackend,
    CandidateSandboxPolicy,
)
from opentulpa.host import evolution_composition
from opentulpa.host import paths as host_paths_module
from opentulpa.host.evolution import (
    HostEvolutionRuntime,
    HostReleaseActivator,
    seed_source_repository,
)
from opentulpa.host.evolution_composition import (
    _TrustedHostWheelReleaseBuilder,
    build_host_evolution_runtime,
)
from opentulpa.host.runtime import RuntimeGenerationSpec, RuntimeUnavailableError

_EVALUATOR_CONTROLLER_ROOT = Path(sys.executable).absolute().parent.parent
_EVALUATOR_WHEELHOUSE = Path(
    os.environ.get("OPENTULPA_TRUSTED_WHEELHOUSE", "/opt/opentulpa-wheelhouse")
)
_EVALUATOR_UNAVAILABLE = IsolatedProcessEvaluationRunner.unavailable_reason(
    controller_root=_EVALUATOR_CONTROLLER_ROOT,
    wheelhouse=_EVALUATOR_WHEELHOUSE,
)


def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "seed"
    package = root / "src" / "opentulpa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Seed\n", encoding="utf-8")
    return root


def test_source_repository_import_is_persistent_and_tracks_new_bundled_source(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path)
    repository = seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")
    first = _git(repository, "rev-parse", "refs/heads/upstream")

    assert seed_source_repository(seed_root=seed, repository=repository) == repository
    assert _git(repository, "rev-parse", "refs/heads/upstream") == first

    (seed / "README.md").write_text("# Updated\n", encoding="utf-8")
    seed_source_repository(seed_root=seed, repository=repository)
    second = _git(repository, "rev-parse", "refs/heads/upstream")

    assert second != first
    assert _git(repository, "show", f"{second}:README.md") == "# Updated"
    assert _git(repository, "merge-base", "--is-ancestor", first, second, check=False) == ""


def test_seed_import_failure_keeps_prior_upstream_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentulpa.host import evolution as host_evolution

    seed = _seed(tmp_path)
    repository = seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")
    previous = _git(repository, "rev-parse", "refs/heads/upstream")
    (seed / "README.md").write_text("# Interrupted update\n", encoding="utf-8")
    original = host_evolution._git

    def fail_update(repository: Path, *arguments: str, **kwargs: Any) -> Any:
        if arguments and arguments[0] == "update-ref":
            raise RuntimeError("injected seed crash")
        return original(repository, *arguments, **kwargs)

    monkeypatch.setattr(host_evolution, "_git", fail_update)

    with pytest.raises(RuntimeError, match="injected seed crash"):
        seed_source_repository(seed_root=seed, repository=repository)

    assert _git(repository, "rev-parse", "refs/heads/upstream") == previous
    assert _git(repository, "show", "refs/heads/upstream:README.md") == "# Seed"


def test_seed_import_rejects_hard_linked_files(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    os.link(seed / "README.md", seed / "README-copy.md")

    with pytest.raises(RuntimeError, match="hard-linked"):
        seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")


def test_seed_import_rejects_dirty_git_worktree(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    _git(seed, "init")
    _git(seed, "config", "user.name", "Seed Test")
    _git(seed, "config", "user.email", "seed@example.test")
    _git(seed, "add", "--all")
    _git(seed, "commit", "-m", "seed")
    (seed / "README.md").write_text("# Dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly clean"):
        seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")


def test_seed_import_detects_source_mutation_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentulpa.host import evolution as host_evolution

    seed = _seed(tmp_path)
    original = host_evolution.shutil.copytree

    def mutate_after_copy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        (seed / "README.md").write_text("# Changed during import\n", encoding="utf-8")
        return result

    monkeypatch.setattr(host_evolution.shutil, "copytree", mutate_after_copy)

    with pytest.raises(RuntimeError, match="changed while it was imported"):
        seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")


def _state_contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=1,
        product_state_schema=1,
        workspace_api=1,
    )


def _generation_release(*, evaluator: str = f"sha256:{'e' * 64}") -> ReleaseRecord:
    generation_id = "a" * 64
    manifest_digest = f"sha256:{'b' * 64}"
    return ReleaseRecord(
        id="release-generation",
        candidate_id="candidate-generation",
        source_commit="c" * 40,
        artifact_digest=manifest_digest,
        manifest_digest=manifest_digest,
        entrypoint=("venv/bin/python", "-I", "-m", "opentulpa"),
        metadata={
            "artifact_kind": "python_generation",
            "image_reference": f"python-generation:{generation_id}",
            "generation_id": generation_id,
            "manifest_digest": manifest_digest,
            "evaluator_fingerprint": evaluator,
            "state_contract_sha256": _state_contract().sha256(),
            "install_profile": "runtime",
            "controller_protocol": 1,
        },
    )


class _GenerationStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.opens: list[tuple[str, dict[str, object]]] = []

    def open(self, generation_id: str, **values: object) -> object:
        self.opens.append((generation_id, values))
        if self.fail:
            raise RuntimeError("corrupt generation provenance")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                identity=SimpleNamespace(source_commit="c" * 40),
            )
        )


class _GenerationRuntime:
    def __init__(
        self,
        previous: RuntimeGenerationSpec | None = None,
        *,
        fail: bool = False,
        failure_status: str = "ready",
    ) -> None:
        self.generation = previous
        self.status = "ready"
        self.endpoint = "http://runtime.test"
        self.fail = fail
        self.failure_status = failure_status
        self.replacements: list[tuple[RuntimeGenerationSpec, object | None]] = []
        self.project_root = Path("/")

    async def replace_generation(
        self,
        generation: RuntimeGenerationSpec,
        *,
        rollback: object | None = None,
    ) -> None:
        self.replacements.append((generation, rollback))
        if self.fail:
            self.status = self.failure_status
            if self.status != "ready":
                self.endpoint = None
            raise RuntimeUnavailableError("candidate failed and prior generation was restored")
        self.generation = generation

    def set_generation(self, generation: RuntimeGenerationSpec) -> None:
        self.generation = generation


def _generation_activator(
    tmp_path: Path,
    *,
    runtime: _GenerationRuntime,
    store: _GenerationStore,
    evaluator_fingerprint_resolver: Callable[[ReleaseRecord], str] | None = None,
) -> HostReleaseActivator:
    del tmp_path
    return HostReleaseActivator(
        runtime=runtime,  # type: ignore[arg-type]
        generation_store=store,  # type: ignore[arg-type]
        state_contract=_state_contract(),
        evaluator_fingerprint=f"sha256:{'e' * 64}",
        evaluator_fingerprint_resolver=evaluator_fingerprint_resolver,
        install_profile="runtime",
        controller_protocol=1,
    )


def test_host_release_activator_rejects_non_generation_artifacts(
    tmp_path: Path,
) -> None:
    release = _generation_release().model_copy(
        update={"metadata": {"artifact_kind": "oci_image"}}
    )
    activator = _generation_activator(
        tmp_path,
        runtime=_GenerationRuntime(),
        store=_GenerationStore(),
    )

    with pytest.raises(RuntimeError, match="immutable Python generations"):
        activator.generation_spec(release)


@pytest.mark.asyncio
async def test_python_generation_activation_accepts_exact_resolved_evaluator_fingerprint(
    tmp_path: Path,
) -> None:
    resolved_fingerprint = f"sha256:{'f' * 64}"
    release = _generation_release(evaluator=resolved_fingerprint)
    store = _GenerationStore()
    runtime = _GenerationRuntime()
    activator = _generation_activator(
        tmp_path,
        runtime=runtime,
        store=store,
        evaluator_fingerprint_resolver=lambda value: str(
            value.metadata["evaluator_fingerprint"]
        ),
    )

    result = await activator.activate(
        release,
        activation_id="activation-resolved-evaluator",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "active"
    assert store.opens[0][1]["expected_evaluator_fingerprint"] == resolved_fingerprint


@pytest.mark.asyncio
async def test_python_generation_activation_verifies_and_uses_runtime_captured_rollback(
    tmp_path: Path,
) -> None:
    release = _generation_release()
    previous = RuntimeGenerationSpec(
        generation_id="d" * 64,
        expected_manifest_digest=f"sha256:{'1' * 64}",
        expected_state_contract_digest=_state_contract().sha256(),
        expected_evaluator_fingerprint=f"sha256:{'e' * 64}",
        expected_install_profile="runtime",
        controller_protocol=1,
    )
    store = _GenerationStore()
    runtime = _GenerationRuntime(previous, fail=True)
    activator = _generation_activator(tmp_path, runtime=runtime, store=store)

    result = await activator.activate(
        release,
        activation_id="activation-generation",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "rolled_back"
    assert store.opens[0][0] == "a" * 64
    assert runtime.replacements[0][1] is None
    assert runtime.generation == previous


@pytest.mark.parametrize(
    ("runtime_status", "failure_code"),
    [("recovery_required", "release_containment_failed"), ("failed", "release_rollback_failed")],
)
@pytest.mark.asyncio
async def test_generation_activation_reports_failed_without_serving_previous(
    tmp_path: Path,
    runtime_status: str,
    failure_code: str,
) -> None:
    previous = RuntimeGenerationSpec(
        generation_id="d" * 64,
        expected_manifest_digest=f"sha256:{'1' * 64}",
        expected_state_contract_digest=_state_contract().sha256(),
        expected_evaluator_fingerprint=f"sha256:{'e' * 64}",
        expected_install_profile="runtime",
        controller_protocol=1,
    )
    runtime = _GenerationRuntime(previous, fail=True, failure_status=runtime_status)
    activator = _generation_activator(tmp_path, runtime=runtime, store=_GenerationStore())
    result = await activator.activate(
        _generation_release(),
        activation_id=f"activation-{runtime_status}",
        origin=None,
        reason="test",
        rollback=False,
    )
    assert result.status == "failed"
    assert result.failure_code == failure_code


@pytest.mark.asyncio
async def test_python_generation_activation_selects_exact_verified_generation(
    tmp_path: Path,
) -> None:
    release = _generation_release()
    store = _GenerationStore()
    runtime = _GenerationRuntime()
    activator = _generation_activator(tmp_path, runtime=runtime, store=store)

    result = await activator.activate(
        release,
        activation_id="activation-generation-success",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "active"
    assert runtime.generation is not None
    assert runtime.generation.generation_id == "a" * 64
    assert store.opens[0][1]["expected_manifest_digest"] == release.manifest_digest


@pytest.mark.asyncio
async def test_python_generation_rejects_corrupt_release_provenance(tmp_path: Path) -> None:
    store = _GenerationStore()
    runtime = _GenerationRuntime()
    activator = _generation_activator(tmp_path, runtime=runtime, store=store)

    result = await activator.activate(
        _generation_release(evaluator=f"sha256:{'f' * 64}"),
        activation_id="activation-corrupt",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "failed"
    assert store.opens == []
    assert runtime.replacements == []


class _Archive:
    def __init__(self, release: ReleaseRecord) -> None:
        self.release = release

    async def start(self) -> None:
        return None

    async def get_current_release(self) -> ReleaseRecord:
        return self.release


class _InitialRelease:
    async def build(self) -> ReleaseRecord:
        raise AssertionError("non-bootstrap current release must not be rebuilt")


@pytest.mark.asyncio
async def test_prepare_selects_archived_python_generation_without_rebuilding(
    tmp_path: Path,
) -> None:
    release = _generation_release().model_copy(
        update={"metadata": {**_generation_release().metadata}}
    )
    store = _GenerationStore()
    runtime = _GenerationRuntime()
    activator = _generation_activator(tmp_path, runtime=runtime, store=store)
    host = HostEvolutionRuntime(
        runtime=runtime,  # type: ignore[arg-type]
        archive=_Archive(release),  # type: ignore[arg-type]
        evolution=object(),  # type: ignore[arg-type]
        activator=activator,
        initial_release=_InitialRelease(),
    )

    await host.prepare()

    assert runtime.generation is not None
    assert runtime.generation.generation_id == "a" * 64
    assert len(store.opens) == 1


def test_missing_wheelhouse_fails_generation_builder_clearly(tmp_path: Path) -> None:
    store = GenerationStore(
        tmp_path / "runtime-generations",
        control_root=tmp_path / "control",
    )
    policy = WheelReleaseBuildPolicy(
        generations_root=store.root,
        build_root=tmp_path / "builds",
        base_dependency_lock_hash="1" * 64,
        state_contract=_state_contract(),
        trusted_metadata_hashes={"pyproject.toml": "2" * 64},
        trusted_wheelhouse=tmp_path / "missing-wheelhouse",
        external_python_runtime_policy_sha256="3" * 64,
    )

    with pytest.raises(ReleaseBuildError, match="offline wheelhouse"):
        _TrustedHostWheelReleaseBuilder(policy=policy, store=store)


def test_nonroot_composition_keeps_immutable_generation_runtime_without_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _seed(tmp_path)
    (seed / "pyproject.toml").write_text(
        "[project]\nname='opentulpa'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    bridge = seed / "railway_sandbox_bridge"
    bridge.mkdir()
    for name in ("bridge.mjs", "package.json", "package-lock.json"):
        (bridge / name).write_text("{}\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o500)
    monkeypatch.setenv("OPENTULPA_SOURCE_SEED_ROOT", str(seed))
    monkeypatch.setenv("OPENTULPA_TRUSTED_WHEELHOUSE", str(wheelhouse))
    monkeypatch.setenv("OPENTULPA_UV_BIN", str(uv))
    monkeypatch.delenv("OPENTULPA_SOURCE_SEED_OID", raising=False)
    monkeypatch.setattr(
        CandidateProcessBackend,
        "unavailable_reason",
        staticmethod(lambda: "strong sandbox requires Linux namespaces"),
    )
    data = tmp_path / "data"
    data.mkdir(mode=0o700)

    composed = build_host_evolution_runtime(
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
        data_root=data,
        control_root=data / "bootstrap",
        settings=Settings(_env_file=None, evolution_enabled=True),
    )

    assert composed is not None
    assert composed.service.source_mutation_enabled is False


def test_source_mutation_requires_both_shell_and_evaluator_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CandidateProcessBackend,
        "unavailable_reason",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        IsolatedProcessEvaluationRunner,
        "unavailable_reason",
        classmethod(lambda cls, **kwargs: "exact evaluator tools are unavailable"),
    )

    reason = evolution_composition._source_mutation_unavailable_reason(  # noqa: SLF001
        controller_root=tmp_path / "controller",
        wheelhouse=tmp_path / "wheelhouse",
    )

    assert (
        reason == "source mutation evaluator is unavailable: exact evaluator tools are unavailable"
    )


def test_host_runtime_and_candidate_identities_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_paths_module.sys, "platform", "linux")
    monkeypatch.setattr(host_paths_module.os, "geteuid", lambda: 0)

    paths = host_paths_module.HostPaths.from_environment(
        {"OPENTULPA_DATA_ROOT": str(tmp_path / "data")}
    )

    assert (paths.runtime_uid, paths.runtime_gid) == (65_532, 65_532)
    assert (paths.candidate_uid, paths.candidate_gid) == (65_533, 65_533)


def test_final_image_installs_strong_sandbox_and_distinct_candidate_identity() -> None:
    recipe = Path("Dockerfile").read_text(encoding="utf-8")

    assert "bubblewrap curl git openssh-client util-linux" in recipe
    assert "opentulpa-runtime:x:65532:65532" in recipe
    assert "opentulpa-candidate:x:65533:65533" in recipe
    assert "test -x /usr/bin/bwrap" in recipe
    assert "stat -c %u /usr/bin/bwrap" in recipe
    assert "find /usr/bin/bwrap -perm /022" in recipe


def test_bootstrap_generation_reuse_requires_complete_release_provenance() -> None:
    from opentulpa.host import evolution as host_evolution

    expected = _generation_release()
    archived = expected.model_copy(deep=True)

    assert host_evolution._same_release_provenance(archived, expected)
    assert not host_evolution._same_release_provenance(
        archived.model_copy(update={"entrypoint": ("venv/bin/other",)}),
        expected,
    )
    assert not host_evolution._same_release_provenance(
        archived.model_copy(
            update={
                "metadata": {
                    **archived.metadata,
                    "evaluator_fingerprint": f"sha256:{'f' * 64}",
                }
            }
        ),
        expected,
    )


@pytest.mark.skipif(
    not CandidateProcessBackend.is_supported(),
    reason=CandidateProcessBackend.unavailable_reason()
    or "root Linux bubblewrap isolation is required",
)
@pytest.mark.slow
def test_candidate_process_backend_drops_identity_and_has_no_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    workspace.mkdir(parents=True)
    allowed.chmod(0o700)
    (workspace / "source.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "must-not-be-visible")
    backend = CandidateProcessBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=False),
    )
    trusted_python = shutil.which("python")
    assert trusted_python is not None

    result = backend.execute(
        'printf \'%s|%s|%s\' "$(id -u)" "${OPENAI_COMPATIBLE_API_KEY-unset}" '
        '"$(command -v python)"; '
        "printf 'after\\n' > source.txt"
    )

    assert result.exit_code == 0
    uid, api_key, python_path = result.output.split("|")
    assert uid == "65533"
    assert api_key == "unset"
    assert python_path == trusted_python
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "after\n"
    assert workspace.stat().st_uid == 0
    assert (workspace / "source.txt").stat().st_uid == 0
    assert allowed.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(
    not CandidateProcessBackend.is_supported(),
    reason=CandidateProcessBackend.unavailable_reason()
    or "root Linux bubblewrap isolation is required",
)
@pytest.mark.slow
def test_candidate_process_cannot_see_host_state_processes_or_network(tmp_path: Path) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    workspace.mkdir(parents=True)
    controller_sentinel = Path("/var/tmp") / f"opentulpa-controller-{os.getpid()}"
    product_sentinel = Path("/var/tmp") / f"opentulpa-product-{os.getpid()}"
    controller_sentinel.write_text("controller-secret", encoding="utf-8")
    product_sentinel.write_text("product-secret", encoding="utf-8")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = int(server.getsockname()[1])
    trusted_python = shutil.which("python3")
    assert trusted_python is not None
    script = (
        "import os,socket; from pathlib import Path; "
        f"assert not Path({str(controller_sentinel)!r}).exists(); "
        f"assert not Path({str(product_sentinel)!r}).exists(); "
        f"assert not Path('/proc/{os.getpid()}').exists(); "
        "s=socket.socket(); s.settimeout(0.2); "
        f"assert s.connect_ex(('127.0.0.1',{port})) != 0"
    )
    backend = CandidateProcessBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=False),
    )

    try:
        result = backend.execute(
            f"{shlex.quote(trusted_python)} -I -c {shlex.quote(script)}",
            timeout=10,
        )
    finally:
        server.close()
        controller_sentinel.unlink(missing_ok=True)
        product_sentinel.unlink(missing_ok=True)

    assert result.exit_code == 0, result.output


def test_candidate_sandbox_allows_only_internal_symlinks_when_enabled(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    target = workspace / "packages" / "shared.txt"
    target.parent.mkdir(parents=True)
    target.write_text("shared\n", encoding="utf-8")
    (workspace / "shared.txt").symlink_to("packages/shared.txt")

    CandidateContainerBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(allow_internal_symlinks=True),
    )

    (workspace / "escaped.txt").symlink_to("/etc/hosts")
    with pytest.raises(RuntimeError, match="escaped"):
        CandidateContainerBackend(
            workspace=workspace,
            allowed_root=allowed,
            policy=CandidateSandboxPolicy(allow_internal_symlinks=True),
        )


@pytest.mark.skipif(
    _EVALUATOR_UNAVAILABLE is not None,
    reason=_EVALUATOR_UNAVAILABLE or "root Linux evaluator isolation is required",
)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_isolated_evaluator_uses_writable_copy_without_mutating_source(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("trusted\n", encoding="utf-8")
    original_mode = source.stat().st_mode
    runner = IsolatedProcessEvaluationRunner(
        controller_root=_EVALUATOR_CONTROLLER_ROOT,
        wheelhouse=_EVALUATOR_WHEELHOUSE,
    )

    result = await runner.run(
        workspace=workspace,
        command=EvaluationCommand(
            name="readonly",
            argv=(
                runner.executables.python,
                "-I",
                "-c",
                "from pathlib import Path; Path('source.txt').write_text('changed')",
            ),
            timeout_seconds=10,
        ),
    )

    assert result.passed is True
    assert source.read_text(encoding="utf-8") == "trusted\n"
    assert source.stat().st_mode == original_mode


@pytest.mark.skipif(
    _EVALUATOR_UNAVAILABLE is not None,
    reason=_EVALUATOR_UNAVAILABLE or "root Linux evaluator isolation is required",
)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_isolated_evaluator_rejects_inherited_path_executable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    runner = IsolatedProcessEvaluationRunner(
        controller_root=_EVALUATOR_CONTROLLER_ROOT,
        wheelhouse=_EVALUATOR_WHEELHOUSE,
    )

    with pytest.raises(ValueError, match="exact trusted executable"):
        await runner.run(
            workspace=workspace,
            command=EvaluationCommand(
                name="trusted-path",
                argv=("python", "-c", "raise SystemExit(0)"),
                timeout_seconds=10,
            ),
        )


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    import subprocess

    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
