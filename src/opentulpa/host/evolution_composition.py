"""Compose immutable Python generation evolution for the stable host."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import sysconfig
from importlib import resources
from pathlib import Path
from typing import Any

import opentulpa
from opentulpa.core.config import Settings
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.dependency_resolver import (
    DependencyResolverPolicy,
    TrustedDependencyResolver,
)
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationExecutables,
    EvaluationRunner,
    LocalEvaluationRunner,
    trusted_default_commands,
)
from opentulpa.evolution.generation import StateContract, canonical_json_bytes
from opentulpa.evolution.generation_store import GenerationStore
from opentulpa.evolution.lineage import GitLineage
from opentulpa.evolution.process import run_bounded_process
from opentulpa.evolution.release import AtomicReleasePointer
from opentulpa.evolution.release_builder import (
    DependencyAwareWheelReleaseBuilder,
    ReleaseBuilder,
    ReleaseBuildError,
    TrustedWheelReleaseBuilder,
    WheelReleaseBuildPolicy,
)
from opentulpa.evolution.sandbox import (
    CandidateSandboxPolicy,
    TrustedLocalCandidateBackend,
)
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.evolution.workspace import GitCandidateWorkspace
from opentulpa.host.evolution import (
    HostEvolutionRuntime,
    HostReleaseActivator,
    RuntimeEvolutionEventSink,
    TrustedGenerationReleaseProvider,
    seed_source_repository,
)
from opentulpa.host.runtime import RuntimeSupervisor

_EVALUATOR_VERSION = "host-python-generation-v1"
_INSTALL_PROFILE = "runtime"
_PACKAGING_METADATA = (
    "MANIFEST.in",
    "hatch.toml",
    "hatch_build.py",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "uv.toml",
)
_BRIDGE_ASSETS = (
    (
        "railway_sandbox_bridge/bridge.mjs",
        "opentulpa/railway_sandbox_bridge/bridge.mjs",
    ),
    (
        "railway_sandbox_bridge/package.json",
        "opentulpa/railway_sandbox_bridge/package.json",
    ),
    (
        "railway_sandbox_bridge/package-lock.json",
        "opentulpa/railway_sandbox_bridge/package-lock.json",
    ),
)


class _TrustedHostWheelReleaseBuilder(TrustedWheelReleaseBuilder):
    """Bind the shared host GenerationStore instead of creating public control state."""

    def __init__(
        self,
        *,
        policy: WheelReleaseBuildPolicy,
        store: GenerationStore,
    ) -> None:
        if store.root != policy.generations_root.expanduser().absolute():
            raise ValueError("host generation store does not match the wheel build policy")
        self._policy = policy
        self._store = store
        self._generations_root = store.root
        raw_build_root = (
            policy.build_root.expanduser()
            if policy.build_root is not None
            else store.control_root / "builds"
        )
        self._build_root = self._secure_controller_directory(
            raw_build_root,
            create=True,
            label="generation build root",
        )
        try:
            self._wheelhouse = self._secure_read_only_directory(
                policy.trusted_wheelhouse,
                label="trusted dependency wheelhouse",
            )
        except ValueError as exc:
            raise ReleaseBuildError(
                "trusted offline wheelhouse is unavailable; source evolution cannot build releases"
            ) from exc
        self._validate_wheelhouse()
        self._runner = run_bounded_process

    @staticmethod
    def _secure_read_only_directory(raw_path: Path, *, label: str) -> Path:
        path = raw_path.expanduser().absolute()
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            if os.path.lexists(current) and current.is_symlink():
                raise ValueError(f"{label} has a symbolic-link ancestor")
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError(f"{label} is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError(f"{label} is not a root-owned read-only directory")
        return path


def build_host_evolution_runtime(
    *,
    runtime: RuntimeSupervisor,
    data_root: Path,
    settings: Settings,
    control_root: Path | None = None,
    product_root: Path | None = None,
    generation_store: GenerationStore | None = None,
) -> HostEvolutionRuntime | None:
    """Build the host evolution controller when its trusted source seed is installed."""

    if not settings.evolution_enabled:
        return None
    configured_seed = str(os.environ.get("OPENTULPA_SOURCE_SEED_ROOT") or "").strip()
    seed_root = Path(configured_seed or "/opt/opentulpa-source").expanduser()
    if not seed_root.is_dir():
        return None
    _verify_installed_source_seed(seed_root)

    resolved_data = data_root.expanduser().absolute()
    resolved_control = (
        control_root.expanduser().absolute()
        if control_root is not None
        else resolved_data / "bootstrap"
    )
    del product_root
    evolution_root = resolved_control / "evolution"
    _private_directory(evolution_root)
    repository = seed_source_repository(
        seed_root=seed_root,
        repository=evolution_root / "source",
    )
    worktrees_root = evolution_root / "worktrees"
    initial_worktrees_root = evolution_root / "initial-worktrees"
    artifacts_root = evolution_root / "artifacts"
    for root in (worktrees_root, initial_worktrees_root, artifacts_root):
        _private_directory(root)

    store = generation_store or GenerationStore(
        resolved_data / "runtime-generations",
        control_root=evolution_root / "generation-store",
        quarantine_root=evolution_root / "generation-quarantine",
    )
    if store.root != (resolved_data / "runtime-generations").absolute():
        raise RuntimeError("host generation storage is outside the dedicated runtime root")

    wheelhouse_value = str(os.environ.get("OPENTULPA_TRUSTED_WHEELHOUSE") or "").strip()
    wheelhouse = Path(wheelhouse_value or "/opt/opentulpa-wheelhouse").expanduser().absolute()
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise RuntimeError(
            "host source evolution requires the trusted offline wheelhouse "
            "(OPENTULPA_TRUSTED_WHEELHOUSE)"
        )
    source_mutation_enabled = True

    state_contract = _packaged_state_contract()
    lock_hash = hashlib.sha256((seed_root / "uv.lock").read_bytes()).hexdigest()
    metadata_hashes = _trusted_metadata_hashes(seed_root)
    bridge_assets = _trusted_bridge_assets(seed_root)
    policy = WheelReleaseBuildPolicy(
        generations_root=store.root,
        build_root=evolution_root / "generation-builds",
        base_dependency_lock_hash=lock_hash,
        state_contract=state_contract,
        trusted_metadata_hashes=metadata_hashes,
        trusted_wheelhouse=wheelhouse,
        external_python_runtime_policy_sha256=_external_python_runtime_policy_sha256(),
        install_profile=_INSTALL_PROFILE,
        uv_cli=_trusted_uv_cli(),
        trusted_bridge_assets=bridge_assets,
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
    )
    try:
        fixed_builder = _TrustedHostWheelReleaseBuilder(policy=policy, store=store)
    except ReleaseBuildError as exc:
        raise RuntimeError(
            "host source evolution cannot use the configured trusted offline wheelhouse"
        ) from exc

    resolver_image = str(
        os.environ.get("OPENTULPA_DEPENDENCY_RESOLVER_IMAGE_DIGEST") or ""
    ).strip()
    resolver_volume = str(
        os.environ.get("OPENTULPA_DEPENDENCY_RESOLVER_VOLUME") or ""
    ).strip()
    resolver_volume_root = Path("/var/lib/opentulpa-dependency-resolver")
    resolver_bases_root = resolved_data / "runtime-dependency-bases"
    resolver_state_root = evolution_root / "dependency-resolver"
    if resolver_volume:
        resolver_bases_root = resolver_volume_root / "bases"
        resolver_state_root = resolver_volume_root / "state"
    dependency_resolver = (
        TrustedDependencyResolver(
            policy=DependencyResolverPolicy(
                bases_root=resolver_bases_root,
                state_root=resolver_state_root,
                trusted_pyproject=seed_root / "pyproject.toml",
                trusted_lock=seed_root / "uv.lock",
                resolver_image_digest=resolver_image,
                container_cli=str(os.environ.get("OPENTULPA_CONTAINER_CLI") or "docker"),
                container_volume_name=resolver_volume or None,
                container_volume_root=resolver_volume_root if resolver_volume else None,
            )
        )
        if resolver_image
        else None
    )
    builder: ReleaseBuilder = fixed_builder
    if dependency_resolver is not None:
        builder = DependencyAwareWheelReleaseBuilder(
            base_builder=fixed_builder,
            base_policy=policy,
            resolver=dependency_resolver,
            builder_factory=lambda selected: _TrustedHostWheelReleaseBuilder(
                policy=selected,
                store=store,
            ),
        )

    sandbox_policy = CandidateSandboxPolicy(
        cpu_limit=settings.sandbox_cpu_limit,
        memory_limit=settings.sandbox_memory_limit,
        pid_limit=settings.sandbox_pid_limit,
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
        network_enabled=False,
    )

    def candidate_backend(workspace: Path) -> TrustedLocalCandidateBackend:
        return TrustedLocalCandidateBackend(
            workspace=workspace,
            allowed_root=worktrees_root,
            policy=sandbox_policy,
        )

    evaluator_executables = _trusted_local_evaluation_executables()
    evaluator_runner: EvaluationRunner
    evaluator_runner = LocalEvaluationRunner(
        fingerprint_context=lambda: _trusted_local_evaluation_context(evaluator_executables)
    )
    evaluator = CandidateEvaluator(
        runner=evaluator_runner,
        commands=trusted_default_commands(
            timeout_seconds=900,
            executables=evaluator_executables,
        ),
    )

    def dependency_evaluator(base: Any) -> CandidateEvaluator:
        extra_env = {
            "MYPYPATH": str(base.dependency_site),
            "PYTHONPATH": str(base.dependency_site),
        }
        runner = LocalEvaluationRunner(
            extra_env=extra_env,
            fingerprint_context=lambda: {
                **_trusted_local_evaluation_context(evaluator_executables),
                "dependency_base_id": str(base.id),
                "dependency_site_sha256": str(base.site_sha256),
            },
        )
        return CandidateEvaluator(
            runner=runner,
            commands=trusted_default_commands(
                timeout_seconds=900,
                executables=evaluator_executables,
            ),
        )

    def release_evaluator_fingerprint(release: Any) -> str:
        lock_hash = str(release.metadata.get("dependency_lock_hash") or "")
        if not lock_hash or lock_hash == policy.base_dependency_lock_hash:
            return evaluator.fingerprint
        if dependency_resolver is None:
            raise RuntimeError("resolved dependency evaluator is unavailable")
        base = dependency_resolver.base_for_lock(lock_hash)
        if base is None:
            raise RuntimeError("resolved dependency base is unavailable")
        expected_metadata = {
            "dependency_base_id": base.id,
            "dependency_inventory_sha256": base.inventory_sha256,
            "dependency_resolver_fingerprint": base.resolver_fingerprint,
            "dependency_site_sha256": base.site_sha256,
            "dependency_wheelhouse_sha256": base.wheelhouse_sha256,
        }
        if any(release.metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError("resolved dependency provenance is inconsistent")
        return dependency_evaluator(base).fingerprint

    lineage = GitLineage(repository, worktrees_root=worktrees_root)
    archive = EvolutionArchive(evolution_root / "archive.db")
    activator = HostReleaseActivator(
        runtime=runtime,
        generation_store=store,
        state_contract=state_contract,
        evaluator_fingerprint=evaluator.fingerprint,
        evaluator_fingerprint_resolver=release_evaluator_fingerprint,
        install_profile=_INSTALL_PROFILE,
        controller_protocol=state_contract.runtime_protocol,
    )
    service = EvolutionSupervisor(
        archive=archive,
        workspaces=GitCandidateWorkspace(
            source_repository=repository,
            worktrees_root=worktrees_root,
            artifacts_root=artifacts_root,
        ),
        candidate_backend_factory=candidate_backend,
        evaluator=evaluator,
        evaluator_version=_EVALUATOR_VERSION,
        release_pointer=AtomicReleasePointer(evolution_root / "current_release.json"),
        source_ref="refs/opentulpa/instance",
        upstream_repository=settings.evolution_upstream_repository,
        upstream_ref=settings.evolution_upstream_ref,
        release_builder=builder,
        release_activator=activator,
        lineage=lineage,
        event_sink=RuntimeEvolutionEventSink(runtime),
        source_mutation_enabled=source_mutation_enabled,
        source_mutation_unavailable_reason=None,
        dependency_resolver=dependency_resolver,
        dependency_evaluator_factory=(
            dependency_evaluator if dependency_resolver is not None else None
        ),
    )
    initial = TrustedGenerationReleaseProvider(
        source_repository=repository,
        worktrees_root=initial_worktrees_root,
        builder=builder,
        evaluator_version=_EVALUATOR_VERSION,
        evaluator_fingerprint=lambda: evaluator.fingerprint,
        state_contract=state_contract,
        install_profile=_INSTALL_PROFILE,
    )
    return HostEvolutionRuntime(
        runtime=runtime,
        archive=archive,
        evolution=service,
        activator=activator,
        initial_release=initial,
    )


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError("host evolution controller root is unsafe")
    path.chmod(0o700)


def _trusted_local_evaluation_executables() -> EvaluationExecutables:
    binary_root = Path(sys.executable).absolute().parent

    def tool(name: str) -> str:
        candidate = binary_root / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        resolved = shutil.which(name)
        if resolved is not None:
            path = Path(resolved).expanduser().absolute()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        raise RuntimeError(f"trusted local evaluation requires {name} on PATH")

    return EvaluationExecutables(
        python=sys.executable,
        ruff=tool("ruff"),
        mypy=tool("mypy"),
        pytest=tool("pytest"),
    )


def _trusted_local_evaluation_context(executables: EvaluationExecutables) -> dict[str, str]:
    context: dict[str, str] = {"mode": "trusted-local"}
    tools = {
        "python": executables.python,
        "ruff": executables.ruff,
        "mypy": executables.mypy,
        "pytest": executables.pytest,
    }
    for name, value in tools.items():
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"trusted local evaluation requires executable {name}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"trusted local evaluation requires executable {name}")
        context[f"tool.{name}.path"] = str(path)
        context[f"tool.{name}.resolved_path"] = str(resolved)
        context[f"tool.{name}.sha256"] = _local_file_sha256(resolved)
    return context


def _local_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_uv_cli() -> str:
    configured = str(os.environ.get("OPENTULPA_UV_BIN") or "").strip()
    candidate = Path(configured or shutil.which("uv") or "").expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("host generation builder requires the installed trusted uv executable")
    if candidate.name != "uv" or not os.access(candidate, os.X_OK):
        raise RuntimeError("host generation builder requires the installed trusted uv executable")
    return str(candidate)


def _verify_installed_source_seed(seed_root: Path) -> None:
    expected = str(os.environ.get("OPENTULPA_SOURCE_SEED_OID") or "").strip().lower()
    expected_digest = str(
        os.environ.get("OPENTULPA_SOURCE_SEED_SHA256") or ""
    ).strip().lower()
    provenance_path = str(
        os.environ.get("OPENTULPA_SOURCE_SEED_PROVENANCE") or ""
    ).strip()
    if not expected_digest and provenance_path:
        provenance = Path(provenance_path).expanduser().absolute()
        metadata = provenance.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("installed source seed provenance is unsafe")
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise RuntimeError("installed source seed provenance is invalid")
        expected_digest = str(payload.get("source_seed_sha256") or "").strip().lower()
    if expected_digest:
        if len(expected_digest) != 64 or any(
            value not in "0123456789abcdef" for value in expected_digest
        ):
            raise RuntimeError("installed source seed digest is invalid")
        if _source_seed_sha256(seed_root) != expected_digest:
            raise RuntimeError("installed source seed bytes do not match recorded provenance")
    if not expected:
        metadata = seed_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise RuntimeError("installed source seed is mutable or unsafe")
        return
    if len(expected) not in {40, 64} or any(value not in "0123456789abcdef" for value in expected):
        raise RuntimeError("installed source seed identity is invalid")
    if not (seed_root / ".git").exists():
        if not expected_digest:
            raise RuntimeError("installed source seed snapshot has no content provenance")
        metadata = seed_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise RuntimeError("installed source seed snapshot is mutable or unsafe")
        return
    head = run_bounded_process(
        ("git", "-C", str(seed_root), "rev-parse", "--verify", "HEAD"),
        cwd=seed_root,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        timeout_seconds=30,
        max_output_bytes=8_192,
    )
    status = run_bounded_process(
        (
            "git",
            "-C",
            str(seed_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
        cwd=seed_root,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        timeout_seconds=30,
        max_output_bytes=64 * 1_024,
    )
    observed = head.output.decode("ascii", errors="ignore").strip().lower()
    if head.returncode != 0 or observed != expected or status.returncode != 0 or status.output:
        raise RuntimeError("installed source seed no longer matches its recorded clean commit")


def _source_seed_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        paths.extend(Path(directory) / name for name in (*directory_names, *file_names))
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind, payload, mode = b"D", b"", 0o755
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = b"F"
            payload = path.read_bytes()
            mode = 0o755 if metadata.st_mode & 0o111 else 0o644
        else:
            raise RuntimeError("installed source seed contains a link or special file")
        digest.update(kind + b"\0")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(f"{mode:o}".encode("ascii") + b"\0")
        digest.update(str(len(payload)).encode("ascii") + b"\0")
        digest.update(payload + b"\0")
    return digest.hexdigest()


def _packaged_state_contract() -> StateContract:
    resource = resources.files(opentulpa).joinpath("resources", "release_contract.json")
    if not resource.is_file():
        raise RuntimeError("packaged generation state contract is unavailable")
    return StateContract.model_validate_json(resource.read_bytes())


def _trusted_metadata_hashes(seed_root: Path) -> dict[str, str]:
    hashes = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative in _PACKAGING_METADATA
        if (path := seed_root / relative).is_file() and not path.is_symlink()
    }
    if "pyproject.toml" not in hashes:
        raise RuntimeError("bundled source has no trusted pyproject metadata")
    return hashes


def _trusted_bridge_assets(seed_root: Path) -> tuple[tuple[str, str, str], ...]:
    assets: list[tuple[str, str, str]] = []
    for source, destination in _BRIDGE_ASSETS:
        path = seed_root / source
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("bundled source has incomplete trusted bridge assets")
        assets.append((source, destination, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(assets)


def _external_python_runtime_policy_sha256() -> str:
    configured = str(
        os.environ.get("OPENTULPA_EXTERNAL_PYTHON_RUNTIME_POLICY_SHA256") or ""
    ).strip()
    if configured:
        if len(configured) != 64 or any(value not in "0123456789abcdef" for value in configured):
            raise RuntimeError("external Python runtime policy digest is invalid")
        return configured
    policy = {
        "policy": "opentulpa-host-cpython-v1",
        "cpython_version": platform.python_version(),
        "cpython_cache_tag": str(sys.implementation.cache_tag or ""),
        "cpython_abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "os_name": os.name,
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "soabi": str(sysconfig.get_config_var("SOABI") or ""),
    }
    return hashlib.sha256(canonical_json_bytes(policy)).hexdigest()


__all__ = ["build_host_evolution_runtime"]
