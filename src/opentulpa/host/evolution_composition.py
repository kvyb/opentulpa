"""Compose trusted source evolution for the stable host."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from opentulpa.core.config import Settings
from opentulpa.host.evolution import (
    HostEvolutionControlService,
    _ActivationJournal,
    _TrustedSourceWorkspace,
    prepare_live_source_repository,
)
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironmentStore,
    RuntimeEnvFileManager,
)


def build_host_evolution_runtime(
    *,
    runtime: RuntimeSupervisor,
    data_root: Path,
    settings: Settings,
    control_root: Path | None = None,
) -> HostEvolutionControlService | None:
    """Build the host-owned trusted source controller."""

    if not settings.evolution_enabled:
        return None
    live_repository = _live_source_repository(runtime)
    if live_repository is None:
        return None

    resolved_data = data_root.expanduser().absolute()
    resolved_control = (
        control_root.expanduser().absolute()
        if control_root is not None
        else resolved_data / "bootstrap"
    )
    evolution_root = resolved_control / "evolution"
    _private_directory(evolution_root)
    max_output_bytes = max(1_024, settings.sandbox_max_output_bytes)
    environment_store = LiveSourceRuntimeEnvironmentStore(
        source_repository=live_repository,
        envs_root=resolved_data / "runtime-source-envs",
        worktrees_root=evolution_root / "runtime-env-worktrees",
        uv_cli=_trusted_uv_cli(),
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=max_output_bytes,
    )
    return HostEvolutionControlService(
        runtime=runtime,
        workspace=_TrustedSourceWorkspace(
            source_repository=live_repository,
            path=evolution_root / "source",
            max_output_bytes=max_output_bytes,
        ),
        journal=_ActivationJournal(evolution_root / "activations.db"),
        runtime_environment_store=environment_store,
        runtime_env_file_manager=RuntimeEnvFileManager(
            source_root=live_repository,
            runtime=runtime,
        ),
        max_output_bytes=max_output_bytes,
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


def _live_source_repository(runtime: RuntimeSupervisor) -> Path | None:
    raw_root = getattr(runtime, "project_root", None)
    if raw_root is None:
        return None
    root = Path(raw_root).expanduser()
    if not (
        root.is_dir()
        and (root / ".git").exists()
        and (root / "src" / "opentulpa" / "__init__.py").is_file()
        and (root / "uv.lock").is_file()
    ):
        return None
    return prepare_live_source_repository(root)


def _trusted_uv_cli() -> str:
    configured = str(os.environ.get("OPENTULPA_UV_BIN") or "").strip()
    candidate = Path(configured or shutil.which("uv") or "").expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("host runtime environment builder requires the trusted uv executable")
    if candidate.name != "uv" or not os.access(candidate, os.X_OK):
        raise RuntimeError("host runtime environment builder requires the trusted uv executable")
    return str(candidate)


__all__ = ["build_host_evolution_runtime"]
