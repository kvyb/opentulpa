"""Compose trusted source evolution for the stable host."""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path

from opentulpa.core.config import Settings
from opentulpa.host.controller_update import SystemdControllerUpdater
from opentulpa.host.evolution import (
    HostEvolutionControlService,
    _ActivationJournal,
    _TrustedSourceWorkspace,
    prepare_live_source_repository,
)
from opentulpa.host.reviewer import DeepAgentReleaseReviewer
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironmentStore,
    RuntimeEnvFileManager,
)


def build_host_evolution_runtime(
    *,
    runtime: RuntimeSupervisor,
    data_root: Path,
    product_root: Path,
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
    resolved_product = product_root.expanduser().absolute()
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
        extras=_runtime_extras(),
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=max_output_bytes,
    )
    workspace = _TrustedSourceWorkspace(
        source_repository=live_repository,
        path=evolution_root / "source",
        max_output_bytes=max_output_bytes,
    )
    return HostEvolutionControlService(
        runtime=runtime,
        workspace=workspace,
        journal=_ActivationJournal(evolution_root / "activations.db"),
        runtime_environment_store=environment_store,
        runtime_env_file_manager=RuntimeEnvFileManager(
            source_root=live_repository,
            runtime=runtime,
        ),
        reviewer=DeepAgentReleaseReviewer(
            runtime,
            runtime_data_root=resolved_product / ".opentulpa",
            api_reasoning_effort=settings.llm_reasoning_effort,
            api_fallback_models=settings.llm_fallback_models,
            provider_order=settings.llm_provider_order,
            max_completion_tokens=settings.agent_max_completion_tokens,
        ),
        controller_updater=_systemd_controller_updater(
            evolution_root=evolution_root,
            source_root=live_repository,
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


def _systemd_controller_updater(
    *,
    evolution_root: Path,
    source_root: Path,
) -> SystemdControllerUpdater | None:
    unit = str(os.environ.get("OPENTULPA_SYSTEMD_UNIT") or "").strip()
    if not unit:
        return None
    if os.geteuid() != 0:
        raise RuntimeError("automatic controller updates require the root-owned host service")
    install_root = _controller_install_root()
    host_executable = Path(
        str(os.environ.get("OPENTULPA_CONTROLLER_HOST_EXECUTABLE") or "")
    ).absolute()
    generation = host_executable.parent.parent
    python = generation / "bin" / "python"
    if (
        not host_executable.is_absolute()
        or not generation.is_relative_to(install_root / "controller" / "generations")
        or generation.is_symlink()
        or not generation.is_dir()
        or not python.exists()
    ):
        raise RuntimeError("automatic controller updates require an installed controller generation")
    try:
        port = int(os.environ.get("PORT") or 8000)
    except ValueError as exc:
        raise RuntimeError("PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise RuntimeError("PORT must be between 1 and 65535")
    return SystemdControllerUpdater(
        state_path=evolution_root / "controller-update.json",
        source_root=source_root,
        install_root=install_root,
        systemd_unit=unit,
        health_url=f"http://127.0.0.1:{port}/agent/healthz",
        systemd_run=_trusted_root_executable("systemd-run"),
        systemctl=_trusted_root_executable("systemctl"),
        git=_trusted_root_executable("git"),
        python_executable=python,
    )


def _controller_install_root() -> Path:
    configured = str(os.environ.get("OPENTULPA_INSTALL_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = str(os.environ.get("XDG_DATA_HOME") or "").strip()
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return (base / "opentulpa" / "install").resolve()


def _trusted_root_executable(name: str) -> Path:
    configured = shutil.which(name, path="/usr/bin:/bin:/usr/sbin:/sbin")
    if configured is None:
        raise RuntimeError(f"automatic controller updates require {name}")
    executable = Path(configured).resolve()
    metadata = executable.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(f"automatic controller updates require trusted {name}")
    return executable


def _runtime_extras() -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[\s,]+", os.environ.get("OPENTULPA_EXTRAS", "")) if part)


__all__ = ["build_host_evolution_runtime"]
