"""Runtime path resolution for source checkouts and installed generations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

APPLICATION_ROOT_ENV = "OPENTULPA_APPLICATION_ROOT"
DATA_ROOT_ENV = "OPENTULPA_DATA_ROOT"
CONFIG_FILE_ENV = "OPENTULPA_CONFIG_FILE"
XDG_DATA_HOME_ENV = "XDG_DATA_HOME"


def _environment_value(environment: Mapping[str, str], name: str) -> str | None:
    value = str(environment.get(name) or "").strip()
    return value or None


def _resolve_configured_path(value: str, *, base: Path | None, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base is None:
            raise ValueError(f"{name} must be an absolute path")
        path = base / path
    return path.resolve()


def _source_checkout_root(package_root: Path) -> Path | None:
    src_root = package_root.parent
    project_root = src_root.parent
    if (
        src_root.name == "src"
        and (project_root / "pyproject.toml").is_file()
        and (project_root / "src" / "opentulpa").resolve() == package_root.resolve()
    ):
        return project_root.resolve()
    return None


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Resolved roots shared by startup and configuration loading."""

    application_root: Path
    data_root: Path | None
    config_file: Path | None
    legacy_source_mode: bool

    @property
    def persistent_root(self) -> Path:
        """Return the root containing the existing ``.opentulpa`` state directory."""

        return self.data_root or self.application_root

    @property
    def installed_generation(self) -> bool:
        return not self.legacy_source_mode

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        package_root: Path | None = None,
        home: Path | None = None,
    ) -> RuntimePaths:
        """Resolve runtime roots without deriving installed writable paths from package files."""

        values = os.environ if environment is None else environment
        package = (
            Path(__file__).resolve().parents[1]
            if package_root is None
            else package_root.expanduser().resolve()
        )
        source_root = _source_checkout_root(package)

        configured_application_root = _environment_value(values, APPLICATION_ROOT_ENV)
        configured_data_root = _environment_value(values, DATA_ROOT_ENV)
        if configured_application_root is not None:
            application_root = _resolve_configured_path(
                configured_application_root,
                base=None,
                name=APPLICATION_ROOT_ENV,
            )
            legacy_source_mode = False
        elif source_root is not None:
            application_root = source_root
            legacy_source_mode = True
        elif configured_data_root is not None:
            application_root = _resolve_configured_path(
                configured_data_root,
                base=None,
                name=DATA_ROOT_ENV,
            )
            legacy_source_mode = False
        else:
            configured_data_home = _environment_value(values, XDG_DATA_HOME_ENV)
            if configured_data_home is not None:
                data_home = _resolve_configured_path(
                    configured_data_home,
                    base=None,
                    name=XDG_DATA_HOME_ENV,
                )
            else:
                home_root = (Path.home() if home is None else home).expanduser()
                if not home_root.is_absolute():
                    raise ValueError("the home directory must be an absolute path")
                data_home = home_root / ".local" / "share"
            application_root = (data_home / "opentulpa").resolve()
            legacy_source_mode = False

        if application_root.exists() and not application_root.is_dir():
            raise ValueError(f"{APPLICATION_ROOT_ENV} must identify a directory")

        data_root = (
            _resolve_configured_path(
                configured_data_root,
                base=application_root,
                name=DATA_ROOT_ENV,
            )
            if configured_data_root is not None
            else None
        )
        if data_root is not None and data_root.exists() and not data_root.is_dir():
            raise ValueError(f"{DATA_ROOT_ENV} must identify a directory")

        configured_config_file = _environment_value(values, CONFIG_FILE_ENV)
        config_file = (
            _resolve_configured_path(
                configured_config_file,
                base=application_root,
                name=CONFIG_FILE_ENV,
            )
            if configured_config_file is not None
            else None
        )
        if config_file is not None and not config_file.is_file():
            raise ValueError(f"{CONFIG_FILE_ENV} must identify an existing regular file")

        return cls(
            application_root=application_root,
            data_root=data_root,
            config_file=config_file,
            legacy_source_mode=legacy_source_mode,
        )


__all__ = [
    "APPLICATION_ROOT_ENV",
    "CONFIG_FILE_ENV",
    "DATA_ROOT_ENV",
    "RuntimePaths",
    "XDG_DATA_HOME_ENV",
]
