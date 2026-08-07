from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from opentulpa import __main__ as main_module
from opentulpa.core.paths import RuntimePaths


def _source_package(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    package_root = project_root / "src" / "opentulpa"
    package_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'opentulpa'\n")
    return project_root, package_root


def test_runtime_paths_preserve_source_checkout_defaults(tmp_path: Path) -> None:
    project_root, package_root = _source_package(tmp_path)

    paths = RuntimePaths.from_environment({}, package_root=package_root, home=tmp_path / "home")

    assert paths.application_root == project_root.resolve()
    assert paths.data_root is None
    assert paths.persistent_root == project_root.resolve()
    assert paths.legacy_source_mode


def test_runtime_paths_use_external_installed_roots(tmp_path: Path) -> None:
    package_root = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "opentulpa"
    package_root.mkdir(parents=True)
    application_root = tmp_path / "application"
    data_root = tmp_path / "data"
    config_file = tmp_path / "config" / "runtime.yaml"
    config_file.parent.mkdir()
    config_file.write_text("port: 9000\n")

    paths = RuntimePaths.from_environment(
        {
            "OPENTULPA_APPLICATION_ROOT": str(application_root),
            "OPENTULPA_DATA_ROOT": str(data_root),
            "OPENTULPA_CONFIG_FILE": str(config_file),
        },
        package_root=package_root,
        home=tmp_path / "home",
    )

    assert paths.application_root == application_root.resolve()
    assert paths.data_root == data_root.resolve()
    assert paths.config_file == config_file.resolve()
    assert paths.installed_package


def test_installed_defaults_use_home_instead_of_package_or_cwd(tmp_path: Path) -> None:
    package_root = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "opentulpa"
    package_root.mkdir(parents=True)
    home = tmp_path / "home"

    paths = RuntimePaths.from_environment({}, package_root=package_root, home=home)

    assert paths.application_root == (home / ".local" / "share" / "opentulpa").resolve()
    assert paths.data_root is None
    assert paths.installed_package


def test_installed_defaults_respect_xdg_data_home(tmp_path: Path) -> None:
    package_root = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "opentulpa"
    package_root.mkdir(parents=True)
    data_home = tmp_path / "xdg-data"

    paths = RuntimePaths.from_environment(
        {"XDG_DATA_HOME": str(data_home)},
        package_root=package_root,
        home=tmp_path / "ignored-home",
    )

    assert paths.application_root == (data_home / "opentulpa").resolve()
    assert paths.installed_package


def test_relative_data_and_config_paths_resolve_against_application_root(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "installed" / "opentulpa"
    package_root.mkdir(parents=True)
    application_root = tmp_path / "application"
    application_root.mkdir()
    config_file = application_root / "config" / "runtime.yaml"
    config_file.parent.mkdir()
    config_file.write_text("port: 9000\n")

    paths = RuntimePaths.from_environment(
        {
            "OPENTULPA_APPLICATION_ROOT": str(application_root),
            "OPENTULPA_DATA_ROOT": "data",
            "OPENTULPA_CONFIG_FILE": "config/runtime.yaml",
        },
        package_root=package_root,
    )

    assert paths.data_root == (application_root / "data").resolve()
    assert paths.config_file == config_file.resolve()


def test_installed_bootstrap_creates_directories_without_symlinks(tmp_path: Path) -> None:
    application_root = tmp_path / "application"
    data_root = tmp_path / "data"

    main_module._bootstrap_persistent_storage(
        application_root,
        str(data_root),
        legacy_source_mode=False,
    )

    assert (data_root / ".opentulpa").is_dir()
    assert not (data_root / ".opentulpa").is_symlink()
    assert (data_root / "tulpa_stuff").is_dir()
    assert not (data_root / "tulpa_stuff").is_symlink()
    assert application_root.is_dir()


def test_legacy_source_bootstrap_keeps_data_aliases(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"

    main_module._bootstrap_persistent_storage(
        project_root,
        str(data_root),
        legacy_source_mode=True,
    )

    assert (project_root / ".opentulpa").is_symlink()
    assert (project_root / ".opentulpa").resolve() == (data_root / ".opentulpa").resolve()
    assert (project_root / "tulpa_stuff").is_symlink()
    assert (project_root / "tulpa_stuff").resolve() == (data_root / "tulpa_stuff").resolve()


def test_main_passes_explicit_application_root_without_package_aliases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    application_root = tmp_path / "application"
    data_root = tmp_path / "data"
    paths = RuntimePaths(
        application_root=application_root,
        data_root=data_root,
        config_file=None,
        legacy_source_mode=False,
    )
    settings = SimpleNamespace(host="127.0.0.1", port=8123)
    composition = SimpleNamespace(
        app=object(),
        langfuse_tracer=None,
        telegram_webhook_secret=None,
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        main_module,
        "RuntimePaths",
        SimpleNamespace(from_environment=lambda: paths),
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    def build_application(*, project_root: Path, settings: Any) -> Any:
        captured["project_root"] = project_root
        captured["settings"] = settings
        return composition

    monkeypatch.setattr(main_module, "build_application", build_application)
    monkeypatch.setattr(
        main_module, "_auto_configure_telegram_webhook", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *args, **kwargs: None)

    main_module.main()

    assert captured == {"project_root": application_root, "settings": settings}
    assert (data_root / ".opentulpa").is_dir()
    assert (data_root / "tulpa_stuff").is_dir()
    assert application_root.is_dir()
