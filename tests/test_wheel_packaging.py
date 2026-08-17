from __future__ import annotations

import json
import os
import shutil
import sys
import sysconfig
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from opentulpa.evolution.process import BoundedProcessResult, run_bounded_process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (PROJECT_ROOT / "src").resolve()
DEPENDENCY_SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"]).resolve()
BUILD_TIMEOUT_SECONDS = 120
SETUP_TIMEOUT_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 30
MAX_SUBPROCESS_OUTPUT_BYTES = 32 * 1024
PACKAGED_CONFIG = "opentulpa/resources/opentulpa.config.yaml"
RELEASE_CONTRACT = "opentulpa/resources/release_contract.json"
REVIEWER_PROMPT = "opentulpa/host/reviewer_prompt.md"
CONSOLE_SCRIPTS = {
    "opentulpa": "opentulpa.host.cli:main",
    "opentulpa-host": "opentulpa.host.cli:serve",
    "opentulpa-migrate-deepagents": "opentulpa.migrations.deepagents:main",
    "opentulpa-sandbox-worker": "opentulpa.sandbox.worker:main",
}
pytestmark = [pytest.mark.slow, pytest.mark.integration]


@dataclass(frozen=True, slots=True)
class InstalledWheel:
    root: Path
    python: Path
    site_packages: Path
    module_file: Path
    sys_path: tuple[str, ...]
    declared_console_scripts: dict[str, str]
    console_module_files: dict[str, str]
    probe_cwd: Path


def _run_command(
    command: list[str | Path],
    *,
    timeout_seconds: int,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> BoundedProcessResult:
    argv = tuple(str(argument) for argument in command)
    result = run_bounded_process(
        argv,
        cwd=cwd,
        env=os.environ.copy() if env is None else env,
        timeout_seconds=timeout_seconds,
        max_output_bytes=MAX_SUBPROCESS_OUTPUT_BYTES,
    )
    output = result.output.decode(errors="replace")
    if result.truncated:
        output += "\n[output truncated]"
    if result.timed_out:
        pytest.fail(
            f"command timed out after {timeout_seconds}s: {argv!r}\n{output}",
            pytrace=False,
        )
    if check and result.returncode != 0:
        pytest.fail(
            f"command failed with exit code {result.returncode}: {argv!r}\n{output}",
            pytrace=False,
        )
    return result


def _output(result: BoundedProcessResult) -> str:
    return result.output.decode(errors="replace")


@pytest.fixture(scope="module")
def uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the wheel in a clean virtualenv")
    return uv


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory, uv_executable: str) -> Path:
    output_dir = tmp_path_factory.mktemp("wheel-dist")
    _run_command(
        [uv_executable, "build", "--wheel", "--out-dir", str(output_dir)],
        timeout_seconds=BUILD_TIMEOUT_SECONDS,
    )
    wheels = list(output_dir.glob("opentulpa-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def installed_wheel(
    tmp_path_factory: pytest.TempPathFactory,
    uv_executable: str,
    built_wheel: Path,
) -> InstalledWheel:
    parent = tmp_path_factory.mktemp("wheel-venv-parent")
    root = parent / "runtime-venv"
    _run_command(
        [uv_executable, "venv", "--python", sys.executable, str(root)],
        timeout_seconds=SETUP_TIMEOUT_SECONDS,
    )
    python = root / "bin" / "python"
    _run_command(
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(built_wheel),
        ],
        timeout_seconds=SETUP_TIMEOUT_SECONDS,
    )
    probe_cwd = parent / "probe-cwd"
    probe_cwd.mkdir()
    script = """
import importlib
import importlib.metadata
import json
import opentulpa
import sys
import sysconfig

expected = json.loads(sys.argv[1])
distribution = importlib.metadata.distribution("opentulpa")
declared = {
    entry.name: entry.value
    for entry in distribution.entry_points
    if entry.group == "console_scripts"
}
module_files = {}
for name, target in declared.items():
    module_name, attribute_path = target.split(":", 1)
    module = importlib.import_module(module_name)
    value = module
    for part in attribute_path.split("."):
        value = getattr(value, part)
    assert callable(value)
    module_files[name] = module.__file__

print(json.dumps({
    "console_module_files": module_files,
    "declared_console_scripts": declared,
    "module_file": opentulpa.__file__,
    "site_packages": sysconfig.get_paths()["purelib"],
    "sys_path": sys.path,
}))
assert declared == expected
"""
    probe_result = _run_command(
        [python, "-c", script, json.dumps(CONSOLE_SCRIPTS, sort_keys=True)],
        cwd=probe_cwd,
        env=_runtime_environment(),
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
    )
    probe = json.loads(_output(probe_result))
    return InstalledWheel(
        root=root.resolve(),
        python=python,
        site_packages=Path(probe["site_packages"]).resolve(),
        module_file=Path(probe["module_file"]).resolve(),
        sys_path=tuple(probe["sys_path"]),
        declared_console_scripts=probe["declared_console_scripts"],
        console_module_files=probe["console_module_files"],
        probe_cwd=probe_cwd,
    )


def _runtime_environment(**updates: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("OPENTULPA_") or name in {
            "BROWSER_USE_USER_DATA_DIR",
            "HOST",
            "LLM_MODEL",
            "OPENAI_COMPATIBLE_API_KEY",
            "OPENAI_COMPATIBLE_BASE_URL",
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "PORT",
            "PYTHONPATH",
            "XDG_DATA_HOME",
        }:
            environment.pop(name, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(DEPENDENCY_SITE_PACKAGES),
            **updates,
        }
    )
    return environment


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_wheel_contains_runtime_resources(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        names = set(archive.namelist())
        assert PACKAGED_CONFIG in names
        assert RELEASE_CONTRACT in names
        assert REVIEWER_PROMPT in names
        assert (
            archive.read(PACKAGED_CONFIG) == (PROJECT_ROOT / "opentulpa.config.yaml").read_bytes()
        )
        contract = json.loads(archive.read(RELEASE_CONTRACT))

    assert contract == {
        "controller_max": 1,
        "controller_min": 1,
        "product_state_schema": 1,
        "runtime_protocol": 1,
        "workspace_api": 1,
    }


def test_wheel_is_installed_non_editably_without_source_paths(
    installed_wheel: InstalledWheel,
) -> None:
    direct_url_files = list(
        installed_wheel.site_packages.glob("opentulpa-*.dist-info/direct_url.json")
    )
    for direct_url_file in direct_url_files:
        direct_url = json.loads(direct_url_file.read_text(encoding="utf-8"))
        assert direct_url.get("dir_info", {}).get("editable") is not True
    for pth_file in installed_wheel.site_packages.glob("*.pth"):
        assert str(SOURCE_ROOT) not in pth_file.read_text(encoding="utf-8")

    assert installed_wheel.module_file.is_relative_to(installed_wheel.site_packages)
    assert str(SOURCE_ROOT) not in installed_wheel.sys_path
    assert list(installed_wheel.probe_cwd.iterdir()) == []


def test_installed_config_ignores_cwd_yaml_and_dotenv(
    installed_wheel: InstalledWheel,
    tmp_path: Path,
) -> None:
    application_root = tmp_path / "application"
    application_root.mkdir()
    (application_root / "opentulpa.config.yaml").write_text(
        "llm_model: from-application-root\n",
        encoding="utf-8",
    )
    explicit_config = tmp_path / "explicit.yaml"
    explicit_config.write_text("port: 8123\n", encoding="utf-8")
    cwd_parent = tmp_path / "contaminated"
    cwd = cwd_parent / "nested"
    cwd.mkdir(parents=True)
    (cwd_parent / "opentulpa.config.yaml").write_text(
        "llm_model: from-cwd\nport: 65535\n",
        encoding="utf-8",
    )
    (cwd / ".env").write_text(
        "LLM_MODEL=from-dotenv\nOPENAI_COMPATIBLE_API_KEY=cwd-secret\n",
        encoding="utf-8",
    )
    script = """
import json
from opentulpa.core.config import Settings

settings = Settings()
print(json.dumps({
    "api_key": settings.openai_compatible_api_key,
    "browser_root": settings.browser_use_user_data_dir,
    "llm_model": settings.llm_model,
    "port": settings.port,
}))
"""
    result = _run_command(
        [installed_wheel.python, "-c", script],
        cwd=cwd,
        env=_runtime_environment(
            OPENTULPA_APPLICATION_ROOT=str(application_root),
            OPENTULPA_CONFIG_FILE=str(explicit_config),
        ),
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
    )
    settings = json.loads(_output(result))

    assert settings == {
        "api_key": None,
        "browser_root": ".opentulpa/browser_profiles",
        "llm_model": "from-application-root",
        "port": 8123,
    }


def test_python_m_opentulpa_bootstraps_only_external_roots(
    installed_wheel: InstalledWheel,
    tmp_path: Path,
) -> None:
    application_root = tmp_path / "application"
    data_root = tmp_path / "data"
    cwd = tmp_path / "contaminated-cwd"
    cwd.mkdir()
    cwd_yaml = cwd / "opentulpa.config.yaml"
    cwd_dotenv = cwd / ".env"
    cwd_yaml.write_text("llm_model: from-cwd\n", encoding="utf-8")
    cwd_dotenv.write_text(
        "OPENAI_COMPATIBLE_API_KEY=cwd-secret\n",
        encoding="utf-8",
    )
    explicit_config = tmp_path / "runtime.yaml"
    explicit_config.write_text("port: 8123\n", encoding="utf-8")
    cwd_before = _tree_snapshot(cwd)
    site_packages_before = _tree_snapshot(installed_wheel.site_packages)

    result = _run_command(
        [installed_wheel.python, "-m", "opentulpa"],
        cwd=cwd,
        env=_runtime_environment(
            OPENTULPA_APPLICATION_ROOT=str(application_root),
            OPENTULPA_CONFIG_FILE=str(explicit_config),
            OPENTULPA_DATA_ROOT=str(data_root),
        ),
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
        check=False,
    )

    assert result.returncode != 0
    assert "OPENAI_COMPATIBLE_API_KEY is required" in _output(result)
    assert application_root.is_dir()
    assert (data_root / ".opentulpa").is_dir()
    assert not (data_root / ".opentulpa").is_symlink()
    assert (data_root / "tulpa_stuff").is_dir()
    assert not (data_root / "tulpa_stuff").is_symlink()
    assert _tree_snapshot(cwd) == cwd_before
    assert _tree_snapshot(installed_wheel.site_packages) == site_packages_before


def test_declared_console_scripts_import_installed_targets(
    installed_wheel: InstalledWheel,
) -> None:
    for script_name in CONSOLE_SCRIPTS:
        assert (installed_wheel.root / "bin" / script_name).is_file()

    assert installed_wheel.declared_console_scripts == CONSOLE_SCRIPTS
    assert CONSOLE_SCRIPTS["opentulpa"] == "opentulpa.host.cli:main"
    assert all(
        Path(module_file).resolve().is_relative_to(installed_wheel.site_packages)
        for module_file in installed_wheel.console_module_files.values()
    )
    assert list(installed_wheel.probe_cwd.iterdir()) == []
