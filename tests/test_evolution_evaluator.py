from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import opentulpa.evolution.evaluator as evaluator_module
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationCommand,
    EvaluationExecutables,
    IsolatedProcessEvaluationRunner,
    LocalEvaluationRunner,
    OciEvaluationRunner,
    trusted_default_commands,
)


def test_evolution_image_installs_trusted_contract_validator() -> None:
    recipe = Path("docker/evolution.Dockerfile").read_text(encoding="utf-8")

    assert "COPY src ./src" in recipe
    assert "uv sync --frozen --all-extras --dev" in recipe
    assert "--no-install-project" not in recipe


@pytest.mark.asyncio
async def test_candidate_evaluator_stops_before_later_stages_after_failure(
    tmp_path: Path,
) -> None:
    evaluator = CandidateEvaluator(
        runner=LocalEvaluationRunner(),
        commands=(
            EvaluationCommand(
                name="public.pass",
                stage="public",
                argv=(sys.executable, "-c", "print('ok')"),
            ),
            EvaluationCommand(
                name="public.fail",
                stage="public",
                argv=(sys.executable, "-c", "raise SystemExit(7)"),
            ),
            EvaluationCommand(
                name="security.never",
                stage="security",
                argv=(sys.executable, "-c", "raise SystemExit(99)"),
            ),
        ),
    )

    results = await evaluator.evaluate(tmp_path)

    assert [result.name for result in results] == ["public.pass", "public.fail"]
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].exit_code == 7


@pytest.mark.asyncio
async def test_local_evaluation_runner_does_not_inherit_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_PRIVATE_TOKEN", "must-not-leak")
    runner = LocalEvaluationRunner()

    result = await runner.run(
        workspace=tmp_path,
        command=EvaluationCommand(
            name="environment",
            argv=(
                sys.executable,
                "-c",
                "import os; print(os.getenv('OPENTULPA_PRIVATE_TOKEN', 'absent'))",
            ),
        ),
    )

    assert result.passed is True
    assert result.output.strip() == "absent"


@pytest.mark.asyncio
async def test_local_evaluation_runner_reports_missing_executable(tmp_path: Path) -> None:
    result = await LocalEvaluationRunner().run(
        workspace=tmp_path,
        command=EvaluationCommand(
            name="missing.executable",
            argv=(str(tmp_path / "missing-tool"), "--version"),
        ),
    )

    assert result.passed is False
    assert result.exit_code == 127
    assert "evaluation executable is unavailable" in result.output


def test_evaluation_contract_rejects_shellless_command_injection() -> None:
    with pytest.raises(ValueError, match="argv"):
        EvaluationCommand(name="bad", argv=())

    with pytest.raises(ValueError, match="image"):
        OciEvaluationRunner(image="image; echo unsafe")


def test_evaluator_requires_unique_named_commands() -> None:
    command = EvaluationCommand(name="same", argv=("true",))

    with pytest.raises(ValueError, match="unique"):
        CandidateEvaluator(
            runner=LocalEvaluationRunner(),
            commands=(command, command),
        )


def test_trusted_default_commands_cover_static_runtime_and_security_gates() -> None:
    commands = trusted_default_commands(timeout_seconds=120)

    assert [command.name for command in commands] == [
        "python.compile",
        "ruff",
        "mypy",
        "pytest",
        "legacy.runtime.absent",
        "source.secret.paths",
        "kernel.contract",
    ]
    assert {command.stage for command in commands} == {
        "build",
        "contract",
        "public",
        "security",
    }
    assert all(command.timeout_seconds == 120 for command in commands)


def test_isolated_default_commands_use_only_exact_controller_executables() -> None:
    tools = EvaluationExecutables.isolated()
    commands = trusted_default_commands(executables=tools)

    assert [command.argv[0] for command in commands] == [
        "/controller/bin/python",
        "/controller/bin/ruff",
        "/controller/bin/mypy",
        "/controller/bin/pytest",
        "/controller/bin/python",
        "/controller/bin/python",
        "/controller/bin/python",
    ]
    assert all(command.argv[0].startswith("/controller/bin/") for command in commands)


def test_isolated_evaluator_support_fails_closed_with_strong_sandbox_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator_module,
        "strong_sandbox_unavailable_reason",
        lambda: "strong sandbox namespaces are blocked",
    )

    reason = IsolatedProcessEvaluationRunner.unavailable_reason(
        controller_root=tmp_path / "controller",
        wheelhouse=tmp_path / "wheelhouse",
    )

    assert reason == "strong sandbox namespaces are blocked"
    assert not IsolatedProcessEvaluationRunner.is_supported(
        controller_root=tmp_path / "controller",
        wheelhouse=tmp_path / "wheelhouse",
    )


def test_isolated_evaluator_accepts_normal_venv_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Ancestor:
        def __init__(self, mode: int, owner: int = 0) -> None:
            self._metadata = SimpleNamespace(st_mode=mode, st_uid=owner)

        def lstat(self) -> SimpleNamespace:
            return self._metadata

    class Target:
        parents = (Ancestor(stat.S_IFDIR | 0o755), Ancestor(stat.S_IFDIR | 0o755))

        @staticmethod
        def stat() -> SimpleNamespace:
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o555, st_uid=0)

        @staticmethod
        def lstat() -> SimpleNamespace:
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o555, st_uid=0)

    target = Target()

    class Executable:
        name = "python"

        @staticmethod
        def lstat() -> SimpleNamespace:
            return SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_uid=0)

        @staticmethod
        def resolve(*, strict: bool) -> Target:
            assert strict is True
            return target

    monkeypatch.setattr(evaluator_module.os, "access", lambda path, mode: True)

    executable = Executable()
    reason = IsolatedProcessEvaluationRunner._executable_reason(  # type: ignore[arg-type]  # noqa: SLF001
        executable
    )

    assert reason is None


def test_isolated_evaluator_rejects_mutable_resolved_executable_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/controller")
    executable = root / "bin" / "python"

    class Entry:
        def __init__(self, mode: int, owner: int = 0) -> None:
            self._metadata = SimpleNamespace(st_mode=mode, st_uid=owner)

        def lstat(self) -> SimpleNamespace:
            return self._metadata

    entries = {
        executable: Entry(stat.S_IFLNK | 0o777),
        root / "bin" / "python.real": Entry(stat.S_IFREG | 0o555),
        root / "bin": Entry(stat.S_IFDIR | 0o755),
        root: Entry(stat.S_IFDIR | 0o775),
        Path("/"): Entry(stat.S_IFDIR | 0o755),
    }

    class Resolved:
        parents = (
            Entry(stat.S_IFDIR | 0o755),
            Entry(stat.S_IFDIR | 0o775),
            Entry(stat.S_IFDIR | 0o755),
        )

        def stat(self) -> SimpleNamespace:
            return entries[root / "bin" / "python.real"]._metadata

        def lstat(self) -> SimpleNamespace:
            return self.stat()

    class Executable:
        name = "python"

        def lstat(self) -> SimpleNamespace:
            return entries[executable]._metadata

        def resolve(self, *, strict: bool) -> Resolved:
            assert strict is True
            return Resolved()

    monkeypatch.setattr(evaluator_module.os, "access", lambda path, mode: True)

    reason = IsolatedProcessEvaluationRunner._executable_reason(  # type: ignore[arg-type]  # noqa: SLF001
        Executable()
    )

    assert reason is not None


def test_isolated_evaluator_fingerprint_covers_exact_tools_packages_and_wheels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = tmp_path / "controller"
    binary_root = controller / "bin"
    packages = controller / "lib" / "python3.12" / "site-packages"
    wheelhouse = tmp_path / "wheelhouse"
    dependency_site = tmp_path / "dependency-site"
    binary_root.mkdir(parents=True)
    wheelhouse.mkdir()
    dependency_site.mkdir()
    (dependency_site / "dependency.py").write_text("VERSION = 1\n", encoding="utf-8")
    for name in ("python", "ruff", "mypy", "pytest"):
        path = binary_root / name
        path.write_bytes(f"{name}-v1".encode())
    for name in ("mypy", "pytest", "ruff"):
        metadata = packages / f"{name}-1.0.dist-info"
        metadata.mkdir(parents=True)
        package = packages / name
        package.mkdir()
        (package / "__init__.py").write_text(f"NAME = {name!r}\n", encoding="utf-8")
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n", encoding="utf-8")
        (metadata / "RECORD").write_text(f"{name}/__init__.py,,\n", encoding="utf-8")
    (wheelhouse / "dependency-1.0-py3-none-any.whl").write_bytes(b"wheel-v1")
    sandbox_tools = []
    for name in ("setpriv", "bwrap", "prlimit"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sandbox_tools.append(path)
    monkeypatch.setattr(
        evaluator_module,
        "_file_sha256",
        lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(evaluator_module, "_TRUSTED_OWNER_UID", os.geteuid())
    runner = IsolatedProcessEvaluationRunner.__new__(IsolatedProcessEvaluationRunner)
    runner._uid = 65_533  # noqa: SLF001
    runner._gid = 65_533  # noqa: SLF001
    runner._memory_bytes = 2 * 1024 * 1024 * 1024  # noqa: SLF001
    runner._pid_limit = 256  # noqa: SLF001
    runner._setpriv, runner._bwrap, runner._prlimit = sandbox_tools  # noqa: SLF001
    runner._controller_root = controller  # noqa: SLF001
    runner._wheelhouse = wheelhouse  # noqa: SLF001
    runner._dependency_site = dependency_site  # noqa: SLF001
    runner._host_executables = {  # noqa: SLF001
        isolated: binary_root / Path(isolated).name
        for isolated in EvaluationExecutables.isolated().values()
    }

    first = runner._input_fingerprint()  # noqa: SLF001
    (binary_root / "ruff").write_bytes(b"ruff-v2")
    second = runner._input_fingerprint()  # noqa: SLF001
    (binary_root / "ruff").write_bytes(b"ruff-v1")
    (packages / "pytest-1.0.dist-info" / "RECORD").write_text(
        "pytest/__init__.py,sha256=changed,1\n",
        encoding="utf-8",
    )
    third = runner._input_fingerprint()  # noqa: SLF001
    (packages / "pytest-1.0.dist-info" / "RECORD").write_text(
        "pytest/__init__.py,,\n",
        encoding="utf-8",
    )
    (dependency_site / "dependency.py").write_text("VERSION = 2\n", encoding="utf-8")
    fourth = runner._input_fingerprint()  # noqa: SLF001

    assert first != second
    assert first != third
    assert first != fourth


@pytest.mark.asyncio
async def test_isolated_python_gate_cannot_be_shadowed_by_candidate_module(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "compileall.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    result = await LocalEvaluationRunner().run(
        workspace=tmp_path,
        command=EvaluationCommand(
            name="isolated.compile",
            argv=(sys.executable, "-I", "-S", "-m", "compileall", "-q", "src"),
        ),
    )

    assert result.passed is False


@pytest.mark.asyncio
async def test_isolated_security_gate_uses_trusted_pathlib(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=private\n", encoding="utf-8")
    (tmp_path / "pathlib.py").write_text(
        "class Path:\n    @classmethod\n    def rglob(cls, pattern):\n        return []\n",
        encoding="utf-8",
    )
    script = (
        "from pathlib import Path; "
        "bad=[p for p in Path('.').rglob('*') if p.is_file() and p.name == '.env']; "
        "assert not bad"
    )

    result = await LocalEvaluationRunner().run(
        workspace=tmp_path,
        command=EvaluationCommand(
            name="isolated.security",
            argv=(sys.executable, "-I", "-S", "-c", script),
        ),
    )

    assert result.passed is False


@pytest.mark.asyncio
async def test_legacy_absence_gate_does_not_match_its_own_implementation(tmp_path: Path) -> None:
    evaluator_source = Path(evaluator_module.__file__).read_text(encoding="utf-8")
    source = tmp_path / "src" / "opentulpa" / "evolution"
    source.mkdir(parents=True)
    (source / "evaluator.py").write_text(evaluator_source, encoding="utf-8")
    command = next(
        item for item in trusted_default_commands() if item.name == "legacy.runtime.absent"
    )

    clean = await LocalEvaluationRunner().run(workspace=tmp_path, command=command)
    assert clean.passed is True

    forbidden_name = "OpenTulpa" + "LangGraphRuntime"
    (tmp_path / "src" / "legacy.py").write_text(forbidden_name, encoding="utf-8")
    detected = await LocalEvaluationRunner().run(workspace=tmp_path, command=command)
    assert detected.passed is False


@pytest.mark.asyncio
async def test_fixed_kernel_contract_is_supervisor_owned(tmp_path: Path) -> None:
    (tmp_path / "src" / "opentulpa" / "deep_agent").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        'dependencies = ["deepagents==0.6.12"]\n',
        encoding="utf-8",
    )
    service = tmp_path / "src" / "opentulpa" / "deep_agent" / "service.py"
    service.write_text(
        "from deepagents import create_deep_agent\nclass DeepAgentService: pass\n",
        encoding="utf-8",
    )
    command = next(item for item in trusted_default_commands() if item.name == "kernel.contract")

    assert (await LocalEvaluationRunner().run(workspace=tmp_path, command=command)).passed

    service.write_text("class DeepAgentService: pass\n", encoding="utf-8")
    assert not (await LocalEvaluationRunner().run(workspace=tmp_path, command=command)).passed
