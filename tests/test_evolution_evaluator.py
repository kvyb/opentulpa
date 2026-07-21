from __future__ import annotations

import sys
from pathlib import Path

import pytest

import opentulpa.evolution.evaluator as evaluator_module
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationCommand,
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
