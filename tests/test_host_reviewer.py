from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

import opentulpa.host.reviewer as reviewer_module
from opentulpa.host.models import HostConfig
from opentulpa.host.reviewer import (
    DeepAgentDeploymentSupervisor,
    DeploymentSupervisionReport,
)
from opentulpa.inference.models import InferenceSelection, ResolvedInferencePlan


class _Runtime:
    def __init__(self) -> None:
        self.status = "ready"
        self.error = None
        self.current_config = HostConfig(
            revision=1,
            status="active",
            api_key=SecretStr("test-key"),
            base_url="https://example.com/v1",
            model="default-model",
            internal_runtime_token=SecretStr("t" * 48),
            created_at=datetime.now(UTC),
        )

    def logs(self) -> list[Any]:
        return []

    async def review_request(self, **_: Any) -> dict[str, Any]:
        return {"ok": True}

    def redact(self, text: str) -> str:
        return text.replace("secret-value", "[redacted]")


def test_supervision_report_has_no_deployment_veto() -> None:
    report = DeploymentSupervisionReport(
        summary="Runtime is healthy.",
        findings=["A non-blocking edge case"],
    )

    assert report.summary == "Runtime is healthy."
    with pytest.raises(ValidationError):
        DeploymentSupervisionReport.model_validate(
            {"approved": False, "summary": "Must not control deployment"}
        )


def test_reviewer_retries_transient_codex_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_model = object()

    class _Inference:
        def resolve_model(self, *_: Any) -> Any:
            return type("Resolved", (), {"model": codex_model, "token_provider": None})()

    monkeypatch.setattr(reviewer_module, "InferenceService", lambda **_: _Inference())
    monkeypatch.setattr(reviewer_module, "load_or_create_host_cipher", lambda _: object())
    reviewer = DeepAgentDeploymentSupervisor(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )
    plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="codex", model="gpt-test"),
        preference_revision=1,
    )

    _, middleware = reviewer._inference_runtime(  # noqa: SLF001
        plan,
        tenant_id="tenant-a",
        api_key="api-key",
        base_url="https://example.com/v1",
        api_default_model="api-model",
    )
    retry = next(
        item for item in middleware if isinstance(item, reviewer_module._ProviderFallbackMiddleware)
    )

    assert retry._fallback_models == (codex_model, codex_model)  # noqa: SLF001


@pytest.mark.asyncio
async def test_reviewer_prefers_connected_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        id = "gpt-review"
        reasoning_efforts = ("low", "high")
        default_reasoning_effort = "low"

    class _Inference:
        validated: InferenceSelection | None = None

        def codex_connected(self, _: str) -> bool:
            return True

        async def models(self, *_: Any) -> tuple[_Model, ...]:
            return (_Model(),)

        async def validate_selection(
            self,
            _: str,
            selection: InferenceSelection,
        ) -> InferenceSelection:
            self.validated = selection
            return selection

    inference = _Inference()
    monkeypatch.setattr(reviewer_module, "InferenceService", lambda **_: inference)
    monkeypatch.setattr(reviewer_module, "load_or_create_host_cipher", lambda _: object())
    reviewer = DeepAgentDeploymentSupervisor(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )
    api_plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="api", model="api-model"),
        preference_revision=4,
    )

    plan = await reviewer._prefer_codex_plan(  # noqa: SLF001
        api_plan,
        tenant_id="tenant-a",
        api_key="api-key",
        base_url="https://example.com/v1",
        api_default_model="api-model",
    )

    assert plan.primary == InferenceSelection(
        provider="codex",
        model="gpt-review",
        reasoning_effort="high",
    )
    assert inference.validated == plan.primary
    assert plan.preference_revision == 4

    codex_plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="codex", model="gpt-pinned", fallback_to_api=True),
        preference_revision=5,
    )
    plan = await reviewer._prefer_codex_plan(  # noqa: SLF001
        codex_plan,
        tenant_id="tenant-a",
        api_key="api-key",
        base_url="https://example.com/v1",
        api_default_model="api-model",
    )

    assert inference.validated == codex_plan.primary
    assert plan.primary.provider == "codex"
    assert plan.primary.fallback_to_api is False

    class _DisconnectedInference:
        def codex_connected(self, _: str) -> bool:
            return False

    monkeypatch.setattr(
        reviewer_module,
        "InferenceService",
        lambda **_: _DisconnectedInference(),
    )
    disconnected = DeepAgentDeploymentSupervisor(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )

    assert (
        await disconnected._prefer_codex_plan(  # noqa: SLF001
            api_plan,
            tenant_id="tenant-b",
            api_key="api-key",
            base_url="https://example.com/v1",
            api_default_model="api-model",
        )
        == api_plan
    )
    plan = await disconnected._prefer_codex_plan(  # noqa: SLF001
        codex_plan,
        tenant_id="tenant-b",
        api_key="api-key",
        base_url="https://example.com/v1",
        api_default_model="api-model",
    )

    assert plan.primary.provider == "api"
    assert plan.primary.model == "api-model"


@pytest.mark.asyncio
async def test_supervisor_is_bounded_redacted_and_has_host_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Graph:
        async def ainvoke(
            self,
            payload: dict[str, Any],
            config: dict[str, Any],
        ) -> dict[str, Any]:
            captured["payload"] = payload
            callback = config["callbacks"][0]
            callback.on_chat_model_start(
                {"name": "review-model"},
                [[{"role": "user", "content": "token=secret-value"}]],
                run_id=uuid4(),
            )
            tool_run_id = uuid4()
            callback.on_tool_start(
                {"name": "run_shell"},
                "password=secret-value",
                run_id=tool_run_id,
            )
            callback.on_tool_end(
                {"output": "secret-value"},
                run_id=tool_run_id,
            )
            return {
                "structured_response": {
                    "summary": "Deployment works with secret-value.",
                    "checks_performed": ["runtime request"],
                    "findings": [],
                    "repair_handoff": "Inspect secret-value.",
                }
            }

    def create_agent(**kwargs: Any) -> _Graph:
        captured.update(kwargs)
        return _Graph()

    def build_model(**kwargs: Any) -> object:
        captured["model_config"] = kwargs
        return object()

    monkeypatch.setattr(reviewer_module, "create_deep_agent", create_agent)
    monkeypatch.setattr(reviewer_module, "build_openrouter_chat_model", build_model)
    real_timeout = asyncio.timeout

    def timeout(seconds: float) -> Any:
        captured["timeout_seconds"] = seconds
        return real_timeout(seconds)

    monkeypatch.setattr(reviewer_module.asyncio, "timeout", timeout)
    plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="api", model="owner-model", reasoning_effort="xhigh"),
        preference_revision=3,
    )
    reviewer = DeepAgentDeploymentSupervisor(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )

    report = await reviewer.observe(
        supervision_id="activation-1",
        release_id="release-1",
        source_commit="a" * 40,
        review_instructions="Verify the deployed endpoint.",
        failure_context={"phase": "runtime switch", "message": "candidate failed"},
        inference_plan=plan,
        tenant_id="tenant-a",
    )

    assert report.summary == "Deployment works with [redacted]."
    assert report.repair_handoff == "Inspect [redacted]."
    assert "deterministic host lifecycle is authoritative" in captured["system_prompt"]
    assert "host shell freely" in captured["system_prompt"]
    assert captured["model_config"]["model_name"] == "owner-model"
    assert captured["model_config"]["reasoning_effort"] == "xhigh"
    assert captured["timeout_seconds"] == 120
    message = json.loads(captured["payload"]["messages"][0]["content"])
    assert message["inference_plan"] == plan.model_dump(mode="json")
    assert message["failure_context"]["phase"] == "runtime switch"
    assert {tool.name for tool in captured["tools"]} == {
        "inspect_runtime",
        "probe_runtime",
        "run_shell",
    }
    shell = next(tool for tool in captured["tools"] if tool.name == "run_shell")
    shell_result = await shell.ainvoke({"command": "printf secret-value", "timeout_seconds": 1})
    assert shell_result == {
        "ok": True,
        "returncode": 0,
        "timed_out": False,
        "truncated": False,
        "cwd": "/",
        "output": "[redacted]",
    }
    probe = next(tool for tool in captured["tools"] if tool.name == "probe_runtime")
    for path in ("/v2/files", "/_runtime/identity"):
        with pytest.raises(ValidationError):
            await probe.ainvoke({"path": path})
    audit_path = tmp_path / "release_reviews" / "activation-1.jsonl"
    audit = audit_path.read_text(encoding="utf-8")
    events = [json.loads(line)["event"] for line in audit.splitlines()]
    assert audit_path.parent.stat().st_mode & 0o777 == 0o700
    assert audit_path.stat().st_mode & 0o777 == 0o600
    assert "secret-value" not in audit
    assert events == [
        "supervision.started",
        "model.start",
        "tool.start",
        "tool.end",
        "supervision.completed",
    ]
