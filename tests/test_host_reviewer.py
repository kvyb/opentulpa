from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr

import opentulpa.host.reviewer as reviewer_module
from opentulpa.host.models import HostConfig
from opentulpa.host.reviewer import DeepAgentReleaseReviewer, ReleaseReviewDecision
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


def test_p0_or_p1_review_requires_repair_handoff() -> None:
    with pytest.raises(ValueError, match="repair handoff"):
        ReleaseReviewDecision(approved=True, summary="Code bug", findings=["[P1] Data loss"])


def test_p2_or_p3_review_is_approved() -> None:
    decision = ReleaseReviewDecision(
        approved=False,
        summary="Minor issue",
        findings=["[P2] A non-blocking edge case"],
        repair_handoff="This must not block the release.",
    )

    assert decision.approved is True
    assert decision.repair_handoff is None


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
    reviewer = DeepAgentReleaseReviewer(
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
        item
        for item in middleware
        if isinstance(item, reviewer_module._ProviderFallbackMiddleware)
    )

    assert retry._fallback_models == (codex_model, codex_model)  # noqa: SLF001


@pytest.mark.asyncio
async def test_reviewer_uses_previous_prompt_owner_plan_and_disposable_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    previous = tmp_path / "previous"
    candidate.mkdir()
    previous.mkdir()
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
                    "approved": True,
                    "summary": "Deployment works.",
                    "checks_performed": ["runtime request"],
                    "findings": [],
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
    monkeypatch.setattr(
        reviewer_module.asyncio,
        "timeout",
        lambda _: pytest.fail("release reviews must not have a whole-review timeout"),
    )
    plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="api", model="owner-model", reasoning_effort="xhigh"),
        preference_revision=3,
    )
    reviewer = DeepAgentReleaseReviewer(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )

    decision = await reviewer.review(
        review_id="activation-1",
        release_id="release-1",
        source_commit="a" * 40,
        changed_paths=("src/opentulpa/example.py",),
        review_instructions="Verify the deployed endpoint.",
        inference_plan=plan,
        tenant_id="tenant-a",
        candidate_root=candidate,
        reviewer_root=previous,
        system_prompt="Previous generation prompt",
    )

    assert decision.approved is True
    assert "# Ponytail" in captured["system_prompt"]
    assert "Only P0 and P1 findings block a release" in captured["system_prompt"]
    assert captured["system_prompt"].endswith("Previous generation prompt")
    assert captured["model_config"]["model_name"] == "owner-model"
    assert captured["model_config"]["reasoning_effort"] == "xhigh"
    message = json.loads(captured["payload"]["messages"][0]["content"])
    assert message["inference_plan"] == plan.model_dump(mode="json")
    shell = next(tool for tool in captured["tools"] if tool.name == "run_shell")
    shell_result = await shell.ainvoke(
        {
            "command": "printf '%s' \"${OPENAI_COMPATIBLE_API_KEY:-safe}\"",
            "working_directory": "candidate",
            "timeout_seconds": 10,
        }
    )
    assert shell_result["cwd"] == str(candidate)
    assert shell_result["output"] == "safe"
    audit_path = tmp_path / "release_reviews" / "activation-1.jsonl"
    audit = audit_path.read_text(encoding="utf-8")
    events = [json.loads(line)["event"] for line in audit.splitlines()]
    assert audit_path.parent.stat().st_mode & 0o777 == 0o700
    assert audit_path.stat().st_mode & 0o777 == 0o600
    assert "secret-value" not in audit
    assert events == [
        "review.started",
        "model.start",
        "tool.start",
        "tool.end",
        "review.completed",
    ]
