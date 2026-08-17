from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def test_rejected_review_requires_repair_handoff() -> None:
    with pytest.raises(ValueError, match="repair handoff"):
        ReleaseReviewDecision(approved=False, summary="Code bug")


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
        async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            captured["payload"] = payload
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
    plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="api", model="owner-model", reasoning_effort="xhigh"),
        preference_revision=3,
    )
    reviewer = DeepAgentReleaseReviewer(
        _Runtime(),  # type: ignore[arg-type]
        runtime_data_root=tmp_path,
    )

    decision = await reviewer.review(
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
    assert captured["system_prompt"] == "Previous generation prompt"
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
