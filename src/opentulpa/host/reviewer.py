"""Stable Deep Agent reviewer for running source releases."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.deep_agent.service import (
    _before_current_run_activity,
    _CodexAuthRetryMiddleware,
    _InferenceMessageMiddleware,
    _ProviderFallbackMiddleware,
    _with_deepagents_context_budget,
    build_openrouter_chat_model,
)
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.inference.codex import is_transient as is_codex_transient
from opentulpa.inference.models import InferenceSelection, ResolvedInferencePlan
from opentulpa.inference.service import InferenceService
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.secrets.host_key import load_or_create_host_cipher

_REVIEW_ID = re.compile(r"^[A-Za-z0-9_-]{1,200}$")


class DeploymentSupervisionReport(BaseModel):
    """Advisory observations; deterministic host checks own deployment state."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=2_000)
    checks_performed: list[str] = Field(default_factory=list, max_length=30)
    findings: list[str] = Field(default_factory=list, max_length=30)


class _ReviewAuditLog(BaseCallbackHandler):
    """Persist redacted LangChain events for one release review."""

    run_inline = True

    def __init__(self, path: Path, *, redact: Any) -> None:
        self._path = path
        self._redact = redact
        self._lock = threading.Lock()

    def record(self, event: str, data: Any = None, **metadata: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **metadata,
            **({"data": data} if data is not None else {}),
        }
        serialized = json.dumps(redact_for_langfuse(payload), ensure_ascii=False, default=str)
        line = self._redact(serialized) + "\n"
        with self._lock:
            descriptor = os.open(
                self._path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(line)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "chain.start",
            {"component": serialized, "inputs": inputs},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "chain.end",
            outputs,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "chain.error",
            {"type": type(error).__name__, "message": str(error)},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "model.start",
            {"component": serialized, "messages": messages},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "model.end",
            response,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "model.error",
            {"type": type(error).__name__, "message": str(error)},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "tool.start",
            {"component": serialized, "input": input_str, "inputs": kwargs.get("inputs")},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "tool.end",
            output,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.record(
            "tool.error",
            {"type": type(error).__name__, "message": str(error)},
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            name=kwargs.get("name"),
        )


class DeepAgentDeploymentSupervisor:
    """Observe a running deployment without controlling its lifecycle."""

    def __init__(
        self,
        runtime: RuntimeSupervisor,
        *,
        runtime_data_root: Path,
        api_reasoning_effort: str | None = "high",
        api_fallback_models: Sequence[str] = (),
        provider_order: Mapping[str, Sequence[str]] | None = None,
        max_completion_tokens: int | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self._runtime = runtime
        self._runtime_data_root = runtime_data_root
        self._api_reasoning_effort = api_reasoning_effort
        self._api_fallback_models = tuple(api_fallback_models)
        self._provider_order = dict(provider_order or {})
        self._max_completion_tokens = max_completion_tokens
        self._timeout_seconds = max(1, int(timeout_seconds))

    async def observe(
        self,
        *,
        supervision_id: str,
        release_id: str,
        source_commit: str,
        review_instructions: str,
        inference_plan: ResolvedInferencePlan | None,
        tenant_id: str,
    ) -> DeploymentSupervisionReport:
        config = self._runtime.current_config
        if config is None:
            raise RuntimeError("deployment supervisor has no active model configuration")
        audit = self._audit_log(supervision_id)
        audit.record(
            "supervision.started",
            {
                "supervision_id": supervision_id,
                "release_id": release_id,
                "source_commit": source_commit,
                "review_instructions": review_instructions,
            },
        )

        @tool
        def inspect_runtime() -> dict[str, Any]:
            """Inspect deployment state and recent redacted application logs."""

            error = str(self._runtime.error or "").strip()
            live_source = getattr(self._runtime, "live_source", None)
            return {
                "status": self._runtime.status,
                "error": self._runtime.redact(error) if error else None,
                "source_commit": getattr(live_source, "source_commit", None),
                "logs": [entry.model_dump(mode="json") for entry in self._runtime.logs()[-200:]],
            }

        @tool
        async def probe_runtime(
            path: Literal["/healthz", "/agent/healthz"],
        ) -> dict[str, Any]:
            """Run one allowlisted authenticated read-only deployment probe."""

            return await self._runtime.review_request(method="GET", path=path)

        plan = inference_plan or ResolvedInferencePlan.resolve(
            InferenceSelection(
                provider="api",
                model=config.model,
                reasoning_effort=self._api_reasoning_effort,
            ),
            preference_revision=0,
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                plan = await self._prefer_codex_plan(
                    plan,
                    tenant_id=tenant_id,
                    api_key=config.api_key.get_secret_value(),
                    base_url=config.base_url,
                    api_default_model=config.model,
                )
                model, middleware = self._inference_runtime(
                    plan,
                    tenant_id=tenant_id,
                    api_key=config.api_key.get_secret_value(),
                    base_url=config.base_url,
                    api_default_model=config.model,
                )
                graph = create_deep_agent(
                    model=model,
                    name="opentulpa_deployment_supervisor",
                    tools=[inspect_runtime, probe_runtime],
                    system_prompt="\n\n".join(
                        (
                            Path(__file__)
                            .with_name("reviewer_policy.md")
                            .read_text(encoding="utf-8"),
                            Path(__file__)
                            .with_name("reviewer_prompt.md")
                            .read_text(encoding="utf-8"),
                        )
                    ),
                    response_format=DeploymentSupervisionReport,
                    middleware=middleware,
                )
                prompt = {
                    "release_id": release_id,
                    "source_commit": source_commit,
                    "review_instructions": review_instructions,
                    "inference_plan": plan.model_dump(mode="json"),
                }
                state = await graph.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                            }
                        ]
                    },
                    config={"callbacks": [audit]},
                )
                response = state.get("structured_response") if isinstance(state, dict) else None
                if isinstance(response, DeploymentSupervisionReport):
                    report = response
                elif isinstance(response, dict):
                    report = DeploymentSupervisionReport.model_validate(response)
                else:
                    raise RuntimeError("deployment supervisor returned no report")
                report = DeploymentSupervisionReport.model_validate_json(
                    self._runtime.redact(report.model_dump_json())
                )
            audit.record("supervision.completed", report.model_dump(mode="json"))
            return report
        except BaseException as exc:
            audit.record(
                "supervision.failed",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise

    async def _prefer_codex_plan(
        self,
        plan: ResolvedInferencePlan,
        *,
        tenant_id: str,
        api_key: str,
        base_url: str,
        api_default_model: str,
    ) -> ResolvedInferencePlan:
        inference = InferenceService(
            db_path=self._runtime_data_root / "deepagents" / "inference.db",
            cipher=load_or_create_host_cipher(self._runtime_data_root),
            api_key=api_key,
            api_base_url=base_url,
            api_default_model=api_default_model,
            api_reasoning_effort=self._api_reasoning_effort,
            api_fallback_models=self._api_fallback_models,
        )
        if not inference.codex_connected(tenant_id):
            if plan.primary.provider == "api":
                return plan
            return ResolvedInferencePlan.resolve(
                InferenceSelection(
                    provider="api",
                    model=api_default_model,
                    reasoning_effort=self._api_reasoning_effort,
                ),
                preference_revision=plan.preference_revision,
            )
        if plan.primary.provider == "codex":
            selection = plan.primary
        else:
            models = await inference.models(tenant_id, "codex")
            if not models:
                raise RuntimeError("release review requires an available Codex model")
            model = models[0]
            selection = InferenceSelection(
                provider="codex",
                model=model.id,
                reasoning_effort=(
                    "high" if "high" in model.reasoning_efforts else model.default_reasoning_effort
                ),
            )
        selection = await inference.validate_selection(tenant_id, selection)
        selection = selection.model_copy(update={"fallback_to_api": False})
        return ResolvedInferencePlan.resolve(
            selection,
            preference_revision=plan.preference_revision,
        )

    def _audit_log(self, review_id: str) -> _ReviewAuditLog:
        safe_id = str(review_id or "").strip()
        if _REVIEW_ID.fullmatch(safe_id) is None:
            raise ValueError("release review id is invalid")
        root = self._runtime_data_root / "release_reviews"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("release review log directory is unsafe")
        root.chmod(0o700)
        path = root / f"{safe_id}.jsonl"
        if os.path.lexists(path):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("release review log path is unsafe")
            path.chmod(0o600)
        return _ReviewAuditLog(path, redact=self._runtime.redact)

    def _inference_runtime(
        self,
        plan: ResolvedInferencePlan,
        *,
        tenant_id: str,
        api_key: str,
        base_url: str,
        api_default_model: str,
    ) -> tuple[Any, list[Any]]:
        fallback_models = tuple(
            build_openrouter_chat_model(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                reasoning_effort=self._api_reasoning_effort,
                max_completion_tokens=self._max_completion_tokens,
                provider_order=self._provider_order.get(model_name, ()),
            )
            for model_name in self._api_fallback_models
            if model_name != api_default_model
        )
        middleware: list[Any] = [_InferenceMessageMiddleware(plan.primary.provider)]
        if plan.primary.provider == "api":
            model = build_openrouter_chat_model(
                api_key=api_key,
                base_url=base_url,
                model_name=plan.primary.model,
                reasoning_effort=plan.primary.reasoning_effort,
                max_completion_tokens=self._max_completion_tokens,
            )
            if fallback_models:
                middleware.append(_ProviderFallbackMiddleware(fallback_models))
            return _with_deepagents_context_budget(model), middleware

        inference = InferenceService(
            db_path=self._runtime_data_root / "deepagents" / "inference.db",
            cipher=load_or_create_host_cipher(self._runtime_data_root),
            api_key=api_key,
            api_base_url=base_url,
            api_default_model=api_default_model,
            api_reasoning_effort=self._api_reasoning_effort,
            api_fallback_models=self._api_fallback_models,
        )
        resolved = inference.resolve_model(tenant_id, plan.primary)
        if plan.primary.fallback_to_api:
            api_primary = build_openrouter_chat_model(
                api_key=api_key,
                base_url=base_url,
                model_name=api_default_model,
                reasoning_effort=self._api_reasoning_effort,
                max_completion_tokens=self._max_completion_tokens,
            )
            middleware.append(
                _ProviderFallbackMiddleware(
                    (api_primary, *fallback_models),
                    eligible=is_codex_transient,
                    allow_request=_before_current_run_activity,
                )
            )
        middleware.append(
            _ProviderFallbackMiddleware(
                (resolved.model, resolved.model),
                eligible=is_codex_transient,
            )
        )
        middleware.append(_CodexAuthRetryMiddleware(resolved))
        return _with_deepagents_context_budget(resolved.model, provider="codex"), middleware


__all__ = ["DeepAgentDeploymentSupervisor", "DeploymentSupervisionReport"]
