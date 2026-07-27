from __future__ import annotations

import inspect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from pydantic import BaseModel

from opentulpa.deep_agent.contracts import AgentRunRequest
from opentulpa.deep_agent.service import (
    DeepAgentService,
    _before_current_run_activity,
    _CodexAuthRetryMiddleware,
    _ProviderFallbackMiddleware,
)
from opentulpa.inference.codex import (
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_REDIRECT_URI,
    CodexProviderError,
    CodexTokenProvider,
    build_codex_model,
    is_transient,
)
from opentulpa.inference.models import InferenceSelection, ResolvedInferencePlan
from opentulpa.inference.service import (
    InferenceConflictError,
    InferenceService,
    ResolvedModel,
)
from opentulpa.inference.store import CodexCredential, DeviceLogin, InferenceCredentialStore
from opentulpa.secrets.cipher import AesGcmHostKeyCipher
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.contract import AgentRunContext


class _ToolCapableTextModel(FakeListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _ToolCapableTextModel:
        del tools, kwargs
        return self


def _cipher() -> AesGcmHostKeyCipher:
    return AesGcmHostKeyCipher(b"i" * 32)


def _inference(root: Path) -> InferenceService:
    return InferenceService(
        db_path=root / "inference.db",
        cipher=_cipher(),
        api_key="api-secret",
        api_base_url="https://openrouter.ai/api/v1",
        api_default_model="test-model",
        api_reasoning_effort="low",
        api_fallback_models=("z-ai/glm-5.2",),
    )


def _context(*, tenant_id: str = "tenant-1", thread_id: str = "thread-1") -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="owner-1",
        thread_id=thread_id,
        channel="web",
        run_kind="owner",
        correlation_id="correlation-1",
        origin=OriginRef(interface="web", source_id="owner-web"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


def test_resolved_plan_digest_is_stable_and_revision_bound() -> None:
    selection = InferenceSelection(
        provider="codex",
        model="gpt-test",
        reasoning_effort="high",
    )

    first = ResolvedInferencePlan.resolve(selection, preference_revision=2)
    same = ResolvedInferencePlan.resolve(selection, preference_revision=2)
    changed = ResolvedInferencePlan.resolve(selection, preference_revision=3)
    fast = ResolvedInferencePlan.resolve(
        selection.model_copy(update={"service_tier": "priority"}),
        preference_revision=2,
    )

    assert first == same
    assert first.digest != changed.digest
    assert first.digest != fast.digest


@pytest.mark.asyncio
async def test_codex_catalog_maps_reasoning_levels_and_service_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _inference(tmp_path)
    service._store.save_credential(  # noqa: SLF001
        "tenant-1",
        CodexCredential(
            access_token="access-secret",
            refresh_token="refresh-secret",
            id_token=None,
            account_id="account-1",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )

    async def request(_: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "visibility": "list",
                        "priority": 1,
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "medium"},
                            {"effort": "high"},
                            {"effort": "xhigh"},
                            {"effort": "max"},
                            {"effort": "ultra"},
                        ],
                        "default_reasoning_level": "low",
                        "service_tiers": [
                            {
                                "id": "priority",
                                "name": "Fast",
                                "description": "1.5x speed, increased usage",
                            }
                        ],
                        "default_service_tier": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(service, "_request_codex_models", request)
    models = await service.models("tenant-1", "codex")

    assert len(models) == 1
    model = models[0]
    assert model.reasoning_efforts == ("low", "medium", "high", "xhigh", "max", "ultra")
    assert model.default_reasoning_effort == "low"
    assert [tier.model_dump() for tier in model.service_tiers] == [
        {
            "id": "priority",
            "name": "Fast",
            "description": "1.5x speed, increased usage",
        }
    ]
    assert model.default_service_tier is None
    selected = await service.validate_selection(
        "tenant-1",
        InferenceSelection(
            provider="codex",
            model=model.id,
            reasoning_effort="ultra",
            service_tier="priority",
        ),
    )
    assert selected.reasoning_effort == "ultra"
    assert selected.service_tier == "priority"
    with pytest.raises(ValueError, match="service tier"):
        await service.validate_selection(
            "tenant-1",
            selected.model_copy(update={"service_tier": "unsupported"}),
        )
    with pytest.raises(ValueError, match="reasoning effort"):
        await service.validate_selection(
            "tenant-1",
            selected.model_copy(update={"reasoning_effort": "unsupported"}),
        )
    sanitized_api = await service.validate_selection(
        "tenant-1",
        InferenceSelection(
            provider="api",
            model="test-model",
            service_tier="priority",
            fallback_to_api=True,
        ),
    )
    assert sanitized_api.service_tier is None
    assert sanitized_api.fallback_to_api is False


def test_codex_credentials_are_encrypted_and_refresh_rotation_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "inference.db"
    store = InferenceCredentialStore(path, cipher=_cipher())
    original = CodexCredential(
        access_token="access-secret",
        refresh_token="refresh-secret",
        id_token="id-secret",
        account_id="account-secret",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    saved = store.save_credential("tenant-1", original)

    rotated = store.refresh_credential(
        "tenant-1",
        lambda current: CodexCredential(
            access_token="new-access-secret",
            refresh_token="new-refresh-secret",
            id_token=current.id_token,
            account_id=current.account_id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
    )

    assert saved.revision == 1
    assert rotated.revision == 2
    assert store.load_credential("tenant-1") == rotated
    raw = path.read_bytes()
    assert b"access-secret" not in raw
    assert b"refresh-secret" not in raw
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT revision FROM codex_credentials WHERE tenant_id = 'tenant-1'"
        ).fetchone() == (2,)


def test_concurrent_refresh_rotates_a_single_time(tmp_path: Path) -> None:
    store = InferenceCredentialStore(tmp_path / "inference.db", cipher=_cipher())
    store.save_credential(
        "tenant-1",
        CodexCredential(
            access_token="old-access",
            refresh_token="old-refresh",
            id_token=None,
            account_id=None,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        ),
    )
    calls: list[str] = []

    def refresh(current: CodexCredential) -> CodexCredential:
        calls.append(current.refresh_token)
        return CodexCredential(
            access_token="fresh-access",
            refresh_token="fresh-refresh",
            id_token=None,
            account_id=None,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.refresh_credential("tenant-1", refresh), range(2)))

    assert calls == ["old-refresh"]
    assert {item.refresh_token for item in results} == {"fresh-refresh"}


def test_refresh_rate_limit_is_not_reported_as_lost_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InferenceCredentialStore(tmp_path / "inference.db", cipher=_cipher())
    provider = CodexTokenProvider(tenant_id="tenant-1", store=store)
    current = CodexCredential(
        access_token="access",
        refresh_token="refresh",
        id_token=None,
        account_id=None,
        expires_at=datetime.now(UTC),
    )
    monkeypatch.setattr(
        "opentulpa.inference.codex.httpx.post",
        lambda *_args, **_kwargs: httpx.Response(429, json={"error": "rate_limited"}),
    )

    with pytest.raises(CodexProviderError) as captured:
        provider._refresh(current)  # noqa: SLF001
    assert captured.value.status_code == 429
    assert is_transient(captured.value) is True


@pytest.mark.asyncio
async def test_device_login_persists_authorized_credential_without_exposing_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expires_at = (datetime.now(UTC) + timedelta(minutes=15)).isoformat()
    responses = iter(
        (
            httpx.Response(
                200,
                json={
                    "device_auth_id": "private-device-auth-id",
                    "user_code": "ABCD-EFGH",
                    "interval": "3",
                    "expires_at": expires_at,
                },
            ),
            httpx.Response(
                200,
                json={
                    "authorization_code": "private-authorization-code",
                    "code_verifier": "private-code-verifier",
                },
            ),
            httpx.Response(
                200,
                json={
                    "access_token": "private-access-token",
                    "refresh_token": "private-refresh-token",
                    "expires_in": 3600,
                },
            ),
        )
    )
    requests: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(self, *_: Any, **kwargs: Any) -> httpx.Response:
            requests.append(kwargs)
            return next(responses)

    monkeypatch.setattr("opentulpa.inference.service.httpx.AsyncClient", Client)
    service = _inference(tmp_path)

    started = await service.start_device_login("tenant-1")
    authorized = await service.get_device_login("tenant-1", started["login_id"])

    assert started["status"] == "pending"
    assert started["verification_url"] == "https://auth.openai.com/codex/device"
    assert authorized is not None and authorized["status"] == "authorized"
    assert "access_token" not in authorized
    assert service.codex_connected("tenant-1") is True
    assert b"private-access-token" not in (tmp_path / "inference.db").read_bytes()
    assert requests[0]["json"] == {"client_id": CHATGPT_CLIENT_ID}
    assert "data" not in requests[0]
    assert requests[1]["json"] == {
        "device_auth_id": "private-device-auth-id",
        "user_code": "ABCD-EFGH",
    }
    assert "data" not in requests[1]
    assert requests[2]["data"] == {
        "grant_type": "authorization_code",
        "code": "private-authorization-code",
        "redirect_uri": CHATGPT_DEVICE_REDIRECT_URI,
        "client_id": CHATGPT_CLIENT_ID,
        "code_verifier": "private-code-verifier",
    }


@pytest.mark.asyncio
async def test_device_login_slow_down_expiry_and_denial_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            httpx.Response(429, json={"error": "slow_down"}),
            httpx.Response(400, json={"error": "access_denied", "error_description": "secret"}),
        )
    )

    class Client:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(self, *_: Any, **__: Any) -> httpx.Response:
            return next(responses)

    monkeypatch.setattr("opentulpa.inference.service.httpx.AsyncClient", Client)
    service = _inference(tmp_path)
    now = datetime.now(UTC)
    pending = DeviceLogin(
        id="login-slow",
        tenant_id="tenant-1",
        status="pending",
        verification_url="https://auth.openai.com/codex/device",
        user_code="ABCD-EFGH",
        device_auth_id="private-device-auth-id",
        interval_seconds=1,
        next_poll_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(minutes=10),
    )
    service._store.create_device_login(pending)  # noqa: SLF001

    slowed = await service.get_device_login("tenant-1", pending.id)
    assert slowed is not None and slowed["status"] == "pending"
    assert slowed["interval_seconds"] == 6
    stored = service._store.load_device_login("tenant-1", pending.id)  # noqa: SLF001
    assert stored is not None
    service._store.update_device_login(  # noqa: SLF001
        service._replace_login(stored, next_poll_at=now - timedelta(seconds=1))  # noqa: SLF001
    )
    denied = await service.get_device_login("tenant-1", pending.id)
    assert denied is not None and denied["status"] == "failed"
    assert denied["error"] == "authorization_failed"
    assert "secret" not in str(denied)

    expired = service._replace_login(  # noqa: SLF001
        pending,
        id="login-expired",
        expires_at=now - timedelta(seconds=1),
    )
    service._store.create_device_login(expired)  # noqa: SLF001
    expired_public = await service.get_device_login("tenant-1", expired.id)
    assert expired_public is not None and expired_public["status"] == "expired"


def test_pinned_langchain_codex_adapter_contract(tmp_path: Path) -> None:
    store = InferenceCredentialStore(tmp_path / "inference.db", cipher=_cipher())
    provider = CodexTokenProvider(tenant_id="tenant-1", store=store)

    model = build_codex_model(
        model="gpt-test",
        reasoning_effort="high",
        token_provider=provider,
        service_tier="priority",
    )
    signature = inspect.signature(type(model))

    assert "token_provider" in signature.parameters
    assert str(model.openai_api_base).rstrip("/") == "https://chatgpt.com/backend-api/codex"
    assert model.streaming is True
    assert model.store is False
    assert "max_completion_tokens" not in model.model_fields_set
    assert model.originator == "opentulpa"
    assert model.include == ["reasoning.encrypted_content"]
    assert model.reasoning == {"effort": "high", "summary": "auto"}
    assert model.service_tier == "priority"
    payload = model._get_request_payload(  # noqa: SLF001
        [SystemMessage(content="system instruction"), HumanMessage(content="hello")],
        _codex_headers={},
    )
    assert payload["instructions"] == "system instruction"
    assert payload["service_tier"] == "priority"
    assert all(item.get("role") != "system" for item in payload["input"])
    bound = model.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert bound.kwargs["tools"][0]["function"]["name"] == "lookup"

    class StructuredAnswer(BaseModel):
        answer: str

    assert model.with_structured_output(StructuredAnswer) is not None

    fallback_model = build_codex_model(
        model="gpt-test",
        reasoning_effort="high",
        token_provider=provider,
        buffer_for_fallback=True,
    )
    assert fallback_model.disable_streaming is True


@pytest.mark.asyncio
async def test_codex_retries_transient_stream_before_first_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InferenceCredentialStore(tmp_path / "inference.db", cipher=_cipher())
    provider = CodexTokenProvider(tenant_id="tenant-1", store=store)
    model = build_codex_model(
        model="gpt-test",
        reasoning_effort="high",
        token_provider=provider,
    )
    calls = 0
    sleeps: list[float] = []

    async def stream(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Our servers are currently overloaded. Please try again later.")
        yield ChatGenerationChunk(message=AIMessageChunk(content="recovered"))

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(_ChatOpenAICodex, "_astream", stream)
    monkeypatch.setattr("opentulpa.inference.codex.asyncio.sleep", sleep)

    chunks = [chunk async for chunk in model._astream([])]  # noqa: SLF001

    assert calls == 2
    assert sleeps == [0.5]
    assert [chunk.text for chunk in chunks] == ["recovered"]


def test_codex_does_not_retry_stream_after_first_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InferenceCredentialStore(tmp_path / "inference.db", cipher=_cipher())
    provider = CodexTokenProvider(tenant_id="tenant-1", store=store)
    model = build_codex_model(
        model="gpt-test",
        reasoning_effort="high",
        token_provider=provider,
    )
    calls = 0

    def stream(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        yield ChatGenerationChunk(message=AIMessageChunk(content="visible"))
        raise RuntimeError("Our servers are currently overloaded. Please try again later.")

    monkeypatch.setattr(_ChatOpenAICodex, "_stream", stream)

    with pytest.raises(RuntimeError, match="overloaded"):
        list(model._stream([]))  # noqa: SLF001

    assert calls == 1


@pytest.mark.asyncio
async def test_codex_401_refreshes_once_and_transient_fallback_stops_after_activity() -> None:
    class Provider:
        refreshes = 0

        async def aforce_refresh(self) -> None:
            self.refreshes += 1

    provider = Provider()
    codex_model = object()
    retry = _CodexAuthRetryMiddleware(  # type: ignore[arg-type]
        ResolvedModel(model=codex_model, token_provider=provider)
    )

    class UnauthorizedError(RuntimeError):
        status_code = 401

    calls = 0

    async def handler(_: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UnauthorizedError()
        return "ok"

    request = type(
        "Request",
        (),
        {"model": codex_model, "messages": [HumanMessage(content="hi")]},
    )()
    assert await retry.awrap_model_call(request, handler) == "ok"
    assert provider.refreshes == 1

    assert _before_current_run_activity(request) is True
    after_tool = type(
        "Request",
        (),
        {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
                ToolMessage(content="done", tool_call_id="1"),
            ]
        },
    )()
    assert _before_current_run_activity(after_tool) is False

    fallback_calls: list[str] = []

    class TransientError(RuntimeError):
        status_code = 503

    class Request:
        model = "codex"
        messages = after_tool.messages

        def override(self, *, model: str) -> Request:
            value = Request()
            value.model = model
            return value

    async def fail(request_value: Request) -> str:
        fallback_calls.append(request_value.model)
        raise TransientError()

    fallback = _ProviderFallbackMiddleware(
        ["api"],
        eligible=lambda error: getattr(error, "status_code", 0) == 503,
        allow_request=_before_current_run_activity,
    )
    with pytest.raises(TransientError):
        await fallback.awrap_model_call(Request(), fail)
    assert fallback_calls == ["codex"]

    try:
        raise httpx.ReadTimeout("timed out")
    except httpx.ReadTimeout as cause:
        wrapped = CodexProviderError("sanitized")
        wrapped.__cause__ = cause
    assert is_transient(wrapped) is True
    assert is_transient(CodexProviderError("rate limited", status_code=429)) is True
    assert (
        is_transient(RuntimeError("Our servers are currently overloaded. Please try again later."))
        is True
    )
    assert is_transient(UnauthorizedError()) is False


def test_graph_cache_separates_provider_model_effort_tier_and_is_bounded(tmp_path: Path) -> None:
    service = DeepAgentService(
        api_key="api-secret",
        base_url="https://openrouter.ai/api/v1",
        model_name="test-model",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        store_db_path=tmp_path / "store.db",
        runs_db_path=tmp_path / "runs.db",
        workspaces_root=tmp_path / "workspaces",
        model=_ToolCapableTextModel(responses=["unused"]),
        inference_service=_inference(tmp_path),
        graph_cache_limit=8,
    )
    plans = (
        ResolvedInferencePlan.resolve(
            InferenceSelection(provider="api", model="one", reasoning_effort="low"),
            preference_revision=1,
        ),
        ResolvedInferencePlan.resolve(
            InferenceSelection(provider="api", model="one", reasoning_effort="high"),
            preference_revision=2,
        ),
        ResolvedInferencePlan.resolve(
            InferenceSelection(provider="api", model="two", reasoning_effort="high"),
            preference_revision=3,
        ),
        ResolvedInferencePlan.resolve(
            InferenceSelection(
                provider="codex",
                model="gpt-test",
                reasoning_effort="high",
            ),
            preference_revision=4,
        ),
        ResolvedInferencePlan.resolve(
            InferenceSelection(
                provider="codex",
                model="gpt-test",
                reasoning_effort="high",
                service_tier="priority",
            ),
            preference_revision=5,
        ),
    )

    keys = {service._inference_cache_key("tenant-1", plan) for plan in plans}  # noqa: SLF001
    assert len(keys) == 5
    for index in range(10):
        service._cache_spec_graph((index,), object())  # noqa: SLF001
    assert list(service._spec_graphs) == [(index,) for index in range(2, 10)]  # noqa: SLF001


@pytest.mark.asyncio
async def test_thread_preference_is_revisioned_owned_and_run_plan_is_pinned(tmp_path: Path) -> None:
    inference = _inference(tmp_path)
    service = DeepAgentService(
        api_key="api-secret",
        base_url="https://openrouter.ai/api/v1",
        model_name="test-model",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        store_db_path=tmp_path / "store.db",
        runs_db_path=tmp_path / "runs.db",
        workspaces_root=tmp_path / "workspaces",
        model=_ToolCapableTextModel(responses=["unused"]),
        inference_service=inference,
    )
    context = _context()
    await service.start()
    try:
        await service.ensure_thread(
            tenant_id=context.tenant_id,
            thread_id=context.thread_id,
            channel="web",
        )
        selected = InferenceSelection(
            provider="api",
            model="test-model",
            reasoning_effort="high",
        )
        updated = await service.update_thread_inference(
            tenant_id=context.tenant_id,
            thread_id=context.thread_id,
            expected_revision=0,
            selection=selected,
        )
        assert updated is not None and updated["revision"] == 1
        routine_context = AgentRunContext(
            tenant_id=context.tenant_id,
            actor_id=context.actor_id,
            thread_id=context.thread_id,
            channel="routine",
            run_kind="routine",
            correlation_id=context.correlation_id,
            origin=OriginRef(interface="routine", source_id="schedule-1"),
            agent_spec=AgentSpecRef(
                tenant_id=context.tenant_id,
                spec_id="routine",
                revision=1,
            ),
            trust_class="background",
        )
        routine_plan = await service._resolve_inference_plan(routine_context)  # noqa: SLF001
        assert routine_plan.primary == inference.default_selection
        with pytest.raises(InferenceConflictError):
            await service.update_thread_inference(
                tenant_id=context.tenant_id,
                thread_id=context.thread_id,
                expected_revision=0,
                selection=None,
            )
        assert (
            await service.get_thread_inference(
                tenant_id="other-tenant",
                thread_id=context.thread_id,
            )
            is None
        )

        prepared = await service._prepare_run(  # noqa: SLF001
            AgentRunRequest(context=context, text="Pinned plan")
        )
        await service.update_thread_inference(
            tenant_id=context.tenant_id,
            thread_id=context.thread_id,
            expected_revision=1,
            selection=None,
        )
        snapshot = await service.get_run(prepared.run_id)
        assert snapshot is not None and snapshot.inference_plan == prepared.inference_plan
        assert snapshot.inference_plan.primary.reasoning_effort == "high"
        await service.cancel(prepared.run_id)
    finally:
        await service.shutdown()
