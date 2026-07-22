"""Provider discovery, device OAuth, and model construction for Deep Agents."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from opentulpa.core.ids import new_short_id
from opentulpa.inference.codex import (
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_REDIRECT_URI,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_TOKEN_URL,
    CODEX_MODELS_URL,
    CodexAuthenticationError,
    CodexProviderError,
    CodexTokenProvider,
    build_codex_model,
    credential_from_oauth_payload,
)
from opentulpa.inference.models import (
    InferenceModel,
    InferenceProvider,
    InferenceSelection,
    InferenceServiceTier,
)
from opentulpa.inference.store import DeviceLogin, InferenceCredentialStore
from opentulpa.secrets.cipher import HostKeySecretCipher

_API_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_MODEL_CACHE_LIMIT = 24
_CATALOG_TTL = timedelta(minutes=5)
_CODEX_DEVICE_VERIFICATION_URL = "https://auth.openai.com/codex/device"


class InferenceConflictError(RuntimeError):
    """An inference preference or credential mutation is unsafe."""


class InferenceUnavailableError(RuntimeError):
    """A selected provider is not currently usable."""


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model: Any
    token_provider: CodexTokenProvider | None = None


class InferenceService:
    """Keep subscription auth separate from the Deep Agents orchestration loop."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        cipher: HostKeySecretCipher,
        api_key: str,
        api_base_url: str,
        api_default_model: str,
        api_reasoning_effort: str | None,
        api_fallback_models: tuple[str, ...] = (),
    ) -> None:
        self._store = InferenceCredentialStore(db_path, cipher=cipher)
        self._api_key = str(api_key or "").strip()
        self._api_base_url = str(api_base_url or "").strip().rstrip("/")
        self._api_default_model = str(api_default_model or "").strip()
        self._api_reasoning_effort = str(api_reasoning_effort or "").strip() or None
        self._api_fallback_models = tuple(api_fallback_models)
        self._models: OrderedDict[
            tuple[str, int, str, str | None, str | None, bool], ResolvedModel
        ] = OrderedDict()
        self._catalogs: dict[tuple[str, int], tuple[datetime, tuple[InferenceModel, ...]]] = {}
        self._device_locks: dict[str, asyncio.Lock] = {}

    @property
    def default_selection(self) -> InferenceSelection:
        return InferenceSelection(
            provider="api",
            model=self._api_default_model,
            reasoning_effort=self._api_reasoning_effort,
        )

    @property
    def api_fallback_models(self) -> tuple[str, ...]:
        return self._api_fallback_models

    def codex_connected(self, tenant_id: str) -> bool:
        return self._store.connected(tenant_id)

    def credential_revision(self, tenant_id: str) -> int:
        return self._store.credential_revision(tenant_id)

    async def status(self, tenant_id: str) -> dict[str, Any]:
        return {
            "api_default": self.default_selection.model_dump(mode="json"),
            "codex": {
                "connected": self.codex_connected(tenant_id),
                "credential_revision": self.credential_revision(tenant_id),
                "experimental": True,
            },
        }

    async def models(
        self,
        tenant_id: str,
        provider: InferenceProvider,
        *,
        query: str = "",
    ) -> tuple[InferenceModel, ...]:
        models = (
            await self._codex_models(tenant_id) if provider == "codex" else await self._api_models()
        )
        search = str(query or "").strip().casefold()
        if search:
            models = tuple(model for model in models if search in model.id.casefold())
        return models

    async def validate_selection(
        self,
        tenant_id: str,
        selection: InferenceSelection,
    ) -> InferenceSelection:
        if selection.provider == "codex":
            if not self.codex_connected(tenant_id):
                raise InferenceUnavailableError("Codex is not connected")
            models = await self._codex_models(tenant_id)
            selected = next((model for model in models if model.id == selection.model), None)
            if selected is None:
                raise ValueError("Codex model is not available for this account")
            if (
                selection.reasoning_effort is not None
                and selection.reasoning_effort not in selected.reasoning_efforts
            ):
                raise ValueError("reasoning effort is not supported by this Codex model")
            service_tier_ids = {tier.id for tier in selected.service_tiers}
            if (
                selection.service_tier is not None
                and selection.service_tier not in service_tier_ids
            ):
                raise ValueError("service tier is not supported by this Codex model")
            return selection
        if selection.fallback_to_api or selection.service_tier is not None:
            selection = selection.model_copy(
                update={"fallback_to_api": False, "service_tier": None}
            )
        return selection

    def resolve_model(self, tenant_id: str, selection: InferenceSelection) -> ResolvedModel:
        if selection.provider != "codex":
            raise ValueError("API models are resolved by the configured model factory")
        revision = self.credential_revision(tenant_id)
        if not self.codex_connected(tenant_id):
            raise InferenceUnavailableError("Codex is not connected")
        key = (
            tenant_id,
            revision,
            selection.model,
            selection.reasoning_effort,
            selection.service_tier,
            selection.fallback_to_api,
        )
        cached = self._models.get(key)
        if cached is not None:
            self._models.move_to_end(key)
            return cached
        provider = CodexTokenProvider(tenant_id=tenant_id, store=self._store)
        resolved = ResolvedModel(
            model=build_codex_model(
                model=selection.model,
                reasoning_effort=selection.reasoning_effort,
                service_tier=selection.service_tier,
                token_provider=provider,
                buffer_for_fallback=selection.fallback_to_api,
            ),
            token_provider=provider,
        )
        self._models[key] = resolved
        while len(self._models) > _MODEL_CACHE_LIMIT:
            self._models.popitem(last=False)
        return resolved

    async def start_device_login(self, tenant_id: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    CHATGPT_DEVICE_CODE_URL,
                    json={"client_id": CHATGPT_CLIENT_ID},
                    headers={"Accept": "application/json"},
                )
            if response.status_code >= 400:
                raise CodexProviderError("Codex device login could not be started")
            payload = response.json()
        except CodexProviderError:
            raise
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise CodexProviderError("Codex device login could not be started") from exc
        if not isinstance(payload, dict):
            raise CodexProviderError("Codex returned an invalid device login response")
        device_auth_id = str(payload.get("device_auth_id") or "").strip()
        user_code = str(payload.get("user_code") or "").strip()
        if not device_auth_id or not user_code:
            raise CodexProviderError("Codex returned an incomplete device login response")
        now = datetime.now(UTC)
        try:
            interval = max(3.0, float(payload.get("interval") or 5.0))
        except (TypeError, ValueError):
            interval = 5.0
        expires_at = self._device_expiry(payload.get("expires_at"), now=now)
        login = DeviceLogin(
            id=new_short_id("login", suffix_chars=12),
            tenant_id=tenant_id,
            status="pending",
            verification_url=_CODEX_DEVICE_VERIFICATION_URL,
            user_code=user_code,
            device_auth_id=device_auth_id,
            interval_seconds=interval,
            next_poll_at=now,
            expires_at=expires_at,
        )
        self._store.create_device_login(login)
        return self._public_login(login)

    async def get_device_login(self, tenant_id: str, login_id: str) -> dict[str, Any] | None:
        lock = self._device_locks.setdefault(login_id, asyncio.Lock())
        async with lock:
            login = self._store.load_device_login(tenant_id, login_id)
            if login is None:
                return None
            now = datetime.now(UTC)
            if login.status != "pending":
                return self._public_login(login)
            if now >= login.expires_at:
                expired = self._replace_login(login, status="expired", error_code="expired")
                self._store.update_device_login(expired)
                return self._public_login(expired)
            if now < login.next_poll_at:
                return self._public_login(login)
            return await self._poll_device_login(login)

    async def cancel_device_login(self, tenant_id: str, login_id: str) -> bool:
        self._device_locks.pop(login_id, None)
        return self._store.delete_device_login(tenant_id, login_id)

    def delete_credential(self, tenant_id: str) -> bool:
        deleted = self._store.delete_credential(tenant_id)
        self._catalogs = {
            key: value for key, value in self._catalogs.items() if key[0] != tenant_id
        }
        self._models = OrderedDict(
            (key, value) for key, value in self._models.items() if key[0] != tenant_id
        )
        return deleted

    async def _poll_device_login(self, login: DeviceLogin) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    CHATGPT_DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": login.device_auth_id,
                        "user_code": login.user_code,
                    },
                    headers={"Accept": "application/json"},
                )
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError):
            pending = self._replace_login(
                login,
                next_poll_at=datetime.now(UTC) + timedelta(seconds=login.interval_seconds),
            )
            self._store.update_device_login(pending)
            return self._public_login(pending)
        if not isinstance(payload, dict):
            payload = {}
        authorization_code = str(payload.get("authorization_code") or "").strip()
        code_verifier = str(payload.get("code_verifier") or "").strip()
        if response.status_code == 200 and authorization_code and code_verifier:
            return await self._exchange_device_login(login, authorization_code, code_verifier)
        error = self._oauth_error_code(payload)
        slow_down = response.status_code == 429 or error == "slow_down"
        interval = login.interval_seconds + (5.0 if slow_down else 0.0)
        if response.status_code in {403, 404, 429} or error in {
            "authorization_pending",
            "slow_down",
        }:
            pending = self._replace_login(
                login,
                interval_seconds=interval,
                next_poll_at=datetime.now(UTC) + timedelta(seconds=interval),
            )
            self._store.update_device_login(pending)
            return self._public_login(pending)
        failed = self._replace_login(login, status="failed", error_code="authorization_failed")
        self._store.update_device_login(failed)
        return self._public_login(failed)

    async def _exchange_device_login(
        self,
        login: DeviceLogin,
        authorization_code: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    CHATGPT_TOKEN_URL,
                    data={
                        "grant_type": "authorization_code",
                        "code": authorization_code,
                        "redirect_uri": CHATGPT_DEVICE_REDIRECT_URI,
                        "client_id": CHATGPT_CLIENT_ID,
                        "code_verifier": code_verifier,
                    },
                    headers={"Accept": "application/json"},
                )
            if response.status_code >= 400:
                raise CodexAuthenticationError("Codex authorization failed")
            payload = response.json()
            if not isinstance(payload, dict):
                raise CodexAuthenticationError("Codex authorization failed")
            credential = credential_from_oauth_payload(payload)
            self._store.save_credential(login.tenant_id, credential)
        except (CodexAuthenticationError, TypeError, ValueError, httpx.HTTPError):
            failed = self._replace_login(
                login,
                status="failed",
                error_code="authorization_failed",
            )
            self._store.update_device_login(failed)
            return self._public_login(failed)
        authorized = self._replace_login(login, status="authorized", error_code=None)
        self._store.update_device_login(authorized)
        self._catalogs = {
            key: value for key, value in self._catalogs.items() if key[0] != login.tenant_id
        }
        self._models = OrderedDict(
            (key, value) for key, value in self._models.items() if key[0] != login.tenant_id
        )
        return self._public_login(authorized)

    @staticmethod
    def _device_expiry(raw: Any, *, now: datetime) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(UTC)
        except (TypeError, ValueError):
            return now + timedelta(minutes=15)
        return parsed if parsed > now else now + timedelta(seconds=30)

    @staticmethod
    def _oauth_error_code(payload: dict[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "").strip()
        return str(error or "").strip()

    async def _codex_models(self, tenant_id: str) -> tuple[InferenceModel, ...]:
        if not self.codex_connected(tenant_id):
            raise InferenceUnavailableError("Codex is not connected")
        revision = self.credential_revision(tenant_id)
        cache_key = (tenant_id, revision)
        cached = self._catalogs.get(cache_key)
        if cached is not None and datetime.now(UTC) - cached[0] < _CATALOG_TTL:
            return cached[1]
        provider = CodexTokenProvider(tenant_id=tenant_id, store=self._store)
        try:
            response = await self._request_codex_models(await provider.aget_token())
            if response.status_code == 401:
                await provider.aforce_refresh()
                response = await self._request_codex_models(await provider.aget_token())
            if response.status_code == 401:
                raise CodexAuthenticationError("Codex authorization must be renewed")
            if response.status_code >= 400:
                raise CodexProviderError(
                    "Codex model discovery failed",
                    status_code=response.status_code,
                )
            payload = response.json()
        except (CodexAuthenticationError, CodexProviderError):
            raise
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise CodexProviderError("Codex model discovery failed") from exc
        raw_models = payload.get("models", []) if isinstance(payload, dict) else []
        ranked: list[tuple[int, InferenceModel]] = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            model_id = str(raw.get("slug") or raw.get("id") or "").strip()
            visibility = str(raw.get("visibility") or "").strip().casefold()
            if not model_id or visibility in {"hide", "hidden"}:
                continue
            efforts = self._reasoning_efforts(raw)
            default_effort = (
                str(
                    raw.get("default_reasoning_level") or raw.get("default_reasoning_effort") or ""
                ).strip()
                or None
            )
            if default_effort and efforts and default_effort not in efforts:
                default_effort = None
            service_tiers = self._service_tiers(raw)
            default_service_tier = str(raw.get("default_service_tier") or "").strip() or None
            if default_service_tier not in {tier.id for tier in service_tiers}:
                default_service_tier = None
            try:
                priority = int(raw.get("priority", 10_000))
            except (TypeError, ValueError):
                priority = 10_000
            ranked.append(
                (
                    priority,
                    InferenceModel(
                        provider="codex",
                        id=model_id,
                        reasoning_efforts=efforts,
                        default_reasoning_effort=default_effort,
                        service_tiers=service_tiers,
                        default_service_tier=default_service_tier,
                    ),
                )
            )
        models = tuple(item for _, item in sorted(ranked, key=lambda item: (item[0], item[1].id)))
        if not models:
            raise CodexProviderError("Codex returned no available models for this account")
        self._catalogs[cache_key] = (datetime.now(UTC), models)
        return models

    @staticmethod
    async def _request_codex_models(token: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token.access_token}"}
        if token.account_id:
            headers["ChatGPT-Account-Id"] = token.account_id
        async with httpx.AsyncClient(timeout=15.0) as client:
            return await client.get(CODEX_MODELS_URL, headers=headers)

    async def _api_models(self) -> tuple[InferenceModel, ...]:
        discovered: list[str] = []
        if self._api_base_url and self._api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{self._api_base_url}/models",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                    )
                if response.status_code < 400:
                    payload = response.json()
                    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
                    for raw in raw_models:
                        if isinstance(raw, dict):
                            model_id = str(raw.get("id") or "").strip()
                            if model_id and model_id not in discovered:
                                discovered.append(model_id)
            except (httpx.HTTPError, TypeError, ValueError):
                pass
        ordered = [self._api_default_model, *self._api_fallback_models, *discovered]
        unique = tuple(dict.fromkeys(model for model in ordered if model))
        return tuple(
            InferenceModel(
                provider="api",
                id=model,
                reasoning_efforts=_API_REASONING_EFFORTS,
                default_reasoning_effort=self._api_reasoning_effort,
            )
            for model in unique
        )

    @staticmethod
    def _reasoning_efforts(raw: dict[str, Any]) -> tuple[str, ...]:
        value = (
            raw.get("supported_reasoning_levels")
            or raw.get("supported_reasoning_efforts")
            or raw.get("reasoning_efforts")
            or []
        )
        efforts: list[str] = []
        if isinstance(value, dict):
            value = list(value)
        if isinstance(value, list | tuple):
            for item in value:
                if isinstance(item, dict):
                    effort = str(item.get("reasoning_effort") or item.get("effort") or "").strip()
                else:
                    effort = str(item or "").strip()
                if effort and effort not in efforts:
                    efforts.append(effort)
        return tuple(efforts)

    @staticmethod
    def _service_tiers(raw: dict[str, Any]) -> tuple[InferenceServiceTier, ...]:
        value = raw.get("service_tiers") or []
        if not isinstance(value, list | tuple):
            return ()
        tiers: list[InferenceServiceTier] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            tier_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or tier_id).strip()
            description = str(item.get("description") or "").strip()
            if not tier_id or not name or tier_id in seen:
                continue
            tiers.append(
                InferenceServiceTier(
                    id=tier_id,
                    name=name,
                    description=description,
                )
            )
            seen.add(tier_id)
        return tuple(tiers)

    @staticmethod
    def _replace_login(login: DeviceLogin, **changes: Any) -> DeviceLogin:
        return replace(login, **changes)

    @staticmethod
    def _public_login(login: DeviceLogin) -> dict[str, Any]:
        return {
            "login_id": login.id,
            "status": login.status,
            "verification_url": login.verification_url,
            "user_code": login.user_code,
            "interval_seconds": login.interval_seconds,
            "expires_at": login.expires_at.astimezone(UTC).isoformat(),
            "error": login.error_code,
        }


__all__ = [
    "InferenceConflictError",
    "InferenceService",
    "InferenceUnavailableError",
    "ResolvedModel",
]
