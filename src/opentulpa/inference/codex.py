"""Isolate the pinned private LangChain Codex adapter behind one module."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import httpx
import openai
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from langchain_openai.chatgpt_oauth import (
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_REDIRECT_URI,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_TOKEN_URL,
    DEFAULT_SCOPE,
    _ChatGPTToken,
    _token_from_response,
)

from opentulpa.inference.store import CodexCredential, InferenceCredentialStore

CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
_INITIAL_STREAM_ATTEMPTS = 3
_INITIAL_STREAM_RETRY_SECONDS = 0.5
logger = logging.getLogger(__name__)


def _repair_missing_function_call_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    """Make stateless Responses API history valid after an interrupted tool call."""

    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return payload
    calls = Counter(
        str(item.get("call_id"))
        for item in input_items
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("call_id")
    )
    outputs = Counter(
        str(item.get("call_id"))
        for item in input_items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id")
    )
    missing = calls - outputs
    if not missing:
        return payload

    repaired: list[Any] = []
    pending_outputs: list[dict[str, str]] = []
    for item in input_items:
        is_function_call = isinstance(item, dict) and item.get("type") == "function_call"
        if pending_outputs and not is_function_call:
            repaired.extend(pending_outputs)
            pending_outputs.clear()
        repaired.append(item)
        if not is_function_call:
            continue
        call_id = str(item.get("call_id") or "")
        if not call_id or missing[call_id] <= 0:
            continue
        missing[call_id] -= 1
        name = str(item.get("name") or "unknown")
        pending_outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": (
                    f"Tool call {name} with id {call_id} was cancelled before it "
                    "could be completed."
                ),
            }
        )
    repaired.extend(pending_outputs)
    return {**payload, "input": repaired}


class CodexAuthenticationError(RuntimeError):
    """Codex authentication is absent or can no longer be refreshed."""


class CodexProviderError(RuntimeError):
    """A sanitized Codex provider operation failed."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


def credential_from_oauth_payload(
    payload: dict[str, Any],
    *,
    fallback_refresh_token: str | None = None,
) -> CodexCredential:
    token = _token_from_response(payload, fallback_refresh_token=fallback_refresh_token)
    return CodexCredential(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        id_token=token.id_token,
        account_id=token.account_id,
        expires_at=token.expires_at,
    )


@dataclass(slots=True)
class CodexTokenProvider:
    tenant_id: str
    store: InferenceCredentialStore
    timeout_seconds: float = 30.0

    def get_token(self) -> _ChatGPTToken:
        try:
            credential = self.store.refresh_credential(
                self.tenant_id,
                self._refresh,
            )
        except FileNotFoundError as exc:
            raise CodexAuthenticationError("Codex is not connected") from exc
        return self._private_token(credential)

    async def aget_token(self) -> _ChatGPTToken:
        return await asyncio.to_thread(self.get_token)

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        return (await self.aget_token()).access_token

    def force_refresh(self) -> None:
        try:
            self.store.refresh_credential(self.tenant_id, self._refresh, force=True)
        except FileNotFoundError as exc:
            raise CodexAuthenticationError("Codex is not connected") from exc

    async def aforce_refresh(self) -> None:
        await asyncio.to_thread(self.force_refresh)

    def _refresh(self, current: CodexCredential) -> CodexCredential:
        try:
            response = httpx.post(
                CHATGPT_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": current.refresh_token,
                    "client_id": CHATGPT_CLIENT_ID,
                },
                headers={"Accept": "application/json"},
                timeout=self.timeout_seconds,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise CodexProviderError(
                    "Codex token refresh is temporarily unavailable",
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                raise CodexAuthenticationError("Codex authorization must be renewed")
            payload = response.json()
            if not isinstance(payload, dict):
                raise CodexAuthenticationError("Codex returned an invalid refresh response")
            return credential_from_oauth_payload(
                payload,
                fallback_refresh_token=current.refresh_token,
            )
        except CodexAuthenticationError:
            raise
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise CodexProviderError("Codex token refresh failed") from exc

    @staticmethod
    def _private_token(credential: CodexCredential) -> _ChatGPTToken:
        return _ChatGPTToken(
            access_token=credential.access_token,
            refresh_token=credential.refresh_token,
            expires_at=credential.expires_at.astimezone(UTC),
            account_id=credential.account_id,
            id_token=credential.id_token,
        )


class _RetryingChatOpenAICodex(_ChatOpenAICodex):
    """Retry transient Codex streams only before any chunk can reach the client."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _repair_missing_function_call_outputs(payload)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for attempt in range(_INITIAL_STREAM_ATTEMPTS):
            emitted = False
            try:
                for chunk in super()._stream(*args, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted or not is_transient(exc) or attempt == _INITIAL_STREAM_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Codex stream failed before its first chunk; retrying attempt %d of %d",
                    attempt + 2,
                    _INITIAL_STREAM_ATTEMPTS,
                )
                time.sleep(_INITIAL_STREAM_RETRY_SECONDS * (2**attempt))

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        for attempt in range(_INITIAL_STREAM_ATTEMPTS):
            emitted = False
            try:
                async for chunk in super()._astream(*args, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted or not is_transient(exc) or attempt == _INITIAL_STREAM_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Codex stream failed before its first chunk; retrying attempt %d of %d",
                    attempt + 2,
                    _INITIAL_STREAM_ATTEMPTS,
                )
                await asyncio.sleep(_INITIAL_STREAM_RETRY_SECONDS * (2**attempt))


def build_codex_model(
    *,
    model: str,
    reasoning_effort: str | None,
    token_provider: CodexTokenProvider,
    service_tier: str | None = None,
    buffer_for_fallback: bool = False,
) -> _ChatOpenAICodex:
    reasoning = (
        {"effort": reasoning_effort, "summary": "auto"} if reasoning_effort else {"summary": "auto"}
    )
    return _RetryingChatOpenAICodex(
        model=model,
        token_provider=token_provider,
        originator="opentulpa",
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
        output_version="responses/v1",
        service_tier=service_tier,
        max_retries=0,
        timeout=60.0,
        # A streamed primary can emit text before middleware sees its terminal
        # failure. Buffer only the opt-in cross-provider fallback mode so a
        # fallback never duplicates already-visible output.
        disable_streaming=buffer_for_fallback,
    )


def is_unauthorized(error: BaseException) -> bool:
    status = int(getattr(error, "status_code", 0) or 0)
    if status == 401:
        return True
    response = getattr(error, "response", None)
    return int(getattr(response, "status_code", 0) or 0) == 401


def is_transient(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(
            current,
            (httpx.TimeoutException, httpx.TransportError, openai.APIConnectionError),
        ):
            return True
        status = int(getattr(current, "status_code", 0) or 0)
        if not status:
            response = getattr(current, "response", None)
            status = int(getattr(response, "status_code", 0) or 0)
        if status == 429 or 500 <= status < 600:
            return True
        message = str(current or "").casefold()
        if (
            "server" in message
            and ("overload" in message or "temporarily unavailable" in message)
        ) or "try again later" in message:
            return True
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            code = str(body.get("code") or body.get("type") or "").casefold()
            if code in {
                "overloaded",
                "overloaded_error",
                "rate_limit_exceeded",
                "server_error",
            }:
                return True
        current = current.__cause__ or current.__context__
    return False


__all__ = [
    "CHATGPT_CLIENT_ID",
    "CHATGPT_DEVICE_CODE_URL",
    "CHATGPT_DEVICE_REDIRECT_URI",
    "CHATGPT_DEVICE_TOKEN_URL",
    "CHATGPT_TOKEN_URL",
    "CODEX_MODELS_URL",
    "DEFAULT_SCOPE",
    "CodexAuthenticationError",
    "CodexProviderError",
    "CodexTokenProvider",
    "build_codex_model",
    "credential_from_oauth_payload",
    "is_transient",
    "is_unauthorized",
]
