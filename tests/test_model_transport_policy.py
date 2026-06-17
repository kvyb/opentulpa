from __future__ import annotations

import httpx

from opentulpa.agent import model_transport_policy as policy
from opentulpa.agent.model_provider_profile import model_provider_profile


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider failed")
        self.status_code = status_code


def test_transport_policy_retries_remote_protocol_error() -> None:
    exc = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body "
        "(incomplete chunked read)"
    )

    assert policy.is_retryable_model_exception(exc) is True


def test_transport_policy_retries_transient_status_code() -> None:
    assert policy.is_retryable_model_exception(_StatusError(503)) is True


def test_transport_policy_does_not_retry_non_transient_status_code() -> None:
    assert policy.is_retryable_model_exception(_StatusError(400)) is False


def test_minimax_stream_chunk_timeout_has_longer_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENTULPA_MINIMAX_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OPENTULPA_ZAI_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("OPENTULPA_MODEL_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", raising=False)

    assert model_provider_profile("minimax/minimax-m3").stream_chunk_timeout_seconds() == 75.0
    assert model_provider_profile("z-ai/glm-5.2").stream_chunk_timeout_seconds() == 75.0


def test_minimax_stream_chunk_timeout_uses_specific_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OPENTULPA_MINIMAX_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", "90")

    assert model_provider_profile("minimax/minimax-m3").stream_chunk_timeout_seconds() == 90.0
    assert model_provider_profile("z-ai/glm-5.1").stream_chunk_timeout_seconds() == 30.0


def test_zai_stream_chunk_timeout_uses_specific_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OPENTULPA_ZAI_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", "90")

    assert model_provider_profile("z-ai/glm-5.2").stream_chunk_timeout_seconds() == 90.0


def test_stream_chunk_timeout_keeps_legacy_first_chunk_env_alias(monkeypatch) -> None:
    monkeypatch.delenv("OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("OPENTULPA_MODEL_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", "0.2")

    assert model_provider_profile("z-ai/glm-5.1").stream_chunk_timeout_seconds() == 0.2
