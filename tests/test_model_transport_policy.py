from __future__ import annotations

import httpx

from opentulpa.agent import model_transport_policy as policy


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
