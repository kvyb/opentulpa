from __future__ import annotations

from typing import Any


class Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | list[Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)
        self.content = b"" if payload is None else b"x"

    def json(self) -> dict[str, Any] | list[Any]:
        return self._payload if self._payload is not None else {}


class DummyRuntime:
    def __init__(
        self,
        responses: list[Response],
        *,
        customer_id: str = "telegram_123",
        thread_id: str = "thread_123",
    ) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._active_customer_id = customer_id
        self._active_thread_id = thread_id

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> Response:
        self.calls.append((method, path, kwargs))
        if not self._responses:
            raise RuntimeError("unexpected internal API call")
        return self._responses.pop(0)
