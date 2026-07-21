"""Async V2 transport used by the local terminal client."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from opentulpa.client.config import Connection


class RemoteError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ClientEvent:
    type: str
    run_id: str
    sequence: int
    timestamp: str
    data: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.type in {"run.completed", "run.failed", "approval.required"}


class OpenTulpaClient:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._client = httpx.AsyncClient(
            base_url=connection.url,
            timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10),
            follow_redirects=False,
            trust_env=False,
            headers=self._headers(),
        )

    async def __aenter__(self) -> OpenTulpaClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def host_status(self) -> dict[str, Any]:
        return await self._json("GET", "/_host/api/status")

    async def run(
        self,
        *,
        thread_id: str,
        text: str,
        file_ids: list[str],
        idempotency_key: str | None = None,
    ) -> AsyncIterator[ClientEvent]:
        async for event in self._stream(
            "POST",
            "/v2/agent/runs",
            json_body={"thread_id": thread_id, "text": text, "file_ids": file_ids},
            headers={"Idempotency-Key": idempotency_key or f"cli-run:{uuid4()}"},
        ):
            yield event

    async def run_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[ClientEvent]:
        async for event in self._stream(
            "GET",
            f"/v2/agent/runs/{run_id}/events",
            params={"after_sequence": max(0, after_sequence)},
        ):
            yield event

    async def resume(
        self,
        run_id: str,
        *,
        approval_id: str,
        decision: str,
        edited_arguments: dict[str, Any] | None = None,
    ) -> AsyncIterator[ClientEvent]:
        async for event in self._stream(
            "POST",
            f"/v2/agent/runs/{run_id}/resume",
            json_body={
                "approval_id": approval_id,
                "decision": decision,
                "edited_arguments": edited_arguments,
            },
        ):
            yield event

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/v2/agent/runs/{run_id}")

    async def cancel(self, run_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/v2/agent/runs/{run_id}/cancel")

    async def upload(self, path: Path) -> dict[str, Any]:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RemoteError(f"Could not read attachment: {path}") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        kind = "image" if media_type.startswith("image/") else "document"
        try:
            response = await self._client.post(
                "/v2/files",
                headers={"Idempotency-Key": f"cli-file:{uuid4()}"},
                files={"upload": (path.name, raw, media_type)},
                data={"kind": kind},
            )
        except httpx.HTTPError as exc:
            raise RemoteError("The attachment upload disconnected.") from exc
        return self._response_json(response)

    async def notifications(
        self,
        *,
        after_id: int,
        wait_seconds: float = 20,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            "/v2/notifications",
            params={"after_id": max(0, after_id), "limit": 100, "wait_seconds": wait_seconds},
        )

    async def acknowledge_notification(self, notification_id: int) -> None:
        try:
            response = await self._client.post(f"/v2/notifications/{notification_id}/ack")
        except httpx.HTTPError as exc:
            raise RemoteError("Could not acknowledge the OpenTulpa notification.") from exc
        if response.status_code != 204:
            self._raise_response(response)

    async def logs(self, *, after: int = 0) -> list[dict[str, Any]]:
        payload = await self._json("GET", "/_host/api/logs", params={"after": max(0, after)})
        logs = payload.get("logs", [])
        return [item for item in logs if isinstance(item, dict)]

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, params=params)
        except httpx.HTTPError as exc:
            raise RemoteError("Could not reach the OpenTulpa server.") from exc
        return self._response_json(response)

    async def _stream(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[ClientEvent]:
        try:
            async with self._client.stream(
                method,
                path,
                json=json_body,
                params=params,
                headers=headers,
            ) as response:
                if not response.is_success:
                    raw = await response.aread()
                    self._raise_response(response, raw=raw)
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                        continue
                    if not data_lines:
                        continue
                    yield _event("\n".join(data_lines))
                    data_lines.clear()
                if data_lines:
                    yield _event("\n".join(data_lines))
        except RemoteError:
            raise
        except httpx.HTTPError as exc:
            raise RemoteError("The OpenTulpa event stream disconnected.") from exc

    def _headers(self) -> dict[str, str]:
        if not self.connection.token:
            return {}
        return {"Authorization": f"Bearer {self.connection.token}"}

    def _response_json(self, response: httpx.Response) -> dict[str, Any]:
        if not response.is_success:
            self._raise_response(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteError("OpenTulpa returned an invalid response.") from exc
        if not isinstance(payload, dict):
            raise RemoteError("OpenTulpa returned an invalid response.")
        return payload

    @staticmethod
    def _raise_response(response: httpx.Response, *, raw: bytes | None = None) -> None:
        content = response.content if raw is None else raw
        message = f"OpenTulpa request failed (HTTP {response.status_code})."
        try:
            payload = json.loads(content)
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, str) and detail.strip():
                message = detail.strip()
        except (ValueError, UnicodeDecodeError):
            pass
        raise RemoteError(message, status_code=response.status_code)


def _event(raw: str) -> ClientEvent:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise ValueError
        return ClientEvent(
            type=str(payload["type"]),
            run_id=str(payload["run_id"]),
            sequence=max(0, int(payload["sequence"])),
            timestamp=str(payload["timestamp"]),
            data=data,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteError("OpenTulpa returned an invalid event.") from exc


__all__ = ["ClientEvent", "OpenTulpaClient", "RemoteError"]
