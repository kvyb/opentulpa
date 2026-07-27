"""HTTP/SSE client for the stable OpenTulpa Agent API."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

import httpx

_NOTIFICATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")


class AgentAPIError(RuntimeError):
    """Sanitized Agent API failure safe for interface-worker logs."""


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: str
    run_id: str
    sequence: int
    timestamp: str
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def settles_stream(self) -> bool:
        return self.type in {"run.completed", "run.failed", "approval.required"}


@dataclass(frozen=True, slots=True)
class AgentNotificationApproval:
    approval_id: str
    tool_name: str
    description: str
    allowed_decisions: tuple[Literal["approve", "edit", "reject"], ...]


@dataclass(frozen=True, slots=True)
class AgentNotification:
    id: int
    kind: str
    text: str
    status: str
    thread_id: str | None
    run_id: str | None
    approvals: tuple[AgentNotificationApproval, ...]
    created_at: str


async def parse_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[AgentEvent]:
    """Parse SSE frames and validate the normalized AgentRunEvent envelope."""

    event_name = ""
    event_id = ""
    data_lines: list[str] = []

    def parse_frame() -> AgentEvent | None:
        nonlocal event_name, event_id, data_lines
        if not data_lines:
            event_name = ""
            event_id = ""
            return None
        try:
            payload = json.loads("\n".join(data_lines))
        except (TypeError, ValueError) as exc:
            raise AgentAPIError("Agent API returned malformed SSE JSON.") from exc
        finally:
            data_lines = []
        if not isinstance(payload, dict):
            raise AgentAPIError("Agent API returned a non-object SSE event.")
        resolved_type = str(payload.get("type") or event_name).strip()
        run_id = str(payload.get("run_id") or "").strip()
        timestamp = str(payload.get("timestamp") or "").strip()
        try:
            sequence = int(payload.get("sequence") or event_id)
        except (TypeError, ValueError) as exc:
            raise AgentAPIError("Agent API returned an invalid event sequence.") from exc
        data = payload.get("data", {})
        event_name = ""
        event_id = ""
        if not resolved_type or not run_id or sequence < 1 or not timestamp:
            raise AgentAPIError("Agent API returned an incomplete SSE event.")
        if not isinstance(data, dict):
            raise AgentAPIError("Agent API returned invalid event data.")
        return AgentEvent(
            type=resolved_type,
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp,
            data=data,
        )

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            event = parse_frame()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field_name, separator, raw_value = line.partition(":")
        value = raw_value[1:] if separator and raw_value.startswith(" ") else raw_value
        if field_name == "event":
            event_name = value
        elif field_name == "id":
            event_id = value
        elif field_name == "data":
            data_lines.append(value)
    event = parse_frame()
    if event is not None:
        yield event


class AgentAPIClient:
    """Authenticated client used by interface and trigger workers."""

    def __init__(
        self,
        *,
        base_url: str,
        credential: str,
        client: httpx.AsyncClient | None = None,
        replay_poll_seconds: float = 0.25,
        replay_timeout_seconds: float = 300,
    ) -> None:
        safe_base = str(base_url or "").strip().rstrip("/")
        if not safe_base.startswith(("http://", "https://")):
            raise ValueError("Agent API base_url must use http or https")
        safe_credential = str(credential or "").strip()
        if not safe_credential:
            raise ValueError("Agent API credential is required")
        if replay_poll_seconds <= 0 or replay_timeout_seconds <= 0:
            raise ValueError("Agent API replay timings must be positive")
        self._base_url = safe_base
        self._credential = safe_credential
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._replay_poll_seconds = replay_poll_seconds
        self._replay_timeout_seconds = replay_timeout_seconds

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(
        self,
        *,
        source_event_id: str | None = None,
        last_event_id: int | None = None,
        origin_conversation_id: str | None = None,
        origin_message_id: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._credential}",
            "Accept": "text/event-stream, application/json",
        }
        if source_event_id:
            headers["Idempotency-Key"] = source_event_id
            headers["X-Correlation-ID"] = source_event_id[-128:]
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(max(0, last_event_id))
        if origin_conversation_id:
            headers["X-OpenTulpa-Origin-Conversation-ID"] = _origin_id(
                origin_conversation_id
            )
        if origin_message_id:
            headers["X-OpenTulpa-Origin-Message-ID"] = _origin_id(origin_message_id)
        return headers

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers(),
                json=dict(json_body) if json_body is not None else None,
                params=dict(params) if params is not None else None,
                timeout=30,
            )
        except httpx.HTTPError as exc:
            raise AgentAPIError("Agent API request failed.") from exc
        if response.status_code != expected_status:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = str(payload.get("detail") or "").strip()
            except ValueError:
                pass
            suffix = f": {detail}" if detail else "."
            raise AgentAPIError(
                f"Agent API returned HTTP {response.status_code}{suffix}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AgentAPIError("Agent API returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise AgentAPIError("Agent API returned an invalid response.")
        return payload

    async def ensure_thread(self, thread_id: str) -> None:
        await self._request_json(
            "PUT",
            f"/v2/agent/threads/{quote(thread_id, safe='')}",
            json_body={},
        )

    async def get_thread_inference(self, thread_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/v2/agent/threads/{quote(thread_id, safe='')}/inference",
        )

    async def update_thread_inference(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        selection: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await self._request_json(
            "PATCH",
            f"/v2/agent/threads/{quote(thread_id, safe='')}/inference",
            json_body={
                "expected_revision": max(0, int(expected_revision)),
                "selection": dict(selection),
            },
        )

    async def inference_status(self) -> dict[str, Any]:
        return await self._request_json("GET", "/v2/inference")

    async def list_models(
        self,
        *,
        provider: Literal["api", "codex"],
        query: str = "",
    ) -> list[dict[str, Any]]:
        payload = await self._request_json(
            "GET",
            "/v2/inference/models",
            params={"provider": provider, "query": str(query or "")[:200]},
        )
        models = payload.get("models")
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise AgentAPIError("Agent API returned invalid inference models.")
        return [dict(item) for item in models]

    async def start_codex_login(self) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/v2/inference/codex/device-logins",
            json_body={},
            expected_status=201,
        )

    async def get_codex_login(self, login_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/v2/inference/codex/device-logins/{quote(login_id, safe='')}",
        )

    async def cancel_thread(self, thread_id: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/v2/agent/threads/{quote(thread_id, safe='')}/cancel",
            json_body={},
        )

    async def upload_file(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str | None,
        kind: str,
        caption: str | None,
        source_event_id: str,
    ) -> str:
        if not content:
            raise AgentAPIError("Telegram attachment was empty.")
        try:
            response = await self._client.post(
                f"{self._base_url}/v2/files",
                headers=self._headers(source_event_id=source_event_id),
                data={"kind": kind, "caption": caption or ""},
                files={
                    "upload": (
                        filename or "file.bin",
                        content,
                        mime_type or "application/octet-stream",
                    )
                },
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise AgentAPIError("Agent API file upload failed.") from exc
        if response.status_code != 201:
            raise AgentAPIError("Agent API rejected a Telegram attachment.")
        try:
            payload = response.json()
            file_id = str(payload["file"]["id"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentAPIError("Agent API returned an invalid file record.") from exc
        if not file_id:
            raise AgentAPIError("Agent API returned an empty file identifier.")
        return file_id

    async def start_run(
        self,
        *,
        thread_id: str,
        text: str,
        file_ids: list[str],
        source_event_id: str,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._stream_and_replay(
            method="POST",
            path="/v2/agent/runs",
            headers=self._headers(
                source_event_id=source_event_id,
                origin_conversation_id=thread_id,
                origin_message_id=source_event_id,
            ),
            json_body={"thread_id": thread_id, "text": text, "file_ids": file_ids},
        ):
            yield event

    async def resume_run(
        self,
        *,
        run_id: str,
        approval_id: str,
        decision: Literal["approve", "edit", "reject"],
        source_event_id: str,
        edited_arguments: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        body: dict[str, Any] = {"approval_id": approval_id, "decision": decision}
        if decision == "edit":
            if edited_arguments is None:
                raise ValueError("edited arguments are required for edit approval")
            body["edited_arguments"] = dict(edited_arguments)
        async for event in self._stream_and_replay(
            method="POST",
            path=f"/v2/agent/runs/{quote(run_id, safe='')}/resume",
            headers=self._headers(
                source_event_id=source_event_id,
                origin_message_id=source_event_id,
            ),
            json_body=body,
        ):
            yield event

    async def replay_run(
        self,
        *,
        run_id: str,
        after_sequence: int,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._replay_until_settled(run_id, after_sequence):
            yield event

    async def _stream_and_replay(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        run_id = ""
        sequence = 0
        settled = False
        try:
            async for event in self._stream(
                method=method,
                path=path,
                headers=headers,
                json_body=json_body,
            ):
                if run_id and event.run_id != run_id:
                    raise AgentAPIError("Agent API changed run identifiers mid-stream.")
                if event.sequence <= sequence:
                    continue
                run_id = event.run_id
                sequence = event.sequence
                settled = event.settles_stream
                yield event
        except httpx.HTTPError:
            if not run_id:
                raise AgentAPIError("Agent API stream failed before returning a run id.") from None
        if settled:
            return
        if not run_id:
            raise AgentAPIError("Agent API stream ended before returning a run id.")
        async for event in self._replay_until_settled(run_id, sequence):
            yield event

    async def _stream(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async with self._client.stream(
            method,
            f"{self._base_url}{path}",
            headers=dict(headers),
            json=dict(json_body) if json_body is not None else None,
            params=dict(params) if params is not None else None,
            timeout=None,
        ) as response:
            if not response.is_success:
                await response.aread()
                raise AgentAPIError(f"Agent API returned HTTP {response.status_code}.")
            async for event in parse_sse_lines(response.aiter_lines()):
                yield event

    async def _replay_until_settled(
        self,
        run_id: str,
        after_sequence: int,
    ) -> AsyncIterator[AgentEvent]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._replay_timeout_seconds
        cursor = max(0, after_sequence)
        terminal_seen_at: float | None = None
        while loop.time() < deadline:
            async for event in self._stream(
                method="GET",
                path=f"/v2/agent/runs/{quote(run_id, safe='')}/events",
                headers=self._headers(last_event_id=cursor),
                params={"after_sequence": cursor},
            ):
                if event.run_id != run_id:
                    raise AgentAPIError("Agent API replay returned the wrong run id.")
                if event.sequence <= cursor:
                    continue
                cursor = event.sequence
                yield event
                if event.settles_stream:
                    return
            snapshot = await self.get_run(run_id)
            status = str(snapshot.get("status") or "")
            if status in {"completed", "failed", "cancelled", "interrupted"}:
                terminal_seen_at = terminal_seen_at or loop.time()
                if loop.time() - terminal_seen_at >= min(5, self._replay_timeout_seconds):
                    raise AgentAPIError(
                        "Agent API terminal state is missing its durable event."
                    )
            else:
                terminal_seen_at = None
            await asyncio.sleep(self._replay_poll_seconds)
        raise AgentAPIError("Agent API replay timed out.")

    async def get_run(self, run_id: str) -> dict[str, Any]:
        try:
            response = await self._client.get(
                f"{self._base_url}/v2/agent/runs/{quote(run_id, safe='')}",
                headers=self._headers(),
                timeout=20,
            )
        except httpx.HTTPError as exc:
            raise AgentAPIError("Agent API run lookup failed.") from exc
        if not response.is_success:
            raise AgentAPIError(f"Agent API run lookup returned HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise AgentAPIError("Agent API returned an invalid run snapshot.") from exc
        if not isinstance(payload, dict):
            raise AgentAPIError("Agent API returned an invalid run snapshot.")
        return payload

    async def list_notifications(
        self,
        *,
        after_id: int,
        limit: int = 100,
        wait_seconds: float = 0,
    ) -> list[AgentNotification]:
        try:
            response = await self._client.get(
                f"{self._base_url}/v2/notifications",
                headers=self._headers(),
                params={
                    "after_id": max(0, int(after_id)),
                    "limit": max(1, min(int(limit), 100)),
                    "wait_seconds": max(0.0, min(float(wait_seconds), 30.0)),
                },
                timeout=max(20.0, min(float(wait_seconds), 30.0) + 5.0),
            )
        except httpx.HTTPError as exc:
            raise AgentAPIError("Agent API notification lookup failed.") from exc
        if not response.is_success:
            raise AgentAPIError(
                f"Agent API notification lookup returned HTTP {response.status_code}."
            )
        try:
            payload = response.json()
            raw_notifications = payload["notifications"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentAPIError("Agent API returned invalid notifications.") from exc
        if not isinstance(raw_notifications, list):
            raise AgentAPIError("Agent API returned invalid notifications.")
        notifications = [_notification(item) for item in raw_notifications]
        if any(
            current.id <= previous.id
            for previous, current in zip(notifications, notifications[1:], strict=False)
        ):
            raise AgentAPIError("Agent API returned notifications out of order.")
        return notifications

    async def acknowledge_notification(self, notification_id: int) -> None:
        identifier = int(notification_id)
        if identifier < 1:
            raise ValueError("notification_id must be positive")
        try:
            response = await self._client.post(
                f"{self._base_url}/v2/notifications/{identifier}/ack",
                headers=self._headers(),
                timeout=20,
            )
        except httpx.HTTPError as exc:
            raise AgentAPIError("Agent API notification acknowledgement failed.") from exc
        if response.status_code != 204:
            raise AgentAPIError(
                "Agent API notification acknowledgement returned "
                f"HTTP {response.status_code}."
            )


def _notification(value: object) -> AgentNotification:
    if not isinstance(value, dict):
        raise AgentAPIError("Agent API returned an invalid notification.")
    try:
        raw_identifier = value["id"]
        if isinstance(raw_identifier, bool):
            raise TypeError
        identifier = int(raw_identifier)
        kind = _required_text(value["kind"])
        text = _required_text(value["text"])
        status = _required_text(value["status"])
        created_at = _required_text(value["created_at"])
        thread_id = _optional_text(value.get("thread_id"))
        run_id = _optional_text(value.get("run_id"))
        raw_approvals = value.get("approvals", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentAPIError("Agent API returned an invalid notification.") from exc
    if (
        identifier < 1
        or _NOTIFICATION_NAME_RE.fullmatch(kind) is None
        or _NOTIFICATION_NAME_RE.fullmatch(status) is None
        or not isinstance(raw_approvals, list)
    ):
        raise AgentAPIError("Agent API returned an invalid notification.")
    approvals = tuple(_notification_approval(item) for item in raw_approvals)
    if approvals and run_id is None:
        raise AgentAPIError("Agent API returned an approval without a run id.")
    return AgentNotification(
        id=identifier,
        kind=kind,
        text=text,
        status=status,
        thread_id=thread_id,
        run_id=run_id,
        approvals=approvals,
        created_at=created_at,
    )


def _notification_approval(value: object) -> AgentNotificationApproval:
    if not isinstance(value, dict):
        raise AgentAPIError("Agent API returned an invalid notification approval.")
    try:
        approval_id = _required_text(value["approval_id"])
        tool_name = _required_text(value["tool_name"])
        description = _required_text(
            value.get("description") or "Approval required."
        )
        raw_allowed = value["allowed_decisions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentAPIError("Agent API returned an invalid notification approval.") from exc
    if not isinstance(raw_allowed, list):
        raise AgentAPIError("Agent API returned an invalid notification approval.")
    if any(not isinstance(item, str) for item in raw_allowed):
        raise AgentAPIError("Agent API returned an invalid notification approval.")
    allowed = tuple(
        decision for decision in raw_allowed if decision in {"approve", "edit", "reject"}
    )
    if not approval_id or not tool_name or not allowed or len(allowed) != len(raw_allowed):
        raise AgentAPIError("Agent API returned an invalid notification approval.")
    return AgentNotificationApproval(
        approval_id=approval_id,
        tool_name=tool_name,
        description=description,
        allowed_decisions=allowed,  # type: ignore[arg-type]
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional protocol text must be a string")
    resolved = value.strip()
    return resolved or None


def _required_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("protocol text must be a string")
    resolved = value.strip()
    if not resolved:
        raise ValueError("protocol text is required")
    return resolved


def _origin_id(value: str) -> str:
    safe = str(value or "").strip()
    if not safe or len(safe) > 200 or any(ord(char) < 32 for char in safe):
        raise AgentAPIError("Interface origin metadata is invalid.")
    return safe


__all__ = [
    "AgentAPIClient",
    "AgentAPIError",
    "AgentEvent",
    "AgentNotification",
    "AgentNotificationApproval",
    "parse_sse_lines",
]
