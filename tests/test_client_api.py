from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from opentulpa.client.api import OpenTulpaClient, RemoteError
from opentulpa.client.config import Connection


def _connection() -> Connection:
    return Connection(
        url="https://tulpa.example",
        token="owner-token",
        thread_id="cli-thread",
        credential_storage="file",
    )


def _sse(*events: dict[str, object]) -> bytes:
    return "".join(
        f"event: {event['type']}\nid: {event['sequence']}\ndata: "
        f"{json.dumps(event)}\n\n"
        for event in events
    ).encode()


@pytest.mark.asyncio
async def test_client_streams_runs_replays_and_preserves_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v2/agent/runs":
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "type": "run.started",
                        "run_id": "run-1",
                        "sequence": 1,
                        "timestamp": "now",
                        "data": {},
                    },
                    {
                        "type": "message.delta",
                        "run_id": "run-1",
                        "sequence": 2,
                        "timestamp": "now",
                        "data": {"text": "hello"},
                    },
                    {
                        "type": "run.completed",
                        "run_id": "run-1",
                        "sequence": 3,
                        "timestamp": "now",
                        "data": {"text": "hello"},
                    },
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path.endswith("/events"):
            return httpx.Response(200, content=b": keepalive\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"detail": "missing"})

    client = OpenTulpaClient(_connection())
    await client._client.aclose()  # noqa: SLF001
    client._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=_connection().url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer owner-token"},
    )
    try:
        events = [
            event
            async for event in client.run(
                thread_id="cli-thread",
                text="hello",
                file_ids=[],
                idempotency_key="run-key",
            )
        ]
        replay = [event async for event in client.run_events("run-1", after_sequence=3)]
    finally:
        await client.aclose()

    assert [event.type for event in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert replay == []
    assert requests[0].headers["authorization"] == "Bearer owner-token"
    assert requests[0].headers["idempotency-key"] == "run-key"
    assert json.loads(requests[0].content)["thread_id"] == "cli-thread"
    assert requests[1].url.params["after_sequence"] == "3"


@pytest.mark.asyncio
async def test_client_sanitizes_http_and_invalid_event_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                content=b'data: {"type":"broken"}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(409, json={"detail": "thread has an unresolved agent run"})

    client = OpenTulpaClient(_connection())
    await client._client.aclose()  # noqa: SLF001
    client._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=_connection().url,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(RemoteError, match="unresolved"):
            await anext(client.run(thread_id="cli-thread", text="hello", file_ids=[]))
        with pytest.raises(RemoteError, match="invalid event"):
            await anext(client.run_events("run-1"))
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_client_uploads_files_with_idempotency(tmp_path: Path) -> None:
    attachment = tmp_path / "note.txt"
    attachment.write_text("hello", encoding="utf-8")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json={"file": {"id": "file-1"}})

    client = OpenTulpaClient(_connection())
    await client._client.aclose()  # noqa: SLF001
    client._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=_connection().url,
        transport=httpx.MockTransport(handler),
    )
    try:
        payload = await client.upload(attachment)
    finally:
        await client.aclose()

    assert payload["file"]["id"] == "file-1"
    assert captured[0].headers["idempotency-key"].startswith("cli-file:")
    assert b"hello" in captured[0].content


@pytest.mark.asyncio
async def test_client_marks_image_uploads_for_inline_vision(tmp_path: Path) -> None:
    attachment = tmp_path / "photo.png"
    attachment.write_bytes(b"png")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json={"file": {"id": "file-image"}})

    client = OpenTulpaClient(_connection())
    await client._client.aclose()  # noqa: SLF001
    client._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=_connection().url,
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.upload(attachment)
    finally:
        await client.aclose()

    assert b'name="kind"' in captured[0].content
    assert b"image" in captured[0].content
    assert b"image/png" in captured[0].content
