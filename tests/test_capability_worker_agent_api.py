from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from opentulpa.capability_workers.agent_api import (
    AgentAPIClient,
    AgentAPIError,
    parse_sse_lines,
)


async def _lines(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value


def _sse(*events: dict[str, object]) -> bytes:
    frames = [
        f"id: {event['sequence']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    ]
    return "".join(frames).encode()


def _event(event_type: str, sequence: int, data: dict[str, object]) -> dict[str, object]:
    return {
        "type": event_type,
        "run_id": "run_1",
        "sequence": sequence,
        "timestamp": "2026-07-20T00:00:00Z",
        "data": data,
    }


@pytest.mark.asyncio
async def test_sse_parser_accepts_comments_crlf_and_multiline_data() -> None:
    events = [
        event
        async for event in parse_sse_lines(
            _lines(
                ": keepalive\r",
                "id: 7\r",
                "event: message.delta\r",
                'data: {"run_id":"run_1","sequence":7,\r',
                'data: "timestamp":"now","data":{"text":"hello"}}\r',
                "\r",
            )
        )
    ]

    assert len(events) == 1
    assert events[0].type == "message.delta"
    assert events[0].sequence == 7
    assert events[0].data == {"text": "hello"}


@pytest.mark.asyncio
async def test_sse_parser_rejects_incomplete_events_without_leaking_payload() -> None:
    with pytest.raises(AgentAPIError, match="incomplete"):
        _ = [
            event
            async for event in parse_sse_lines(
                _lines('data: {"type":"message.delta","sequence":1}', "")
            )
        ]


@pytest.mark.asyncio
async def test_start_run_sends_public_contract_and_replays_disconnected_stream() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v2/agent/runs":
            assert json.loads(request.content) == {
                "thread_id": "thread_1",
                "text": "hello",
                "file_ids": ["file_1"],
            }
            return httpx.Response(200, content=_sse(_event("run.started", 1, {})))
        if request.url.path == "/v2/agent/runs/run_1/events":
            assert request.url.params["after_sequence"] == "1"
            assert request.headers["Last-Event-ID"] == "1"
            return httpx.Response(
                200,
                content=_sse(
                    _event("message.delta", 2, {"text": "hello"}),
                    _event("run.completed", 3, {"text": "hello"}),
                ),
            )
        raise AssertionError(request.url)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
        replay_poll_seconds=0.001,
    )
    events = [
        event
        async for event in client.start_run(
            thread_id="thread_1",
            text="hello",
            file_ids=["file_1"],
            source_event_id="telegram:99:12",
        )
    ]

    assert [event.type for event in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert requests[0].headers["Authorization"] == "Bearer private-token"
    assert requests[0].headers["Idempotency-Key"] == "telegram:99:12"
    assert requests[0].headers["X-Correlation-ID"] == "telegram:99:12"
    assert requests[0].headers["X-OpenTulpa-Origin-Conversation-ID"] == "thread_1"
    assert requests[0].headers["X-OpenTulpa-Origin-Message-ID"] == "telegram:99:12"
    await http.aclose()


@pytest.mark.asyncio
async def test_resume_and_upload_use_v2_routes_and_sanitized_failures() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v2/files":
            assert b'name="upload"' in request.content
            assert b"notes.txt" in request.content
            return httpx.Response(201, json={"file": {"id": "file_9"}})
        if request.url.path == "/v2/agent/runs/run_1/resume":
            assert json.loads(request.content) == {
                "approval_id": "approval_1",
                "decision": "edit",
                "edited_arguments": {"recipient": "owner@example.com"},
            }
            return httpx.Response(
                200,
                content=_sse(_event("run.completed", 4, {"text": "sent"})),
            )
        raise AssertionError(request.url)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
    )
    file_id = await client.upload_file(
        filename="notes.txt",
        content=b"hello",
        mime_type="text/plain",
        kind="document",
        caption=None,
        source_event_id="telegram:99:1:file:0",
    )
    resumed = [
        event
        async for event in client.resume_run(
            run_id="run_1",
            approval_id="approval_1",
            decision="edit",
            edited_arguments={"recipient": "owner@example.com"},
            source_event_id="telegram:99:2",
        )
    ]

    assert file_id == "file_9"
    assert resumed[-1].type == "run.completed"
    assert seen[1].headers["Idempotency-Key"] == "telegram:99:2"
    assert seen[1].headers["X-OpenTulpa-Origin-Message-ID"] == "telegram:99:2"
    await http.aclose()


@pytest.mark.asyncio
async def test_protocol_corruption_fails_closed_after_run_id() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                _event("run.started", 1, {}),
                {**_event("run.completed", 2, {}), "run_id": "run_other"},
            ),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
    )
    with pytest.raises(AgentAPIError, match="identifiers"):
        _ = [
            event
            async for event in client.start_run(
                thread_id="thread_1",
                text="hello",
                file_ids=[],
                source_event_id="telegram:99:1",
            )
        ]
    await http.aclose()


@pytest.mark.asyncio
async def test_notifications_use_scoped_list_and_ack_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v2/notifications":
            assert request.url.params["after_id"] == "4"
            assert request.url.params["limit"] == "20"
            return httpx.Response(
                200,
                json={
                    "notifications": [
                        {
                            "id": 5,
                            "kind": "approval.required",
                            "text": "Approval is waiting.",
                            "status": "interrupted",
                            "thread_id": "trigger:daily",
                            "run_id": "run-5",
                            "approvals": [
                                {
                                    "approval_id": "approval-5",
                                    "tool_name": "integration_invoke",
                                    "description": "Send an email.",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                            "created_at": "2026-07-20T00:00:00+00:00",
                        }
                    ],
                    "next_after_id": 5,
                },
            )
        if request.url.path == "/v2/notifications/5/ack":
            return httpx.Response(204)
        raise AssertionError(request.url)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
    )

    notifications = await client.list_notifications(after_id=4, limit=20)
    await client.acknowledge_notification(5)

    assert notifications[0].run_id == "run-5"
    assert notifications[0].approvals[0].allowed_decisions == (
        "approve",
        "reject",
    )
    assert all(request.headers["Authorization"] == "Bearer private-token" for request in requests)
    assert requests[1].method == "POST"
    await http.aclose()


@pytest.mark.asyncio
async def test_notifications_reject_out_of_order_or_incomplete_payloads() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "notifications": [
                    {
                        "id": 2,
                        "kind": "run.completed",
                        "text": "Done.",
                        "status": "completed",
                        "thread_id": None,
                        "run_id": None,
                        "approvals": [],
                        "created_at": "now",
                    },
                    {
                        "id": 1,
                        "kind": "run.completed",
                        "text": "Earlier.",
                        "status": "completed",
                        "thread_id": None,
                        "run_id": None,
                        "approvals": [],
                        "created_at": "now",
                    },
                ]
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
    )

    with pytest.raises(AgentAPIError, match="out of order"):
        await client.list_notifications(after_id=0)
    await http.aclose()


@pytest.mark.asyncio
async def test_interface_controls_use_scoped_agent_api_routes() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v2/agent/threads/thread_1" and request.method == "PUT":
            return httpx.Response(200, json={"thread_id": "thread_1"})
        if path == "/v2/inference":
            return httpx.Response(200, json={"codex": {"connected": False}})
        if path.endswith("/inference") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "revision": 0,
                    "effective": {"provider": "api", "model": "openai/gpt-5.2"},
                },
            )
        if path.endswith("/inference") and request.method == "PATCH":
            assert json.loads(request.content) == {
                "expected_revision": 0,
                "selection": {
                    "provider": "codex",
                    "model": "gpt-5.2-codex",
                },
            }
            return httpx.Response(
                200,
                json={
                    "revision": 1,
                    "effective": {"provider": "codex", "model": "gpt-5.2-codex"},
                },
            )
        if path == "/v2/inference/models":
            assert request.url.params["provider"] == "codex"
            assert request.url.params["query"] == "gpt"
            return httpx.Response(
                200,
                json={
                    "provider": "codex",
                    "models": [{"id": "gpt-5.2-codex", "reasoning_efforts": ["high"]}],
                },
            )
        if path == "/v2/inference/codex/device-logins" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "login_id": "login-1",
                    "verification_url": "https://auth.openai.com/device",
                    "user_code": "ABCD-EFGH",
                },
            )
        if path.endswith("/device-logins/login-1"):
            return httpx.Response(
                200,
                json={"login_id": "login-1", "status": "pending"},
            )
        if path.endswith("/cancel"):
            return httpx.Response(200, json={"status": "cancelled"})
        raise AssertionError(request.url)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AgentAPIClient(
        base_url="http://agent.test",
        credential="private-token",
        client=http,
    )

    await client.ensure_thread("thread_1")
    current = await client.get_thread_inference("thread_1")
    updated = await client.update_thread_inference(
        "thread_1",
        expected_revision=0,
        selection={"provider": "codex", "model": "gpt-5.2-codex"},
    )
    models = await client.list_models(provider="codex", query="gpt")
    status = await client.inference_status()
    login = await client.start_codex_login()
    login_status = await client.get_codex_login("login-1")
    cancelled = await client.cancel_thread("thread_1")

    assert current["revision"] == 0
    assert updated["revision"] == 1
    assert models[0]["id"] == "gpt-5.2-codex"
    assert status["codex"]["connected"] is False
    assert login["login_id"] == "login-1"
    assert login_status["status"] == "pending"
    assert cancelled["status"] == "cancelled"
    assert all(request.headers["Authorization"] == "Bearer private-token" for request in requests)
    await http.aclose()
