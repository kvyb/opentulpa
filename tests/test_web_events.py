from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.web_events import register_web_event_routes
from opentulpa.web.events import WebEventStore, append_web_event, set_default_web_event_store


def test_web_event_store_lists_events_after_cursor(tmp_path: Path) -> None:
    store = WebEventStore(tmp_path / "web_events.db")
    first_id = store.append(
        customer_id="telegram_1",
        thread_id="chat-1",
        source="chat",
        kind="assistant_message",
        text="First",
    )
    second_id = store.append(
        customer_id="telegram_1",
        thread_id="chat-1",
        source="routine",
        kind="proactive_message",
        text="Second",
    )

    events = store.list_events(after_id=first_id)

    assert second_id > first_id
    assert [event["id"] for event in events] == [second_id]
    assert events[0]["text"] == "Second"


def test_web_events_route_requires_bearer_token(tmp_path: Path) -> None:
    store = WebEventStore(tmp_path / "web_events.db")
    store.append(
        customer_id="telegram_1",
        thread_id="chat-1",
        source="chat",
        kind="assistant_message",
        text="Hello dashboard",
    )
    app = FastAPI()
    register_web_event_routes(
        app,
        settings=type("Settings", (), {"opentulpa_web_token": "secret"})(),
        get_web_events=lambda: store,
    )

    with TestClient(app) as client:
        rejected = client.get("/web/events")
        accepted = client.get("/web/events", headers={"authorization": "Bearer secret"})

    assert rejected.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["events"][0]["text"] == "Hello dashboard"


def test_append_web_event_never_raises_when_store_fails() -> None:
    class BrokenStore:
        def append(self, **_: object) -> int:
            raise RuntimeError("sqlite unavailable")

    set_default_web_event_store(BrokenStore())  # type: ignore[arg-type]
    try:
        event_id = append_web_event(
            customer_id="telegram_1",
            thread_id="chat-1",
            source="chat",
            kind="assistant_message",
            text="Hello",
        )
    finally:
        set_default_web_event_store(None)

    assert event_id == 0
