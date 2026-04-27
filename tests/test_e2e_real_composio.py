from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.e2e.harness.real_composio import (
    RecordingComposioService,
    build_recording_live_googlesheets_service,
    discover_local_customer_ids,
)


class _FakeLiveComposio:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_connected_accounts(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "list_connected_accounts",
                "customer_id": customer_id,
                "toolkits": toolkits or [],
                "statuses": statuses or [],
                "limit": limit,
            }
        )
        if customer_id == "telegram_123":
            return {
                "ok": True,
                "items": [{"id": "ca_live_123", "toolkit_slug": "googlesheets", "status": "ACTIVE"}],
            }
        return {"ok": True, "items": []}

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "execute_tool",
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": dict(arguments or {}),
                "connected_account_id": connected_account_id,
                "text": text,
            }
        )
        if tool_slug == "GOOGLESHEETS_CREATE_GOOGLE_SHEET1":
            return {
                "successful": True,
                "error": None,
                "data": {
                    "spreadsheetId": "sheet_live_123",
                    "display_url": "https://docs.google.com/spreadsheets/d/sheet_live_123/edit",
                },
            }
        return {"successful": True, "error": None, "data": {"updatedRows": 1}}


def test_discover_local_customer_ids_from_opentulpa_state(tmp_path: Path) -> None:
    state_path = tmp_path / ".opentulpa" / "telegram_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "sessions": {
                    "999": {"customer_id": "telegram_123", "user_id": 456},
                    "1000": {"customer_id": "invalid", "user_id": ""},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".opentulpa" / "file_vault" / "telegram_789").mkdir(parents=True)

    assert discover_local_customer_ids(project_root=tmp_path) == [
        "telegram_123",
        "telegram_456",
        "telegram_789",
    ]


def test_build_recording_live_googlesheets_service_discovers_and_creates_sheet(
    tmp_path: Path,
) -> None:
    (tmp_path / ".opentulpa" / "file_vault" / "telegram_123").mkdir(parents=True)
    fake = _FakeLiveComposio()

    service = build_recording_live_googlesheets_service(
        composio=fake,  # type: ignore[arg-type]
        project_root=tmp_path,
    )

    assert isinstance(service, RecordingComposioService)
    assert service.live_google_sheets_target.customer_id == "telegram_123"
    assert service.live_google_sheets_target.connected_account_id == "ca_live_123"
    assert service.live_google_sheets_target.spreadsheet_id == "sheet_live_123"
    assert service.live_google_sheets_target.sheet_name == "Bookings"
