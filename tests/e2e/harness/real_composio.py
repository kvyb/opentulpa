from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.integrations.composio import ComposioService

_CUSTOMER_ID_RE = re.compile(r"\b[a-z][a-z0-9_-]*_[a-zA-Z0-9][a-zA-Z0-9_-]*\b")
_SPREADSHEET_URL_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")
_E2E_SHEET_NAME = "Bookings"


@dataclass(frozen=True)
class LiveGoogleSheetsTarget:
    customer_id: str
    connected_account_id: str
    spreadsheet_id: str
    sheet_name: str
    display_url: str = ""
    created_title: str = ""


@dataclass
class RecordingComposioService:
    """Delegate to real Composio while preserving E2E sheet-write artifacts."""

    delegate: ComposioService
    live_google_sheets_target: LiveGoogleSheetsTarget
    calls: list[dict[str, Any]] = field(default_factory=list)
    sheet_writes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.delegate, "enabled", False))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append(
            {
                "method": "execute_tool",
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": args,
                "connected_account_id": connected_account_id,
                "text": text,
            }
        )
        result = self.delegate.execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=args,
            connected_account_id=connected_account_id,
            text=text,
        )
        if str(tool_slug or "").upper() == "GOOGLESHEETS_UPSERT_ROWS":
            write: dict[str, Any] = {
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": args,
                "connected_account_id": connected_account_id,
                "successful": bool(result.get("successful", False)),
                "error": result.get("error"),
                "data": result.get("data"),
            }
            headers = args.get("headers")
            rows = args.get("rows")
            if isinstance(headers, list) and isinstance(rows, list):
                write["normalized_rows"] = [
                    dict(zip([str(item) for item in headers], list(row), strict=False))
                    for row in rows
                    if isinstance(row, list)
                ]
            self.sheet_writes.append(write)
        return result


def discover_local_customer_ids(*, project_root: Path) -> list[str]:
    """Find local customer ids already known to this OpenTulpa checkout."""

    root = project_root.resolve()
    candidates: set[str] = set()
    _collect_ids_from_telegram_state(root / ".opentulpa" / "telegram_state.json", candidates)

    for db_path, table, columns in (
        (root / ".opentulpa" / "customer_profiles.db", "customer_profiles", ("customer_id",)),
        (root / ".opentulpa" / "context_events.db", "context_events", ("customer_id",)),
        (root / ".opentulpa" / "file_vault.db", "uploaded_files", ("customer_id",)),
        (root / ".opentulpa" / "skills.db", "skills", ("customer_id",)),
        (root / ".opentulpa" / "telegram_business.db", "telegram_business_connections", ("customer_id",)),
        (root / ".opentulpa" / "telegram_business.db", "telegram_business_messages", ("customer_id",)),
    ):
        _collect_ids_from_sqlite(db_path, table, columns, candidates)

    vault_dir = root / ".opentulpa" / "file_vault"
    if vault_dir.exists():
        for child in vault_dir.iterdir():
            if child.is_dir():
                _add_customer_id(candidates, child.name)

    return sorted(candidates)


def build_recording_live_googlesheets_service(
    *,
    composio: ComposioService,
    project_root: Path,
) -> RecordingComposioService | None:
    target_account = _discover_connected_googlesheets_account(
        composio=composio,
        customer_ids=discover_local_customer_ids(project_root=project_root),
    )
    if target_account is None:
        return None

    title = "OpenTulpa E2E AutoSpa Bookings " + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    created = composio.execute_tool(
        customer_id=target_account["customer_id"],
        connected_account_id=target_account["connected_account_id"],
        tool_slug="GOOGLESHEETS_CREATE_GOOGLE_SHEET1",
        arguments={"title": title},
    )
    if not bool(created.get("successful", False)):
        raise RuntimeError(f"failed to create live E2E Google Sheet: {created}")

    data = created.get("data")
    spreadsheet_id = _extract_spreadsheet_id(data)
    if not spreadsheet_id:
        raise RuntimeError(f"could not extract spreadsheet id from create response: {created}")

    sheet_name = _E2E_SHEET_NAME
    added_sheet = composio.execute_tool(
        customer_id=target_account["customer_id"],
        connected_account_id=target_account["connected_account_id"],
        tool_slug="GOOGLESHEETS_ADD_SHEET",
        arguments={"spreadsheet_id": spreadsheet_id, "title": _E2E_SHEET_NAME},
    )
    if not bool(added_sheet.get("successful", False)):
        sheet_name = "Sheet1"

    display_url = _extract_display_url(data)
    target = LiveGoogleSheetsTarget(
        customer_id=target_account["customer_id"],
        connected_account_id=target_account["connected_account_id"],
        spreadsheet_id=spreadsheet_id,
        sheet_name=sheet_name,
        display_url=display_url,
        created_title=title,
    )
    return RecordingComposioService(delegate=composio, live_google_sheets_target=target)


def _discover_connected_googlesheets_account(
    *,
    composio: ComposioService,
    customer_ids: list[str],
) -> dict[str, str] | None:
    for customer_id in customer_ids:
        if not customer_id.startswith("telegram_"):
            continue
        accounts = composio.list_connected_accounts(
            customer_id=customer_id,
            toolkits=["googlesheets"],
            statuses=["ACTIVE"],
            limit=10,
        )
        for item in accounts.get("items") or []:
            connected_account_id = str(item.get("id", "") or "").strip()
            if connected_account_id:
                return {
                    "customer_id": customer_id,
                    "connected_account_id": connected_account_id,
                }
    return None


def _collect_ids_from_telegram_state(path: Path, candidates: set[str]) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(sessions, dict):
        return
    for slot in sessions.values():
        if not isinstance(slot, dict):
            continue
        _add_customer_id(candidates, slot.get("customer_id"))
        user_id = str(slot.get("user_id", "") or "").strip()
        if user_id:
            _add_customer_id(candidates, f"telegram_{user_id}")


def _collect_ids_from_sqlite(
    db_path: Path,
    table: str,
    columns: tuple[str, ...],
    candidates: set[str],
) -> None:
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(db_path) as conn:
            names = {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                if len(row) > 1
            }
            if not names:
                return
            for column in columns:
                if column not in names:
                    continue
                for row in conn.execute(f"SELECT DISTINCT {column} FROM {table}"):
                    _add_customer_id(candidates, row[0] if row else "")
    except sqlite3.Error:
        return


def _add_customer_id(candidates: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    match = _CUSTOMER_ID_RE.fullmatch(text)
    if match:
        candidates.add(text)


def _extract_spreadsheet_id(value: Any) -> str:
    for key in ("spreadsheetId", "spreadsheet_id", "id"):
        found = _find_nested_key(value, key)
        if found:
            text = str(found).strip()
            if "/" not in text:
                return text
            url_match = _SPREADSHEET_URL_RE.search(text)
            if url_match:
                return url_match.group(1)
    display_url = _extract_display_url(value)
    url_match = _SPREADSHEET_URL_RE.search(display_url)
    return url_match.group(1) if url_match else ""


def _extract_display_url(value: Any) -> str:
    for key in ("display_url", "displayUrl", "spreadsheetUrl", "url"):
        found = _find_nested_key(value, key)
        if found:
            return str(found).strip()
    return ""


def _find_nested_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_nested_key(child, key)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_nested_key(child, key)
            if found:
                return found
    return None
