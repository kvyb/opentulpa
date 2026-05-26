"""Shared helpers for intake sink configuration and execution."""

from __future__ import annotations

from typing import Any


def incoming_user_id(conversation_summary: dict[str, Any]) -> str:
    return str(
        conversation_summary.get("incoming_user_id")
        or conversation_summary.get("latest_inbound_sender_id")
        or conversation_summary.get("latest_inbound_sender_user_id")
        or ""
    ).strip()


def incoming_username(conversation_summary: dict[str, Any]) -> str:
    return str(
        conversation_summary.get("username")
        or conversation_summary.get("latest_inbound_sender_username")
        or ""
    ).strip()


def clean_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_field in value.items():
        key = str(raw_key or "").strip()
        field = str(raw_field or "").strip()
        if key and field:
            out[key] = field
    return out


def sheet_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def normalize_google_sheets_arguments(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    for canonical, aliases in {
        "spreadsheetId": ("spreadsheet_id",),
        "sheetName": ("sheet_name", "worksheet", "worksheet_name", "tab_name"),
    }.items():
        if str(out.get(canonical, "") or "").strip():
            continue
        for alias in aliases:
            alias_value = out.pop(alias, None)
            if str(alias_value or "").strip():
                out[canonical] = alias_value
                break
    return out


def google_sheets_top_level_arguments(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "spreadsheetId",
            "spreadsheet_id",
            "sheetName",
            "sheet_name",
            "worksheet",
            "worksheet_name",
            "tab_name",
        )
        if key in value
    }


def normalize_google_sheets_field_mapping(
    field_mapping: dict[str, str],
    *,
    payload_keys: set[str],
) -> dict[str, str]:
    """Return source-field -> sheet-header mapping."""

    out: dict[str, str] = {}
    for raw_key, raw_value in field_mapping.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if not key or not value:
            continue
        if key in payload_keys:
            out[key] = value
            continue
        if value in payload_keys:
            out[value] = key
            continue
        out[key] = value
    return out


def normalize_toolkit_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "")


def normalize_composio_tool_slug(value: Any) -> str:
    safe = str(value or "").strip()
    if not safe:
        return ""
    if "_" not in safe:
        return safe
    prefix, remainder = safe.split("_", 1)
    if prefix and prefix == prefix.lower():
        upper_prefix = prefix.upper()
        if remainder.upper().startswith(f"{upper_prefix}_"):
            return remainder
    return safe


def infer_toolkit_from_tool_slug(value: Any) -> str:
    safe = normalize_composio_tool_slug(value)
    if not safe:
        return ""
    if "_" not in safe:
        return normalize_toolkit_slug(safe)
    prefix, _ = safe.split("_", 1)
    return normalize_toolkit_slug(prefix)


def infer_operation_hint_from_tool_slug(value: Any) -> str:
    safe = normalize_composio_tool_slug(value)
    if not safe:
        return ""
    if "_" in safe:
        _, remainder = safe.split("_", 1)
    else:
        remainder = safe
    return str(remainder).replace("_", " ").strip().lower()
