"""Google Sheets-specific Composio adapter."""

from __future__ import annotations

from typing import Any


class GoogleSheetsComposioAdapter:
    def __init__(self, core: Any) -> None:
        self.core = core

    def list_tab_names(
        self,
        *,
        customer_id: str,
        spreadsheet_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_spreadsheet_id = str(spreadsheet_id or "").strip()
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_spreadsheet_id:
            raise ValueError("spreadsheet_id is required")

        last_error = ""
        for slug in self._candidate_sheet_metadata_tools():
            for arguments in (
                {"spreadsheetId": safe_spreadsheet_id},
                {"spreadsheet_id": safe_spreadsheet_id},
            ):
                result = self.core.execute_tool(
                    customer_id=safe_customer,
                    tool_slug=slug,
                    arguments=arguments,
                    connected_account_id=connected_account_id,
                )
                if not bool(result.get("successful", False)):
                    last_error = str(result.get("error") or "sheet metadata lookup failed")
                    continue
                sheet_names = _extract_google_sheet_names(result.get("data"))
                if sheet_names:
                    return {
                        "ok": True,
                        "spreadsheet_id": safe_spreadsheet_id,
                        "sheet_names": sheet_names,
                        "tool_slug": slug,
                    }
                last_error = f"{slug} returned no sheet names"
        return {
            "ok": False,
            "spreadsheet_id": safe_spreadsheet_id,
            "sheet_names": [],
            "error": last_error or "unable to discover Google Sheets worksheet names",
        }

    def _candidate_sheet_metadata_tools(self) -> list[str]:
        candidate_slugs = ["GOOGLESHEETS_GET_SHEET_NAMES"]
        try:
            with_tool_search = self.core.search_tools(
                query="list sheets in google spreadsheet",
                toolkits=["googlesheets"],
                limit=20,
            )
        except Exception:
            with_tool_search = {}
        for item in _safe_list(with_tool_search.get("items")):
            slug = str(_safe_dict(item).get("slug", "") or "").strip()
            upper_slug = slug.upper()
            if slug and (
                "GET_SHEET_NAMES" in upper_slug or "GET_SPREADSHEET_INFO" in upper_slug
            ):
                candidate_slugs.append(slug)
        return _unique_strings(candidate_slugs)


def _extract_google_sheet_names(value: Any) -> list[str]:
    names: list[Any] = []

    def visit(node: Any, *, sheet_context: bool = False) -> None:
        if isinstance(node, list):
            _visit_sheet_list(node, names=names, sheet_context=sheet_context, visit=visit)
            return
        if not isinstance(node, dict):
            return
        _collect_sheet_name_keys(node, names=names)
        _collect_sheet_container_keys(node, names=names, visit=visit)
        for key in ("data", "result"):
            raw = node.get(key)
            if isinstance(raw, dict | list):
                visit(raw, sheet_context=sheet_context)

    visit(value)
    return _unique_strings(names)


def _visit_sheet_list(
    node: list[Any],
    *,
    names: list[Any],
    sheet_context: bool,
    visit: Any,
) -> None:
    if all(isinstance(item, str) for item in node):
        names.extend(node)
        return
    for item in node:
        if isinstance(item, str) and sheet_context:
            names.append(item)
            continue
        visit(item, sheet_context=sheet_context)


def _collect_sheet_name_keys(node: dict[str, Any], *, names: list[Any]) -> None:
    for key in ("sheet_names", "sheetNames", "worksheet_names", "worksheetNames"):
        raw = node.get(key)
        if isinstance(raw, list):
            names.extend(raw)


def _collect_sheet_container_keys(node: dict[str, Any], *, names: list[Any], visit: Any) -> None:
    for key in ("sheets", "worksheets", "tabs"):
        raw = node.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                candidate = _sheet_name_from_dict(item)
                names.append(candidate) if candidate else visit(item, sheet_context=True)


def _sheet_name_from_dict(item: dict[str, Any]) -> str:
    props = _safe_dict(item.get("properties"))
    candidate = (
        item.get("sheetName")
        or item.get("sheet_name")
        or item.get("name")
        or item.get("title")
        or props.get("title")
        or props.get("sheetName")
        or props.get("name")
    )
    return str(candidate or "").strip()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out
