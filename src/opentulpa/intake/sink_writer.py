"""Sink execution for intake workflow bookings."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from opentulpa.intake.sink_utils import (
    clean_mapping as _clean_mapping,
)
from opentulpa.intake.sink_utils import (
    google_sheets_top_level_arguments as _google_sheets_top_level_arguments,
)
from opentulpa.intake.sink_utils import (
    incoming_user_id as _incoming_user_id,
)
from opentulpa.intake.sink_utils import (
    incoming_username as _incoming_username,
)
from opentulpa.intake.sink_utils import (
    normalize_google_sheets_arguments as _normalize_google_sheets_arguments,
)
from opentulpa.intake.sink_utils import (
    normalize_google_sheets_field_mapping as _normalize_google_sheets_field_mapping,
)
from opentulpa.intake.sink_utils import (
    normalize_toolkit_slug as _normalize_toolkit_slug,
)
from opentulpa.intake.sink_utils import (
    sheet_cell_value as _sheet_cell_value,
)
from opentulpa.intake.workflow_runtime import (
    safe_dict as _safe_dict,
)
from opentulpa.intake.workflow_runtime import (
    safe_list as _safe_list,
)
from opentulpa.intake.workflow_runtime import (
    utc_now_iso as _utc_now_iso,
)

_LOCAL_CSV_SYSTEM_COLUMNS = frozenset(
    {
        "booking_id",
        "workflow_id",
        "workflow_name",
        "conversation_id",
        "customer_id",
        "status",
        "completed_at",
    }
)


class SinkWriter:
    """Writes completed or partial bookings to configured sinks."""

    def __init__(self, *, project_root: Path, composio: Any | None) -> None:
        self._project_root = project_root
        self._composio = composio

    def write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        sink_arguments: dict[str, Any] | None = None,
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        if sink_type == "local_csv":
            return self.write_to_local_csv(
                workflow=workflow,
                booking=booking,
                payload=payload,
                record_status=record_status,
            )
        if sink_type in {"google_sheets_composio", "generic_composio_write"}:
            return self.write_to_composio_sink(
                workflow=workflow,
                booking=booking,
                conversation_summary=conversation_summary,
                payload=payload,
                sink_arguments=sink_arguments,
                record_status=record_status,
            )
        return {}, f"unsupported sink_type={sink_type}"

    def write_to_local_csv(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        sink_config = _safe_dict(workflow.get("sink_config"))
        relative_path = str(sink_config.get("file_path", "") or "").strip()
        if not relative_path:
            return {}, "local_csv sink is missing file_path"
        absolute_path = (self._project_root / relative_path).resolve()
        base_row = self._local_csv_row(
            workflow=workflow,
            booking=booking,
            payload=payload,
            record_status=record_status,
        )
        rows, fieldnames = self._read_existing_csv(absolute_path, base_row)
        self._upsert_csv_row(rows=rows, row=base_row, booking_id=str(booking["booking_id"]))
        self._write_csv_rows(absolute_path=absolute_path, fieldnames=fieldnames, rows=rows)
        return {
            "sink_type": "local_csv",
            "file_path": relative_path,
            "booking_id": str(booking["booking_id"]),
        }, None

    def write_to_composio_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        sink_arguments: dict[str, Any] | None = None,
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return {}, "Composio is not available for sink execution"
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        tool_slug = self.resolve_composio_sink_tool_slug(
            sink_type=sink_type,
            sink_config=sink_config,
        )
        if not tool_slug:
            return {}, f"could not resolve a Composio tool for toolkit={toolkit or 'unknown'}"
        arguments_result = self._build_composio_arguments(
            workflow=workflow,
            booking=booking,
            conversation_summary=conversation_summary,
            payload=payload,
            sink_arguments=sink_arguments,
            record_status=record_status,
        )
        if isinstance(arguments_result, str):
            return {}, arguments_result
        connected_account_id = str(sink_config.get("connected_account_id", "") or "").strip() or None
        try:
            result = self._composio.execute_tool(
                customer_id=str(workflow["customer_id"]),
                tool_slug=tool_slug,
                arguments=arguments_result,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            return {}, f"sink execution failed: {exc}"
        if not bool(result.get("successful", False)):
            return {}, str(result.get("error") or "sink execution failed")
        return {
            "sink_type": str(workflow["sink_type"]),
            "toolkit": toolkit,
            "booking_id": str(booking["booking_id"]),
            "data": result.get("data"),
        }, None

    def resolve_composio_sink_tool_slug(
        self,
        *,
        sink_type: str,
        sink_config: dict[str, Any],
    ) -> str:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            raise ValueError("Composio is not available for sink tool resolution")
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        if not toolkit:
            raise ValueError("sink_config.toolkit is required")
        operation_hint = str(sink_config.get("operation_hint", "") or "").strip().lower()
        candidates: list[dict[str, Any]] = []
        for query in self._tool_search_queries(sink_type, operation_hint):
            result = self._composio.search_tools(query=query, toolkits=[toolkit], limit=20)
            if not bool(result.get("ok", False)):
                continue
            candidates.extend(item for item in _safe_list(result.get("items")) if isinstance(item, dict))
            selected = self.select_composio_sink_candidate(
                sink_type=sink_type,
                toolkit=toolkit,
                operation_hint=operation_hint or query,
                candidates=candidates,
            )
            if selected:
                return selected
        selected = self.select_composio_sink_candidate(
            sink_type=sink_type,
            toolkit=toolkit,
            operation_hint=operation_hint,
            candidates=candidates,
        )
        if selected:
            return selected
        raise ValueError(f"no matching tool found in toolkit={toolkit}")

    @staticmethod
    def select_composio_sink_candidate(
        *,
        sink_type: str,
        toolkit: str,
        operation_hint: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        best_slug = ""
        best_score = -1
        hint_tokens = [token for token in operation_hint.replace("_", " ").split() if len(token) > 2]
        for item in candidates:
            slug = str(item.get("slug", "") or "").strip()
            if not slug:
                continue
            upper_slug = slug.upper()
            haystack = " ".join(
                str(item.get(key, "") or "").lower()
                for key in ("slug", "name", "description")
            )
            score = 0
            if _normalize_toolkit_slug(toolkit).upper() in upper_slug:
                score += 20
            if sink_type == "google_sheets_composio":
                if "SHEET" in upper_slug:
                    score += 20
                if "ROW" in upper_slug:
                    score += 15
            for token in hint_tokens:
                if token in haystack:
                    score += 8
            input_schema = item.get("input_schema")
            if isinstance(input_schema, dict):
                schema_text = str(input_schema).lower()
                if "rows" in schema_text:
                    score += 10
                if "headers" in schema_text:
                    score += 6
                if "keycolumn" in schema_text:
                    score += 6
            if score > best_score:
                best_score = score
                best_slug = slug
        return best_slug

    def _build_composio_arguments(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        sink_arguments: dict[str, Any] | None,
        record_status: str | None,
    ) -> dict[str, Any] | str:
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        override_arguments = _safe_dict(sink_arguments)
        if sink_type == "google_sheets_composio":
            static_result = self._google_sheets_static_arguments(
                customer_id=str(workflow["customer_id"]),
                sink_config=sink_config,
                static_arguments=static_arguments,
            )
            if isinstance(static_result, str):
                return static_result
            static_arguments = static_result
            override_arguments = _normalize_google_sheets_arguments(override_arguments)
        enriched_payload = self._enriched_payload(
            workflow=workflow,
            booking=booking,
            conversation_summary=conversation_summary,
            payload=payload,
            record_status=record_status,
        )
        if sink_type == "google_sheets_composio":
            arguments = self._google_sheets_arguments(
                static_arguments,
                field_mapping,
                enriched_payload,
            )
            arguments.update(override_arguments)
            return arguments
        arguments = dict(static_arguments)
        for target_key, source_key in field_mapping.items():
            arguments[target_key] = enriched_payload.get(source_key)
        arguments.update(override_arguments)
        return arguments

    def _google_sheets_static_arguments(
        self,
        *,
        customer_id: str,
        sink_config: dict[str, Any],
        static_arguments: dict[str, Any],
    ) -> dict[str, Any] | str:
        top_level_arguments = _google_sheets_top_level_arguments(sink_config)
        normalized = _normalize_google_sheets_arguments({**top_level_arguments, **static_arguments})
        try:
            return self.resolve_google_sheets_sheet_name_for_sink(
                customer_id=customer_id,
                static_arguments=normalized,
                connected_account_id=str(sink_config.get("connected_account_id", "") or "").strip()
                or None,
                validate_target=True,
            )
        except ValueError as exc:
            return str(exc)

    def resolve_google_sheets_sheet_name_for_sink(
        self,
        *,
        customer_id: str,
        static_arguments: dict[str, Any],
        connected_account_id: str | None,
        validate_target: bool,
    ) -> dict[str, Any]:
        normalized = _normalize_google_sheets_arguments(static_arguments)
        spreadsheet_id = str(normalized.get("spreadsheetId", "") or "").strip()
        if not spreadsheet_id:
            raise ValueError("google_sheets_composio requires static_arguments.spreadsheetId")
        if str(normalized.get("sheetName", "") or "").strip() or not validate_target:
            return normalized
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return normalized
        list_tabs = getattr(self._composio, "list_google_sheets_tab_names", None)
        if not callable(list_tabs):
            return normalized
        try:
            result = list_tabs(
                customer_id=customer_id,
                spreadsheet_id=spreadsheet_id,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            raise ValueError(
                "unable to inspect Google Sheets tabs; specify "
                "sink_config.static_arguments.sheetName"
            ) from exc
        return self._resolve_sheet_name_result(normalized, spreadsheet_id, result)

    @staticmethod
    def _resolve_sheet_name_result(
        normalized: dict[str, Any],
        spreadsheet_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        sheet_names = [
            str(item or "").strip()
            for item in _safe_list(_safe_dict(result).get("sheet_names"))
            if str(item or "").strip()
        ]
        if len(sheet_names) == 1:
            return {**normalized, "sheetName": sheet_names[0]}
        if len(sheet_names) > 1:
            preview = ", ".join(sheet_names[:10])
            raise ValueError(
                "google_sheets_composio requires sink_config.static_arguments.sheetName "
                f"because spreadsheetId={spreadsheet_id} has multiple sheets: {preview}"
            )
        if bool(_safe_dict(result).get("ok", False)):
            raise ValueError(
                "unable to find any worksheets in the Google Sheets target; specify "
                "sink_config.static_arguments.sheetName"
            )
        raise ValueError(
            "unable to inspect Google Sheets tabs; specify "
            "sink_config.static_arguments.sheetName"
        )

    @staticmethod
    def _google_sheets_arguments(
        static_arguments: dict[str, Any],
        field_mapping: dict[str, str],
        enriched_payload: dict[str, Any],
    ) -> dict[str, Any]:
        mapping = _normalize_google_sheets_field_mapping(
            field_mapping,
            payload_keys=set(enriched_payload.keys()),
        )
        key_source = "booking_id"
        key_header = str(mapping.get(key_source, "Booking ID") or "Booking ID").strip()
        headers = [key_header]
        row = [_sheet_cell_value(enriched_payload.get(key_source))]
        for source_key, header_name in mapping.items():
            safe_source = str(source_key or "").strip()
            safe_header = str(header_name or "").strip()
            if not safe_source or not safe_header or safe_source == key_source:
                continue
            headers.append(safe_header)
            row.append(_sheet_cell_value(enriched_payload.get(safe_source)))
        return {**static_arguments, "headers": headers, "rows": [row], "keyColumn": key_header}

    @staticmethod
    def _enriched_payload(
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None,
    ) -> dict[str, Any]:
        enriched = {
            **payload,
            "booking_id": str(booking["booking_id"]),
            "workflow_id": str(workflow["workflow_id"]),
            "conversation_id": str(booking["conversation_id"]),
            "customer_id": str(workflow["customer_id"]),
            "incoming_user_id": _incoming_user_id(conversation_summary),
            "latest_inbound_sender_id": _incoming_user_id(conversation_summary),
            "username": _incoming_username(conversation_summary),
            "latest_inbound_sender_username": _incoming_username(conversation_summary),
        }
        if record_status:
            enriched["status"] = str(record_status).strip()
        return enriched

    @staticmethod
    def _local_csv_row(
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None,
    ) -> dict[str, str]:
        base_row = {
            "booking_id": str(booking["booking_id"]),
            "workflow_id": str(workflow["workflow_id"]),
            "workflow_name": str(workflow["name"]),
            "conversation_id": str(booking["conversation_id"]),
            "customer_id": str(workflow["customer_id"]),
            "status": str(record_status or "completed").strip() or "completed",
            "completed_at": _utc_now_iso(),
        }
        for key, value in payload.items():
            if str(key) in _LOCAL_CSV_SYSTEM_COLUMNS:
                continue
            base_row[str(key)] = str(value or "")
        return base_row

    @staticmethod
    def _read_existing_csv(
        absolute_path: Path,
        base_row: dict[str, str],
    ) -> tuple[list[dict[str, str]], list[str]]:
        rows: list[dict[str, str]] = []
        fieldnames = list(base_row.keys())
        if not absolute_path.exists():
            return rows, fieldnames
        with absolute_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                rows.append({str(k): str(v or "") for k, v in item.items()})
            for field in list(reader.fieldnames or []):
                if field not in fieldnames:
                    fieldnames.append(field)
        for field in base_row:
            if field not in fieldnames:
                fieldnames.append(field)
        return rows, fieldnames

    @staticmethod
    def _upsert_csv_row(
        *,
        rows: list[dict[str, str]],
        row: dict[str, str],
        booking_id: str,
    ) -> None:
        for existing in rows:
            if str(existing.get("booking_id", "")).strip() != booking_id:
                continue
            existing.update(row)
            return
        rows.append(row)

    @staticmethod
    def _write_csv_rows(
        *,
        absolute_path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        with absolute_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    @staticmethod
    def _tool_search_queries(sink_type: str, operation_hint: str) -> list[str]:
        queries: list[str] = []
        if operation_hint:
            queries.append(operation_hint)
        if sink_type == "google_sheets_composio":
            queries.extend(["upsert rows", "append rows", "add row", "rows"])
        elif operation_hint:
            queries.append("write")
        seen: set[str] = set()
        out: list[str] = []
        for query in queries:
            safe_query = str(query or "").strip().lower()
            if safe_query and safe_query not in seen:
                seen.add(safe_query)
                out.append(safe_query)
        return out
