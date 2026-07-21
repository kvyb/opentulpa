"""Sink execution for intake workflow bookings."""

from __future__ import annotations

import csv
import io
import logging
import os
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
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
from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
    IdempotencyStore,
)
from opentulpa.persistence.tenant_namespace import tenant_namespace_label

logger = logging.getLogger(__name__)
_PUBLIC_SINK_ERROR = "intake sink execution failed"
_PUBLIC_INDETERMINATE_SINK_ERROR = "intake sink outcome is indeterminate"
_MAX_LOCAL_CSV_BYTES = 10 * 1024 * 1024
_FORMULA_PREFIXES = frozenset({"=", "+", "-", "@", "\t", "\r"})

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


def normalize_local_csv_path(value: Any, *, workflow_id: str) -> str:
    """Return a confined tenant-relative CSV path."""

    requested = str(value or "").strip()
    raw = requested or f"intake_{workflow_id or 'workflow'}.csv"
    if len(raw) > 240 or "\\" in raw or any(ord(char) < 32 for char in raw):
        raise ValueError("local_csv file_path is invalid")
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("local_csv file_path must be a relative path without traversal")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.suffix.casefold() != ".csv":
        raise ValueError("local_csv file_path must be a relative .csv path")
    return path.as_posix()


class SinkWriter:
    """Writes completed or partial bookings to configured sinks."""

    def __init__(
        self,
        *,
        sink_root: Path,
        composio: Any | None,
        idempotency: IdempotencyStore,
    ) -> None:
        requested_root = sink_root.expanduser().resolve(strict=False)
        requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if requested_root.is_symlink():
            raise ValueError("intake sink root cannot be a symbolic link")
        self._sink_root = requested_root.resolve()
        os.chmod(self._sink_root, 0o700)
        self._sink_root_fd = os.open(
            self._sink_root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        self._composio = composio
        self._idempotency = idempotency
        self._csv_locks_guard = threading.Lock()
        self._csv_locks: dict[str, threading.Lock] = {}

    def close(self) -> None:
        descriptor = self._sink_root_fd
        if descriptor < 0:
            return
        self._sink_root_fd = -1
        with suppress(OSError):
            os.close(descriptor)

    def __del__(self) -> None:
        if hasattr(self, "_sink_root_fd"):
            self.close()

    def write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
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
        try:
            relative_path = normalize_local_csv_path(
                relative_path,
                workflow_id=str(workflow.get("workflow_id") or "workflow"),
            )
        except (OSError, ValueError):
            return {}, "local_csv sink path is invalid"
        base_row = self._local_csv_row(
            workflow=workflow,
            booking=booking,
            payload=payload,
            record_status=record_status,
        )
        customer_id = str(workflow.get("customer_id") or "")
        lock = self._csv_lock(f"{tenant_namespace_label(customer_id)}/{relative_path}")
        try:
            with lock:
                path = PurePosixPath(relative_path)
                with self._open_csv_directory(
                    customer_id=customer_id,
                    directory_parts=path.parent.parts if path.parent != PurePosixPath(".") else (),
                ) as directory_fd:
                    rows, fieldnames = self._read_existing_csv(
                        directory_fd,
                        path.name,
                        base_row,
                    )
                    self._upsert_csv_row(
                        rows=rows,
                        row=base_row,
                        booking_id=str(booking["booking_id"]),
                    )
                    self._write_csv_rows(
                        directory_fd=directory_fd,
                        filename=path.name,
                        fieldnames=fieldnames,
                        rows=rows,
                    )
        except (OSError, ValueError):
            logger.exception(
                "intake local CSV sink failed",
                extra={"workflow_id": str(workflow.get("workflow_id") or "")},
            )
            return {}, _PUBLIC_SINK_ERROR
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
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return {}, "Composio is not available for sink execution"
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        tool_slug = str(sink_config.get("tool_slug") or "").strip()
        connected_account_id = str(sink_config.get("connected_account_id", "") or "").strip()
        if not toolkit or not tool_slug or not connected_account_id:
            return {}, "intake sink requires a pinned tenant connection and tool"
        arguments_result = self._build_composio_arguments(
            workflow=workflow,
            booking=booking,
            conversation_summary=conversation_summary,
            payload=payload,
            record_status=record_status,
        )
        if isinstance(arguments_result, str):
            return {}, arguments_result
        try:
            self._composio.resolve_sink_binding(
                tenant_id=str(workflow["customer_id"]),
                toolkit=toolkit,
                connected_account_id=connected_account_id,
                tool_slug=tool_slug,
                operation_hint="",
                required_arguments=set(arguments_result),
                allow_discovery=False,
            )
        except Exception as exc:
            logger.error(
                "intake sink binding validation failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"workflow_id": str(workflow.get("workflow_id") or "")},
            )
            return {}, _PUBLIC_SINK_ERROR
        operation = f"intake_sink.{sink_type}.{tool_slug}"
        effect_arguments = {
            "booking_id": str(booking["booking_id"]),
            "connected_account_id": connected_account_id,
            "arguments": arguments_result,
        }
        phase = str(record_status or "partial").strip().casefold() or "partial"
        revision = int(booking.get("sink_effect_revision") or 0)
        if revision <= 0:
            return {}, "intake sink effect revision is missing"
        idempotency_key = self.effect_idempotency_key(
            booking_id=str(booking["booking_id"]),
            phase=phase,
            revision=revision,
        )
        try:
            claim = self._idempotency.claim(
                tenant_id=str(workflow["customer_id"]),
                idempotency_key=idempotency_key,
                operation=operation,
                arguments=effect_arguments,
            )
        except IdempotencyPendingError:
            return {}, _PUBLIC_INDETERMINATE_SINK_ERROR
        except IdempotencyConflictError:
            return {}, _PUBLIC_INDETERMINATE_SINK_ERROR
        if not claim.created:
            return dict(claim.result or {}), None
        try:
            result = self._composio.execute_sink(
                tenant_id=str(workflow["customer_id"]),
                toolkit=toolkit,
                connected_account_id=connected_account_id,
                tool_slug=tool_slug,
                arguments=arguments_result,
            )
        except Exception as exc:
            logger.error(
                "intake Composio sink execution raised",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "workflow_id": str(workflow.get("workflow_id") or ""),
                    "booking_id": str(booking.get("booking_id") or ""),
                },
            )
            return {}, _PUBLIC_INDETERMINATE_SINK_ERROR
        if not bool(result.get("successful", False)):
            logger.error(
                "intake Composio sink provider rejected the request: %r",
                result.get("error"),
                extra={
                    "workflow_id": str(workflow.get("workflow_id") or ""),
                    "booking_id": str(booking.get("booking_id") or ""),
                },
            )
            return {}, _PUBLIC_INDETERMINATE_SINK_ERROR
        sink_result = {
            "sink_type": str(workflow["sink_type"]),
            "toolkit": toolkit,
            "tool_slug": tool_slug,
            "booking_id": str(booking["booking_id"]),
            "data": result.get("data"),
        }
        try:
            self._idempotency.complete(
                tenant_id=str(workflow["customer_id"]),
                idempotency_key=idempotency_key,
                result=sink_result,
            )
        except Exception as exc:
            logger.error(
                "intake sink result could not be committed: error_type=%s",
                type(exc).__name__,
                extra={
                    "workflow_id": str(workflow.get("workflow_id") or ""),
                    "booking_id": str(booking.get("booking_id") or ""),
                },
            )
            return {}, _PUBLIC_INDETERMINATE_SINK_ERROR
        return sink_result, None

    @staticmethod
    def effect_idempotency_key(*, booking_id: str, phase: str, revision: int) -> str:
        safe_phase = str(phase or "").strip().casefold()
        if not booking_id or not safe_phase or revision <= 0:
            raise ValueError("sink effect identity is incomplete")
        return f"intake-sink:{booking_id}:{safe_phase}:r{revision}"

    def _build_composio_arguments(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None,
    ) -> dict[str, Any] | str:
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        if sink_type == "google_sheets_composio":
            static_result = self._google_sheets_static_arguments(
                customer_id=str(workflow["customer_id"]),
                sink_config=sink_config,
                static_arguments=static_arguments,
            )
            if isinstance(static_result, str):
                return static_result
            static_arguments = static_result
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
            return arguments
        arguments = dict(static_arguments)
        for target_key, source_key in field_mapping.items():
            arguments[target_key] = enriched_payload.get(source_key)
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
            logger.error(
                "intake Google Sheets sink validation failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"customer_id": customer_id},
            )
            return _PUBLIC_SINK_ERROR

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
            base_row[_neutralize_csv_cell(str(key))] = str(value or "")
        return {key: _neutralize_csv_cell(value) for key, value in base_row.items()}

    @staticmethod
    def _read_existing_csv(
        directory_fd: int,
        filename: str,
        base_row: dict[str, str],
    ) -> tuple[list[dict[str, str]], list[str]]:
        rows: list[dict[str, str]] = []
        fieldnames = list(base_row.keys())
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return rows, fieldnames
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > _MAX_LOCAL_CSV_BYTES
            ):
                raise ValueError("local CSV sink target is invalid")
            with os.fdopen(descriptor, "rb", closefd=False) as raw:
                content = raw.read(_MAX_LOCAL_CSV_BYTES + 1)
            if len(content) > _MAX_LOCAL_CSV_BYTES:
                raise ValueError("local CSV sink exceeds its size limit")
            reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
            for item in reader:
                rows.append(
                    {
                        _neutralize_csv_cell(str(key)): _neutralize_csv_cell(str(value or ""))
                        for key, value in item.items()
                    }
                )
            for field in list(reader.fieldnames or []):
                safe_field = _neutralize_csv_cell(field)
                if safe_field not in fieldnames:
                    fieldnames.append(safe_field)
        finally:
            os.close(descriptor)
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
        directory_fd: int,
        filename: str,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        safe_fieldnames = [_neutralize_csv_cell(field) for field in fieldnames]
        temporary_name = f".{filename}.{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=safe_fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            safe_field: _neutralize_csv_cell(row.get(field, ""))
                            for field, safe_field in zip(
                                fieldnames,
                                safe_fieldnames,
                                strict=True,
                            )
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            metadata = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
            if metadata.st_size > _MAX_LOCAL_CSV_BYTES:
                raise ValueError("local CSV sink exceeds its size limit")
            try:
                existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and (
                not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
            ):
                raise ValueError("local CSV sink target is invalid")
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)

    @contextmanager
    def _open_csv_directory(
        self,
        *,
        customer_id: str,
        directory_parts: tuple[str, ...],
    ) -> Iterator[int]:
        descriptor = os.dup(self._sink_root_fd)
        try:
            for part in (tenant_namespace_label(customer_id), *directory_parts):
                if not part or part in {".", ".."} or "/" in part:
                    raise ValueError("local CSV directory is invalid")
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                os.fchmod(next_descriptor, 0o700)
                os.close(descriptor)
                descriptor = next_descriptor
            yield descriptor
        finally:
            os.close(descriptor)

    def _csv_lock(self, path: str) -> threading.Lock:
        with self._csv_locks_guard:
            return self._csv_locks.setdefault(path, threading.Lock())


def _neutralize_csv_cell(value: Any) -> str:
    text = str(value or "")
    stripped = text.lstrip()
    if (text and text[0] in {"\t", "\r"}) or (
        stripped and stripped[0] in _FORMULA_PREFIXES
    ):
        return f"'{text}"
    return text
