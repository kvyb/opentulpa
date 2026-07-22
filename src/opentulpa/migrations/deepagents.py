"""Idempotent product-data migration for the Deep Agents big-bang cutover."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from langgraph.store.sqlite import SqliteStore

from opentulpa.intake.drafts.service import IntakeDraftService
from opentulpa.intake.drafts.store import IntakeDraftStore
from opentulpa.migrations.memory_sources import JsonMemorySource, QdrantMem0Source
from opentulpa.migrations.models import (
    ComponentMigrationReport,
    DeepAgentsMigrationConfig,
    DeepAgentsMigrationReport,
    LegacyMemoryRecord,
    LegacyMemorySource,
    MigrationIssue,
    PreservedDatasetName,
    PreservedDatasetReport,
    PreservedProductDataReport,
)
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.persistence.tenant_namespace import tenant_store_namespace
from opentulpa.schedules.service import ScheduleService
from opentulpa.specs import (
    AgentSpecService,
    AgentSpecStore,
    TriggerSpecService,
    TriggerSpecStore,
)
from opentulpa.specs.defaults import DEFAULT_ROUTINE_SPEC_ID

_EMPTY_CHECKSUM = hashlib.sha256(b"[]").hexdigest()
_MEMORY_INDEX_START = "<!-- opentulpa-mem0-migration:start -->"
_MEMORY_INDEX_END = "<!-- opentulpa-mem0-migration:end -->"
_GENERATED_WORKFLOW_SOURCES = {
    "intake_workflow",
    "intake-workflow",
    "workflow_generated",
    "workflow-generated",
}


@dataclass(frozen=True, slots=True)
class _PreservedTableSpec:
    name: str
    required_columns: tuple[str, ...]
    count_records: bool = False
    where: str | None = None


@dataclass(frozen=True, slots=True)
class _PreservedDatasetSpec:
    name: PreservedDatasetName
    database_path: Path
    tables: tuple[_PreservedTableSpec, ...]
    file_vault_root: Path | None = None


def _checksum(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _readonly_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_rows(
    path: Path | None,
    *,
    table: str,
    columns: tuple[str, ...],
    order_by: str,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with closing(_readonly_connection(path)) as conn:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if present is None:
            return []
        available = {
            str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        missing = set(columns) - available
        if missing:
            raise ValueError(f"legacy {table} table is missing columns: {sorted(missing)}")
        selected = ", ".join(columns)
        rows = conn.execute(f"SELECT {selected} FROM {table} ORDER BY {order_by}").fetchall()
    return [dict(row) for row in rows]


def _parse_timestamp(value: object, *, field: str, required: bool = True) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise ValueError(f"{field} is required")
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_legacy_{digest}"


def _safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    if safe == value and len(safe) <= 80:
        return safe
    safe = safe[:60].rstrip(".-") or "record"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"


def _skill_frontmatter_name(markdown: str) -> str:
    if not markdown.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = markdown.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter terminator not found")
    fields: dict[str, str] = {}
    for line in markdown[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        raw_value = value.strip()
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
            raw_value = raw_value[1:-1].strip()
        fields[key.strip().casefold()] = raw_value
    name = fields.get("name", "")
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", name) is None:
        raise ValueError("SKILL.md frontmatter requires a valid lowercase slug name")
    if not fields.get("description", "").strip():
        raise ValueError("SKILL.md frontmatter requires a non-empty description")
    return name


def _component_report(
    *,
    scanned: int,
    eligible: int,
    migrated: int,
    skipped: int,
    invalid: int,
    disabled: int,
    source_checksum: str,
    issues: list[MigrationIssue],
) -> ComponentMigrationReport:
    return ComponentMigrationReport(
        scanned=scanned,
        eligible=eligible,
        migrated=migrated,
        skipped=skipped,
        invalid=invalid,
        disabled=disabled,
        source_checksum=source_checksum,
        issues=issues,
    )


def _empty_component_report() -> ComponentMigrationReport:
    return _component_report(
        scanned=0,
        eligible=0,
        migrated=0,
        skipped=0,
        invalid=0,
        disabled=0,
        source_checksum=_EMPTY_CHECKSUM,
        issues=[],
    )


def _quoted_identifier(value: str) -> str:
    if re.fullmatch(r"[a-z_][a-z0-9_]*", value) is None:
        raise ValueError(f"invalid SQLite identifier: {value!r}")
    return f'"{value}"'


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def _canonical_table(
    conn: sqlite3.Connection,
    spec: _PreservedTableSpec,
    *,
    visit_row: Callable[[dict[str, Any]], None] | None = None,
    canonicalize_row: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    table = _quoted_identifier(spec.name)
    present = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (spec.name,),
    ).fetchone()
    if present is None:
        raise ValueError(f"required product table is missing: {spec.name}")
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    columns = [str(row["name"]) for row in info]
    missing = sorted(set(spec.required_columns) - set(columns))
    if missing:
        raise ValueError(f"product table {spec.name} is missing columns: {missing}")
    order_columns = [
        str(row["name"])
        for row in sorted(info, key=lambda item: int(item["pk"]) or len(info) + 1)
        if int(row["pk"]) > 0
    ]
    if not order_columns:
        order_columns = columns
    selected = ", ".join(_quoted_identifier(column) for column in columns)
    ordered = ", ".join(_quoted_identifier(column) for column in order_columns)
    where = f" WHERE {spec.where}" if spec.where else ""
    cursor = conn.execute(
        f"SELECT {selected} FROM {table}{where} ORDER BY {ordered}"
    )
    row_count = 0
    digest = hashlib.sha256()
    for row in cursor:
        source_row = {column: _sqlite_value(row[column]) for column in columns}
        canonical_row = (
            canonicalize_row(source_row) if canonicalize_row is not None else source_row
        )
        if set(canonical_row) != set(columns):
            raise ValueError(f"canonicalizer changed columns for product table {spec.name}")
        encoded = json.dumps(
            canonical_row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        row_count += 1
        if visit_row is not None:
            visit_row(canonical_row)
    return {
        "table": spec.name,
        "columns": columns,
        "row_count": row_count,
        "source_checksum": digest.hexdigest(),
    }


def _sqlite_snapshot(source: Path, destination: Path) -> None:
    """Copy one live SQLite view, including committed WAL pages, into a disposable DB."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(_readonly_connection(source)) as source_conn, closing(
        sqlite3.connect(destination)
    ) as destination_conn:
        source_conn.backup(destination_conn)


class DeepAgentsMigrator:
    """Translate legacy orchestration data while preserving product-owned tables."""

    def __init__(
        self,
        config: DeepAgentsMigrationConfig,
        *,
        memory_source: LegacyMemorySource | None = None,
    ) -> None:
        self._config = config
        self._memory_source = memory_source
        self._native_store: SqliteStore | None = None

    def run(self, *, dry_run: bool = False) -> DeepAgentsMigrationReport:
        preserved_data = self._verify_preserved_product_data()
        if not preserved_data.verified:
            empty = _empty_component_report()
            return DeepAgentsMigrationReport(
                dry_run=dry_run,
                status="blocked",
                preserved_data=preserved_data,
                routines=empty,
                drafts=empty,
                memories=empty,
                skills=empty,
                combined_checksum=_checksum(
                    {
                        "status": "blocked",
                        "preserved_data": preserved_data.combined_checksum,
                    }
                ),
            )
        pending_file_paths = self._file_paths_requiring_rebase()
        rebased_file_paths = 0
        if not dry_run and pending_file_paths:
            rebased_file_paths = self._rebase_uploaded_file_paths()
            verified_after_rebase = self._verify_preserved_product_data()
            if (
                not verified_after_rebase.verified
                or verified_after_rebase.combined_checksum != preserved_data.combined_checksum
            ):
                raise RuntimeError("uploaded file path rebasing changed preserved product data")
            preserved_data = verified_after_rebase
        routines = self._migrate_routines(dry_run=dry_run)
        drafts = self._migrate_drafts(dry_run=dry_run)
        if dry_run:
            memories = self._migrate_memories(dry_run=True)
            skills = self._migrate_skills(dry_run=True)
        else:
            store_path = self._config.store_db_path.expanduser().resolve()
            store_path.parent.mkdir(parents=True, exist_ok=True)
            with SqliteStore.from_conn_string(str(store_path)) as store:
                store.setup()
                self._native_store = store
                try:
                    memories = self._migrate_memories(dry_run=False)
                    skills = self._migrate_skills(dry_run=False)
                finally:
                    self._native_store = None
        status: Literal["completed", "blocked"] = (
            "blocked"
            if any(
                issue.disposition == "conflict"
                for component in (routines, drafts, memories, skills)
                for issue in component.issues
            )
            else "completed"
        )
        combined_checksum = _checksum(
            {
                "status": status,
                "routines": routines.source_checksum,
                "drafts": drafts.source_checksum,
                "memories": memories.source_checksum,
                "preserved_data": preserved_data.combined_checksum,
                "skills": skills.source_checksum,
            }
        )
        return DeepAgentsMigrationReport(
            dry_run=dry_run,
            status=status,
            preserved_data=preserved_data,
            routines=routines,
            drafts=drafts,
            memories=memories,
            skills=skills,
            combined_checksum=combined_checksum,
            file_paths_pending_rebase=pending_file_paths,
            file_paths_rebased=rebased_file_paths,
        )

    def _verify_preserved_product_data(self) -> PreservedProductDataReport:
        reports = [self._verify_preserved_dataset(spec) for spec in self._preserved_specs()]
        accepted_statuses = (
            {"ok", "missing"}
            if self._config.allow_missing_preserved_data
            else {"ok"}
        )
        verified = all(report.status in accepted_statuses for report in reports)
        combined_checksum = _checksum(
            [
                {
                    "dataset": report.dataset,
                    "status": report.status,
                    "record_count": report.record_count,
                    "table_counts": report.table_counts,
                    "source_checksum": report.source_checksum,
                }
                for report in reports
            ]
        )
        return PreservedProductDataReport(
            verified=verified,
            datasets=reports,
            combined_checksum=combined_checksum,
        )

    def _preserved_specs(self) -> tuple[_PreservedDatasetSpec, ...]:
        config = self._config
        return (
            _PreservedDatasetSpec(
                name="profiles",
                database_path=config.customer_profiles_db_path,
                tables=(
                    _PreservedTableSpec(
                        name="customer_profiles",
                        required_columns=(
                            "customer_id",
                            "directive_text",
                            "utc_offset",
                            "locale",
                            "source",
                            "updated_at",
                        ),
                        count_records=True,
                    ),
                    _PreservedTableSpec(
                        name="customer_identity_aliases",
                        required_columns=(
                            "alias_user_id",
                            "user_id",
                            "storage_user_id",
                            "alias_kind",
                            "provider",
                            "provider_user_id",
                            "created_at",
                            "updated_at",
                        ),
                    ),
                ),
            ),
            _PreservedDatasetSpec(
                name="files",
                database_path=config.file_vault_db_path,
                file_vault_root=config.file_vault_root_path,
                tables=(
                    _PreservedTableSpec(
                        name="uploaded_files",
                        required_columns=(
                            "id",
                            "customer_id",
                            "stored_path",
                            "size_bytes",
                        ),
                        count_records=True,
                    ),
                ),
            ),
            _PreservedDatasetSpec(
                name="knowledge",
                database_path=config.knowledge_db_path,
                tables=(
                    _PreservedTableSpec(
                        name="knowledge_sources",
                        required_columns=(
                            "customer_id",
                            "scope_type",
                            "scope_id",
                            "file_id",
                            "source_hash",
                            "status",
                        ),
                        count_records=True,
                    ),
                    _PreservedTableSpec(
                        name="knowledge_sections",
                        required_columns=(
                            "section_id",
                            "customer_id",
                            "scope_type",
                            "scope_id",
                            "file_id",
                            "content",
                        ),
                    ),
                    _PreservedTableSpec(
                        name="knowledge_preflight_cache",
                        required_columns=(
                            "customer_id",
                            "cache_key",
                            "source_signature",
                            "result_json",
                            "updated_at",
                        ),
                    ),
                ),
            ),
            _PreservedDatasetSpec(
                name="active_workflows",
                database_path=config.intake_workflows_db_path,
                tables=(
                    _PreservedTableSpec(
                        name="intake_workflows",
                        required_columns=(
                            "workflow_id",
                            "customer_id",
                            "name",
                            "channel",
                            "provider",
                            "source_config_json",
                            "intent_description",
                            "required_fields_json",
                            "field_guidance_json",
                            "sink_type",
                            "sink_config_json",
                            "schedule",
                            "notify_user",
                            "enabled",
                            "routine_id",
                            "created_at",
                            "updated_at",
                        ),
                        count_records=True,
                    ),
                    _PreservedTableSpec(
                        name="intake_conversation_cursors",
                        required_columns=(
                            "workflow_id",
                            "conversation_id",
                            "last_seen_inbound_message_id",
                            "updated_at",
                        ),
                    ),
                    _PreservedTableSpec(
                        name="intake_pending_runs",
                        required_columns=(
                            "workflow_id",
                            "conversation_id",
                            "customer_id",
                            "generation",
                            "status",
                            "due_at",
                            "updated_at",
                        ),
                    ),
                ),
            ),
            _PreservedDatasetSpec(
                name="bookings",
                database_path=config.intake_workflows_db_path,
                tables=(
                    _PreservedTableSpec(
                        name="intake_bookings",
                        required_columns=(
                            "booking_id",
                            "workflow_id",
                            "customer_id",
                            "conversation_id",
                            "status",
                            "extracted_fields_json",
                            "sink_write_status",
                        ),
                        count_records=True,
                    ),
                ),
            ),
            _PreservedDatasetSpec(
                name="integration_connections",
                database_path=config.integration_connections_db_path,
                tables=(
                    _PreservedTableSpec(
                        name="telegram_business_connections",
                        required_columns=(
                            "business_connection_id",
                            "customer_id",
                            "user_id",
                            "user_chat_id",
                            "is_enabled",
                            "connection_json",
                        ),
                        count_records=True,
                    ),
                    _PreservedTableSpec(
                        name="telegram_business_messages",
                        required_columns=(
                            "business_connection_id",
                            "customer_id",
                            "chat_id",
                            "message_id",
                            "sender_role",
                            "raw_json",
                            "updated_at",
                        ),
                    ),
                ),
            ),
        )

    def _verify_preserved_dataset(
        self,
        spec: _PreservedDatasetSpec,
    ) -> PreservedDatasetReport:
        database_path = spec.database_path.expanduser().resolve()
        if not database_path.exists():
            return PreservedDatasetReport(
                dataset=spec.name,
                database_path=str(database_path),
                status="missing",
                record_count=0,
                table_counts={},
                source_checksum=_EMPTY_CHECKSUM,
            )
        try:
            file_rows: list[dict[str, Any]] = []
            with closing(_readonly_connection(database_path)) as conn:
                tables: list[dict[str, Any]] = []
                for table in spec.tables:
                    is_uploaded_files = table.name == "uploaded_files"
                    canonicalizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None
                    if is_uploaded_files:
                        if spec.file_vault_root is None:
                            raise ValueError("uploaded file verification requires a vault root")
                        root = spec.file_vault_root

                        def canonicalize_uploaded_file(
                            row: dict[str, Any],
                            *,
                            root: Path = root,
                        ) -> dict[str, Any]:
                            return self._canonical_file_row(row, root)

                        canonicalizer = canonicalize_uploaded_file
                    tables.append(
                        _canonical_table(
                            conn,
                            table,
                            visit_row=file_rows.append if is_uploaded_files else None,
                            canonicalize_row=canonicalizer,
                        )
                    )
            table_counts = {
                str(table["table"]): int(table["row_count"])
                for table in tables
            }
            record_count = sum(
                table_counts[table.name] for table in spec.tables if table.count_records
            )
            file_fingerprints = (
                self._file_fingerprints(file_rows, spec.file_vault_root)
                if spec.file_vault_root is not None
                else []
            )
            return PreservedDatasetReport(
                dataset=spec.name,
                database_path=str(database_path),
                status="ok",
                record_count=record_count,
                table_counts=table_counts,
                source_checksum=_checksum(
                    {
                        "dataset": spec.name,
                        "tables": tables,
                        "files": file_fingerprints,
                    }
                ),
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            status: Literal["invalid", "unreadable"] = "unreadable"
            message = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status = "invalid"
            message = f"{type(exc).__name__}: {exc}"
        return PreservedDatasetReport(
            dataset=spec.name,
            database_path=str(database_path),
            status=status,
            record_count=0,
            table_counts={},
            source_checksum=_EMPTY_CHECKSUM,
            message=message,
        )

    @staticmethod
    def _canonical_file_row(row: dict[str, Any], file_vault_root: Path) -> dict[str, Any]:
        normalized = dict(row)
        target = DeepAgentsMigrator._project_file_path(row, file_vault_root)
        root = file_vault_root.expanduser().resolve()
        normalized["stored_path"] = target.relative_to(root).as_posix()
        return normalized

    @staticmethod
    def _project_file_path(row: dict[str, Any], file_vault_root: Path) -> Path:
        file_id = str(row.get("id") or "").strip()
        tenant_id = str(row.get("customer_id") or "").strip()
        stored_path = Path(str(row.get("stored_path") or ""))
        if not file_id or not tenant_id or not stored_path.name:
            raise ValueError("uploaded file metadata is missing identity or stored_path")
        root = file_vault_root.expanduser().resolve()
        tenant_segment = re.sub(r"[^A-Za-z0-9._-]+", "_", tenant_id)
        candidate = (root / tenant_segment / stored_path.name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("uploaded file path escapes the configured vault root") from exc
        return candidate

    @staticmethod
    def _file_fingerprints(
        rows: list[dict[str, Any]],
        file_vault_root: Path,
    ) -> list[dict[str, Any]]:
        root = file_vault_root.expanduser().resolve()
        fingerprints: list[dict[str, Any]] = []
        for row in rows:
            file_id = str(row.get("id") or "").strip()
            tenant_id = str(row.get("customer_id") or "").strip()
            stored_path = Path(str(row.get("stored_path") or ""))
            if not file_id or not tenant_id or not stored_path.name:
                raise ValueError("uploaded file metadata is missing identity or stored_path")
            tenant_segment = re.sub(r"[^A-Za-z0-9._-]+", "_", tenant_id)
            candidate = root / tenant_segment / stored_path.name
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError("uploaded file path escapes the configured vault root") from exc
            if candidate.is_symlink() or not resolved.is_file():
                raise ValueError(f"uploaded file bytes are not a regular file: {file_id}")
            size = resolved.stat().st_size
            raw_expected_size = row.get("size_bytes")
            if not isinstance(raw_expected_size, int | str):
                raise ValueError(f"uploaded file size is invalid: {file_id}")
            expected_size = int(raw_expected_size)
            if size != expected_size:
                raise ValueError(f"uploaded file size mismatch: {file_id}")
            digest = hashlib.sha256()
            with resolved.open("rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprints.append(
                {
                    "file_id": file_id,
                    "tenant_id": tenant_id,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
        return sorted(fingerprints, key=lambda item: (item["tenant_id"], item["file_id"]))

    def _file_paths_requiring_rebase(self) -> int:
        path = self._config.file_vault_db_path.expanduser().resolve()
        if not path.exists():
            return 0
        with closing(_readonly_connection(path)) as conn:
            rows = conn.execute(
                """
                SELECT id, customer_id, stored_path, size_bytes
                FROM uploaded_files ORDER BY customer_id, id
                """
            ).fetchall()
        return sum(
            str(row["stored_path"])
            != str(
                self._project_file_path(
                    dict(row),
                    self._config.file_vault_root_path,
                )
            )
            for row in rows
        )

    def _rebase_uploaded_file_paths(self) -> int:
        path = self._config.file_vault_db_path.expanduser().resolve()
        if not path.exists():
            return 0
        changed = 0
        with closing(connect_sqlite(path, wal=True)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id, customer_id, stored_path, size_bytes
                FROM uploaded_files ORDER BY customer_id, id
                """
            ).fetchall()
            for row in rows:
                source = dict(row)
                target = self._project_file_path(
                    source,
                    self._config.file_vault_root_path,
                )
                resolved = target.resolve(strict=True)
                if target.is_symlink() or not resolved.is_file():
                    raise ValueError(f"uploaded file bytes are not a regular file: {row['id']}")
                if resolved.stat().st_size != int(row["size_bytes"]):
                    raise ValueError(f"uploaded file size mismatch: {row['id']}")
                if str(row["stored_path"]) == str(resolved):
                    continue
                cursor = conn.execute(
                    """
                    UPDATE uploaded_files SET stored_path=?
                    WHERE id=? AND customer_id=? AND stored_path=?
                    """,
                    (
                        str(resolved),
                        str(row["id"]),
                        str(row["customer_id"]),
                        str(row["stored_path"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("uploaded file metadata changed during migration")
                changed += 1
            conn.commit()
        return changed

    def _migrate_routines(self, *, dry_run: bool) -> ComponentMigrationReport:
        source_path = self._config.legacy_routines_db_path
        if source_path is None or not source_path.exists():
            return _component_report(
                scanned=0,
                eligible=0,
                migrated=0,
                skipped=0,
                invalid=0,
                disabled=0,
                source_checksum=_EMPTY_CHECKSUM,
                issues=[],
            )
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="opentulpa-schedule-dry-run-") as temp_dir:
                agent_specs_path = Path(temp_dir) / "agent_specs.db"
                trigger_specs_path = Path(temp_dir) / "trigger_specs.db"
                _sqlite_snapshot(self._config.agent_specs_db_path, agent_specs_path)
                _sqlite_snapshot(self._config.trigger_specs_db_path, trigger_specs_path)
                service = self._schedule_service(
                    agent_specs_path=agent_specs_path,
                    trigger_specs_path=trigger_specs_path,
                )
                legacy = service.migrate_legacy_routines(
                    legacy_db_path=source_path,
                    default_timezone=self._config.default_timezone,
                    dry_run=True,
                )
        else:
            service = self._schedule_service(
                agent_specs_path=self._config.agent_specs_db_path,
                trigger_specs_path=self._config.trigger_specs_db_path,
            )
            legacy = service.migrate_legacy_routines(
                legacy_db_path=source_path,
                default_timezone=self._config.default_timezone,
                dry_run=False,
            )

        invalid_ids = {
            issue.routine_id for issue in legacy.issues if issue.disposition == "invalid"
        }
        disabled_ids = self._disable_routines(invalid_ids) if not dry_run else set()
        issues = [
            MigrationIssue(
                component="routines",
                legacy_id=issue.routine_id,
                disposition=issue.disposition,
                message=issue.error,
                disabled=issue.routine_id in disabled_ids,
            )
            for issue in legacy.issues
        ]
        return _component_report(
            scanned=legacy.scanned,
            eligible=legacy.eligible,
            migrated=legacy.migrated,
            skipped=legacy.skipped,
            invalid=legacy.invalid,
            disabled=len(disabled_ids),
            source_checksum=legacy.source_checksum,
            issues=issues,
        )

    @staticmethod
    def _schedule_service(
        *,
        agent_specs_path: Path,
        trigger_specs_path: Path,
    ) -> ScheduleService:
        agent_store = AgentSpecStore(agent_specs_path)
        trigger_store = TriggerSpecStore(trigger_specs_path, agent_specs=agent_store)
        agent_service = AgentSpecService(agent_store)

        def resolve_routine(tenant_id: str):
            active = agent_service.get_active(
                tenant_id=tenant_id,
                spec_id=DEFAULT_ROUTINE_SPEC_ID,
            )
            if active is None:
                seeded = agent_service.seed_defaults(
                    tenant_id=tenant_id,
                    actor_id="migration:legacy-routines",
                )
                active = next(
                    spec for spec in seeded if spec.id == DEFAULT_ROUTINE_SPEC_ID
                )
            return active.ref

        return ScheduleService(
            TriggerSpecService(trigger_store),
            resolve_agent_spec=resolve_routine,
        )

    def _disable_routines(self, routine_ids: set[str]) -> set[str]:
        path = self._config.legacy_routines_db_path
        if path is None or not routine_ids:
            return set()
        disabled: set[str] = set()
        with closing(connect_sqlite(path, wal=True)) as conn:
            for routine_id in sorted(routine_ids):
                cursor = conn.execute(
                    "UPDATE routines SET enabled=0 WHERE id=? AND enabled != 0",
                    (routine_id,),
                )
                if cursor.rowcount == 1:
                    disabled.add(routine_id)
            conn.commit()
        return disabled

    def _migrate_drafts(self, *, dry_run: bool) -> ComponentMigrationReport:
        columns = (
            "session_id",
            "customer_id",
            "thread_id",
            "status",
            "mode",
            "target_workflow_id",
            "target_workflow_snapshot_json",
            "draft_upsert_json",
            "scratchpad_json",
            "last_proposed_draft_hash",
            "confirmed_draft_hash",
            "created_or_updated_workflow_id",
            "created_at",
            "updated_at",
            "completed_at",
        )
        rows = _table_rows(
            self._config.legacy_setup_db_path,
            table="intake_workflow_setup_sessions",
            columns=columns,
            order_by="created_at ASC, session_id ASC",
        )
        source_checksum = _checksum(rows)
        scanned = len(rows)
        eligible = migrated = skipped = invalid = disabled = 0
        issues: list[MigrationIssue] = []
        draft_store = IntakeDraftStore(self._config.intake_drafts_db_path) if not dry_run else None

        for row in rows:
            session_id = str(row.get("session_id") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            if status in {"completed", "cancelled"}:
                skipped += 1
                continue
            try:
                tenant_id, workflow_id, draft_id, payload, updated_at = self._draft_source(row)
                existing = self._existing_draft(tenant_id=tenant_id, draft_id=draft_id)
                if existing is not None:
                    if existing["workflow_id"] == workflow_id and existing["payload"] == payload:
                        skipped += 1
                        continue
                    raise _MigrationConflictError("destination draft exists with different content")
                eligible += 1
                if dry_run:
                    continue
                assert draft_store is not None
                service = IntakeDraftService(
                    draft_store,
                    workflow_activator=_UnexpectedActivation(),
                    clock=_fixed_clock(updated_at),
                )
                service.save(
                    tenant_id=tenant_id,
                    actor_id="migration",
                    draft_id=draft_id,
                    workflow_id=workflow_id,
                    patch=payload,
                )
                migrated += 1
            except Exception as exc:
                disposition: Literal["invalid", "conflict"] = (
                    "conflict" if isinstance(exc, _MigrationConflictError) else "invalid"
                )
                was_disabled = False
                if not dry_run and disposition == "invalid":
                    was_disabled = self._disable_setup_session(session_id)
                    disabled += int(was_disabled)
                invalid += 1
                issues.append(
                    MigrationIssue(
                        component="drafts",
                        legacy_id=session_id,
                        disposition=disposition,
                        message=str(exc) or type(exc).__name__,
                        disabled=was_disabled,
                    )
                )

        return _component_report(
            scanned=scanned,
            eligible=eligible,
            migrated=migrated,
            skipped=skipped,
            invalid=invalid,
            disabled=disabled,
            source_checksum=source_checksum,
            issues=issues,
        )

    @staticmethod
    def _draft_source(
        row: dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any], datetime]:
        session_id = str(row.get("session_id") or "").strip()
        tenant_id = str(row.get("customer_id") or "").strip()
        thread_id = str(row.get("thread_id") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        mode = str(row.get("mode") or "").strip().lower()
        target_workflow_id = str(row.get("target_workflow_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        if not tenant_id or len(tenant_id) > 200:
            raise ValueError("customer_id must contain at most 200 characters")
        if not thread_id:
            raise ValueError("thread_id is required")
        if status not in {"active", "paused"}:
            raise ValueError("live setup status must be active or paused")
        if mode not in {"create", "edit"}:
            raise ValueError("setup mode must be create or edit")
        if mode == "edit" and not target_workflow_id:
            raise ValueError("edit setup is missing target_workflow_id")
        workflow_id = target_workflow_id or _stable_id("iwf", session_id)
        if len(workflow_id) > 100:
            raise ValueError("target_workflow_id must be at most 100 characters")
        try:
            payload = json.loads(str(row.get("draft_upsert_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("draft_upsert_json is invalid") from exc
        if not isinstance(payload, dict) or not payload:
            raise ValueError("draft_upsert_json must contain a non-empty object")
        payload = cast(dict[str, Any], IntakeDraftService._validated_patch(payload))
        _parse_timestamp(row.get("created_at"), field="created_at")
        updated_at = _parse_timestamp(row.get("updated_at"), field="updated_at")
        assert updated_at is not None
        return tenant_id, workflow_id, _stable_id("idft", session_id), payload, updated_at

    def _existing_draft(self, *, tenant_id: str, draft_id: str) -> dict[str, Any] | None:
        path = self._config.intake_drafts_db_path
        if not path.exists():
            return None
        with closing(_readonly_connection(path)) as conn:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='intake_drafts'"
            ).fetchone()
            if table is None:
                return None
            row = conn.execute(
                "SELECT workflow_id, payload_json FROM intake_drafts WHERE tenant_id=? AND id=?",
                (tenant_id, draft_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "workflow_id": str(row["workflow_id"]),
            "payload": json.loads(str(row["payload_json"])),
        }

    def _disable_setup_session(self, session_id: str) -> bool:
        path = self._config.legacy_setup_db_path
        if path is None:
            return False
        with closing(connect_sqlite(path, wal=True)) as conn:
            cursor = conn.execute(
                """
                UPDATE intake_workflow_setup_sessions
                SET status='cancelled'
                WHERE session_id=? AND status != 'cancelled'
                """,
                (session_id,),
            )
            conn.commit()
        return cursor.rowcount == 1

    def _migrate_memories(self, *, dry_run: bool) -> ComponentMigrationReport:
        records = self._memory_source.records() if self._memory_source is not None else []
        canonical_records = sorted(
            (record.model_dump(mode="json") for record in records),
            key=lambda item: (str(item["tenant_id"]), str(item["legacy_id"])),
        )
        source_checksum = _checksum(canonical_records)
        scanned = len(records)
        eligible = migrated = skipped = invalid = disabled = 0
        issues: list[MigrationIssue] = []
        index_entries: dict[str, list[tuple[LegacyMemoryRecord, str]]] = defaultdict(list)
        seen: dict[tuple[str, str], str] = {}

        for record in records:
            try:
                key, rendered = self._memory_file(record)
                identity = (record.tenant_id.strip(), key)
                prior = seen.get(identity)
                if prior is not None:
                    if prior == rendered:
                        skipped += 1
                        continue
                    raise _MigrationConflictError(
                        "multiple memory records map to the same native file"
                    )
                seen[identity] = rendered
                existing = self._native_value(
                    namespace=tenant_store_namespace(identity[0], "memory"),
                    key=key,
                )
                if existing is not None:
                    if existing.get("content") == rendered:
                        skipped += 1
                        index_entries[identity[0]].append((record, key))
                        continue
                    raise _MigrationConflictError(
                        "destination memory file exists with different content"
                    )
                eligible += 1
                index_entries[identity[0]].append((record, key))
                if dry_run:
                    continue
                self._put_native(
                    namespace=tenant_store_namespace(identity[0], "memory"),
                    key=key,
                    content=rendered,
                    created_at=record.created_at,
                    modified_at=record.updated_at,
                )
                migrated += 1
            except Exception as exc:
                disposition: Literal["invalid", "conflict"] = (
                    "conflict" if isinstance(exc, _MigrationConflictError) else "invalid"
                )
                was_disabled = False
                if (
                    not dry_run
                    and disposition == "invalid"
                    and self._memory_source is not None
                ):
                    try:
                        was_disabled = self._memory_source.disable(record, reason=str(exc))
                    except Exception:
                        was_disabled = False
                disabled += int(was_disabled)
                invalid += 1
                issues.append(
                    MigrationIssue(
                        component="memories",
                        legacy_id=record.legacy_id,
                        disposition=disposition,
                        message=str(exc) or type(exc).__name__,
                        disabled=was_disabled,
                    )
                )

        if not dry_run:
            for tenant_id, entries in sorted(index_entries.items()):
                self._write_memory_index(tenant_id=tenant_id, entries=entries)
        return _component_report(
            scanned=scanned,
            eligible=eligible,
            migrated=migrated,
            skipped=skipped,
            invalid=invalid,
            disabled=disabled,
            source_checksum=source_checksum,
            issues=issues,
        )

    @staticmethod
    def _memory_file(record: LegacyMemoryRecord) -> tuple[str, str]:
        legacy_id = record.legacy_id.strip()
        tenant_id = record.tenant_id.strip()
        content = record.content
        if not legacy_id:
            raise ValueError("memory id is required")
        if not tenant_id or len(tenant_id) > 200:
            raise ValueError("memory tenant id must contain at most 200 characters")
        if not content.strip():
            raise ValueError("memory content is required")
        if record.created_at:
            _parse_timestamp(record.created_at, field="created_at")
        if record.updated_at:
            _parse_timestamp(record.updated_at, field="updated_at")
        key = f"/{_safe_file_stem(legacy_id)}.md"
        if key.casefold() == "/agents.md":
            key = f"/mem0-{_safe_file_stem(legacy_id)}.md"
        frontmatter = [
            "---",
            f"legacy_id: {json.dumps(legacy_id, ensure_ascii=False)}",
            "source: mem0",
            f"created_at: {json.dumps(record.created_at, ensure_ascii=False)}",
            f"updated_at: {json.dumps(record.updated_at, ensure_ascii=False)}",
            "---",
            "",
        ]
        suffix = "" if content.endswith("\n") else "\n"
        return key, "\n".join(frontmatter) + content + suffix

    def _write_memory_index(
        self,
        *,
        tenant_id: str,
        entries: list[tuple[LegacyMemoryRecord, str]],
    ) -> None:
        namespace = tenant_store_namespace(tenant_id, "memory")
        key = "/AGENTS.md"
        lines = [
            _MEMORY_INDEX_START,
            "# Migrated Memories",
            "",
            "Legacy Mem0 records preserved during the Deep Agents cutover:",
            "",
        ]
        for record, file_key in sorted(entries, key=lambda item: item[0].legacy_id):
            summary = " ".join(record.content.split())[:120]
            created = f"; created {record.created_at}" if record.created_at else ""
            lines.append(f"- [`{record.legacy_id}`]({Path(file_key).name}){created}: {summary}")
        lines.extend([_MEMORY_INDEX_END, ""])
        generated = "\n".join(lines)
        existing = self._native_value(namespace=namespace, key=key)
        existing_content = str(existing.get("content") or "") if existing else ""
        merged = _merge_memory_index(existing_content, generated)
        if merged == existing_content:
            return
        self._put_native(namespace=namespace, key=key, content=merged)

    def _migrate_skills(self, *, dry_run: bool) -> ComponentMigrationReport:
        columns = (
            "scope",
            "customer_id",
            "name",
            "description",
            "source",
            "enabled",
            "skill_path",
            "created_at",
            "updated_at",
        )
        rows = _table_rows(
            self._config.legacy_skills_db_path,
            table="skills",
            columns=columns,
            order_by="scope ASC, customer_id ASC, name ASC",
        )
        checksum_rows: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            path = self._legacy_skill_path(row)
            try:
                item["skill_markdown"] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                item["skill_markdown_error"] = type(exc).__name__
            checksum_rows.append(item)
        source_checksum = _checksum(checksum_rows)
        scanned = len(rows)
        eligible = migrated = skipped = invalid = disabled = 0
        issues: list[MigrationIssue] = []

        for row in rows:
            scope = str(row.get("scope") or "").strip().lower()
            raw_tenant_id = str(row.get("customer_id") or "")
            raw_name = str(row.get("name") or "")
            tenant_id = raw_tenant_id.strip()
            name = raw_name.strip().lower()
            source = str(row.get("source") or "").strip().lower()
            identity = f"{tenant_id}:{name}".strip(":")
            if scope != "user" or not bool(row.get("enabled")):
                skipped += 1
                continue
            if name.startswith("intake-workflow-") or source in _GENERATED_WORKFLOW_SOURCES:
                skipped += 1
                issues.append(
                    MigrationIssue(
                        component="skills",
                        legacy_id=identity,
                        disposition="skipped",
                        message="generated intake workflow skills are not migrated",
                    )
                )
                continue
            try:
                if not tenant_id or len(tenant_id) > 200:
                    raise ValueError("user skill customer_id must contain at most 200 characters")
                if not name:
                    raise ValueError("skill name is required")
                path = self._legacy_skill_path(row)
                markdown = path.read_text(encoding="utf-8")
                parsed_name = _skill_frontmatter_name(markdown)
                if parsed_name != name:
                    raise ValueError("SKILL.md frontmatter name does not match the legacy row")
                created_at = _parse_timestamp(row.get("created_at"), field="created_at")
                updated_at = _parse_timestamp(row.get("updated_at"), field="updated_at")
                assert created_at is not None and updated_at is not None
                key = f"/{parsed_name}/SKILL.md"
                existing = self._native_value(
                    namespace=tenant_store_namespace(tenant_id, "skills"),
                    key=key,
                )
                if existing is not None:
                    if existing.get("content") == markdown:
                        skipped += 1
                        continue
                    raise _MigrationConflictError("destination skill exists with different content")
                eligible += 1
                if dry_run:
                    continue
                self._put_native(
                    namespace=tenant_store_namespace(tenant_id, "skills"),
                    key=key,
                    content=markdown,
                    created_at=created_at.isoformat(),
                    modified_at=updated_at.isoformat(),
                )
                migrated += 1
            except Exception as exc:
                disposition: Literal["invalid", "conflict"] = (
                    "conflict" if isinstance(exc, _MigrationConflictError) else "invalid"
                )
                was_disabled = False
                if not dry_run and disposition == "invalid":
                    was_disabled = self._disable_skill(
                        tenant_id=raw_tenant_id,
                        name=raw_name,
                    )
                    disabled += int(was_disabled)
                invalid += 1
                issues.append(
                    MigrationIssue(
                        component="skills",
                        legacy_id=identity,
                        disposition=disposition,
                        message=str(exc) or type(exc).__name__,
                        disabled=was_disabled,
                    )
                )

        return _component_report(
            scanned=scanned,
            eligible=eligible,
            migrated=migrated,
            skipped=skipped,
            invalid=invalid,
            disabled=disabled,
            source_checksum=source_checksum,
            issues=issues,
        )

    def _legacy_skill_path(self, row: dict[str, Any]) -> Path:
        root = self._config.legacy_skills_root_path
        if root is None:
            return Path(str(row.get("skill_path") or ""))
        scope = str(row.get("scope") or "").strip().lower()
        name = str(row.get("name") or "").strip().lower()
        if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", name) is None:
            return root.expanduser().resolve() / "__invalid_skill_name__"
        if scope == "global":
            return root.expanduser().resolve() / "global" / name / "SKILL.md"
        customer_id = str(row.get("customer_id") or "").strip()
        customer_segment = re.sub(r"[^a-zA-Z0-9._-]+", "_", customer_id)
        return root.expanduser().resolve() / "users" / customer_segment / name / "SKILL.md"

    def _disable_skill(self, *, tenant_id: str, name: str) -> bool:
        path = self._config.legacy_skills_db_path
        if path is None:
            return False
        with closing(connect_sqlite(path, wal=True)) as conn:
            cursor = conn.execute(
                """
                UPDATE skills SET enabled=0
                WHERE scope='user' AND customer_id=? AND name=? AND enabled != 0
                """,
                (tenant_id, name),
            )
            conn.commit()
        return cursor.rowcount == 1

    def _native_value(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
    ) -> dict[str, Any] | None:
        if self._native_store is not None:
            item = self._native_store.get(namespace, key)
            return dict(item.value) if item is not None else None
        path = self._config.store_db_path
        if not path.exists():
            return None
        with closing(_readonly_connection(path)) as conn:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='store'"
            ).fetchone()
            if table is None:
                return None
            row = conn.execute(
                "SELECT value FROM store WHERE prefix=? AND key=?",
                (".".join(namespace), key),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row["value"]))
        return dict(value) if isinstance(value, dict) else None

    def _put_native(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        content: str,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> None:
        if self._native_store is None:
            raise RuntimeError("native store is not open")
        value: dict[str, Any] = {"content": content, "encoding": "utf-8"}
        if created_at:
            value["created_at"] = created_at
        if modified_at:
            value["modified_at"] = modified_at
        self._native_store.put(namespace, key, value, index=False)


class _MigrationConflictError(RuntimeError):
    pass


class _UnexpectedActivation:
    def activate_draft(self, **_: Any) -> Any:
        raise RuntimeError("migration must not activate intake workflows")


def _fixed_clock(timestamp: datetime) -> Callable[[], datetime]:
    def clock() -> datetime:
        return timestamp

    return clock


def _merge_memory_index(existing: str, generated: str) -> str:
    if not existing.strip():
        return generated
    start = existing.find(_MEMORY_INDEX_START)
    end = existing.find(_MEMORY_INDEX_END, start + len(_MEMORY_INDEX_START))
    if start >= 0 and end >= 0:
        end += len(_MEMORY_INDEX_END)
        merged = existing[:start].rstrip() + "\n\n" + generated.rstrip() + existing[end:]
        return merged.lstrip("\n")
    return existing.rstrip() + "\n\n" + generated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opentulpa-migrate-deepagents",
        description="Migrate legacy OpenTulpa orchestration data to Deep Agents stores.",
    )
    parser.add_argument("--data-root", type=Path, default=Path(".opentulpa"))
    parser.add_argument("--customer-profiles-db", type=Path)
    parser.add_argument("--file-vault-db", type=Path)
    parser.add_argument("--file-vault-root", type=Path)
    parser.add_argument("--knowledge-db", type=Path)
    parser.add_argument("--intake-workflows-db", type=Path)
    parser.add_argument("--integration-connections-db", type=Path)
    parser.add_argument("--legacy-routines-db", type=Path)
    parser.add_argument("--agent-specs-db", type=Path)
    parser.add_argument("--trigger-specs-db", type=Path)
    parser.add_argument("--legacy-setup-db", type=Path)
    parser.add_argument("--intake-drafts-db", type=Path)
    parser.add_argument("--legacy-skills-db", type=Path)
    parser.add_argument("--legacy-skills-root", type=Path)
    parser.add_argument("--store-db", type=Path)
    memory_source = parser.add_mutually_exclusive_group()
    memory_source.add_argument("--memory-json", type=Path)
    memory_source.add_argument("--mem0-qdrant-path", type=Path)
    memory_source.add_argument("--skip-memories", action="store_true")
    parser.add_argument("--mem0-collection", default="mem0")
    parser.add_argument("--default-timezone", default="UTC")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow absent preserved product databases. Use only for a verified new "
            "installation; cutover migration fails closed by default."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.data_root.expanduser().resolve()
    config = DeepAgentsMigrationConfig(
        customer_profiles_db_path=args.customer_profiles_db or root / "customer_profiles.db",
        file_vault_db_path=args.file_vault_db or root / "file_vault.db",
        file_vault_root_path=args.file_vault_root or root / "file_vault",
        knowledge_db_path=args.knowledge_db or root / "knowledge" / "knowledge.db",
        intake_workflows_db_path=args.intake_workflows_db or root / "intake_workflows.db",
        integration_connections_db_path=(
            args.integration_connections_db or root / "telegram_business.db"
        ),
        legacy_routines_db_path=args.legacy_routines_db or root / "scheduler.db",
        agent_specs_db_path=args.agent_specs_db or root / "deepagents" / "agent_specs.db",
        trigger_specs_db_path=(
            args.trigger_specs_db or root / "deepagents" / "trigger_specs.db"
        ),
        legacy_setup_db_path=args.legacy_setup_db or root / "intake_workflow_setup.db",
        intake_drafts_db_path=args.intake_drafts_db or root / "deepagents" / "intake_drafts.db",
        legacy_skills_db_path=args.legacy_skills_db or root / "skills.db",
        legacy_skills_root_path=args.legacy_skills_root or root / "skills",
        store_db_path=args.store_db or root / "deepagents" / "store.db",
        default_timezone=args.default_timezone,
        allow_missing_preserved_data=args.allow_missing,
    )
    memory_source: LegacyMemorySource | None = None
    close_source: QdrantMem0Source | None = None
    if not args.skip_memories:
        if args.memory_json is not None:
            memory_source = JsonMemorySource(args.memory_json)
        else:
            source = QdrantMem0Source(
                path=args.mem0_qdrant_path or root / "qdrant",
                collection_name=args.mem0_collection,
            )
            memory_source = source
            close_source = source
    try:
        report = DeepAgentsMigrator(config, memory_source=memory_source).run(dry_run=args.dry_run)
    finally:
        if close_source is not None:
            close_source.close()
    print(report.model_dump_json(indent=2))
    return 0 if report.status == "completed" else 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = ["DeepAgentsMigrator", "main"]
