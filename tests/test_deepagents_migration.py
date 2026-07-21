from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from langgraph.store.sqlite.aio import AsyncSqliteStore
from qdrant_client import QdrantClient, models

from opentulpa.context.file_vault import FileVaultService
from opentulpa.intake.drafts.store import IntakeDraftStore
from opentulpa.migrations.deepagents import DeepAgentsMigrator, main
from opentulpa.migrations.memory_sources import QdrantMem0Source
from opentulpa.migrations.models import DeepAgentsMigrationConfig, LegacyMemoryRecord
from opentulpa.persistence.tenant_namespace import tenant_store_namespace
from opentulpa.schedules.models import AgentJob, Cron, ScheduleWrite
from opentulpa.schedules.service import ScheduleService
from opentulpa.specs import AgentSpecStore, TriggerSpecService, TriggerSpecStore


class _MemorySource:
    def __init__(self, records: list[LegacyMemoryRecord]) -> None:
        self._records = records
        self.disabled: list[str] = []

    def records(self) -> list[LegacyMemoryRecord]:
        return list(self._records)

    def disable(self, record: LegacyMemoryRecord, *, reason: str) -> bool:
        assert reason
        self.disabled.append(record.legacy_id)
        return True


def _create_routines(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE routines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                schedule TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                is_cron INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO routines VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "rtn_valid",
                    "Daily brief",
                    "0 9 * * *",
                    json.dumps(
                        {
                            "customer_id": "tenant-a",
                            "instruction": "Prepare the daily brief",
                            "notify_user": True,
                            "timezone": "Europe/Moscow",
                        }
                    ),
                    1,
                    1,
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "rtn_invalid",
                    "Missing owner",
                    "0 8 * * *",
                    json.dumps({"instruction": "Cannot safely migrate"}),
                    1,
                    1,
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "rtn_intake",
                    "Intake polling",
                    "*/5 * * * *",
                    json.dumps(
                        {
                            "customer_id": "tenant-a",
                            "instruction": "Poll intake",
                            "workflow_type": "intake_workflow",
                        }
                    ),
                    1,
                    1,
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "rtn_bad_timestamp",
                    "Bad timestamp",
                    "0 7 * * *",
                    json.dumps(
                        {
                            "customer_id": "tenant-a",
                            "instruction": "Do not guess this timestamp",
                        }
                    ),
                    1,
                    1,
                    "not-a-timestamp",
                    "2026-07-02T00:00:00+00:00",
                ),
            ],
        )


def _create_setup_sessions(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE intake_workflow_setup_sessions (
                session_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                target_workflow_id TEXT,
                target_workflow_snapshot_json TEXT NOT NULL DEFAULT '{}',
                draft_upsert_json TEXT NOT NULL,
                scratchpad_json TEXT NOT NULL DEFAULT '{}',
                last_proposed_draft_hash TEXT,
                confirmed_draft_hash TEXT,
                created_or_updated_workflow_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE bookings (
                booking_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            INSERT INTO bookings VALUES ('booking-preserved', '{"name":"Ada"}');
            """
        )
        conn.executemany(
            """
            INSERT INTO intake_workflow_setup_sessions (
                session_id, customer_id, thread_id, status, mode,
                target_workflow_id, draft_upsert_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "setup-create",
                    "tenant-a",
                    "thread-a",
                    "paused",
                    "create",
                    None,
                    json.dumps({"name": "New intake", "required_fields": ["name"]}),
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "setup-edit",
                    "tenant-a",
                    "thread-b",
                    "active",
                    "edit",
                    "iwf_existing",
                    json.dumps({"name": "Edited intake"}),
                    "2026-07-03T00:00:00+00:00",
                    "2026-07-04T00:00:00+00:00",
                ),
                (
                    "setup-invalid",
                    "tenant-a",
                    "thread-c",
                    "active",
                    "edit",
                    None,
                    "{}",
                    "2026-07-03T00:00:00+00:00",
                    "2026-07-04T00:00:00+00:00",
                ),
                (
                    "setup-complete",
                    "tenant-a",
                    "thread-d",
                    "completed",
                    "create",
                    None,
                    json.dumps({"name": "Already handled"}),
                    "2026-07-03T00:00:00+00:00",
                    "2026-07-04T00:00:00+00:00",
                ),
            ],
        )


def _skill_markdown(name: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: Test skill\n---\n\n# {name}\n\n{body}\n"


def _create_skills(path: Path, root: Path) -> None:
    valid_path = root / "users" / "tenant-a" / "quote-builder" / "SKILL.md"
    generated_path = root / "users" / "tenant-a" / "intake-workflow-iwf-1" / "SKILL.md"
    mismatched_path = root / "users" / "tenant-a" / "bad-frontmatter" / "SKILL.md"
    valid_path.parent.mkdir(parents=True)
    generated_path.parent.mkdir(parents=True)
    mismatched_path.parent.mkdir(parents=True)
    valid_path.write_text(_skill_markdown("quote-builder", "Build a quote."), encoding="utf-8")
    generated_path.write_text(
        _skill_markdown("intake-workflow-iwf-1", "Generated workflow policy."),
        encoding="utf-8",
    )
    mismatched_path.write_text(
        _skill_markdown("different-name", "Do not guess the requested name."),
        encoding="utf-8",
    )
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE skills (
                scope TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                skill_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scope, customer_id, name)
            );
            """
        )
        conn.executemany(
            "INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "user",
                    "tenant-a",
                    "quote-builder",
                    "Test skill",
                    "agent",
                    1,
                    "/original-data/skills/users/tenant-a/quote-builder/SKILL.md",
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "user",
                    "tenant-a",
                    "intake-workflow-iwf-1",
                    "Generated",
                    "agent",
                    1,
                    "/original-data/skills/users/tenant-a/intake-workflow-iwf-1/SKILL.md",
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "user",
                    "tenant-a",
                    "missing-file",
                    "Broken",
                    "agent",
                    1,
                    "/original-data/skills/users/tenant-a/missing-file/SKILL.md",
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
                (
                    "user",
                    "tenant-a",
                    "bad-frontmatter",
                    "Broken",
                    "agent",
                    1,
                    "/original-data/skills/users/tenant-a/bad-frontmatter/SKILL.md",
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-02T00:00:00+00:00",
                ),
            ],
        )


def _create_preserved_product_data(config: DeepAgentsMigrationConfig) -> None:
    config.customer_profiles_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.customer_profiles_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE customer_profiles (
                customer_id TEXT PRIMARY KEY,
                directive_text TEXT,
                utc_offset TEXT,
                locale TEXT,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE customer_identity_aliases (
                alias_user_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                storage_user_id TEXT NOT NULL,
                alias_kind TEXT NOT NULL,
                provider TEXT,
                provider_user_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO customer_profiles VALUES (
                'tenant-a', 'Keep replies concise', '+03:00', 'en', 'owner',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO customer_identity_aliases VALUES (
                'telegram_42', 'owner', 'tenant-a', 'provider', 'telegram', '42',
                '2026-07-01T00:00:00+00:00', '2026-07-02T00:00:00+00:00'
            );
            """
        )

    raw_file = b"preserved uploaded file\n"
    stored_file = config.file_vault_root_path / "tenant-a" / "source.txt"
    stored_file.parent.mkdir(parents=True, exist_ok=True)
    stored_file.write_bytes(raw_file)
    with sqlite3.connect(config.file_vault_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE uploaded_files (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                chat_id INTEGER,
                telegram_file_id TEXT,
                kind TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER NOT NULL,
                caption TEXT,
                summary TEXT,
                text_excerpt TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO uploaded_files (
                id, customer_id, kind, original_filename, stored_path, mime_type,
                size_bytes, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "file-1",
                "tenant-a",
                "document",
                "source.txt",
                "/legacy/data/file_vault/tenant-a/source.txt",
                "text/plain",
                len(raw_file),
                "Preserved file",
                "2026-07-02T00:00:00+00:00",
            ),
        )

    config.knowledge_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.knowledge_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE knowledge_sources (
                customer_id TEXT NOT NULL,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (customer_id, scope_type, scope_id, file_id)
            );
            CREATE TABLE knowledge_sections (
                section_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE TABLE knowledge_preflight_cache (
                customer_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                source_signature TEXT NOT NULL,
                result_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (customer_id, cache_key)
            );
            INSERT INTO knowledge_sources VALUES (
                'tenant-a', 'customer_business', 'library', 'file-1', 'hash-1', 'ready'
            );
            INSERT INTO knowledge_sections VALUES (
                'section-1', 'tenant-a', 'customer_business', 'library', 'file-1',
                'Grounded business fact'
            );
            INSERT INTO knowledge_preflight_cache VALUES (
                'tenant-a', 'preflight-1', 'source-signature', '{"ok":true}',
                '2026-07-02T00:00:00+00:00'
            );
            """
        )

    with sqlite3.connect(config.intake_workflows_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE intake_workflows (
                workflow_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                name TEXT NOT NULL,
                channel TEXT NOT NULL,
                provider TEXT NOT NULL,
                source_config_json TEXT NOT NULL,
                intent_description TEXT NOT NULL,
                required_fields_json TEXT NOT NULL,
                field_guidance_json TEXT NOT NULL,
                assistant_instructions TEXT NOT NULL DEFAULT '',
                business_facts_json TEXT NOT NULL DEFAULT '{}',
                knowledge_file_ids_json TEXT NOT NULL DEFAULT '[]',
                sink_type TEXT NOT NULL,
                sink_config_json TEXT NOT NULL,
                schedule TEXT NOT NULL,
                notify_user INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                routine_id TEXT NOT NULL,
                reply_mode TEXT NOT NULL DEFAULT 'auto',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE intake_bookings (
                booking_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                extracted_fields_json TEXT NOT NULL,
                sink_write_status TEXT NOT NULL
            );
            CREATE TABLE intake_conversation_cursors (
                workflow_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                last_seen_inbound_message_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workflow_id, conversation_id)
            );
            CREATE TABLE intake_pending_runs (
                workflow_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                due_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workflow_id, conversation_id)
            );
            INSERT INTO intake_workflows (
                workflow_id, customer_id, revision, name, channel, provider,
                source_config_json, intent_description, required_fields_json,
                field_guidance_json, sink_type, sink_config_json, schedule,
                notify_user, enabled, routine_id, created_at, updated_at
            ) VALUES (
                'workflow-active', 'tenant-a', 3, 'Lead intake', 'instagram_dm',
                'composio', '{"channel":"instagram"}', 'Capture leads', '["name"]',
                '{}', 'google_sheets', '{"sheet":"leads"}', '*/5 * * * *', 1, 1,
                'routine-active', '2026-07-01T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO intake_workflows (
                workflow_id, customer_id, revision, name, channel, provider,
                source_config_json, intent_description, required_fields_json,
                field_guidance_json, sink_type, sink_config_json, schedule,
                notify_user, enabled, routine_id, created_at, updated_at
            ) VALUES (
                'workflow-disabled', 'tenant-a', 1, 'Old intake', 'instagram_dm',
                'composio', '{}', 'Old workflow', '[]', '{}', 'none', '{}', '', 0, 0,
                '', '2026-06-01T00:00:00+00:00', '2026-06-02T00:00:00+00:00'
            );
            INSERT INTO intake_bookings VALUES (
                'booking-1', 'workflow-active', 'tenant-a', 'conversation-1', 'completed',
                '{"name":"Ada"}', 'written'
            );
            INSERT INTO intake_conversation_cursors VALUES (
                'workflow-active', 'conversation-1', 'message-7',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO intake_pending_runs VALUES (
                'workflow-active', 'conversation-1', 'tenant-a', 4, 'pending',
                '2026-07-02T00:01:00+00:00', '2026-07-02T00:00:00+00:00'
            );
            """
        )

    with sqlite3.connect(config.integration_connections_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE telegram_business_connections (
                business_connection_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_chat_id TEXT NOT NULL,
                is_enabled INTEGER NOT NULL,
                connection_json TEXT NOT NULL
            );
            CREATE TABLE telegram_business_messages (
                business_connection_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                sender_role TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (business_connection_id, chat_id, message_id)
            );
            INSERT INTO telegram_business_connections VALUES (
                'connection-1', 'tenant-a', '42', '84', 1, '{"rights":["reply"]}'
            );
            INSERT INTO telegram_business_messages VALUES (
                'connection-1', 'tenant-a', 'chat-1', 'message-7', 'customer',
                '{"text":"hello"}', '2026-07-02T00:00:00+00:00'
            );
            """
        )


def _create_origin_main_intake_data(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE intake_workflows (
                workflow_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                name TEXT NOT NULL,
                channel TEXT NOT NULL,
                provider TEXT NOT NULL,
                source_config_json TEXT NOT NULL,
                intent_description TEXT NOT NULL,
                required_fields_json TEXT NOT NULL,
                field_guidance_json TEXT NOT NULL,
                assistant_instructions TEXT NOT NULL DEFAULT '',
                business_facts_json TEXT NOT NULL DEFAULT '{}',
                knowledge_file_ids_json TEXT NOT NULL DEFAULT '[]',
                sink_type TEXT NOT NULL,
                sink_config_json TEXT NOT NULL,
                schedule TEXT NOT NULL,
                notify_user INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                routine_id TEXT NOT NULL,
                reply_mode TEXT NOT NULL DEFAULT 'auto',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE intake_conversation_cursors (
                workflow_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                last_seen_inbound_message_id TEXT,
                last_seen_inbound_message_time TEXT,
                last_seen_conversation_updated_time TEXT,
                last_seen_latest_outbound_message_id TEXT,
                last_agent_action_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workflow_id, conversation_id)
            );
            CREATE TABLE intake_pending_runs (
                workflow_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                owner_chat_id TEXT NOT NULL DEFAULT '',
                generation INTEGER NOT NULL,
                running_generation INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                due_at TEXT NOT NULL,
                last_inbound_message_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (workflow_id, conversation_id)
            );
            CREATE TABLE intake_bookings (
                booking_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                extracted_fields_json TEXT NOT NULL,
                sink_write_status TEXT NOT NULL,
                sink_record_ref_json TEXT NOT NULL,
                conversation_summary TEXT NOT NULL,
                last_customer_message_at TEXT,
                opened_at TEXT NOT NULL,
                completed_at TEXT,
                edit_window_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO intake_workflows (
                workflow_id, customer_id, name, channel, provider, source_config_json,
                intent_description, required_fields_json, field_guidance_json, sink_type,
                sink_config_json, schedule, notify_user, enabled, routine_id, created_at,
                updated_at
            ) VALUES (
                'legacy-active', 'tenant-a', 'Legacy intake', 'telegram_business_dm',
                'telegram_bot_api', '{"business_connection_id":"connection-1"}',
                'Book appointments', '["name","time"]', '{}', 'none', '{}',
                '*/5 * * * *', 1, 1, 'routine-legacy', '2026-07-01T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO intake_workflows (
                workflow_id, customer_id, name, channel, provider, source_config_json,
                intent_description, required_fields_json, field_guidance_json, sink_type,
                sink_config_json, schedule, notify_user, enabled, routine_id, created_at,
                updated_at
            ) VALUES (
                'legacy-disabled', 'tenant-a', 'Disabled intake', 'telegram_business_dm',
                'telegram_bot_api', '{}', 'Old flow', '[]', '{}', 'none', '{}', '', 0, 0,
                '', '2026-06-01T00:00:00+00:00', '2026-06-02T00:00:00+00:00'
            );
            INSERT INTO intake_bookings (
                booking_id, workflow_id, customer_id, conversation_id, status,
                extracted_fields_json, sink_write_status, sink_record_ref_json,
                conversation_summary, opened_at, created_at, updated_at
            ) VALUES (
                'legacy-booking', 'legacy-active', 'tenant-a', 'conversation-1', 'active',
                '{"name":"Ada"}', 'pending', '{}', 'Appointment request',
                '2026-07-02T00:00:00+00:00', '2026-07-02T00:00:00+00:00',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO intake_conversation_cursors (
                workflow_id, conversation_id, last_seen_inbound_message_id, updated_at
            ) VALUES (
                'legacy-active', 'conversation-1', 'message-1',
                '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO intake_pending_runs (
                workflow_id, conversation_id, customer_id, event_type, generation,
                status, due_at, created_at, updated_at
            ) VALUES (
                'legacy-active', 'conversation-1', 'tenant-a', 'scheduled', 2,
                'pending', '2026-07-02T00:01:00+00:00',
                '2026-07-02T00:00:00+00:00', '2026-07-02T00:00:00+00:00'
            );
            """
        )


def _config(tmp_path: Path) -> DeepAgentsMigrationConfig:
    return DeepAgentsMigrationConfig(
        customer_profiles_db_path=tmp_path / "customer_profiles.db",
        file_vault_db_path=tmp_path / "file_vault.db",
        file_vault_root_path=tmp_path / "file_vault",
        knowledge_db_path=tmp_path / "knowledge" / "knowledge.db",
        intake_workflows_db_path=tmp_path / "intake_workflows.db",
        integration_connections_db_path=tmp_path / "telegram_business.db",
        legacy_routines_db_path=tmp_path / "scheduler.db",
        agent_specs_db_path=tmp_path / "new" / "agent_specs.db",
        trigger_specs_db_path=tmp_path / "new" / "trigger_specs.db",
        legacy_setup_db_path=tmp_path / "setup.db",
        intake_drafts_db_path=tmp_path / "new" / "drafts.db",
        legacy_skills_db_path=tmp_path / "skills.db",
        legacy_skills_root_path=tmp_path / "legacy-skills",
        store_db_path=tmp_path / "new" / "store.db",
        default_timezone="UTC",
    )


def _legacy_value(path: Path, query: str, params: tuple[str, ...]) -> object:
    with sqlite3.connect(path) as conn:
        row = conn.execute(query, params).fetchone()
    assert row is not None
    return row[0]


def test_preserved_verifier_accepts_origin_main_intake_workflow_schema(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path).model_copy(
        update={"allow_missing_preserved_data": True}
    )
    _create_origin_main_intake_data(config.intake_workflows_db_path)
    migrator = DeepAgentsMigrator(config, memory_source=_MemorySource([]))

    first = migrator.run(dry_run=True)
    repeated = migrator.run(dry_run=True)

    assert first.status == "completed"
    assert first.preserved_data.verified is True
    workflows = next(
        item for item in first.preserved_data.datasets if item.dataset == "active_workflows"
    )
    bookings = next(item for item in first.preserved_data.datasets if item.dataset == "bookings")
    assert workflows.status == "ok"
    assert workflows.record_count == 2
    assert workflows.table_counts == {
        "intake_conversation_cursors": 1,
        "intake_pending_runs": 1,
        "intake_workflows": 2,
    }
    assert bookings.status == "ok"
    assert bookings.record_count == 1
    assert workflows.source_checksum == next(
        item
        for item in repeated.preserved_data.datasets
        if item.dataset == "active_workflows"
    ).source_checksum
    assert repeated.preserved_data.combined_checksum == first.preserved_data.combined_checksum


def test_preserved_verifier_blocks_unsupported_intake_workflow_schema(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with sqlite3.connect(config.intake_workflows_db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE intake_workflows (
                workflow_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                source_config_json TEXT NOT NULL,
                sink_config_json TEXT NOT NULL,
                enabled INTEGER NOT NULL
            );
            CREATE TABLE intake_bookings (
                booking_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                extracted_fields_json TEXT NOT NULL,
                sink_write_status TEXT NOT NULL
            );
            """
        )

    report = DeepAgentsMigrator(config, memory_source=_MemorySource([])).run(dry_run=True)

    assert report.status == "blocked"
    assert report.preserved_data.verified is False
    workflows = next(
        item for item in report.preserved_data.datasets if item.dataset == "active_workflows"
    )
    assert workflows.status == "invalid"
    assert workflows.message is not None
    assert "missing columns" in workflows.message
    assert "channel" in workflows.message


def test_cutover_migration_dry_run_is_read_only_and_reports_checksums(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_routines(config.legacy_routines_db_path)
    _create_setup_sessions(config.legacy_setup_db_path)
    _create_skills(config.legacy_skills_db_path, tmp_path / "legacy-skills")
    _create_preserved_product_data(config)
    memory_source = _MemorySource(
        [
            LegacyMemoryRecord(
                legacy_id="memory-1",
                tenant_id="tenant-a",
                content="The owner prefers concise replies.",
                created_at="2026-06-01T10:00:00+00:00",
                updated_at="2026-06-02T10:00:00+00:00",
            ),
            LegacyMemoryRecord(legacy_id="memory-invalid", tenant_id="", content="orphan"),
        ]
    )

    report = DeepAgentsMigrator(config, memory_source=memory_source).run(dry_run=True)
    repeated = DeepAgentsMigrator(config, memory_source=memory_source).run(dry_run=True)

    assert report.dry_run is True
    assert report.status == "completed"
    assert report.preserved_data.verified is True
    preserved = {item.dataset: item for item in report.preserved_data.datasets}
    assert {name: item.record_count for name, item in preserved.items()} == {
        "profiles": 1,
        "files": 1,
        "knowledge": 1,
        "active_workflows": 2,
        "bookings": 1,
        "integration_connections": 1,
    }
    assert preserved["profiles"].table_counts == {
        "customer_profiles": 1,
        "customer_identity_aliases": 1,
    }
    assert preserved["knowledge"].table_counts == {
        "knowledge_preflight_cache": 1,
        "knowledge_sources": 1,
        "knowledge_sections": 1,
    }
    assert preserved["active_workflows"].table_counts == {
        "intake_conversation_cursors": 1,
        "intake_pending_runs": 1,
        "intake_workflows": 2,
    }
    assert preserved["integration_connections"].table_counts == {
        "telegram_business_connections": 1,
        "telegram_business_messages": 1,
    }
    assert all(len(item.source_checksum) == 64 for item in preserved.values())
    assert report.routines.model_dump(exclude={"issues"}) == {
        "scanned": 4,
        "eligible": 1,
        "migrated": 0,
        "skipped": 1,
        "invalid": 2,
        "disabled": 0,
        "source_checksum": report.routines.source_checksum,
    }
    assert report.drafts.eligible == 2
    assert report.drafts.invalid == 1
    assert report.memories.eligible == 1
    assert report.memories.invalid == 1
    assert report.skills.eligible == 1
    assert report.skills.skipped == 1
    assert report.skills.invalid == 2
    assert report.checkpoints_migrated == 0
    assert report.checkpoint_policy == "fresh"
    assert len(report.combined_checksum) == 64
    assert repeated.combined_checksum == report.combined_checksum
    assert repeated.routines.source_checksum == report.routines.source_checksum
    assert repeated.drafts.source_checksum == report.drafts.source_checksum
    assert repeated.memories.source_checksum == report.memories.source_checksum
    assert repeated.skills.source_checksum == report.skills.source_checksum
    assert repeated.preserved_data.combined_checksum == report.preserved_data.combined_checksum
    assert not (tmp_path / "new").exists()
    assert report.file_paths_pending_rebase == 1
    assert report.file_paths_rebased == 0
    assert (
        _legacy_value(
            config.file_vault_db_path,
            "SELECT stored_path FROM uploaded_files WHERE id=?",
            ("file-1",),
        )
        == "/legacy/data/file_vault/tenant-a/source.txt"
    )
    assert memory_source.disabled == []
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_invalid",),
        )
        == 1
    )
    assert (
        _legacy_value(
            config.legacy_setup_db_path,
            "SELECT status FROM intake_workflow_setup_sessions WHERE session_id=?",
            ("setup-invalid",),
        )
        == "active"
    )


@pytest.mark.asyncio
async def test_cutover_migration_writes_native_store_disables_invalid_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _create_routines(config.legacy_routines_db_path)
    _create_setup_sessions(config.legacy_setup_db_path)
    _create_skills(config.legacy_skills_db_path, tmp_path / "legacy-skills")
    _create_preserved_product_data(config)
    memory_source = _MemorySource(
        [
            LegacyMemoryRecord(
                legacy_id="memory-1",
                tenant_id="tenant-a",
                content="The owner prefers concise replies.",
                created_at="2026-06-01T10:00:00+00:00",
                updated_at="2026-06-02T10:00:00+00:00",
            ),
            LegacyMemoryRecord(legacy_id="memory-invalid", tenant_id="", content="orphan"),
        ]
    )
    migrator = DeepAgentsMigrator(config, memory_source=memory_source)
    dry_run = migrator.run(dry_run=True)

    first = migrator.run()

    assert first.status == "completed"
    assert first.preserved_data.verified is True
    assert first.preserved_data.combined_checksum == dry_run.preserved_data.combined_checksum
    assert first.combined_checksum == dry_run.combined_checksum
    assert first.file_paths_pending_rebase == 1
    assert first.file_paths_rebased == 1
    rebased_path = Path(
        str(
            _legacy_value(
                config.file_vault_db_path,
                "SELECT stored_path FROM uploaded_files WHERE id=?",
                ("file-1",),
            )
        )
    )
    assert rebased_path == (config.file_vault_root_path / "tenant-a" / "source.txt").resolve()
    file_vault = FileVaultService(
        root_dir=config.file_vault_root_path,
        db_path=config.file_vault_db_path,
    )
    assert file_vault.read_file_bytes("tenant-a", "file-1") == b"preserved uploaded file\n"
    assert first.routines.migrated == 1
    assert first.routines.disabled == 2
    assert first.drafts.migrated == 2
    assert first.drafts.disabled == 1
    assert first.memories.migrated == 1
    assert first.memories.disabled == 1
    assert first.skills.migrated == 1
    assert first.skills.disabled == 2
    agent_specs = AgentSpecStore(config.agent_specs_db_path)
    routine_ref = agent_specs.get_active_ref(tenant_id="tenant-a", spec_id="routine")
    assert routine_ref is not None
    schedule_service = ScheduleService(
        TriggerSpecService(
            TriggerSpecStore(config.trigger_specs_db_path, agent_specs=agent_specs)
        ),
        resolve_agent_spec=lambda _: routine_ref,
    )
    assert schedule_service.get(tenant_id="tenant-a", schedule_id="rtn_valid") is not None
    drafts = IntakeDraftStore(config.intake_drafts_db_path).list(tenant_id="tenant-a")
    assert len(drafts) == 2
    assert {draft.workflow_id for draft in drafts if draft.payload["name"] == "Edited intake"} == {
        "iwf_existing"
    }
    created_draft = next(draft for draft in drafts if draft.payload["name"] == "New intake")
    assert created_draft.workflow_id.startswith("iwf_legacy_")
    async with AsyncSqliteStore.from_conn_string(str(config.store_db_path)) as store:
        await store.setup()
        memory = await store.aget(tenant_store_namespace("tenant-a", "memory"), "/memory-1.md")
        index = await store.aget(tenant_store_namespace("tenant-a", "memory"), "/AGENTS.md")
        skill = await store.aget(
            tenant_store_namespace("tenant-a", "skills"),
            "/quote-builder/SKILL.md",
        )
    assert memory is not None
    assert 'legacy_id: "memory-1"' in memory.value["content"]
    assert memory.value["created_at"] == "2026-06-01T10:00:00+00:00"
    assert index is not None and "memory-1" in index.value["content"]
    assert skill is not None and "Build a quote." in skill.value["content"]
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_invalid",),
        )
        == 0
    )
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_bad_timestamp",),
        )
        == 0
    )
    assert (
        _legacy_value(
            config.legacy_setup_db_path,
            "SELECT status FROM intake_workflow_setup_sessions WHERE session_id=?",
            ("setup-invalid",),
        )
        == "cancelled"
    )
    assert (
        _legacy_value(
            config.legacy_skills_db_path,
            "SELECT enabled FROM skills WHERE customer_id=? AND name=?",
            ("tenant-a", "missing-file"),
        )
        == 0
    )
    assert (
        _legacy_value(
            config.legacy_setup_db_path,
            "SELECT payload_json FROM bookings WHERE booking_id=?",
            ("booking-preserved",),
        )
        == '{"name":"Ada"}'
    )
    assert memory_source.disabled == ["memory-invalid"]

    second = migrator.run()

    assert second.routines.migrated == 0
    assert second.drafts.migrated == 0
    assert second.memories.migrated == 0
    assert second.skills.migrated == 0
    assert second.memories.skipped == 1
    assert second.skills.skipped == 4
    assert second.preserved_data.combined_checksum == first.preserved_data.combined_checksum
    assert len(IntakeDraftStore(config.intake_drafts_db_path).list(tenant_id="tenant-a")) == 2
    dry_after_apply = migrator.run(dry_run=True)
    assert dry_after_apply.routines.eligible == 0
    assert dry_after_apply.routines.skipped == 2
    assert dry_after_apply.file_paths_pending_rebase == 0


def test_migration_command_requires_explicit_allow_missing_for_empty_data_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocked = main(["--data-root", str(tmp_path), "--dry-run", "--skip-memories"])

    assert blocked == 2
    blocked_report = json.loads(capsys.readouterr().out)
    assert blocked_report["status"] == "blocked"
    assert blocked_report["preserved_data"]["verified"] is False

    exit_code = main(
        [
            "--data-root",
            str(tmp_path),
            "--dry-run",
            "--skip-memories",
            "--allow-missing",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["dry_run"] is True
    assert report["checkpoints_migrated"] == 0
    assert report["routines"]["scanned"] == 0
    assert report["status"] == "completed"
    assert report["preserved_data"]["verified"] is True
    assert {item["status"] for item in report["preserved_data"]["datasets"]} == {
        "missing"
    }


def test_invalid_preserved_store_blocks_all_migration_writes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_routines(config.legacy_routines_db_path)
    config.customer_profiles_db_path.write_bytes(b"not a sqlite database")

    report = DeepAgentsMigrator(config, memory_source=_MemorySource([])).run()

    assert report.status == "blocked"
    assert report.preserved_data.verified is False
    profiles = next(
        item for item in report.preserved_data.datasets if item.dataset == "profiles"
    )
    assert profiles.status == "unreadable"
    assert profiles.message
    assert report.routines.scanned == 0
    assert not config.agent_specs_db_path.exists()
    assert not config.trigger_specs_db_path.exists()
    assert not config.intake_drafts_db_path.exists()
    assert not config.store_db_path.exists()
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_invalid",),
        )
        == 1
    )


def test_migration_command_reports_invalid_store_and_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "customer_profiles.db").write_bytes(b"corrupt")

    exit_code = main(["--data-root", str(tmp_path), "--skip-memories"])

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert report["preserved_data"]["verified"] is False
    profiles = next(
        item
        for item in report["preserved_data"]["datasets"]
        if item["dataset"] == "profiles"
    )
    assert profiles["status"] == "unreadable"


def test_file_dataset_checksum_includes_preserved_file_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_preserved_product_data(config)
    migrator = DeepAgentsMigrator(config, memory_source=_MemorySource([]))

    before = migrator.run(dry_run=True)
    stored_file = config.file_vault_root_path / "tenant-a" / "source.txt"
    stored_file.write_bytes(b"changed uploaded content\n")
    with sqlite3.connect(config.file_vault_db_path) as conn:
        conn.execute(
            "UPDATE uploaded_files SET size_bytes=? WHERE id='file-1'",
            (stored_file.stat().st_size,),
        )
    after = migrator.run(dry_run=True)

    before_files = next(
        item for item in before.preserved_data.datasets if item.dataset == "files"
    )
    after_files = next(
        item for item in after.preserved_data.datasets if item.dataset == "files"
    )
    assert before_files.source_checksum != after_files.source_checksum
    assert before.preserved_data.combined_checksum != after.preserved_data.combined_checksum


def test_preserved_checksum_is_stable_across_copied_data_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_config = _config(source_root)
    _create_preserved_product_data(source_config)
    source_report = DeepAgentsMigrator(
        source_config,
        memory_source=_MemorySource([]),
    ).run(dry_run=True)

    copied_root = tmp_path / "copied"
    shutil.copytree(source_root, copied_root)
    copied_report = DeepAgentsMigrator(
        _config(copied_root),
        memory_source=_MemorySource([]),
    ).run(dry_run=True)

    assert copied_report.preserved_data.combined_checksum == (
        source_report.preserved_data.combined_checksum
    )


def test_uploaded_file_path_rebase_rolls_back_as_one_transaction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_preserved_product_data(config)
    second_file = config.file_vault_root_path / "tenant-a" / "second.txt"
    second_file.write_bytes(b"second preserved file\n")
    with sqlite3.connect(config.file_vault_db_path) as conn:
        conn.execute(
            """
            INSERT INTO uploaded_files (
                id, customer_id, kind, original_filename, stored_path, mime_type,
                size_bytes, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "file-2",
                "tenant-a",
                "document",
                "second.txt",
                "/legacy/data/file_vault/tenant-a/second.txt",
                "text/plain",
                second_file.stat().st_size,
                "Second preserved file",
                "2026-07-02T00:00:00+00:00",
            ),
        )
        conn.executescript(
            """
            CREATE TRIGGER reject_second_file_rebase
            BEFORE UPDATE OF stored_path ON uploaded_files
            WHEN OLD.id = 'file-2'
            BEGIN
                SELECT RAISE(ABORT, 'injected rebase failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected rebase failure"):
        DeepAgentsMigrator(config, memory_source=_MemorySource([])).run()

    with sqlite3.connect(config.file_vault_db_path) as conn:
        paths = dict(conn.execute("SELECT id, stored_path FROM uploaded_files").fetchall())
    assert paths == {
        "file-1": "/legacy/data/file_vault/tenant-a/source.txt",
        "file-2": "/legacy/data/file_vault/tenant-a/second.txt",
    }


def test_preserved_checksums_include_cutover_continuity_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _create_preserved_product_data(config)
    migrator = DeepAgentsMigrator(config, memory_source=_MemorySource([]))

    before = {
        item.dataset: item.source_checksum
        for item in migrator.run(dry_run=True).preserved_data.datasets
    }
    with sqlite3.connect(config.intake_workflows_db_path) as conn:
        conn.execute(
            "UPDATE intake_workflows SET name='Changed disabled' WHERE workflow_id='workflow-disabled'"
        )
        conn.execute(
            """
            UPDATE intake_conversation_cursors SET last_seen_inbound_message_id='message-8'
            WHERE workflow_id='workflow-active' AND conversation_id='conversation-1'
            """
        )
        conn.execute(
            """
            UPDATE intake_pending_runs SET generation=5
            WHERE workflow_id='workflow-active' AND conversation_id='conversation-1'
            """
        )
    with sqlite3.connect(config.integration_connections_db_path) as conn:
        conn.execute(
            "UPDATE telegram_business_messages SET raw_json='{}' WHERE message_id='message-7'"
        )
    with sqlite3.connect(config.knowledge_db_path) as conn:
        conn.execute(
            "UPDATE knowledge_preflight_cache SET result_json='{}' WHERE cache_key='preflight-1'"
        )

    after_report = migrator.run(dry_run=True).preserved_data
    after = {item.dataset: item.source_checksum for item in after_report.datasets}

    assert after["active_workflows"] != before["active_workflows"]
    assert after["integration_connections"] != before["integration_connections"]
    assert after["knowledge"] != before["knowledge"]


def test_routine_destination_conflict_blocks_without_changing_legacy_source(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "legacy_setup_db_path": None,
            "legacy_skills_db_path": None,
            "legacy_skills_root_path": None,
        }
    )
    _create_routines(config.legacy_routines_db_path)
    _create_preserved_product_data(config)
    schedules = DeepAgentsMigrator._schedule_service(
        agent_specs_path=config.agent_specs_db_path,
        trigger_specs_path=config.trigger_specs_db_path,
    )
    schedules.save(
        tenant_id="tenant-a",
        schedule_id="rtn_valid",
        actor_id="existing-owner",
        write=ScheduleWrite(
            name="Unrelated schedule",
            trigger=Cron(expression="0 1 * * *", timezone="UTC"),
            action=AgentJob(instruction="Do something else"),
            notify_owner=False,
        ),
    )
    migrator = DeepAgentsMigrator(config, memory_source=_MemorySource([]))

    dry_run = migrator.run(dry_run=True)

    assert dry_run.status == "blocked"
    conflict = next(
        issue for issue in dry_run.routines.issues if issue.legacy_id == "rtn_valid"
    )
    assert conflict.disposition == "conflict"
    assert conflict.disabled is False
    assert dry_run.routines.eligible == 0
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_valid",),
        )
        == 1
    )

    applied = migrator.run()

    assert applied.status == "blocked"
    applied_conflict = next(
        issue for issue in applied.routines.issues if issue.legacy_id == "rtn_valid"
    )
    assert applied_conflict.disposition == "conflict"
    assert applied_conflict.disabled is False
    assert (
        _legacy_value(
            config.legacy_routines_db_path,
            "SELECT enabled FROM routines WHERE id=?",
            ("rtn_valid",),
        )
        == 1
    )
    existing = schedules.get(tenant_id="tenant-a", schedule_id="rtn_valid")
    assert existing is not None
    assert existing.name == "Unrelated schedule"
    assert isinstance(existing.action, AgentJob)
    assert existing.action.instruction == "Do something else"


def test_destination_conflicts_preserve_legacy_draft_memory_and_skill_sources(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _create_routines(config.legacy_routines_db_path)
    _create_setup_sessions(config.legacy_setup_db_path)
    _create_skills(config.legacy_skills_db_path, tmp_path / "legacy-skills")
    _create_preserved_product_data(config)
    initial_memory = _MemorySource(
        [
            LegacyMemoryRecord(
                legacy_id="memory-1",
                tenant_id="tenant-a",
                content="Original memory",
            )
        ]
    )
    first = DeepAgentsMigrator(config, memory_source=initial_memory).run()
    assert first.status == "completed"

    with sqlite3.connect(config.legacy_setup_db_path) as conn:
        conn.execute(
            """
            UPDATE intake_workflow_setup_sessions SET draft_upsert_json=?
            WHERE session_id='setup-create'
            """,
            (json.dumps({"name": "Changed draft", "required_fields": ["name"]}),),
        )
    skill_path = (
        config.legacy_skills_root_path
        / "users"
        / "tenant-a"
        / "quote-builder"
        / "SKILL.md"
    )
    skill_path.write_text(
        _skill_markdown("quote-builder", "Changed legacy source."),
        encoding="utf-8",
    )
    changed_memory = _MemorySource(
        [
            LegacyMemoryRecord(
                legacy_id="memory-1",
                tenant_id="tenant-a",
                content="Changed legacy memory",
            )
        ]
    )

    second = DeepAgentsMigrator(config, memory_source=changed_memory).run()

    assert second.status == "blocked"
    assert any(issue.disposition == "conflict" for issue in second.drafts.issues)
    assert any(issue.disposition == "conflict" for issue in second.memories.issues)
    assert any(issue.disposition == "conflict" for issue in second.skills.issues)
    assert changed_memory.disabled == []
    assert (
        _legacy_value(
            config.legacy_setup_db_path,
            "SELECT status FROM intake_workflow_setup_sessions WHERE session_id=?",
            ("setup-create",),
        )
        == "paused"
    )
    assert (
        _legacy_value(
            config.legacy_skills_db_path,
            "SELECT enabled FROM skills WHERE customer_id=? AND name=?",
            ("tenant-a", "quote-builder"),
        )
        == 1
    )


def test_qdrant_mem0_source_preserves_identity_and_marks_invalid_records(tmp_path: Path) -> None:
    qdrant_path = tmp_path / "qdrant"
    client = QdrantClient(path=str(qdrant_path))
    client.create_collection(
        collection_name="mem0",
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="mem0",
        points=[
            models.PointStruct(
                id=7,
                vector=[1.0],
                payload={
                    "user_id": "tenant-a",
                    "data": "Remember the original record.",
                    "created_at": "2026-06-01T10:00:00+00:00",
                    "updated_at": "2026-06-02T10:00:00+00:00",
                },
            )
        ],
    )
    client.close()

    source = QdrantMem0Source(path=qdrant_path)
    records = source.records()

    assert records == [
        LegacyMemoryRecord(
            legacy_id="7",
            tenant_id="tenant-a",
            content="Remember the original record.",
            created_at="2026-06-01T10:00:00+00:00",
            updated_at="2026-06-02T10:00:00+00:00",
        )
    ]
    assert source.disable(records[0], reason="test invalid record") is True
    source.close()

    verifier = QdrantClient(path=str(qdrant_path))
    point = verifier.retrieve(collection_name="mem0", ids=[7], with_payload=True)[0]
    verifier.close()
    assert point.payload is not None
    assert point.payload["opentulpa_migration_disabled"] is True
    assert point.payload["opentulpa_migration_error"] == "test invalid record"
