from __future__ import annotations

import pytest

import opentulpa.interfaces.telegram.attachments as attachments_module
from opentulpa.context.file_vault import FileVaultService
from opentulpa.interfaces.telegram.attachments import (
    XLSX_MIME_TYPE,
    build_uploaded_files_context,
    extract_attachments,
    ingest_attachments,
)
from opentulpa.interfaces.telegram.models import TelegramAttachment


def test_extract_attachments_includes_video_note() -> None:
    attachments = extract_attachments(
        {
            "video_note": {
                "file_id": "vid-note-1",
                "file_unique_id": "uniq-vid-note",
                "mime_type": "video/mp4",
            }
        }
    )

    assert len(attachments) == 1
    item = attachments[0]
    assert item.kind == "video_note"
    assert item.file_id == "vid-note-1"
    assert item.filename == "uniq-vid-note.mp4"
    assert item.mime_type == "video/mp4"


def test_extract_attachments_preserves_file_size() -> None:
    attachments = extract_attachments(
        {
            "document": {
                "file_id": "doc-1",
                "file_name": "clip.MOV",
                "mime_type": "video/quicktime",
                "file_size": 20_500_000,
            }
        }
    )

    assert len(attachments) == 1
    assert attachments[0].filename == "clip.MOV"
    assert attachments[0].mime_type == "video/quicktime"
    assert attachments[0].file_size == 20_500_000


def test_uploaded_files_context_is_internal_and_avoids_paths() -> None:
    context = build_uploaded_files_context(
        [
            {
                "id": "file_1",
                "original_filename": "prices.xlsx",
                "kind": "document",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "stored_path": "/app/opentulpa_data/.opentulpa/file_vault/customer/prices.xlsx",
                "local_path": "tulpa_stuff/uploads/customer/prices.xlsx",
                "created_at": "2026-04-27T13:16:49Z",
                "summary": "Workbook with price sheets.",
            }
        ]
    )

    assert "Do not quote this metadata verbatim" in context
    assert "file_id=file_1" in context
    assert "prices.xlsx" in context
    assert "/app/opentulpa_data" not in context
    assert "tulpa_stuff/uploads" not in context


def test_uploaded_files_context_sanitizes_stale_xlsx_no_text_summary() -> None:
    context = build_uploaded_files_context(
        [
            {
                "id": "file_1",
                "original_filename": "prices.xlsx",
                "kind": "document",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "summary": (
                    "document file 'prices.xlsx' | ai_summary=Uploaded document file "
                    "'prices.xlsx'. No extractable text was available."
                ),
            }
        ]
    )

    assert "No extractable text was available" not in context
    assert "Spreadsheet file stored" in context
    assert "business_knowledge_index" in context


@pytest.mark.asyncio
async def test_document_ingest_skips_auto_llm_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class FakeTelegramClient:
        def __init__(self, token: str) -> None:
            self.token = token

        async def download_file(self, *, file_id: str):
            return {
                "raw_bytes": b"not-a-real-workbook",
                "file_path": "prices.xlsx",
                "mime_type": XLSX_MIME_TYPE,
            }

        async def aclose(self) -> None:
            return None

    class Runtime:
        called = False

        async def summarize_uploaded_blob(self, **kwargs):
            self.called = True
            raise AssertionError(
                "document uploads should be indexed/queryable, not auto-summarized"
            )

    monkeypatch.setattr(attachments_module, "TelegramClient", FakeTelegramClient)
    runtime = Runtime()
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.sqlite")

    records = await ingest_attachments(
        attachments=[
            TelegramAttachment(
                kind="document",
                file_id="tg-file-1",
                filename="prices.xlsx",
                mime_type=XLSX_MIME_TYPE,
            )
        ],
        bot_token="token",
        file_vault=vault,
        memory=None,
        agent_runtime=runtime,
        customer_id="telegram_1",
        chat_id=1,
        caption=None,
    )

    assert len(records) == 1
    assert records[0]["original_filename"] == "prices.xlsx"
    assert "ai_summary=" not in records[0]["summary"]
    assert runtime.called is False


@pytest.mark.asyncio
async def test_ingest_attachments_records_unavailable_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeTelegramClient:
        def __init__(self, token: str) -> None:
            self.token = token

        async def download_file(self, *, file_id: str):
            assert file_id == "tg-video-1"
            return None

        async def aclose(self) -> None:
            return None

    class Memory:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_text(
            self, text: str, *, user_id: str, metadata: dict[str, object], infer: bool
        ) -> None:
            self.calls.append(
                {
                    "text": text,
                    "user_id": user_id,
                    "metadata": metadata,
                    "infer": infer,
                }
            )

    monkeypatch.setattr(attachments_module, "TelegramClient", FakeTelegramClient)
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.sqlite")
    memory = Memory()

    records = await ingest_attachments(
        attachments=[
            TelegramAttachment(
                kind="video",
                file_id="tg-video-1",
                filename="clip.MOV",
                mime_type="video/quicktime",
                file_size=20_500_000,
            )
        ],
        bot_token="token",
        file_vault=vault,
        memory=memory,
        agent_runtime=None,
        customer_id="telegram_1",
        chat_id=1,
        caption=None,
    )

    assert len(records) == 1
    assert records[0]["kind"] == "unavailable_video"
    assert "clip.MOV" in records[0]["summary"]
    assert "could not be downloaded" in records[0]["summary"]
    assert memory.calls
    assert memory.calls[0]["infer"] is False
