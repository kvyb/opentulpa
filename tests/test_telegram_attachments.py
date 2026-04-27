from __future__ import annotations

from opentulpa.interfaces.telegram.attachments import (
    build_uploaded_files_context,
    extract_attachments,
)


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
    assert "uploaded_file_inspect_structure" in context
