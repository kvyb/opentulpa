from __future__ import annotations

from opentulpa.interfaces.telegram.attachments import extract_attachments


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
