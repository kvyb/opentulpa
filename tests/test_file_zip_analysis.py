from __future__ import annotations

import stat
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

import opentulpa.business_knowledge.extraction as extraction
from opentulpa.context.file_vault import FileVaultService
from opentulpa.files.analysis import FileAnalysisJobArguments, FileAnalysisService
from opentulpa.jobs import JobExecutionContext


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _inspect_zip(tmp_path: Path, raw_bytes: bytes) -> dict[str, object]:
    files = FileVaultService(
        root_dir=tmp_path / "vault",
        db_path=tmp_path / "files.sqlite",
    )
    record = files.ingest_file(
        customer_id="tenant-a",
        chat_id=123,
        kind="document",
        telegram_file_id="telegram-file",
        original_filename="v2-redesign-parcel.zip",
        mime_type="application/zip",
        caption="what's this",
        raw_bytes=raw_bytes,
    )
    return FileAnalysisService(files).inspect(
        tenant_id="tenant-a",
        file_id=str(record["id"]),
        question="What is this redesign parcel?",
    )


def test_zip_inspection_lists_members_and_reads_parcel_markdown(tmp_path: Path) -> None:
    result = _inspect_zip(
        tmp_path,
        _zip_bytes(
            {
                "v2-redesign-parcel/CONTEXT.md": b"# Context\nImplement the validated mobile redesign.",
                "v2-redesign-parcel/SOURCES.md": b"# Sources\n- User research",
                "v2-redesign-parcel/README.md": (
                    b"# V2 Redesign Parcel\nA portable handoff for implementing the interface redesign."
                ),
            }
        ),
    )

    assert result["source_kind"] == "zip_archive"
    assert result["warnings"] == []
    sections = result["sections"]
    assert isinstance(sections, list)
    assert sections[0]["source_kind"] == "zip_archive"
    assert "v2-redesign-parcel/README.md" in sections[0]["preview"]
    assert sections[1]["metadata"]["archive_member"] == "v2-redesign-parcel/README.md"
    previews = "\n".join(str(section["preview"]) for section in sections)
    assert "portable handoff for implementing the interface redesign" in previews
    assert "Implement the validated mobile redesign" in previews


@pytest.mark.asyncio
async def test_zip_analysis_job_returns_parcel_text_instead_of_unsupported(
    tmp_path: Path,
) -> None:
    files = FileVaultService(
        root_dir=tmp_path / "vault",
        db_path=tmp_path / "files.sqlite",
    )
    record = files.ingest_file(
        customer_id="tenant-a",
        chat_id=123,
        kind="document",
        telegram_file_id="telegram-file",
        original_filename="v2-redesign-parcel.zip",
        mime_type="application/zip",
        caption="what's this",
        raw_bytes=_zip_bytes(
            {
                "v2-redesign-parcel/README.md": (
                    b"# V2 Redesign Parcel\nA portable handoff for implementing the interface redesign."
                )
            }
        ),
    )
    progress_events: list[dict[str, Any]] = []

    async def progress(payload: dict[str, Any]) -> None:
        progress_events.append(payload)

    service = FileAnalysisService(files)
    result = await service._analyze_job(  # noqa: SLF001
        FileAnalysisJobArguments(
            file_id=str(record["id"]),
            instruction="Explain this redesign parcel",
        ),
        JobExecutionContext(
            tenant_id="tenant-a",
            job_id="job-zip",
            idempotency_key="analyze-zip",
            attempt=1,
            _emit_progress=progress,
        ),
    )

    assert result.data["source_kind"] == "zip_archive"
    assert result.data["section_count"] == 2
    assert "V2 Redesign Parcel" in result.data["summary"]
    assert "No extractable text was available" not in result.data["summary"]
    assert progress_events == [{"stage": "parsing"}, {"stage": "completed"}]


def test_zip_inspection_skips_unsafe_special_and_nested_members(tmp_path: Path) -> None:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("README.md", b"# Safe parcel\nInspect this content.")
        archive.writestr("../escape.md", b"must not be inspected")
        archive.writestr("nested.zip", _zip_bytes({"hidden.md": b"nested content"}))
        symlink = ZipInfo("linked.md")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "README.md")

    result = _inspect_zip(tmp_path, output.getvalue())

    previews = "\n".join(str(section["preview"]) for section in result["sections"])
    warnings = result["warnings"]
    assert "Inspect this content" in previews
    assert "must not be inspected" not in previews
    assert "nested content" not in previews
    assert any("unsafe ZIP member path" in warning for warning in warnings)
    assert any("non-regular ZIP member" in warning for warning in warnings)
    assert any("nested ZIP member" in warning for warning in warnings)


def test_zip_inspection_enforces_entry_and_expanded_size_limits(monkeypatch) -> None:
    raw_bytes = _zip_bytes({"one.md": b"one", "two.md": b"two"})
    monkeypatch.setattr(extraction, "_MAX_ZIP_ENTRIES", 1)

    sections, warnings, source_kind = extraction.extract_source_sections(
        record={
            "id": "file_zip",
            "original_filename": "bundle.zip",
            "mime_type": "application/zip",
        },
        raw_bytes=raw_bytes,
    )

    assert sections == []
    assert source_kind == "zip_archive"
    assert warnings == ["ZIP archive has too many entries: 2 > 1"]

    monkeypatch.setattr(extraction, "_MAX_ZIP_ENTRIES", 512)
    monkeypatch.setattr(extraction, "_MAX_ZIP_EXPANDED_BYTES", 5)
    sections, warnings, source_kind = extraction.extract_source_sections(
        record={
            "id": "file_zip",
            "original_filename": "bundle.zip",
            "mime_type": "application/zip",
        },
        raw_bytes=raw_bytes,
    )

    assert sections == []
    assert source_kind == "zip_archive"
    assert warnings == ["ZIP archive expands beyond the inspection limit: 6 > 5 bytes"]


def test_zip_inspection_rejects_suspicious_compression_ratio(monkeypatch) -> None:
    monkeypatch.setattr(extraction, "_MIN_ZIP_RATIO_CHECK_BYTES", 1)
    monkeypatch.setattr(extraction, "_MAX_ZIP_COMPRESSION_RATIO", 2)

    sections, warnings, source_kind = extraction.extract_source_sections(
        record={
            "id": "file_zip",
            "original_filename": "bundle.zip",
            "mime_type": "application/zip",
        },
        raw_bytes=_zip_bytes({"large.md": b"a" * 1_000}),
    )

    assert sections == []
    assert source_kind == "zip_archive"
    assert warnings == ["ZIP archive contains a suspiciously compressed member: large.md"]
