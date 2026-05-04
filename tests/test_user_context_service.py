from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from opentulpa.business_knowledge.service import BusinessKnowledgeService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.user_context import USER_CONTEXT_SCOPE_TYPE, UserContextService


class _Oracle:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def extract_intent(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "mode": "specific_fact",
            "target_terms": [str(kwargs.get("query", ""))],
            "qualifier_terms": [],
            "ignore_terms": [],
        }

    def answer(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "Grounded answer from user context."


def _services(tmp_path: Path) -> tuple[FileVaultService, _Oracle, UserContextService]:
    vault = FileVaultService(
        root_dir=tmp_path / "vault",
        db_path=tmp_path / "vault.db",
    )
    oracle = _Oracle()
    knowledge = BusinessKnowledgeService(
        root_dir=tmp_path / "knowledge",
        db_path=tmp_path / "knowledge.db",
        file_vault=vault,
        oracle_client=oracle,  # type: ignore[arg-type]
    )
    service = UserContextService(
        db_path=tmp_path / "user_context.db",
        knowledge_service=knowledge,
        file_vault=vault,
    )
    return vault, oracle, service


def test_user_context_add_files_indexes_user_context_scope(tmp_path: Path) -> None:
    vault, _oracle, service = _services(tmp_path)
    record = vault.ingest_file(
        customer_id="cust_1",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="blog.md",
        mime_type="text/markdown",
        caption=None,
        raw_bytes=b"# Voice\nShort punchy hooks.",
    )

    result = service.add_files(customer_id="cust_1", file_ids=[record["id"]])

    assert result["scope_type"] == USER_CONTEXT_SCOPE_TYPE
    assert result["scope_id"] == "cust_1"
    assert result["indexed"]["scope_type"] == USER_CONTEXT_SCOPE_TYPE
    assert result["sources"][0]["file_id"] == record["id"]
    assert result["sources"][0]["status"] == "indexed"
    assert result["source_refs"][0]["file_id"] == record["id"]
    assert result["source_refs"][0]["filename"] == "blog.md"
    assert result["source_refs"][0]["source_kind"] == "local_source"


def test_user_context_indexes_prepared_summary_for_scanned_pdf(tmp_path: Path) -> None:
    vault, _oracle, service = _services(tmp_path)
    pdf = PdfWriter()
    pdf.add_blank_page(width=200, height=200)
    raw_pdf = BytesIO()
    pdf.write(raw_pdf)
    record = vault.ingest_file(
        customer_id="cust_1",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="scan.pdf",
        mime_type="application/pdf",
        caption=None,
        raw_bytes=raw_pdf.getvalue(),
    )
    vault.set_ai_summary(
        "cust_1",
        record["id"],
        "Visible text: BLOG SPRINT CTA. Visual facts from scanned PDF.",
    )

    result = service.add_files(customer_id="cust_1", file_ids=[record["id"]])

    source = result["indexed"]["sources"][0]
    assert source["status"] == "indexed"
    assert source["source_kind"] == "derived_from_media"
    assert source["section_count"] == 1
    assert result["source_refs"][0]["filename"] == "scan.pdf"
    assert result["source_refs"][0]["source_kind"] == "derived_from_media"
    assert result["source_refs"][0]["locator"] == "derived media summary"


def test_user_context_indexes_pdf_text_and_prepared_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Page:
        def extract_text(self) -> str:
            return "Local PDF text: retainer floor is $1200."

    class _Reader:
        def __init__(self, _raw: Any) -> None:
            self.pages = [_Page()]

    monkeypatch.setattr("pypdf.PdfReader", _Reader)
    vault, _oracle, service = _services(tmp_path)
    record = vault.ingest_file(
        customer_id="cust_1",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="deck.pdf",
        mime_type="application/pdf",
        caption=None,
        raw_bytes=b"%PDF fake enough for monkeypatched reader",
    )
    vault.set_ai_summary(
        "cust_1",
        record["id"],
        "Visible slide note: BLOG SPRINT CTA appears in the hero chart.",
    )

    result = service.add_files(customer_id="cust_1", file_ids=[record["id"]])

    source = result["indexed"]["sources"][0]
    assert source["status"] == "indexed"
    assert source["source_kind"] == "local_source"
    assert source["section_count"] == 2
    source_kinds = {ref["source_kind"] for ref in result["source_refs"]}
    assert source_kinds == {"local_source", "derived_from_media"}
    assert {ref["locator"] for ref in result["source_refs"]} == {"page 1", "derived media summary"}


def test_user_context_archive_excludes_source_from_queries(tmp_path: Path) -> None:
    vault, oracle, service = _services(tmp_path)
    keep = vault.ingest_file(
        customer_id="cust_1",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="keep.md",
        mime_type="text/markdown",
        caption=None,
        raw_bytes=b"Keep this source.",
    )
    archived = vault.ingest_file(
        customer_id="cust_1",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="old.md",
        mime_type="text/markdown",
        caption=None,
        raw_bytes=b"Do not use this old source.",
    )
    service.add_files(customer_id="cust_1", file_ids=[keep["id"], archived["id"]])
    service.archive_sources(customer_id="cust_1", file_ids=[archived["id"]])

    result = service.query(customer_id="cust_1", query="what should I use?")

    assert result["ok"] is True
    assert result["answer_extract"] == "Grounded answer from user context."
    assert result["source_refs"][0]["filename"] == "keep.md"
    source_pack = oracle.calls[-1]["source_pack"]
    assert "keep.md" in source_pack
    assert "old.md" not in source_pack
