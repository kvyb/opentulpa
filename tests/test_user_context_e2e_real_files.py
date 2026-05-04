from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from opentulpa.api.routes.user_context import register_user_context_routes
from opentulpa.business_knowledge.service import BusinessKnowledgeService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.context.user_context import UserContextService

CUSTOMER_ID = "telegram_real_files"


class _Oracle:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.intent_calls: list[dict[str, Any]] = []

    def extract_intent(self, **kwargs: Any) -> dict[str, Any]:
        self.intent_calls.append(kwargs)
        return {
            "mode": "specific_fact",
            "target_terms": ["Launch package"],
            "qualifier_terms": ["CTA", "retainer"],
            "ignore_terms": [],
        }

    def answer(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        source_pack = str(kwargs.get("source_pack", ""))
        if (
            "Launch package" in source_pack
            and "Retainer offer" in source_pack
            and "Visual CTA" in source_pack
        ):
            return "Use the launch package, retainer offer, and the visual CTA from the image."
        return "NO_SOURCE"


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    async def summarize_uploaded_blob(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        filename = str(kwargs.get("filename") or "")
        if filename.endswith(".png"):
            return "Visual CTA: Start your blog sprint. Palette: black text on yellow button."
        if filename.endswith(".mp4"):
            return "00:00-00:30 Transcript: creator says to reuse this clip as launch proof."
        return "Media summary unavailable."

    def record_observability_event(self, *, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})


def _services(tmp_path: Path) -> tuple[FileVaultService, _Oracle, _Runtime, UserContextService]:
    vault = FileVaultService(root_dir=tmp_path / "vault", db_path=tmp_path / "vault.db")
    oracle = _Oracle()
    knowledge = BusinessKnowledgeService(
        root_dir=tmp_path / "knowledge",
        db_path=tmp_path / "knowledge.db",
        file_vault=vault,
        oracle_client=oracle,  # type: ignore[arg-type]
    )
    user_context = UserContextService(
        db_path=tmp_path / "user_context.db",
        knowledge_service=knowledge,
        file_vault=vault,
    )
    runtime = _Runtime()
    return vault, oracle, runtime, user_context


def _client(
    *,
    vault: FileVaultService,
    runtime: _Runtime,
    user_context: UserContextService,
) -> TestClient:
    app = FastAPI()
    register_user_context_routes(
        app,
        get_user_context_service=lambda: user_context,
        get_file_vault=lambda: vault,
        get_agent_runtime=lambda: runtime,
    )
    return TestClient(app)


def _xlsx_bytes() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Offers"
    ws.append(["Offer", "Price", "Notes"])
    ws.append(["Launch package", "$900", "Includes blog ideas and scripts"])
    out = BytesIO()
    wb.save(out)
    return out.getvalue()


def _docx_bytes() -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Retainer offer: weekly scenario writing and idea bank refresh.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    out = BytesIO()
    with ZipFile(out, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return out.getvalue()


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xf8\x0f"
        b"\x00\x01\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _ingest_real_files(vault: FileVaultService) -> list[str]:
    records = [
        vault.ingest_file(
            customer_id=CUSTOMER_ID,
            chat_id=None,
            kind="document",
            telegram_file_id=None,
            original_filename="offers.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            caption=None,
            raw_bytes=_xlsx_bytes(),
        ),
        vault.ingest_file(
            customer_id=CUSTOMER_ID,
            chat_id=None,
            kind="document",
            telegram_file_id=None,
            original_filename="positioning.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            caption=None,
            raw_bytes=_docx_bytes(),
        ),
        vault.ingest_file(
            customer_id=CUSTOMER_ID,
            chat_id=None,
            kind="photo",
            telegram_file_id=None,
            original_filename="cta.png",
            mime_type="image/png",
            caption=None,
            raw_bytes=_png_bytes(),
        ),
    ]
    return [str(record["id"]) for record in records]


def test_user_context_routes_index_and_query_mixed_real_files(tmp_path: Path) -> None:
    vault, oracle, runtime, user_context = _services(tmp_path)
    file_ids = _ingest_real_files(vault)
    client = _client(vault=vault, runtime=runtime, user_context=user_context)

    add_response = client.post(
        "/internal/user_context/add_files",
        json={"customer_id": CUSTOMER_ID, "file_ids": file_ids},
    )
    query_response = client.post(
        "/internal/user_context/query",
        json={"customer_id": CUSTOMER_ID, "query": "launch package retainer visual CTA"},
    )

    assert add_response.status_code == 200
    indexed = add_response.json()["indexed"]["sources"]
    assert {source["source_kind"] for source in indexed} == {
        "structured_table",
        "local_source",
        "derived_from_media",
    }
    assert [call["filename"] for call in runtime.calls] == ["cta.png"]
    assert query_response.status_code == 200
    assert query_response.json()["answer_extract"] == (
        "Use the launch package, retainer offer, and the visual CTA from the image."
    )
    source_refs = query_response.json()["source_refs"]
    assert {item["filename"] for item in source_refs} >= {"offers.xlsx", "positioning.docx", "cta.png"}
    assert any(item.get("sheet") == "Offers" and item.get("locator") for item in source_refs)
    assert any(item.get("locator") == "derived media summary" for item in source_refs)
    source_pack = oracle.calls[-1]["source_pack"]
    assert "Launch package" in source_pack
    assert "Retainer offer" in source_pack
    assert "Visual CTA" in source_pack
    assert query_response.json()["diagnostics"]["source_pack"]["supplemental_section_count"] == 2
    assert [item["event"] for item in runtime.events] == [
        "user_context.media_prepare_succeeded",
        "user_context.add_files",
        "user_context.query",
    ]
    assert runtime.events[1]["file_count"] == 3
    assert runtime.events[1]["prepared_count"] == 1


def test_user_context_promotes_real_files_to_intake_scope(tmp_path: Path) -> None:
    vault, oracle, runtime, user_context = _services(tmp_path)
    file_ids = _ingest_real_files(vault)
    client = _client(vault=vault, runtime=runtime, user_context=user_context)
    client.post(
        "/internal/user_context/add_files",
        json={"customer_id": CUSTOMER_ID, "file_ids": file_ids},
    )

    promote_response = client.post(
        "/internal/user_context/promote_to_intake",
        json={"customer_id": CUSTOMER_ID, "workflow_id": "iwf_real_files", "file_ids": file_ids},
    )
    query_result = user_context.knowledge_service.query(
        customer_id=CUSTOMER_ID,
        scope_type="intake_workflow",
        scope_id="iwf_real_files",
        query="launch package retainer visual CTA",
    )

    assert promote_response.status_code == 200
    assert promote_response.json()["indexed"]["scope_type"] == "intake_workflow"
    assert {item["filename"] for item in promote_response.json()["source_refs"]} == {
        "offers.xlsx",
        "positioning.docx",
        "cta.png",
    }
    assert query_result.ok is True
    assert "Launch package" in oracle.calls[-1]["source_pack"]
    assert "Retainer offer" in oracle.calls[-1]["source_pack"]
    assert "Visual CTA" in oracle.calls[-1]["source_pack"]
