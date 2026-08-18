from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from opentulpa.villas import (
    VillaInventoryService,
    VillaRepository,
    VillaWorkbookError,
    parse_master_villas_xlsx,
)

HEADERS = (
    "Property",
    "Location",
    "Owner / Agency",
    "Type",
    "Bedrooms",
    "Bathrooms",
    "Monthly (IDR)",
    "Yearly (IDR)",
    "Weekly (IDR)",
    "Daily (IDR)",
    "Available",
    "Pet Friendly",
    "Pool",
    "Parking",
    "Construction",
    "Deposit Monthly",
    "Deposit Yearly",
    "Commission",
    "Included",
    "Excluded",
    "Map / Link",
    "Source Sheet",
    "Raw Notes",
)


def _column(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell(reference: str, value: object) -> str:
    if isinstance(value, int | float):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _workbook(rows: list[list[object]]) -> bytes:
    sheet_rows: list[str] = []
    numbered_rows: list[tuple[int, list[object]]] = [(4, list(HEADERS))]
    numbered_rows.extend((row_number, values) for row_number, values in enumerate(rows, start=5))
    for row_number, values in numbered_rows:
        cells = "".join(
            _cell(f"{_column(index)}{row_number}", value)
            for index, value in enumerate(values, start=1)
            if value not in (None, "")
        )
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')
    end_row = 4 + len(rows)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <sheets><sheet name="MASTER VILLAS" sheetId="1" r:id="sheet"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="sheet" Target="worksheets/sheet.xml"
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
            </Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet.xml",
            f"""<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <sheetData>{''.join(sheet_rows)}</sheetData></worksheet>""",
        )
        archive.writestr(
            "xl/worksheets/_rels/sheet.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="table" Target="../tables/table.xml"
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table"/>
            </Relationships>""",
        )
        archive.writestr(
            "xl/tables/table.xml",
            f'<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ref="A4:W{end_row}"/>',
        )
    return buffer.getvalue()


def _row(name: str, *, price: int = 30_000_000, notes: str = "Heading\nListing one") -> list[object]:
    return [
        name,
        "Pererenan",
        "Agency",
        "Villa",
        2,
        2,
        price,
        300_000_000,
        "",
        "",
        "Now",
        "Yes",
        "Yes",
        "Yes",
        "No construction",
        5_000_000,
        10_000_000,
        "internal",
        "WiFi; cleaning",
        "Electricity",
        "https://maps.example/villa",
        "MASTER",
        notes,
    ]


def test_parser_reads_master_table_and_separates_duplicate_names() -> None:
    records = parse_master_villas_xlsx(
        _workbook(
            [
                _row("4BR PERERENAN", notes="Heading\nRooftop terrace"),
                _row("4BR PERERENAN", notes="Heading\nPrivate gym"),
            ]
        )
    )

    assert len(records) == 2
    assert records[0].identity_fingerprint != records[1].identity_fingerprint
    assert records[0].monthly_idr == 30_000_000
    assert records[0].pet_friendly is True


def test_parser_rejects_wrong_sheet() -> None:
    with pytest.raises(VillaWorkbookError, match="worksheet not found"):
        parse_master_villas_xlsx(_workbook([_row("Villa One")]), sheet_name="Other")


def test_repository_is_idempotent_and_preserves_manual_state(tmp_path: Path) -> None:
    raw = _workbook([_row("Villa One")])
    records = parse_master_villas_xlsx(raw)
    repository = VillaRepository(tmp_path / "villas" / "villas.db")

    first = repository.import_records(
        tenant_id="tenant-a",
        file_id="file-one",
        filename="villas.xlsx",
        sheet_name="MASTER VILLAS",
        source_sha256="sha-one",
        records=records,
    )
    with repository._conn() as connection:
        connection.execute(
            "UPDATE villas SET manual_status='reserved', manual_overrides_json=? WHERE tenant_id=?",
            ('{"monthly_idr":31000000}', "tenant-a"),
        )
        connection.commit()
    second = repository.import_records(
        tenant_id="tenant-a",
        file_id="file-two",
        filename="villas-new.xlsx",
        sheet_name="MASTER VILLAS",
        source_sha256="sha-two",
        records=parse_master_villas_xlsx(_workbook([_row("Villa One", price=32_000_000)])),
    )

    assert first.inserted_count == 1
    assert second.updated_count == 1
    assert repository.counts(tenant_id="tenant-a") == {
        "total": 1,
        "active": 1,
        "source_records": 2,
    }
    villa = repository.list_villas(tenant_id="tenant-a")[0]
    assert villa["monthly_idr"] == 32_000_000
    assert villa["manual_status"] == "reserved"
    assert villa["manual_overrides_json"] == '{"monthly_idr":31000000}'
    assert repository.list_villas(tenant_id="tenant-b") == []


class _Files:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def get_file(self, customer_id: str, file_id: str) -> dict[str, object] | None:
        if (customer_id, file_id) != ("owner", "file-one"):
            return None
        return {"original_filename": "villas.xlsx", "mime_type": "application/octet-stream"}

    def read_file_bytes(self, customer_id: str, file_id: str) -> bytes | None:
        return self.raw if (customer_id, file_id) == ("owner", "file-one") else None


def test_service_reads_only_the_authenticated_tenant_file(tmp_path: Path) -> None:
    raw = _workbook([_row("Villa One")])
    repository = VillaRepository(tmp_path / "villas.db")
    service = VillaInventoryService(repository=repository, file_vault=_Files(raw))

    result = service.import_file(tenant_id="owner", file_id="file-one")

    assert result.parsed_count == 1
    assert repository.counts(tenant_id="owner")["active"] == 1
    with pytest.raises(KeyError):
        service.import_file(tenant_id="another", file_id="file-one")
