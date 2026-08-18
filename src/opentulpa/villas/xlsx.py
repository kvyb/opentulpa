"""Small dependency-free OOXML reader for the curated MASTER VILLAS sheet."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import unicodedata
from io import BytesIO
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from opentulpa.villas.models import VillaRecord

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS = {"m": _MAIN_NS, "r": _REL_NS}
_EXPECTED_HEADERS = (
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
_MAX_XLSX_BYTES = 20 * 1024 * 1024
_MAX_ARCHIVE_ENTRIES = 512
_MAX_UNCOMPRESSED_BYTES = 96 * 1024 * 1024
_MAX_XML_PART_BYTES = 32 * 1024 * 1024
_MAX_CELL_CHARS = 100_000
_CELL_REF_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
_RANGE_RE = re.compile(r"^([A-Z]+)([1-9][0-9]*):([A-Z]+)([1-9][0-9]*)$")


class VillaWorkbookError(ValueError):
    """Workbook is unsafe or does not match the villa inventory contract."""


def _read_part(zf: ZipFile, part: str, *, label: str) -> bytes:
    try:
        info = zf.getinfo(part)
    except KeyError as exc:
        raise VillaWorkbookError(f"{label} part is missing") from exc
    if info.file_size > _MAX_XML_PART_BYTES:
        raise VillaWorkbookError(f"{label} part exceeds the supported limit")
    with zf.open(info) as stream:
        raw = stream.read(_MAX_XML_PART_BYTES + 1)
    if len(raw) > _MAX_XML_PART_BYTES:
        raise VillaWorkbookError(f"{label} part exceeds the supported limit")
    return raw


def _safe_xml(raw: bytes, *, label: str) -> ElementTree.Element:
    lowered = raw.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise VillaWorkbookError(f"unsafe declarations in {label} XML")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise VillaWorkbookError(f"invalid {label} XML") from exc


def _xml_part(zf: ZipFile, part: str, *, label: str) -> ElementTree.Element:
    return _safe_xml(_read_part(zf, part, label=label), label=label)


def _canonical_part(base_part: str, target: str) -> str:
    raw = str(target or "").replace("\\", "/")
    if raw.startswith("/"):
        part = posixpath.normpath(raw.lstrip("/"))
    else:
        part = posixpath.normpath(posixpath.join(posixpath.dirname(base_part), raw))
    if not part or part == ".." or part.startswith("../"):
        raise VillaWorkbookError("workbook relationship escapes the archive")
    return part


def _relationship_map(zf: ZipFile, rels_part: str, *, source_part: str) -> dict[str, str]:
    try:
        root = _xml_part(zf, rels_part, label="relationship")
    except VillaWorkbookError as exc:
        raise VillaWorkbookError("workbook relationships are missing or invalid") from exc
    result: dict[str, str] = {}
    for node in root.findall(f"{{{_PACKAGE_REL_NS}}}Relationship"):
        relationship_id = str(node.attrib.get("Id") or "")
        target = str(node.attrib.get("Target") or "")
        if relationship_id and target:
            result[relationship_id] = _canonical_part(source_part, target)
    return result


def _shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = _xml_part(zf, "xl/sharedStrings.xml", label="shared strings")
    except VillaWorkbookError as exc:
        if "part is missing" in str(exc):
            return []
        raise
    return ["".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")) for item in root]


def _worksheet_part(zf: ZipFile, sheet_name: str) -> str:
    workbook = _xml_part(zf, "xl/workbook.xml", label="workbook")
    relationships = _relationship_map(
        zf,
        "xl/_rels/workbook.xml.rels",
        source_part="xl/workbook.xml",
    )
    sheets = workbook.find("m:sheets", _NS)
    if sheets is None:
        raise VillaWorkbookError("workbook contains no sheets")
    for node in sheets:
        if str(node.attrib.get("name") or "") != sheet_name:
            continue
        relationship_id = str(node.attrib.get(f"{{{_REL_NS}}}id") or "")
        part = relationships.get(relationship_id)
        if not part:
            raise VillaWorkbookError("worksheet relationship is missing")
        return part
    raise VillaWorkbookError(f"worksheet not found: {sheet_name}")


def _column_number(reference: str) -> int:
    match = _CELL_REF_RE.fullmatch(reference)
    if match is None:
        raise VillaWorkbookError("invalid cell reference")
    number = 0
    for char in match.group(1):
        number = number * 26 + ord(char) - 64
    return number


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> str | int | float | bool | None:
    cell_type = str(cell.attrib.get("t") or "")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", _NS)
        value: str | int | float | bool | None = (
            "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))
            if inline is not None
            else ""
        )
    else:
        node = cell.find("m:v", _NS)
        raw = node.text if node is not None and node.text is not None else ""
        if cell_type == "s":
            try:
                value = shared[int(raw)]
            except (ValueError, IndexError) as exc:
                raise VillaWorkbookError("invalid shared string reference") from exc
        elif cell_type in {"str", "e"}:
            value = raw
        elif cell_type == "b":
            value = raw == "1"
        elif not raw:
            value = None
        else:
            try:
                numeric = float(raw)
            except ValueError:
                value = raw
            else:
                value = int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        raise VillaWorkbookError("cell text exceeds the supported limit")
    return value


def _worksheet_rows(
    zf: ZipFile,
    worksheet_part: str,
    shared: list[str],
) -> dict[int, dict[int, str | int | float | bool | None]]:
    root = _xml_part(zf, worksheet_part, label="worksheet")
    rows: dict[int, dict[int, str | int | float | bool | None]] = {}
    for row in root.findall(".//m:sheetData/m:row", _NS):
        try:
            row_number = int(str(row.attrib.get("r") or "0"))
        except ValueError as exc:
            raise VillaWorkbookError("invalid worksheet row number") from exc
        if row_number <= 0 or row_number > 100_000:
            raise VillaWorkbookError("worksheet row is outside the supported range")
        cells: dict[int, str | int | float | bool | None] = {}
        for cell in row.findall("m:c", _NS):
            reference = str(cell.attrib.get("r") or "")
            column = _column_number(reference)
            if column > 256:
                raise VillaWorkbookError("worksheet exceeds the supported column count")
            cells[column] = _cell_value(cell, shared)
        rows[row_number] = cells
    return rows


def _table_bounds(zf: ZipFile, worksheet_part: str, worksheet_rows: dict[int, dict[int, Any]]) -> tuple[int, int, int, int]:
    rels_part = posixpath.join(
        posixpath.dirname(worksheet_part),
        "_rels",
        posixpath.basename(worksheet_part) + ".rels",
    )
    table_ranges: list[str] = []
    try:
        relationships = _relationship_map(zf, rels_part, source_part=worksheet_part)
    except VillaWorkbookError:
        relationships = {}
    for part in relationships.values():
        if "/tables/" not in f"/{part}":
            continue
        try:
            table = _xml_part(zf, part, label="table")
        except VillaWorkbookError as exc:
            if "part is missing" in str(exc):
                continue
            raise
        reference = str(table.attrib.get("ref") or "")
        if reference:
            table_ranges.append(reference)
    for reference in table_ranges:
        match = _RANGE_RE.fullmatch(reference)
        if match is None:
            continue
        first_col = _column_number(f"{match.group(1)}{match.group(2)}")
        last_col = _column_number(f"{match.group(3)}{match.group(4)}")
        first_row = int(match.group(2))
        last_row = int(match.group(4))
        header = tuple(
            str(worksheet_rows.get(first_row, {}).get(column) or "").strip()
            for column in range(first_col, last_col + 1)
        )
        if header == _EXPECTED_HEADERS:
            return first_col, first_row, last_col, last_row
    # The curated workbook contract has its header on row 4. This fallback keeps
    # sanitized fixtures and table-less exports importable without guessing columns.
    for row_number, cells in worksheet_rows.items():
        header = tuple(str(cells.get(column) or "").strip() for column in range(1, 24))
        if header == _EXPECTED_HEADERS:
            return 1, row_number, 23, max(worksheet_rows, default=row_number)
    raise VillaWorkbookError("MASTER VILLAS headers do not match the expected schema")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    normalized = re.sub(r"[^0-9.\-]+", "", str(value))
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _normalize_text(value)
    if normalized in {"yes", "true", "1", "y"}:
        return True
    if normalized in {"no", "false", "0", "n"}:
        return False
    return None


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    return " ".join(normalized.split())


def _second_nonempty_line(value: Any) -> str:
    lines = [_normalize_text(line) for line in _text(value).splitlines() if _normalize_text(line)]
    return lines[1] if len(lines) > 1 else (lines[0] if lines else "")


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(source_row: int, values: dict[str, Any]) -> VillaRecord:
    property_name = _text(values["Property"])
    if not property_name:
        raise VillaWorkbookError(f"property name is missing on row {source_row}")
    identity = {
        "source_sheet": _normalize_text(values["Source Sheet"]),
        "owner_agency": _normalize_text(values["Owner / Agency"]),
        "property": _normalize_text(property_name),
        "location": _normalize_text(values["Location"]),
        "type": _normalize_text(values["Type"]),
        "bedrooms": _number(values["Bedrooms"]),
        "bathrooms": _number(values["Bathrooms"]),
        "map_link": _normalize_text(values["Map / Link"]),
        "listing_hint": _second_nonempty_line(values["Raw Notes"]),
    }
    fingerprint = _json_hash(identity)
    source_values = {header: values.get(header) for header in _EXPECTED_HEADERS}
    return VillaRecord(
        identity_fingerprint=fingerprint,
        source_key=f"master-villas:v1:{fingerprint}",
        source_hash=_json_hash(source_values),
        source_row=source_row,
        property_name=property_name,
        location=_text(values["Location"]),
        owner_agency=_text(values["Owner / Agency"]),
        property_type=_text(values["Type"]),
        bedrooms=_number(values["Bedrooms"]),
        bathrooms=_number(values["Bathrooms"]),
        monthly_idr=_integer(values["Monthly (IDR)"]),
        yearly_idr=_integer(values["Yearly (IDR)"]),
        weekly_idr=_integer(values["Weekly (IDR)"]),
        daily_idr=_integer(values["Daily (IDR)"]),
        availability_text=_text(values["Available"]),
        pet_friendly=_boolean(values["Pet Friendly"]),
        pool=_boolean(values["Pool"]),
        parking=_boolean(values["Parking"]),
        construction_text=_text(values["Construction"]),
        deposit_monthly_idr=_integer(values["Deposit Monthly"]),
        deposit_yearly_idr=_integer(values["Deposit Yearly"]),
        commission_text=_text(values["Commission"]),
        included_text=_text(values["Included"]),
        excluded_text=_text(values["Excluded"]),
        map_link=_text(values["Map / Link"]),
        source_sheet=_text(values["Source Sheet"]),
        raw_notes=_text(values["Raw Notes"]),
        source_values=source_values,
    )


def parse_master_villas_xlsx(
    raw_bytes: bytes,
    *,
    sheet_name: str = "MASTER VILLAS",
) -> list[VillaRecord]:
    """Parse and validate the curated master sheet without loading spreadsheet code."""

    if not raw_bytes or len(raw_bytes) > _MAX_XLSX_BYTES:
        raise VillaWorkbookError("workbook size is unsupported")
    try:
        with ZipFile(BytesIO(raw_bytes)) as zf:
            infos = zf.infolist()
            if len(infos) > _MAX_ARCHIVE_ENTRIES:
                raise VillaWorkbookError("workbook contains too many archive entries")
            total_size = 0
            for info in infos:
                name = str(info.filename).replace("\\", "/")
                if name.startswith("/") or name == ".." or name.startswith("../") or "/../" in name:
                    raise VillaWorkbookError("workbook contains an unsafe archive path")
                if info.flag_bits & 0x1:
                    raise VillaWorkbookError("encrypted workbooks are unsupported")
                total_size += int(info.file_size)
            if total_size > _MAX_UNCOMPRESSED_BYTES:
                raise VillaWorkbookError("workbook expands beyond the supported limit")
            worksheet_part = _worksheet_part(zf, sheet_name)
            shared = _shared_strings(zf)
            rows = _worksheet_rows(zf, worksheet_part, shared)
            first_col, header_row, last_col, last_row = _table_bounds(
                zf,
                worksheet_part,
                rows,
            )
    except BadZipFile as exc:
        raise VillaWorkbookError("file is not a valid XLSX workbook") from exc

    if last_col - first_col + 1 != len(_EXPECTED_HEADERS):
        raise VillaWorkbookError("MASTER VILLAS has an unexpected column count")
    records: list[VillaRecord] = []
    fingerprints: set[str] = set()
    for row_number in range(header_row + 1, last_row + 1):
        cells = rows.get(row_number, {})
        values = {
            header: cells.get(first_col + offset)
            for offset, header in enumerate(_EXPECTED_HEADERS)
        }
        if not any(value not in (None, "") for value in values.values()):
            continue
        record = _record(row_number, values)
        if record.identity_fingerprint in fingerprints:
            raise VillaWorkbookError(f"duplicate villa identity on row {row_number}")
        fingerprints.add(record.identity_fingerprint)
        records.append(record)
    if not records:
        raise VillaWorkbookError("MASTER VILLAS contains no inventory rows")
    return records
