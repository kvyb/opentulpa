"""Typed villa inventory records and import summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VillaRecord:
    """One normalized source row from the master villa workbook."""

    identity_fingerprint: str
    source_key: str
    source_hash: str
    source_row: int
    property_name: str
    location: str
    owner_agency: str
    property_type: str
    bedrooms: float | None
    bathrooms: float | None
    monthly_idr: int | None
    yearly_idr: int | None
    weekly_idr: int | None
    daily_idr: int | None
    availability_text: str
    pet_friendly: bool | None
    pool: bool | None
    parking: bool | None
    construction_text: str
    deposit_monthly_idr: int | None
    deposit_yearly_idr: int | None
    commission_text: str
    included_text: str
    excluded_text: str
    map_link: str
    source_sheet: str
    raw_notes: str
    source_values: dict[str, str | int | float | bool | None]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VillaImportResult:
    """Public summary of one atomic workbook import."""

    import_run_id: str
    file_id: str
    filename: str
    sheet_name: str
    source_sha256: str
    parsed_count: int
    inserted_count: int
    updated_count: int
    unchanged_count: int
    missing_count: int
    replayed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
