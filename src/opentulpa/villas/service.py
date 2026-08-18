"""File-vault backed villa inventory import service."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from opentulpa.villas.models import VillaImportResult
from opentulpa.villas.repository import VillaRepository
from opentulpa.villas.xlsx import parse_master_villas_xlsx


class VillaFileVault(Protocol):
    def get_file(self, customer_id: str, file_id: str) -> dict[str, Any] | None: ...

    def read_file_bytes(self, customer_id: str, file_id: str) -> bytes | None: ...


class VillaInventoryService:
    """Import tenant-owned XLSX inventory into the separate villas database."""

    def __init__(self, *, repository: VillaRepository, file_vault: VillaFileVault) -> None:
        self.repository = repository
        self.file_vault = file_vault

    def import_file(
        self,
        *,
        tenant_id: str,
        file_id: str,
        sheet_name: str = "MASTER VILLAS",
    ) -> VillaImportResult:
        tenant = str(tenant_id or "").strip()
        source_file = str(file_id or "").strip()
        source_sheet = str(sheet_name or "").strip()
        if not tenant or not source_file or not source_sheet:
            raise ValueError("tenant_id, file_id, and sheet_name are required")
        metadata = self.file_vault.get_file(tenant, source_file)
        raw_bytes = self.file_vault.read_file_bytes(tenant, source_file)
        if metadata is None or raw_bytes is None:
            raise KeyError(source_file)
        filename = str(metadata.get("original_filename") or source_file)
        mime_type = str(metadata.get("mime_type") or "").partition(";")[0].strip().lower()
        if not filename.casefold().endswith(".xlsx") and mime_type != (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            raise ValueError("villa inventory source must be an XLSX workbook")
        records = parse_master_villas_xlsx(raw_bytes, sheet_name=source_sheet)
        return self.repository.import_records(
            tenant_id=tenant,
            file_id=source_file,
            filename=filename,
            sheet_name=source_sheet,
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            records=records,
        )
