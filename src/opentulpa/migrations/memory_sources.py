"""Offline adapters for reading legacy Mem0 records without model dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opentulpa.migrations.models import LegacyMemoryRecord


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _record_from_mapping(value: dict[str, Any], *, fallback_id: object = "") -> LegacyMemoryRecord:
    metadata = _mapping(value.get("metadata"))
    for key, item in value.items():
        if key not in {
            "content",
            "created_at",
            "data",
            "id",
            "memory",
            "memory_id",
            "metadata",
            "tenant_id",
            "text",
            "updated_at",
            "user_id",
        }:
            metadata.setdefault(str(key), item)
    if value.get("opentulpa_migration_disabled"):
        metadata["opentulpa_migration_disabled"] = True
    tenant_id = str(
        value.get("user_id")
        or value.get("tenant_id")
        or metadata.get("user_id")
        or metadata.get("tenant_id")
        or ""
    )
    content = str(
        value.get("memory") or value.get("data") or value.get("text") or value.get("content") or ""
    )
    legacy_id = str(value.get("id") or value.get("memory_id") or fallback_id or "")
    created_at = value.get("created_at") or metadata.get("created_at")
    updated_at = value.get("updated_at") or metadata.get("updated_at")
    return LegacyMemoryRecord(
        legacy_id=legacy_id,
        tenant_id=tenant_id,
        content=content,
        created_at=str(created_at) if created_at is not None else None,
        updated_at=str(updated_at) if updated_at is not None else None,
        metadata=metadata,
    )


class JsonMemorySource:
    """Read an explicit Mem0 JSON export for rehearsals and deterministic tests."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve()

    def records(self) -> list[LegacyMemoryRecord]:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        raw_records = payload.get("memories", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_records, list):
            raise ValueError("memory export must be a list or an object containing memories")
        records: list[LegacyMemoryRecord] = []
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                records.append(
                    LegacyMemoryRecord(
                        legacy_id=f"json-index-{index}",
                        tenant_id="",
                        content="",
                    )
                )
                continue
            records.append(_record_from_mapping(raw, fallback_id=f"json-index-{index}"))
        return records

    def disable(self, record: LegacyMemoryRecord, *, reason: str) -> bool:
        # JSON exports are immutable rehearsal inputs. Invalid records remain in the report.
        _ = (record, reason)
        return False


class QdrantMem0Source:
    """Read and mark records directly in Mem0's embedded Qdrant collection."""

    def __init__(self, *, path: Path, collection_name: str = "mem0") -> None:
        self._path = path.expanduser().resolve()
        self._collection_name = str(collection_name or "mem0").strip()
        self._client: Any | None = None
        self._point_ids: dict[str, Any] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=str(self._path))
        return self._client

    def records(self) -> list[LegacyMemoryRecord]:
        if not self._path.exists():
            return []
        client = self._get_client()
        if not client.collection_exists(self._collection_name):
            return []
        records: list[LegacyMemoryRecord] = []
        offset: Any | None = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                point_id = str(point.id)
                payload = _mapping(point.payload)
                record = _record_from_mapping(payload, fallback_id=point_id)
                self._point_ids[record.legacy_id] = point.id
                records.append(record)
            if offset is None:
                break
        return records

    def disable(self, record: LegacyMemoryRecord, *, reason: str) -> bool:
        client = self._get_client()
        point_id = self._point_ids.get(record.legacy_id, record.legacy_id)
        client.set_payload(
            collection_name=self._collection_name,
            payload={
                "opentulpa_migration_disabled": True,
                "opentulpa_migration_error": str(reason or "invalid legacy memory")[:1000],
            },
            points=[point_id],
            wait=True,
        )
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["JsonMemorySource", "QdrantMem0Source"]
