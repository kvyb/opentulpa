from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def add(self, kind: str, **payload: Any) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "kind": str(kind or "").strip(),
            **payload,
        }
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def count(self, kind: str) -> int:
        return len([item for item in self.entries if item.get("kind") == kind])

    def slice(self, kind: str, start: int = 0) -> list[dict[str, Any]]:
        items = [item for item in self.entries if item.get("kind") == kind]
        return items[start:]
