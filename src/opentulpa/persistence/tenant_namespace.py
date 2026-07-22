"""Stable tenant namespaces shared by the Deep Agents runtime and migrations."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

TenantStoreKind = Literal["memory", "skills"]


def tenant_namespace_label(tenant_id: str) -> str:
    """Encode an opaque tenant ID as one valid, collision-resistant store label."""
    tenant = str(tenant_id or "").strip()
    if not tenant:
        raise ValueError("tenant_id is required")
    slug = re.sub(r"[^a-z0-9_-]+", "-", tenant.casefold()).strip("-_")[:32]
    digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:24]
    return f"ot-{slug or 'tenant'}-{digest}"


def tenant_store_namespace(
    tenant_id: str,
    kind: TenantStoreKind,
) -> tuple[str, str, str]:
    return ("tenant", tenant_namespace_label(tenant_id), kind)


__all__ = ["TenantStoreKind", "tenant_namespace_label", "tenant_store_namespace"]
