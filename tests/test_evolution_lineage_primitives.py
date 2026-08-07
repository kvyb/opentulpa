from __future__ import annotations

import pytest
from pydantic import ValidationError

from opentulpa.evolution.canonical_json import canonical_json_bytes
from opentulpa.evolution.lineage import UpstreamLineage


def test_canonical_json_bytes_sorts_keys_and_rejects_nonfinite_numbers() -> None:
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'

    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_upstream_lineage_round_trips_and_requires_coupled_commits() -> None:
    lineage = UpstreamLineage(
        upstream_commit="a" * 40,
        merge_base_commit="b" * 40,
    )
    stored = lineage.model_dump(mode="json")

    assert UpstreamLineage.model_validate(stored) == lineage
    with pytest.raises(ValidationError, match="recorded together"):
        UpstreamLineage.model_validate({"upstream_commit": "a" * 40})
    with pytest.raises(ValidationError, match="same object format"):
        UpstreamLineage(upstream_commit="a" * 64, merge_base_commit="b" * 40)
