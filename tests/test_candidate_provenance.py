from __future__ import annotations

import pytest
from pydantic import ValidationError

from opentulpa.evolution.candidate_provenance import CandidateSourceProvenance
from opentulpa.evolution.generation import UPSTREAM_LINEAGE_METADATA_KEY


def test_candidate_source_provenance_preserves_flat_metadata_and_unknown_keys() -> None:
    metadata = {
        "label": "keep me",
        "changed_paths": ["src/opentulpa/app.py", "uv.lock"],
        "diff_sha256": "a" * 64,
        "promotion_eligible": False,
        "accepted_upstream_commit": "b" * 40,
        UPSTREAM_LINEAGE_METADATA_KEY: {
            "upstream_commit": "b" * 40,
            "merge_base_commit": "c" * 40,
        },
        "opentulpa.evolution.upstream_merge_commit": "d" * 40,
    }

    provenance = CandidateSourceProvenance.from_metadata(metadata)

    assert provenance.changed_paths == ("src/opentulpa/app.py", "uv.lock")
    assert provenance.promotion_eligible is False
    assert provenance.apply_to_metadata(metadata) == metadata


def test_candidate_source_provenance_preserves_absent_optional_keys() -> None:
    metadata = {"source_session": True, "requested_by": {"tenant_id": "tenant_1"}}

    provenance = CandidateSourceProvenance.from_metadata(metadata)

    assert provenance.changed_paths is None
    assert provenance.promotion_eligible is None
    assert provenance.apply_to_metadata(metadata) == metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"changed_paths": ["src/app.py"], "diff_sha256": "a" * 63},
        {"accepted_upstream_commit": "b" * 39},
        {"promotion_eligible": "yes"},
    ],
)
def test_candidate_source_provenance_rejects_invalid_integrity_evidence(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CandidateSourceProvenance.from_metadata(metadata)


def test_candidate_source_provenance_preserves_legacy_path_values() -> None:
    metadata = {
        "changed_paths": ["../legacy.py", "../legacy.py", "/absolute/legacy.py"],
        "diff_sha256": "a" * 64,
    }

    provenance = CandidateSourceProvenance.from_metadata(metadata)

    assert provenance.apply_to_metadata(metadata) == metadata
