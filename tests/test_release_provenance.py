from __future__ import annotations

import pytest
from pydantic import ValidationError

from opentulpa.evolution.generation import StateContract
from opentulpa.evolution.release_provenance import ReleaseArtifactProvenance


def _contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=2,
        product_state_schema=1,
        workspace_api=1,
    )


def _generation_metadata() -> dict[str, object]:
    generation_id = "a" * 64
    contract = _contract()
    return {
        "artifact_kind": "python_generation",
        "image_reference": f"python-generation:{generation_id}",
        "generation_id": generation_id,
        "dependency_lock_hash": "b" * 64,
        "evaluator_fingerprint": f"sha256:{'c' * 64}",
        "state_contract": contract.model_dump(mode="json"),
        "state_contract_sha256": contract.sha256(),
        "install_profile": "runtime",
        "controller_protocol": 1,
    }


def _generation(metadata: dict[str, object] | None = None) -> ReleaseArtifactProvenance:
    manifest_digest = f"sha256:{'d' * 64}"
    return ReleaseArtifactProvenance.from_values(
        source_commit="e" * 40,
        artifact_digest=manifest_digest,
        manifest_digest=manifest_digest,
        entrypoint=("venv/bin/python", "-I", "-m", "opentulpa"),
        metadata=metadata or _generation_metadata(),
    )


def test_generation_provenance_round_trips_canonical_flat_metadata() -> None:
    provenance = _generation()

    assert provenance.generation_id == "a" * 64
    assert provenance.state_contract == _contract()
    assert provenance.release_metadata() == _generation_metadata()


@pytest.mark.parametrize(
    "change",
    [
        {"generation_id": "f" * 64},
        {"state_contract_digest": "f" * 64},
        {"dependency_base_id": "f" * 64},
    ],
)
def test_generation_provenance_rejects_inconsistent_or_partial_metadata(
    change: dict[str, object],
) -> None:
    metadata = {**_generation_metadata(), **change}

    with pytest.raises((ValidationError, ValueError)):
        _generation(metadata)


def test_oci_provenance_rejects_generation_identity() -> None:
    metadata = {
        "artifact_kind": "oci_image",
        "image_reference": "registry.example/opentulpa@sha256:" + "a" * 64,
        "generation_id": "b" * 64,
    }

    with pytest.raises(ValidationError, match="cannot carry a generation"):
        ReleaseArtifactProvenance.from_values(
            source_commit="c" * 40,
            artifact_digest=f"sha256:{'d' * 64}",
            manifest_digest=f"sha256:{'e' * 64}",
            entrypoint=("/app/start",),
            metadata=metadata,
        )
