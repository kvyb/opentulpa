from __future__ import annotations

import pytest
from pydantic import ValidationError

from opentulpa.evolution.release_provenance import (
    ReleaseArtifactProvenance,
    live_repo_artifact_digest,
)


def test_live_repo_provenance_round_trips_commit_artifact() -> None:
    source_commit = "a" * 40
    digest = live_repo_artifact_digest(source_commit)

    provenance = ReleaseArtifactProvenance.from_values(
        source_commit=source_commit,
        artifact_digest=digest,
        manifest_digest=digest,
        entrypoint=("python", "-P", "-m", "opentulpa"),
        metadata={
            "artifact_kind": "live_repo",
            "image_reference": f"git-commit:{source_commit}",
            "dependency_lock_hash": "b" * 64,
            "evaluator_fingerprint": f"sha256:{'c' * 64}",
        },
    )

    assert provenance.release_metadata() == {
        "artifact_kind": "live_repo",
        "image_reference": f"git-commit:{source_commit}",
        "dependency_lock_hash": "b" * 64,
        "evaluator_fingerprint": f"sha256:{'c' * 64}",
    }


@pytest.mark.parametrize(
    ("metadata", "artifact_digest", "manifest_digest"),
    [
        ({"artifact_kind": "container_image"}, f"sha256:{'b' * 64}", f"sha256:{'b' * 64}"),
        ({"artifact_kind": "wheel_archive"}, f"sha256:{'b' * 64}", f"sha256:{'b' * 64}"),
        (
            {"artifact_kind": "live_repo", "image_reference": "git-commit:" + "c" * 40},
            live_repo_artifact_digest("a" * 40),
            live_repo_artifact_digest("a" * 40),
        ),
        (
            {"artifact_kind": "live_repo", "image_reference": "git-commit:" + "a" * 40},
            f"sha256:{'b' * 64}",
            f"sha256:{'b' * 64}",
        ),
    ],
)
def test_live_repo_provenance_rejects_non_live_or_unbound_metadata(
    metadata: dict[str, object],
    artifact_digest: str,
    manifest_digest: str,
) -> None:
    source_commit = "a" * 40

    with pytest.raises(ValidationError):
        ReleaseArtifactProvenance.from_values(
            source_commit=source_commit,
            artifact_digest=artifact_digest,
            manifest_digest=manifest_digest,
            entrypoint=("python", "-P", "-m", "opentulpa"),
            metadata={
                "image_reference": f"git-commit:{source_commit}",
                **metadata,
            },
        )


def test_live_repo_provenance_rejects_partial_dependency_base() -> None:
    source_commit = "a" * 40
    digest = live_repo_artifact_digest(source_commit)

    with pytest.raises(ValidationError, match="dependency base provenance"):
        ReleaseArtifactProvenance.from_values(
            source_commit=source_commit,
            artifact_digest=digest,
            manifest_digest=digest,
            entrypoint=("python", "-P", "-m", "opentulpa"),
            metadata={
                "artifact_kind": "live_repo",
                "image_reference": f"git-commit:{source_commit}",
                "dependency_base_id": "b" * 64,
            },
        )
