"""Release-bound attestations for capabilities bundled with OpenTulpa."""

from __future__ import annotations

from collections.abc import Sequence

from opentulpa.capabilities.bundled import BUNDLED_CAPABILITY_TEMPLATES
from opentulpa.capabilities.models import (
    CapabilityManifest,
    CapabilityTestCheck,
    CapabilityTestStatus,
    canonical_json_digest,
)


class BundledCapabilityEvaluator:
    """Attest only exact manifests already covered by the release test gates.

    Generated capabilities enter the next release through source evolution and its
    isolated evaluator. This runtime evaluator never executes tenant-supplied code.
    """

    def __init__(
        self,
        bundled: Sequence[CapabilityManifest] = BUNDLED_CAPABILITY_TEMPLATES,
    ) -> None:
        self._digests = {manifest.name: _body_digest(manifest) for manifest in bundled}

    async def evaluate(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> Sequence[CapabilityTestCheck]:
        del tenant_id
        matches_release = manifest.seed and self._digests.get(manifest.name) == _body_digest(
            manifest
        )
        return (
            CapabilityTestCheck(
                name="bundled_release",
                status=(
                    CapabilityTestStatus.PASSED if matches_release else CapabilityTestStatus.FAILED
                ),
                message=(
                    "Exact bundled manifest is covered by the release evaluation."
                    if matches_release
                    else "Capability is not an exact manifest from this release."
                ),
            ),
        )


def _body_digest(manifest: CapabilityManifest) -> str:
    return canonical_json_digest(manifest.model_dump(mode="json", exclude={"revision"}))


__all__ = ["BundledCapabilityEvaluator"]
