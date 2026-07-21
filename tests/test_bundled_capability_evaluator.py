from __future__ import annotations

import pytest

from opentulpa.capabilities import (
    TELEGRAM_CAPABILITY,
    BundledCapabilityEvaluator,
    CapabilityTestStatus,
)


@pytest.mark.asyncio
async def test_evaluator_attests_only_exact_bundled_manifest() -> None:
    evaluator = BundledCapabilityEvaluator()
    exact = TELEGRAM_CAPABILITY.model_copy(update={"revision": 7})
    changed = exact.model_copy(update={"version": "1.0.1"})

    exact_checks = await evaluator.evaluate(tenant_id="tenant-1", manifest=exact)
    changed_checks = await evaluator.evaluate(tenant_id="tenant-1", manifest=changed)

    assert exact_checks[0].status is CapabilityTestStatus.PASSED
    assert changed_checks[0].status is CapabilityTestStatus.FAILED
