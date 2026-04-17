from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

from opentulpa.core.config import get_settings

_E2E_ROOT = Path(__file__).resolve().parent
if str(_E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(_E2E_ROOT))

from harness.runner import E2EHarness, build_harness, close_harness
from mocks.composio_instagram import FakeComposioInstagramService, build_instagram_conversation


pytestmark = [pytest.mark.e2e]


def _has_live_llm_key() -> bool:
    settings = get_settings()
    return bool(str(settings.openai_compatible_api_key or "").strip())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end test suite")
    config.addinivalue_line("markers", "live_llm: requires OPENAI_COMPATIBLE_API_KEY")
    config.addinivalue_line("markers", "telegram: exercises telegram webhook path")
    config.addinivalue_line("markers", "ingress: exercises instagram ingress/intake path")


@pytest.fixture()
def composio_instagram_fixture() -> FakeComposioInstagramService:
    service = FakeComposioInstagramService()
    service.conversations["conv_e2e_1"] = build_instagram_conversation(
        conversation_id="conv_e2e_1",
        recipient_id="178900001",
        inbound_text=(
            "Hi! I'd like to book a table for 2 on Friday April 18 at 7pm. "
            "Name: Alex Rivera. Phone: +1 415 555 1234."
        ),
    )
    return service


@pytest.fixture()
def e2e_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    composio_instagram_fixture: FakeComposioInstagramService,
) -> Iterator[E2EHarness]:
    if not _has_live_llm_key():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) required for e2e live_llm suite")
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="suite",
        composio_service=composio_instagram_fixture,
    )
    try:
        yield harness
    finally:
        close_harness(harness)
