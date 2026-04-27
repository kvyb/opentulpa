from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from opentulpa.core.config import get_settings

_E2E_ROOT = Path(__file__).resolve().parent
if str(_E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(_E2E_ROOT))

from harness.runner import E2EHarness, build_harness, close_harness
from harness.real_composio import build_recording_live_googlesheets_service
from mocks.composio_instagram import FakeComposioInstagramService, build_instagram_conversation

from opentulpa.integrations.composio import ComposioService


pytestmark = [pytest.mark.e2e]


def _has_live_llm_key() -> bool:
    settings = get_settings()
    return bool(str(settings.openai_compatible_api_key or "").strip())


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-e2e", action="store_true", default=False, help="run e2e tests")
    parser.addoption("--run-live-llm", action="store_true", default=False, help="run live LLM e2e tests")
    parser.addoption(
        "--run-real-composio",
        action="store_true",
        default=False,
        help="allow opted-in e2e tests to use real Composio connections discovered from local OpenTulpa state",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end test suite")
    config.addinivalue_line("markers", "live_llm: requires OPENAI_COMPATIBLE_API_KEY")
    config.addinivalue_line("markers", "real_composio: may create/write real Composio-backed resources")
    config.addinivalue_line("markers", "telegram: exercises telegram webhook path")
    config.addinivalue_line("markers", "ingress: exercises instagram ingress/intake path")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_e2e = bool(config.getoption("--run-e2e"))
    run_live_llm = bool(config.getoption("--run-live-llm"))
    skip_e2e = pytest.mark.skip(reason="set --run-e2e to run e2e tests")
    skip_live_llm = pytest.mark.skip(reason="set --run-live-llm to run live LLM e2e tests")
    for item in items:
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
        if "live_llm" in item.keywords and not run_live_llm:
            item.add_marker(skip_live_llm)


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
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    composio_instagram_fixture: FakeComposioInstagramService,
) -> Iterator[E2EHarness]:
    if not _has_live_llm_key():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) required for e2e live_llm suite")
    composio_service: Any = composio_instagram_fixture
    if bool(request.config.getoption("--run-real-composio")) and "real_composio" in request.node.keywords:
        settings = get_settings()
        if not str(settings.composio_api_key or "").strip():
            pytest.skip("COMPOSIO_API_KEY is required for --run-real-composio")
        live_composio = build_recording_live_googlesheets_service(
            composio=ComposioService(
                api_key=str(settings.composio_api_key or "").strip(),
                default_callback_url=str(settings.composio_default_callback_url or "").strip()
                or None,
            ),
            project_root=Path.cwd(),
        )
        if live_composio is None:
            pytest.skip(
                "--run-real-composio requested, but no active Google Sheets Composio account "
                "was found for local OpenTulpa customer ids"
            )
        composio_service = live_composio
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="suite",
        composio_service=composio_service,
    )
    try:
        yield harness
    finally:
        close_harness(harness)
