from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from opentulpa.interfaces.telegram.status_generation import generate_llm_status_message


class _StructuredTimeoutRuntime:
    _workflow_setup_input_classifier_model = object()
    _workflow_setup_input_classifier_model_name = "google/gemini-3-flash-preview"

    async def _invoke_structured_model(self, **kwargs: Any) -> tuple[None, None]:
        del kwargs
        await asyncio.sleep(10.0)
        return None, None


@pytest.mark.asyncio
async def test_generate_llm_status_message_logs_timeout_without_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="opentulpa.interfaces.telegram.status_generation")

    result = await generate_llm_status_message(
        runtime=_StructuredTimeoutRuntime(),
        customer_id="telegram_1",
        thread_id="chat_1",
        context={"stage": "first_token"},
        timeout_seconds=0.01,
    )

    assert result is None
    assert any(
        "telegram.status_generation structured generator timed out" in record.getMessage()
        for record in caplog.records
    )
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
