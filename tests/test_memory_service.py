from __future__ import annotations

import logging

from opentulpa.memory.service import MemoryService


def test_memory_service_filters_only_mem0_noop_logs(caplog) -> None:
    MemoryService()
    logger = logging.getLogger("mem0.memory.main")

    with caplog.at_level(logging.INFO, logger="mem0.memory.main"):
        logger.info("NOOP for Memory.")
        logger.info("useful mem0 info")

    assert "NOOP for Memory." not in caplog.text
    assert "useful mem0 info" in caplog.text
