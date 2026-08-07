from __future__ import annotations

import pytest

from opentulpa.evolution.archive import EvolutionArchiveCorruptionError
from opentulpa.evolution.event_publisher import EvolutionEventPublisher
from opentulpa.evolution.models import Candidate, CandidateStatus, EvolutionEvent


class _RecoverConflictArchive:
    def __init__(
        self,
        message: str = "evolution event key is bound to another payload",
    ) -> None:
        self._message = message

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | None = None,
        limit: int = 1_000,
    ) -> list[Candidate]:
        del limit
        if status is CandidateStatus.READY:
            return [
                Candidate(
                    id="candidate-1",
                    base_commit="a" * 40,
                    requested_improvement="test",
                    source_commit="b" * 40,
                    artifact_digest="sha256:" + "c" * 64,
                    evaluator_fingerprint="sha256:" + "d" * 64,
                    status=CandidateStatus.READY,
                )
            ]
        return []

    async def list_release_history(self, *, limit: int = 1_000) -> list[object]:
        del limit
        return []

    async def enqueue_event(self, event: EvolutionEvent) -> EvolutionEvent:
        del event
        raise EvolutionArchiveCorruptionError(self._message)

    async def pending_events(self, *, limit: int = 100) -> list[EvolutionEvent]:
        del limit
        return []


@pytest.mark.asyncio
async def test_recover_terminal_events_skips_conflicting_outbox_payload() -> None:
    publisher = EvolutionEventPublisher(
        archive=_RecoverConflictArchive(),  # type: ignore[arg-type]
        sink=None,
    )

    await publisher.recover_terminal_events()


@pytest.mark.asyncio
async def test_recover_terminal_events_keeps_other_corruption_strict() -> None:
    publisher = EvolutionEventPublisher(
        archive=_RecoverConflictArchive("invalid persisted event JSON"),  # type: ignore[arg-type]
        sink=None,
    )

    with pytest.raises(EvolutionArchiveCorruptionError, match="invalid persisted event JSON"):
        await publisher.recover_terminal_events()
