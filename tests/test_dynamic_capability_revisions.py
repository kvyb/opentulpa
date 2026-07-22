from datetime import UTC, datetime

import pytest

from opentulpa.capabilities import (
    CapabilityManifest,
    CapabilityRevisionConflictError,
    CapabilityRevisionNotFoundError,
    CapabilityRevisionStore,
    EvalCommand,
    WorkerKind,
    WorkerSpec,
)


def _manifest(revision: int, version: str) -> CapabilityManifest:
    return CapabilityManifest(
        name="telegram",
        version=version,
        revision=revision,
        workers=(
            WorkerSpec(
                name="telegram_interface",
                kind=WorkerKind.INTERFACE,
                protocol="agent-interface-v1",
                command=("telegram-worker",),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


def test_revision_store_is_append_only_and_activation_is_cas(tmp_path) -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3", clock=lambda: now)
    first = store.append(
        namespace="tenant-a",
        manifest=_manifest(1, "1.0.0"),
        expected_latest_revision=None,
    )
    second = store.append(
        namespace="tenant-a",
        manifest=_manifest(2, "1.1.0"),
        expected_latest_revision=1,
    )

    assert store.list(namespace="tenant-a", capability_name="telegram") == [first, second]
    active_first = store.activate(
        namespace="tenant-a",
        capability_name="telegram",
        revision=1,
        expected_generation=None,
    )
    active_second = store.activate(
        namespace="tenant-a",
        capability_name="telegram",
        revision=2,
        expected_generation=active_first.generation,
    )
    rollback = store.activate(
        namespace="tenant-a",
        capability_name="telegram",
        revision=1,
        expected_generation=active_second.generation,
    )

    assert (active_first.generation, active_second.generation, rollback.generation) == (1, 2, 3)
    assert rollback.revision == 1
    assert store.active(namespace="tenant-a", capability_name="telegram") == rollback

    with pytest.raises(CapabilityRevisionConflictError):
        store.activate(
            namespace="tenant-a",
            capability_name="telegram",
            revision=2,
            expected_generation=1,
        )
    with pytest.raises(CapabilityRevisionConflictError):
        store.append(
            namespace="tenant-a",
            manifest=_manifest(2, "9.9.9"),
            expected_latest_revision=1,
        )
    with pytest.raises(CapabilityRevisionNotFoundError):
        store.activate(
            namespace="tenant-a",
            capability_name="telegram",
            revision=99,
            expected_generation=rollback.generation,
        )


def test_reactivating_current_revision_is_idempotent(tmp_path) -> None:
    store = CapabilityRevisionStore(tmp_path / "capabilities.sqlite3")
    store.append(
        namespace="tenant-a",
        manifest=_manifest(1, "1.0.0"),
        expected_latest_revision=None,
    )
    active = store.activate(
        namespace="tenant-a",
        capability_name="telegram",
        revision=1,
        expected_generation=None,
    )

    assert (
        store.activate(
            namespace="tenant-a",
            capability_name="telegram",
            revision=1,
            expected_generation=active.generation,
        )
        == active
    )
