"""Application services for revisioned AgentSpec and TriggerSpec control planes."""

from __future__ import annotations

from collections.abc import Callable

from opentulpa.core.ids import new_short_id
from opentulpa.specs.defaults import DEFAULT_OWNER_SPEC_ID, default_agent_spec_writes
from opentulpa.specs.models import AgentSpec, AgentSpecWrite, TriggerSpec, TriggerSpecWrite
from opentulpa.specs.protocol import AgentSpecRef
from opentulpa.specs.store import (
    AgentSpecStore,
    SpecConflictError,
    SpecNotFoundError,
    TriggerSpecStore,
)


class AgentSpecService:
    """Tenant-safe AgentSpec revisions, activation, rollback, and seed defaults."""

    def __init__(
        self,
        store: AgentSpecStore,
        *,
        validate_activation: Callable[[AgentSpec], None] | None = None,
    ) -> None:
        self._store = store
        self._validate_activation = validate_activation

    def list_latest(self, *, tenant_id: str) -> list[AgentSpec]:
        return self._store.list_latest(tenant_id=tenant_id)

    def list_revisions(self, *, tenant_id: str, spec_id: str) -> list[AgentSpec]:
        return self._store.list_revisions(tenant_id=tenant_id, spec_id=spec_id)

    def get_active(self, *, tenant_id: str, spec_id: str) -> AgentSpec | None:
        return self._store.get_active(tenant_id=tenant_id, spec_id=spec_id)

    def get_revision(
        self,
        *,
        tenant_id: str,
        spec_id: str,
        revision: int,
    ) -> AgentSpec | None:
        return self._store.get_revision(
            AgentSpecRef(tenant_id=tenant_id, spec_id=spec_id, revision=revision)
        )

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: AgentSpecWrite,
        spec_id: str | None = None,
        expected_revision: int | None = None,
    ) -> AgentSpec:
        return self._store.create_revision(
            tenant_id=tenant_id,
            spec_id=str(spec_id or "").strip() or new_short_id("spec", suffix_chars=12),
            write=write,
            expected_revision=expected_revision,
            created_by=actor_id,
        )

    def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        spec_id: str,
        revision: int,
        expected_active_revision: int | None,
    ) -> AgentSpec:
        ref = AgentSpecRef(tenant_id=tenant_id, spec_id=spec_id, revision=revision)
        spec = self._store.get_revision(ref)
        if spec is None:
            raise SpecNotFoundError(ref)
        if self._validate_activation is not None:
            self._validate_activation(spec)
        self._store.activate(
            ref,
            expected_active_revision=expected_active_revision,
            updated_by=actor_id,
        )
        return spec

    def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        spec_id: str,
        expected_active_revision: int,
    ) -> AgentSpec:
        active = self._store.get_active(tenant_id=tenant_id, spec_id=spec_id)
        if active is None:
            raise SpecNotFoundError(spec_id)
        if active.revision != expected_active_revision:
            raise SpecConflictError(
                f"active AgentSpec revision is {active.revision}, "
                f"expected {expected_active_revision}"
            )
        target = self._store.previous_revision(
            tenant_id=tenant_id,
            spec_id=spec_id,
            before_revision=active.revision,
        )
        if target is None:
            raise SpecConflictError("AgentSpec has no previous revision")
        if self._validate_activation is not None:
            self._validate_activation(target)
        self._store.activate(
            target.ref,
            expected_active_revision=active.revision,
            updated_by=actor_id,
        )
        return target

    def deactivate(
        self,
        *,
        tenant_id: str,
        spec_id: str,
        expected_active_revision: int,
    ) -> AgentSpecRef:
        return self._store.deactivate(
            tenant_id=tenant_id,
            spec_id=spec_id,
            expected_active_revision=expected_active_revision,
        )

    def seed_defaults(self, *, tenant_id: str, actor_id: str) -> list[AgentSpec]:
        active_specs: list[AgentSpec] = []
        for spec_id, write in default_agent_spec_writes().items():
            active = self._store.get_active(tenant_id=tenant_id, spec_id=spec_id)
            if active is not None:
                active = self._upgrade_legacy_owner_default(
                    active=active,
                    desired=write,
                    actor_id=actor_id,
                )
                active_specs.append(active)
                continue
            latest = self._store.get_latest(tenant_id=tenant_id, spec_id=spec_id)
            if latest is None:
                latest = self._store.create_revision(
                    tenant_id=tenant_id,
                    spec_id=spec_id,
                    write=write,
                    expected_revision=None,
                    created_by=actor_id,
                )
            self._store.activate(
                latest.ref,
                expected_active_revision=None,
                updated_by=actor_id,
            )
            active_specs.append(latest)
        return active_specs

    def _upgrade_legacy_owner_default(
        self,
        *,
        active: AgentSpec,
        desired: AgentSpecWrite,
        actor_id: str,
    ) -> AgentSpec:
        if active.id != DEFAULT_OWNER_SPEC_ID:
            return active
        active_write = AgentSpecWrite.model_validate(
            active.model_dump(include=set(AgentSpecWrite.model_fields))
        )
        legacy = desired.model_copy(
            update={
                "max_runtime_seconds": 900,
                "max_model_calls": 100,
            }
        )
        if active_write != legacy:
            return active
        latest = self._store.get_latest(tenant_id=active.tenant_id, spec_id=active.id)
        if latest is not None and latest.revision != active.revision:
            latest_write = AgentSpecWrite.model_validate(
                latest.model_dump(include=set(AgentSpecWrite.model_fields))
            )
            if latest_write != desired:
                return active
            upgraded = latest
        else:
            upgraded = self._store.create_revision(
                tenant_id=active.tenant_id,
                spec_id=active.id,
                write=desired,
                expected_revision=active.revision,
                created_by=actor_id,
            )
        self._store.activate(
            upgraded.ref,
            expected_active_revision=active.revision,
            updated_by=actor_id,
        )
        return upgraded


class TriggerSpecService:
    """Tenant-safe TriggerSpec revisions, activation, rollback, and deactivation."""

    def __init__(
        self,
        store: TriggerSpecStore,
        *,
        validate_activation: Callable[[TriggerSpec], None] | None = None,
    ) -> None:
        self._store = store
        self._validate_activation = validate_activation

    def list_latest(self, *, tenant_id: str) -> list[TriggerSpec]:
        return self._store.list_latest(tenant_id=tenant_id)

    def list_active(self, *, tenant_id: str) -> list[TriggerSpec]:
        return self._store.list_active(tenant_id=tenant_id)

    def list_tenant_ids(self) -> list[str]:
        return self._store.list_tenant_ids()

    def list_revisions(self, *, tenant_id: str, trigger_id: str) -> list[TriggerSpec]:
        return self._store.list_revisions(tenant_id=tenant_id, trigger_id=trigger_id)

    def get_active(self, *, tenant_id: str, trigger_id: str) -> TriggerSpec | None:
        return self._store.get_active(tenant_id=tenant_id, trigger_id=trigger_id)

    def get_latest(self, *, tenant_id: str, trigger_id: str) -> TriggerSpec | None:
        return self._store.get_latest(tenant_id=tenant_id, trigger_id=trigger_id)

    def get_revision(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        revision: int,
    ) -> TriggerSpec | None:
        return self._store.get_revision(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            revision=revision,
        )

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: TriggerSpecWrite,
        trigger_id: str | None = None,
        expected_revision: int | None = None,
    ) -> TriggerSpec:
        return self._store.create_revision(
            tenant_id=tenant_id,
            trigger_id=str(trigger_id or "").strip()
            or new_short_id("trigger", suffix_chars=12),
            write=write,
            expected_revision=expected_revision,
            created_by=actor_id,
        )

    def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        trigger_id: str,
        revision: int,
        expected_active_revision: int | None,
    ) -> TriggerSpec:
        trigger = self._store.get_revision(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            revision=revision,
        )
        if trigger is None:
            raise SpecNotFoundError(trigger_id)
        if self._validate_activation is not None:
            self._validate_activation(trigger)
        return self._store.activate(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            revision=revision,
            expected_active_revision=expected_active_revision,
            updated_by=actor_id,
        )

    def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        trigger_id: str,
        expected_active_revision: int,
    ) -> TriggerSpec:
        active = self._store.get_active(tenant_id=tenant_id, trigger_id=trigger_id)
        if active is None:
            raise SpecNotFoundError(trigger_id)
        if active.revision != expected_active_revision:
            raise SpecConflictError(
                f"active TriggerSpec revision is {active.revision}, "
                f"expected {expected_active_revision}"
            )
        target = self._store.previous_revision(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            before_revision=active.revision,
        )
        if target is None:
            raise SpecConflictError("TriggerSpec has no previous revision")
        if self._validate_activation is not None:
            self._validate_activation(target)
        return self._store.activate(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            revision=target.revision,
            expected_active_revision=active.revision,
            updated_by=actor_id,
        )

    def deactivate(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        expected_active_revision: int,
    ) -> int:
        return self._store.deactivate(
            tenant_id=tenant_id,
            trigger_id=trigger_id,
            expected_active_revision=expected_active_revision,
        )


def seed_default_agent_spec_refs(
    store: AgentSpecStore,
    *,
    tenant_id: str,
    actor_id: str,
) -> dict[str, AgentSpecRef]:
    """Idempotently seed one tenant and return refs keyed by runtime profile."""

    specs = AgentSpecService(store).seed_defaults(
        tenant_id=tenant_id,
        actor_id=actor_id,
    )
    return {spec.runtime_profile: spec.ref for spec in specs}


__all__ = [
    "AgentSpecService",
    "TriggerSpecService",
    "seed_default_agent_spec_refs",
]
