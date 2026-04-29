# SQLite runtime store policy

## Inventory of OpenTulpa-owned runtime SQLite stores

These stores are owned by OpenTulpa runtime code and now use the shared connection policy helper:

- `src/opentulpa/tasks/service.py`
- `src/opentulpa/tasks/wake_queue.py`
- `src/opentulpa/scheduler/service.py`
- `src/opentulpa/intake/service.py`
- `src/opentulpa/intake/workflow_setup_store.py`
- `src/opentulpa/approvals/store.py`
- `src/opentulpa/context/service.py`
- `src/opentulpa/context/customer_profiles.py`
- `src/opentulpa/context/link_aliases.py`
- `src/opentulpa/context/thread_rollups.py`
- `src/opentulpa/context/file_vault.py`
- `src/opentulpa/interfaces/telegram/business.py`
- `src/opentulpa/skills/service.py`
- `src/opentulpa/business_knowledge/service.py`

LangGraph checkpoint internals are intentionally out of scope.

## Standard connection policy

`opentulpa.persistence.sqlite.connect_sqlite` centralizes:

- nonzero `timeout` on `sqlite3.connect`
- `PRAGMA busy_timeout`
- `PRAGMA synchronous=NORMAL`
- optional `PRAGMA journal_mode=WAL` (enabled for runtime stores above)

## SQLAlchemy/SQLModel boundary decision

For this refactor, we keep raw `sqlite3` for OpenTulpa-owned stores and standardize behavior through a thin helper.

Rationale:

- The lock failures came from inconsistent connection pragmas rather than query composition.
- Most stores are small, simple, and already stable with direct SQL statements.
- Migrating to SQLAlchemy Core/SQLModel would add dependency and migration overhead without directly reducing lock contention.

Boundary:

- OpenTulpa durable runtime stores should use `connect_sqlite`.
- If future modules require cross-database portability, rich transaction orchestration, or declarative schema management, SQLAlchemy Core can be adopted module-by-module rather than globally.
