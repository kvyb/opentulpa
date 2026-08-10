# SQLite Persistence Policy

OpenTulpa is a single-active-process deployment by default. Product services use `opentulpa.persistence.sqlite.connect_sqlite` to apply a nonzero connection timeout, `PRAGMA busy_timeout`, `PRAGMA synchronous=NORMAL`, and WAL where appropriate.

The intake workflow and intake draft databases deliberately use rollback journals instead of WAL. Draft activation attaches both files and commits the active workflow row plus the consumed confirmation in one SQLite super-journal transaction with `synchronous=FULL`; SQLite cannot guarantee an atomic multi-database commit when either database uses WAL.

Deep Agents checkpoints and interrupts use `AsyncSqliteSaver`; native tenant memory and
skills use `AsyncSqliteStore`. Runs, notifications, jobs, immutable AgentSpec and
TriggerSpec revisions, intake, files, knowledge, profiles, integration state,
capability generations, encrypted secret handles, idempotency records, and Telegram
state remain explicit product stores. The simple schedule API is a projection over
TriggerSpec revisions and has no second schedule database.

The stable host adds `bootstrap/evolution/activations.db` for source releases, the
active/previous/last-known-good decision, idempotent activation attempts, and terminal
notification state. It lives outside the trusted source repository and mutable child.

Raw `sqlite3` remains appropriate for these small stores. Moving to multiple active replicas requires replacing the Deep Agents saver/store and any concurrently written product stores with shared database implementations before adding workers.
