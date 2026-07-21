# Deployment

OpenTulpa has three explicit host shapes:

- **Host** is the default. It starts before model configuration and owns setup, owner
  authentication, encrypted credentials, child process health, redacted logs, and the
  public proxy. The Deep Agents application runs as its replaceable child.

- **Direct** starts the mutable FastAPI release itself. Use it for development or a
  platform that already owns image rollout and rollback. Reviewed capability workers
  may run as subprocesses in this mode.
- **Managed** starts the immutable bootstrap gateway. The bootstrap builds the first
  release from a canonical Git commit, then owns staging, activation, and rollback.

Do not treat direct mode as a safe self-replacement mechanism. It deliberately has no
access to the stable bootstrap evolution API.

## Dependencies

Required in every mode:

- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- writable persistent storage.

The default host does not require `OPENAI_COMPATIBLE_API_KEY` to start. Configure it in
`/_host` or provide it for non-interactive first boot.

Local loopback startup generates and privately persists owner authentication and
selects `${XDG_DATA_HOME:-$HOME/.local/share}/opentulpa` automatically. Public host
deployments can be claimed with their one-time setup token; managed deployments still
require an explicit `OPENTULPA_WEB_TOKEN` and persistent storage paths.

The launcher installs only the lean core unless `OPENTULPA_EXTRAS` is set. Use
`OPENTULPA_EXTRAS=bundled` or a comma-separated subset of `browser`, `integrations`,
`documents`, and `research`; managed installation bakes the same selection into the
reviewed runtime base.

Tenant shell tools need an isolated OCI engine because `TenantContainerBackend` never
executes commands directly on the application host. Direct startup auto-detects
rootless Podman/Docker and recognizes Docker Desktop or OrbStack's macOS VM boundary.
If none is available, chat starts with shell execution disabled. Managed mode never
uses that degraded path and additionally requires:

- a rootless Docker or Podman engine available to the bootstrap;
- a persistent canonical Git checkout;
- a reviewed runtime base image and evaluator image;
- an administrator-created production egress network;
- separate persistent bootstrap state and release workspace paths.

## Stable Host

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./start.sh
```

The host opens its setup console at `http://127.0.0.1:8000/_host`. It remains healthy
while unconfigured or while a candidate child fails. Local owner access needs no token.
Remote first boot prints a one-time setup token, and the returned owner token can be
used by the browser or `opentulpa connect`. Health checks are:

- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

The first endpoint reports host health. The second returns `503` until the child is
ready. Use `./start.sh server` to run the mutable application directly for development.

`./start.sh local` remains a convenience mode for a host-configured Telegram bot. It
starts the direct app, a temporary Cloudflare tunnel, and webhook synchronization. It
requires `TELEGRAM_BOT_TOKEN` plus an owner username or numeric ID allowlist. It is a
development convenience, not the dynamic Telegram capability path.

## Managed Runtime

Managed mode keeps a stable public gateway while release containers change behind it.
The mutable release never receives the source repository, bootstrap database,
environment file, or OCI socket.

### 1. Configure the host

Start from `.env.example` and set at least:

```env
OPENAI_COMPATIBLE_API_KEY=...
OPENTULPA_WEB_TOKEN=...
EVOLUTION_ENABLED=true

OPENTULPA_BOOTSTRAP_STATE_ROOT=/absolute/persistent/opentulpa-bootstrap
OPENTULPA_RELEASE_WORKSPACE=/absolute/persistent/opentulpa-release-data
OPENTULPA_RELEASE_BASE_IMAGE=opentulpa-runtime-base:0.1.0
OPENTULPA_RELEASE_EGRESS_NETWORK=opentulpa-release-egress
OPENTULPA_CONTAINER_CLI=docker
OPENTULPA_RECOVERY_TOKEN=<32-or-more-random-characters>
OPENTULPA_INGRESS_TOKEN=<32-or-more-random-characters>
```

Run the managed launcher from the canonical checkout. It must contain `.git` and
`uv.lock`. The runtime setting `EVOLUTION_SOURCE_REPOSITORY` supports an explicit
canonical checkout for custom hosts, but the bundled `doctor managed` validates the
launcher checkout.

The state and workspace roots must be different from each other and from the source
checkout. The release workspace is mounted read-write at `/workspace`; it holds the
product databases, checkpoints, memories, skills, capabilities, and tenant workspace
directories. Bootstrap state holds leases, releases, activation records, the evolution
archive, private candidate refs, and its generated internal token.

The bootstrap creates its internal evolution credential as a mode `0600` file.
Recovery and durable ingress use the required, stable high-entropy deployment tokens
shown above; they are never passed to a mutable release.

### 2. Create the production network

The bootstrap refuses to launch a production release without an explicitly named,
non-internal OCI network. Create it before startup:

```bash
docker network create opentulpa-release-egress
```

A normal bridge network has broad internet access. Restrict this network with the host
firewall, rootless engine configuration, or an egress proxy to the destinations needed
by the model and enabled adapters. The setting is an explicit boundary, not an
automatic hostname allowlist.

Staging and candidate evaluation use isolated no-network containers. Production
release egress is the only declared runtime network path. Interactive tenant and source
shells use outbound bridge networking by default so the owner agent can fetch dependencies,
inspect public repositories, and run networked experiments. They still receive no host or
service credentials. Apply stricter destination controls at the OCI host or egress proxy
when the deployment requires them.

### 3. Install reviewed images

```bash
./start.sh install managed
```

Managed installation builds:

- `OPENTULPA_RELEASE_BASE_IMAGE` from the reviewed root `Dockerfile`;
- `opentulpa-evolution:0.1.0` from `docker/evolution.Dockerfile` unless the configured
  evaluator/sandbox image names override it;
- `SANDBOX_IMAGE` (default `opentulpa-tenant-sandbox:0.1.0`) from
  `docker/tenant-sandbox.Dockerfile`.

At bootstrap startup every reviewed tag is inspected and converted to its immutable
local `sha256:` image ID. The tenant image ID and OCI engine remain in the stable host.
The mutable production release receives private sandbox and capability-worker
URL/tokens plus its release ID, lease epoch, and short-lived control credential. Every
call is fenced to the current production lease; old and staging releases fail closed.
The sandbox endpoint accepts only a hidden tenant ID, bounded command, and timeout. The
capability endpoint accepts only a reviewed package module, exact bundled manifest,
tenant config, and manifest-declared ephemeral grants. Neither endpoint accepts an
image, mount, runtime user, container socket, or OCI arguments. The stable host derives
the active release image and mounts only per-tenant/per-capability `/state` into a
rootless bounded worker, never the product `/workspace`, source, databases, or host
credentials.

`OPENTULPA_CAPABILITY_ALLOWED_HOSTS` is the administrator ceiling for manifest-declared
worker egress. The configured capability/release network must enforce that ceiling at
the host firewall or proxy layer; an OCI network name alone is not destination
filtering. Requests outside the declared ceiling fail before container launch.

The runtime base contains the lockfile-matched virtual environment. Candidate release
building exports an exact evaluated Git commit and replaces the complete secret-free
application snapshot. Core, API, product-service, config, integrations, interfaces,
tools, web assets, tests, and new source files may all evolve. The candidate's Dockerfile
and `.dockerignore` are copied as source artifacts but never executed by the trusted
builder. Secret paths, links, special files, and dependency-lock changes fail closed.

Every candidate image must carry
`org.opentulpa.release.runtime-overlay=full-source-v1`.
The immutable bootstrap verifies this exact label before it prepares the image.

If the managed release needs all optional adapters, build the runtime base with the
bundled extra before startup:

```bash
docker build \
  --build-arg OPENTULPA_EXTRAS=bundled \
  -t opentulpa-runtime-base:0.1.0 .
```

Pin production image references and preserve the built image until its release is no
longer a rollback target. A dependency-lock change cannot be deployed as a normal
candidate; rebuild and review the trusted runtime base first.

### 4. Check and start

```bash
./start.sh doctor managed
./start.sh run managed
```

Or install and start in one command:

```bash
./start.sh managed
```

`doctor managed` checks required environment, launcher Git source, OCI availability,
all three reviewed images, and the explicit egress network. Startup does not silently
fall back to direct mode. The bootstrap additionally validates state/workspace
separation and writability while starting.

On an empty state store, the bootstrap builds the canonical checkout as the first
content-addressed release, starts it with ingress disabled, verifies production health,
and only then accepts public traffic. Subsequent approved candidates follow:

```text
queue -> prepare -> isolated staging -> drain old release -> start new release
      -> production health -> probation -> active
```

A failure before cutover leaves the current release serving. A failure after cutover
automatically restores the prior image and lease. Only release-coupled capability state
is restored; product records created during probation remain intact. Exact bundled
seed capability pointers are reconciled to the restored image, while candidate-only
seeds are deactivated without deleting history. If restoration is impossible, the
gateway enters safe mode rather than forwarding to an unverified process.

Probation serves live traffic with production credentials. Automatic rollback restores
the release process and release-coupled capability state, but it cannot retract external
messages, purchases, authorization changes, or provider writes. Rehearse changes near
external effects with fake sinks and retain normal approval and idempotency controls.

Recovery is host-shell only. There is intentionally no browser console: `/recovery` and
all children are reserved by the stable gateway and always return `404` instead of being
proxied to mutable release assets. Recovery APIs require the separate bearer token and
reject requests carrying `Origin`, `Referer`, or any `Sec-Fetch-*` header.

Set the token in the host environment, never in a command argument or URL:

```bash
export OPENTULPA_RECOVERY_URL=http://127.0.0.1:8000
export OPENTULPA_RECOVERY_TOKEN='<the separately stored recovery token>'
opentulpa-recovery status
opentulpa-recovery rollback --reason 'Restore the previous healthy release'
opentulpa-recovery restart
opentulpa-recovery safe-mode
```

`opentulpa-recovery rollback`, `opentulpa-recovery restart`, and
`opentulpa-recovery safe-mode` remain available even when the mutable release is dead.
The CLI uses `httpx` with redirects, proxy-environment inheritance, and browser headers
disabled.

Ordinary source promotion has one persisted Deep Agents approval in the owner chat. That
approval is application policy, not a cryptographic defense against an already malicious
active release. The stable bootstrap independently binds and verifies the source commit,
fixed evaluator evidence, artifact digest, staging health, and rollback path, but does not
prove owner intent. Deployments that require a hostile-release threat model need a separate
host-controlled signing or approval authority, which is intentionally not part of this
minimal mode.

### 5. Back up the right state

Back up these together:

- `OPENTULPA_BOOTSTRAP_STATE_ROOT`;
- `OPENTULPA_RELEASE_WORKSPACE`;
- the canonical Git repository including private `refs/opentulpa/*` refs;
- every OCI image still referenced by a current or previous release.

Restoring only the SQLite product databases is insufficient for source lineage and
release rollback. Restoring only bootstrap state is insufficient for conversations,
memory, capabilities, and tenant workspaces.

## Dynamic Telegram From Web

To let the running agent establish Telegram itself, do **not** set a host
`TELEGRAM_BOT_TOKEN`.

1. Start web-only direct or managed mode with `OPENTULPA_WEB_TOKEN`.
2. Open `/` and authenticate.
3. Ask OpenTulpa to enable Telegram and paste the BotFather token in that message.
4. Credential ingress encrypts the value and replaces it with a `secret://` handle
   before checkpointing.
5. The agent seeds the Telegram manifest, runs its deterministic tests, and requests
   activation approval.
6. Approve the exact capability revision and secret-handle binding.
7. In Telegram, send `/start <code>`. The default one-time code is the final eight
   characters of the bot token; `OPENTULPA_TELEGRAM_PAIRING_CODE` can override it.
8. The worker begins Telegram long polling and uses only scoped Agent API, file,
   approval, replay, and notification endpoints.

The worker uses the same tenant and Deep Agents checkpoints as web. Rotating the bound
secret creates a new vault revision and reconciles the active worker. A source release
restart restores the active capability generation from persistent state.
In managed mode, Telegram runs in the stable host's rootless capability container with
only its private `/state/telegram.json` mount. Revision changes stop the old long poll
before starting the new generation. If startup or activation persistence fails, the
prior generation is restarted against the same state and the activation pointer is not
advanced, so two pollers never intentionally share the bot token.

If `TELEGRAM_BOT_TOKEN` is set on the host, OpenTulpa uses the bundled webhook route at
`/webhook/telegram`. Configure:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=...
TELEGRAM_ALLOWED_USER_IDS=123456789
PUBLIC_BASE_URL=https://opentulpa.example.com
```

The dynamic Telegram capability is blocked in that configuration to prevent duplicate
consumers. Telegram Business intake also uses the webhook form and requires Business
Mode plus the appropriate inbox permissions.

## Optional Adapters

The base dependency set contains Deep Agents, LangChain/LangGraph, FastAPI, SQLite
checkpoint support, APScheduler, HTTP, encryption, Langfuse hooks, MCP adapters, and
the OpenRouter chat adapter. FastAPI is the universal API, not an optional capability
dependency.

Install only what the deployment uses:

| Extra | Adds | Runtime behavior |
|---|---|---|
| `browser` | Browser Use Cloud SDK, Playwright CDP client | Hosted Chromium with required `BROWSER_USE_API_KEY`; no host-browser fallback |
| `integrations` | Composio SDK and LangChain provider | Tenant-owned SaaS connections when `COMPOSIO_API_KEY` is set |
| `documents` | PDF, workbook, and encoding parsers | Additional file-analysis formats |
| `research` | Crawl4AI | Richer content extraction; bounded built-in extraction remains available without it |
| `bundled` | All of the above | Convenience image for a full adapter set |

Examples:

```bash
uv sync --no-dev --extra browser
```

```env
BROWSER_USE_API_KEY=...  # required when browser tools are enabled
COMPOSIO_API_KEY=...     # optional SaaS integration provider
```

No optional adapter receives credentials through model-visible arguments. Browser and
Composio actions with unknown or risky effects require persisted owner approval.
Browser navigation requires explicit allowed domains and rejects direct private and
link-local targets. Chromium and target-network access run inside Browser Use Cloud;
OpenTulpa controls it over CDP and cannot DNS-pin the vendor browser's connections.

## Docker Compose And Railway

The included `docker-compose.yml` and `railway.toml` start the stable host plus its
Deep Agents child. They do not provide the rootless OCI engine, canonical Git checkout,
and release network needed for managed source self-replacement.

For Docker Compose:

```bash
docker compose up --build
```

Persist the volume mounted at `/app/opentulpa_data` and set
`OPENTULPA_DATA_ROOT=/app/opentulpa_data`.

For Railway, configure:

- `OPENTULPA_DATA_ROOT=/app/opentulpa_data` with a persistent volume;
- optionally `OPENAI_COMPATIBLE_API_KEY` and `OPENTULPA_WEB_TOKEN` for unattended boot.

Without an owner token, read the one-time setup token from the first startup log and
claim the deployment at `/_host`. Configure Telegram there; polling needs no webhook.

The Docker and Railway entrypoint is `./start.sh serve --run-only`. A VM can initialize
and start both interfaces directly:

```bash
./start.sh serve \
  --api-key '<openai-compatible-key>' \
  --telegram-bot-token '<telegram-bot-token>' \
  --telegram-user-id 123456789 \
  --public-url https://tulpa.example.com
```

One host is one OpenTulpa installation. Its owner token and optional Telegram owner ID
determine the owner identity; there is no tenant argument. Secrets are encrypted in the
host database rather than saved to `.env`. Product, host, and Deep Agent state live
under `OPENTULPA_DATA_ROOT`.

Source evolution is unavailable in this direct shape. Do not mount a platform container
socket into the app to imitate managed mode. Run the immutable bootstrap on a host that
can enforce its OCI and storage contract, or let the platform own reviewed source
deployments.

## Migration Cutover

Historical chat checkpoints are intentionally not migrated. Product data, memories,
and user-authored skills are.

Rehearse against a copied data directory with fake external sinks:

```bash
uv run --extra migration opentulpa-migrate-deepagents \
  --data-root /path/to/copied-data --dry-run
uv run --extra migration opentulpa-migrate-deepagents \
  --data-root /path/to/copied-data
```

The command fails closed when any preserved product database is absent. This catches a
wrong data root before migration writes begin. Use `--allow-missing` only for a verified
new installation that has no legacy product stores; never use it for a cutover.
The dry run checks existing AgentSpec/TriggerSpec destinations, so a same-ID schedule
with different content is reported as a blocking conflict rather than silently treated
as migrated. Applied migration rebases legacy absolute uploaded-file paths to the
current copied data root in one SQLite transaction and verifies the resulting file
bytes. Destination conflicts never disable the corresponding legacy routine, setup
session, memory, or user skill; resolve the conflict and rerun the idempotent command.
The preservation checksum includes disabled workflows, intake cursors and pending
runs, Telegram Business messages, and the knowledge preflight cache.

For the single production cutover:

1. pause ingress consumers and trigger dispatch;
2. take a complete product-data and source-lineage backup;
3. run migration and retain its count/checksum report;
4. deploy the V2 API and coordinated clients;
5. smoke test web, Telegram, intake, schedules, approvals, and optional adapters;
6. resume consumers and triggers.

Cross-tenant access, duplicate external effects, migration loss, or an approval that
cannot resume are immediate rollback conditions. Migration rollback restores the old
image/client and pre-cutover data snapshot. Managed candidate rollback is a separate
release operation and does not reverse product-data migration.
