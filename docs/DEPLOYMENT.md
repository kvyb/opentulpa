# Deployment Guide

This guide covers the practical ways to run OpenTulpa today.

If you just want the fastest local path, do this:

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
cp .env.example .env
```

Set:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Then run:

```bash
./start.sh server
```

Health checks:

- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

## Choose a runtime mode

`start.sh` supports two useful modes:

- `server`: run the FastAPI app directly
- `local`: run the app through the Cloudflare tunnel and Telegram webhook sync flow

In practice:

- use `./start.sh server` for direct app server runs
- use `./start.sh local` or `./start.sh` when you want the managed local Telegram flow with `cloudflared`

## Local setup

Requirements:

- Python `3.12` for local startup. `./start.sh` asks uv for Python 3.12 by default.
- [`uv`](https://docs.astral.sh/uv/) (`start.sh` can install it if missing)
- an OpenAI-compatible API key

Base setup:

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
cp .env.example .env
```

Required `.env` value:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Run locally:

```bash
./start.sh server
```

## Telegram setup

Telegram is the main operator interface.

For local use:

1. create a bot in `@BotFather`
2. set `TELEGRAM_BOT_TOKEN` in `.env`
3. run `./start.sh local`

When you use local mode, `start.sh` will also handle dependency setup for Playwright Chromium and `cloudflared` if needed.

## Telegram Business intake

Telegram Business uses the same bot token and webhook surface, but Telegram has extra setup requirements:

1. create the bot in `@BotFather`
2. enable Business Mode for that bot
3. connect the bot to the Telegram Business account
4. grant the required business inbox permissions

Once connected, OpenTulpa can ingest inbound Telegram Business leads from `/webhook/telegram`, persist their state locally, and continue those conversations across multiple turns.

## Optional integrations

### Composio

If you want OpenTulpa to authenticate into supported third-party services:

```bash
COMPOSIO_API_KEY=...
```

OpenTulpa derives the Composio callback URL from your public base URL when possible. Override only if you need to force a specific callback:

```bash
COMPOSIO_DEFAULT_CALLBACK_URL=https://your-public-base/webhook/composio/callback
```

### Browser automation

Browser Use and Playwright Chromium are installed by default when `./start.sh` runs.

Skip browser installation with:

```bash
./start.sh --no-browser-use
```

For browser-heavy workflows, configure Browser Use Cloud:

```bash
BROWSER_USE_API_KEY=...
```

With this key, OpenTulpa keeps its local browser worker loop but runs it against
Browser Use Cloud hosted sessions via CDP. This is the recommended browser
backend when you need live owner handoff URLs, persisted cloud profiles,
proxying, and managed browser infrastructure. Non-secret defaults such as
`browser_use_cloud_proxy_country_code` and `browser_use_cloud_timeout_minutes`
live in `opentulpa.config.yaml`.

Optional CAPTCHA solving for Browser Use is disabled unless you configure
CapSolver:

```bash
CAPSOLVER_API_KEY=...
```

When configured, OpenTulpa registers a `solve_captcha_with_capsolver` Browser
Use action for supported reCAPTCHA v2/v3 and Cloudflare Turnstile pages. Without
the key, the solver is not registered and normal Browser Use behavior is
unchanged.

## Useful startup commands

| Command | Meaning |
|---|---|
| `./start.sh` | Install and run local Telegram mode |
| `./start.sh local` | Install and run app + Cloudflare tunnel + Telegram webhook sync |
| `./start.sh server` | Install and run the plain app server |
| `./start.sh install` | Install only |
| `./start.sh run server` | Run the plain app server without installing |
| `./start.sh doctor` | Check local startup readiness |

Useful `.env` knobs:

- `START_MODE=local|server|auto`
- `INSTALL_BROWSER_USE=1|0`
- `INSTALL_CLOUDFLARED=auto|1|0`
- `INSTALL_UV=1|auto|0` controls uv bootstrap; default `1` installs uv when missing after first checking `PATH`
- `UV_PYTHON=3.12` controls the Python interpreter uv uses for local startup
- `AGENT_PROMPT_CACHING_ENABLED=1|0`

Compatibility aliases still work but are deprecated: `--app` maps to `server`, and `--manager` maps to `local`.

## Docker

The included `Dockerfile` already installs Python dependencies, Node.js/npm, and Playwright Chromium.

Run with Docker Compose:

```bash
docker compose up --build
```

Compose is optional. It loads `.env`, maps port `8000`, mounts a persistent volume at `/app/opentulpa_data`, and starts `./start.sh run server` inside the container.

## Railway

Railway builds from the included `Dockerfile` and starts through the same server entrypoint: `./start.sh run server`.

### Required settings

- `OPENAI_COMPATIBLE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`
- `PUBLIC_BASE_URL=https://your-service.up.railway.app` or Railway's `RAILWAY_PUBLIC_DOMAIN` fallback
- `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
- `TELEGRAM_ALLOWED_USER_IDS` or `TELEGRAM_ALLOWED_USERNAMES`

### Recommended settings

- `COMPOSIO_API_KEY` for connector integrations such as Google Sheets and Instagram
- Model defaults live in `opentulpa.config.yaml` (`LLM_MODEL=z-ai/glm-5.1`, `LLM_REASONING_EFFORT=medium`, `WAKE_EXECUTION_MODEL=z-ai/glm-5.1`, Gemini Flash for memory/media, Gemini Flash Lite for the business knowledge oracle)

Browser Use reuses `MULTIMODAL_LLM` by default unless `BROWSER_USE_MODEL` is set.

If `OPENAI_COMPATIBLE_BASE_URL` is not OpenRouter, review `opentulpa.config.yaml` before startup. The provider must have valid model IDs for `llm_model`, `wake_execution_model`, `workflow_setup_input_classifier_model`, `memory_llm_model`, `multimodal_llm`, `business_knowledge_oracle_model`, `openai_compatible_embedding_model`, and optional `browser_use_model`. File, image, browser, memory, workflow setup, and source-grounded knowledge features will not work correctly if those roles point at unavailable or incompatible models. When an API key is present, `start.sh` calls the provider's OpenAI-compatible `/models` endpoint and warns if configured model IDs are missing from the catalog; it still cannot infer capabilities such as multimodal support from providers that do not expose those flags.

### Optional settings

- `COMPOSIO_DEFAULT_CALLBACK_URL`
- `AGENT_PROMPT_CACHING_ENABLED=1|0`
- `TELEGRAM_SUPPORT_USER_IDS` or `TELEGRAM_SUPPORT_USERNAMES`

### Railway setup checklist

1. Create a Railway project from this repo
2. Add one volume mounted at `/app/opentulpa_data`
3. Set:
   - `OPENAI_COMPATIBLE_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_WEBHOOK_SECRET`
   - `PUBLIC_BASE_URL` if you do not want to rely on Railway's `RAILWAY_PUBLIC_DOMAIN` fallback
   - `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
   - `TELEGRAM_ALLOWED_USERNAMES` or `TELEGRAM_ALLOWED_USER_IDS`
4. Optionally set:
   - `COMPOSIO_API_KEY`
   - `COMPOSIO_DEFAULT_CALLBACK_URL`
   - `TELEGRAM_SUPPORT_USER_IDS` or `TELEGRAM_SUPPORT_USERNAMES`
5. Deploy

### What happens automatically

- Railway builds the Docker image
- Python dependencies are installed
- Playwright Chromium is installed
- Telegram webhook is auto-registered when a public URL is available
- Composio callback URL is derived from the public base URL when Composio is configured unless you override it

### Telegram Business notes for production

- the business account owner must connect the bot inside Telegram after deploy
- `PUBLIC_BASE_URL` should be set so webhook registration is explicit; when unset on Railway, OpenTulpa falls back to `RAILWAY_PUBLIC_DOMAIN`
- the same deployed bot and webhook handle both ordinary Telegram chat and Telegram Business updates
- OpenTulpa persists Telegram Business inbox state locally, so use persistent storage
- connected Telegram Business accounts are not env vars; Telegram sends `business_connection` updates and OpenTulpa stores the resulting `business_connection_id` under a `customer_id`
- Telegram Business intake workflows bind to that stored `business_connection_id`; if a Business account is not visible, inspect `getWebhookInfo`, `/debug_logs`, and `.opentulpa/telegram_business.db` instead of adding account ids to Railway variables

## Telegram owner and support access

`TELEGRAM_ALLOWED_USERNAMES` and `TELEGRAM_ALLOWED_USER_IDS` are the owner/operator allowlist for normal bot chat. Configure at least one of them.

Allowed users are not automatically pooled into one owner tenant. Each normal allowed Telegram chat gets its own owner session by default, with a default `customer_id` shaped like `telegram_<user_id>`. If several humans should operate on the same customer tenant without sharing the owner's chat history, configure them as support operators instead and have them bind to that tenant.

## Support operator access

Support mode is optional. If no support allowlist is configured, support commands are disabled.

Use support mode when an operator needs to set up or debug a customer's OpenTulpa tenant without sharing the owner's Telegram chat history.

Configure one or both:

```bash
TELEGRAM_SUPPORT_USER_IDS=123456789,987654321
TELEGRAM_SUPPORT_USERNAMES=operator1,operator2
```

Support operators can use:

- `/support_customers` to list known customer tenants and operational signals
- `/support_bind <number-or-customer_id>` to act as that customer tenant
- `/support_whoami` to inspect the current binding and support thread
- `/support_unbind` to clear the binding

Support chat history stays in a support-specific thread. It does not pollute the owner's main chat history. Customer-facing proactive events still go to the owner by default.

## Persistence

OpenTulpa stores durable state in:

- `.opentulpa`
- `tulpa_stuff`

For Railway, use one mounted volume and set:

```bash
OPENTULPA_DATA_ROOT=/app/opentulpa_data
```

Startup aliases both storage directories into that mounted root.
