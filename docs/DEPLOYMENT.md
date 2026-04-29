# Deployment Guide

This guide covers the practical ways to run OpenTulpa today.

If you just want the fastest local path, do this:

```bash
git clone <repo-url>
cd opentulpa
cp .env.example .env
```

Set:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Then run:

```bash
./start.sh --app
```

Health checks:

- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

## Choose a runtime mode

`start.sh` supports two useful modes:

- `--app`: run the FastAPI app directly
- default manager mode: run the app through the quick-tunnel manager flow

In practice:

- use `./start.sh --app` for direct local app runs
- use `./start.sh` when you want the managed local Telegram flow with `cloudflared`

## Local setup

Requirements:

- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)
- an OpenAI-compatible API key

Base setup:

```bash
git clone <repo-url>
cd opentulpa
cp .env.example .env
```

Required `.env` value:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Run locally:

```bash
./start.sh --app
```

## Telegram setup

Telegram is the main operator interface.

For local use:

1. create a bot in `@BotFather`
2. set `TELEGRAM_BOT_TOKEN` in `.env`
3. run `./start.sh`

When you use the default manager flow, `start.sh` will also handle dependency setup for Playwright Chromium and `cloudflared` if needed.

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
| `./start.sh` | Install and run in manager mode |
| `./start.sh --app` | Install and run in direct app mode |
| `./start.sh install` | Install only |
| `./start.sh run --app` | Run only |

Useful `.env` knobs:

- `START_MODE=auto|app|manager`
- `INSTALL_BROWSER_USE=1|0`
- `INSTALL_CLOUDFLARED=auto|1|0`
- `AGENT_PROMPT_CACHING_ENABLED=1|0`

## Docker

The included `Dockerfile` already installs Python dependencies, Node.js/npm, and Playwright Chromium.

Build and run:

```bash
docker build -t opentulpa .
docker run --rm -p 8000:8000 --env-file .env opentulpa
```

Use Docker for local API testing or as the base for cloud deployment.

## Railway

Railway builds from the included `Dockerfile`.

### Required settings

- `OPENAI_COMPATIBLE_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Recommended settings

- `TELEGRAM_WEBHOOK_SECRET`
- `PUBLIC_BASE_URL=https://your-service.up.railway.app`
- `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
- Model defaults live in `opentulpa.config.yaml` (`LLM_MODEL=z-ai/glm-5.1`, `LLM_REASONING_EFFORT=medium`, `WAKE_EXECUTION_MODEL=z-ai/glm-5.1`, Gemini Flash for memory/media/guardrails, Gemini Flash Lite for the business knowledge oracle)

Browser Use reuses `MULTIMODAL_LLM` by default unless `BROWSER_USE_MODEL` is set.

### Optional settings

- `COMPOSIO_API_KEY`
- `COMPOSIO_DEFAULT_CALLBACK_URL`
- `AGENT_PROMPT_CACHING_ENABLED=1|0`
- `TELEGRAM_ALLOWED_USER_IDS` or `TELEGRAM_ALLOWED_USERNAMES`
- `TELEGRAM_SUPPORT_USER_IDS` or `TELEGRAM_SUPPORT_USERNAMES`

### Railway setup checklist

1. Create a Railway project from this repo
2. Add one volume mounted at `/app/opentulpa_data`
3. Set:
   - `OPENAI_COMPATIBLE_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
4. Optionally set:
   - `TELEGRAM_WEBHOOK_SECRET`
   - `PUBLIC_BASE_URL`
   - `COMPOSIO_API_KEY`
   - `COMPOSIO_DEFAULT_CALLBACK_URL`
   - `TELEGRAM_ALLOWED_USERNAMES`
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
- `PUBLIC_BASE_URL` should be set so webhook registration is correct
- the same deployed bot and webhook handle both ordinary Telegram chat and Telegram Business updates
- OpenTulpa persists Telegram Business inbox state locally, so use persistent storage

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
