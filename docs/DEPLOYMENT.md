# Deployment Guide

## Local

Requirements:
- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)
- an OpenAI-compatible API key

Setup:

```bash
git clone <repo-url>
cd opentulpa
cp .env.example .env
```

Set in `.env`:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Install and run:

```bash
./start.sh --app
```

Health checks:
- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

Telegram is the primary interface. For local use:

1. set `TELEGRAM_BOT_TOKEN`
2. run `./start.sh`

Optional Composio support:

```bash
COMPOSIO_API_KEY=...
```

OpenTulpa computes the Composio callback URL from your public base URL when possible. You only need this override if you want to force a specific callback URL:

```bash
COMPOSIO_DEFAULT_CALLBACK_URL=https://your-public-base/webhook/composio/callback
```

If you want Browser Use locally:

- it is installed by default when `./start.sh` runs
- use `./start.sh --no-browser-use` to skip it

Useful script modes:
- `./start.sh`
- `./start.sh --app`
- `./start.sh install`
- `./start.sh run --app`

Useful `.env` knobs:
- `START_MODE=auto|app|manager`
- `INSTALL_BROWSER_USE=1|0`
- `INSTALL_CLOUDFLARED=auto|1|0`
- `AGENT_PROMPT_CACHING_ENABLED=1|0`

## Docker

The included `Dockerfile` already installs Python dependencies, Node.js/npm, and Playwright Chromium:

```bash
docker build -t opentulpa .
docker run --rm -p 8000:8000 --env-file .env opentulpa
```

Use Docker for local API testing or as the base for cloud deploys.

## Railway

Railway builds from the included `Dockerfile`.

### Required

- `OPENAI_COMPATIBLE_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Recommended

- `TELEGRAM_WEBHOOK_SECRET`
- `PUBLIC_BASE_URL=https://your-service.up.railway.app`
- `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
- `LLM_MODEL=z-ai/glm-5.1:nitro`
- `WAKE_EXECUTION_MODEL=z-ai/glm-5.1:nitro`
- `MEMORY_LLM_MODEL=google/gemini-3-flash-preview`
- `TELEGRAM_MEDIA_MODEL=google/gemini-3-flash-preview`
- `GUARDRAIL_CLASSIFIER_MODEL=google/gemini-3-flash-preview`

### Optional

- `COMPOSIO_API_KEY`
- `COMPOSIO_DEFAULT_CALLBACK_URL`
- `AGENT_PROMPT_CACHING_ENABLED=1|0`

### Setup

1. Create a Railway project from this repo.
2. Add one volume mounted at `/app/opentulpa_data`.
3. Set:
   - `OPENAI_COMPATIBLE_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
4. Optionally set:
   - `TELEGRAM_WEBHOOK_SECRET`
   - `PUBLIC_BASE_URL`
   - `COMPOSIO_API_KEY`
   - `COMPOSIO_DEFAULT_CALLBACK_URL`
5. Deploy.

### What happens automatically

- Railway builds the Docker image
- Python dependencies are installed
- Playwright Chromium is installed
- Telegram webhook is auto-registered when a public URL is available
- Composio callback URL is derived from the public base URL when Composio is configured, unless you override it explicitly

### Persistence

OpenTulpa stores state in:
- `.opentulpa`
- `tulpa_stuff`

For Railway, use one mounted volume and set:

```bash
OPENTULPA_DATA_ROOT=/app/opentulpa_data
```

Startup aliases both storage directories into that mounted root.
