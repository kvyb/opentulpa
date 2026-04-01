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
OPENROUTER_API_KEY=...
```

Install and run:

```bash
./start.sh --app
```

Health checks:
- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

If you want Telegram locally:

1. set `TELEGRAM_BOT_TOKEN`
2. run `./start.sh`

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

- `OPENROUTER_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Recommended

- `TELEGRAM_WEBHOOK_SECRET`
- `PUBLIC_BASE_URL=https://your-service.up.railway.app`
- `OPENTULPA_DATA_ROOT=/app/opentulpa_data`

### Setup

1. Create a Railway project from this repo.
2. Add one volume mounted at `/app/opentulpa_data`.
3. Set:
   - `OPENROUTER_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
4. Optionally set:
   - `TELEGRAM_WEBHOOK_SECRET`
   - `PUBLIC_BASE_URL`
5. Deploy.

### What happens automatically

- Railway builds the Docker image
- Python dependencies are installed
- Playwright Chromium is installed
- Telegram webhook is auto-registered when a public URL is available

### Persistence

OpenTulpa stores state in:
- `.opentulpa`
- `tulpa_stuff`

For Railway, use one mounted volume and set:

```bash
OPENTULPA_DATA_ROOT=/app/opentulpa_data
```

Startup aliases both storage directories into that mounted root.
