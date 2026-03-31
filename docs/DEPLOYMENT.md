# Deployment Guide

This guide covers production deployment for OpenTulpa using Docker and Railway.

## Docker / Railway (Env-Only)

The repo includes a production `Dockerfile` so Railway can deploy directly.

### Required env vars

- `OPENROUTER_API_KEY` for the configured OpenAI-compatible model endpoint
- `TELEGRAM_BOT_TOKEN`

### Optional env vars

- `OPENROUTER_BASE_URL` (defaults to OpenRouter; can point at another OpenAI-compatible endpoint)
- `OPENTULPA_DATA_ROOT` (single persistent data root; aliases both `.opentulpa` and `tulpa_stuff` into one mounted volume)
- `TELEGRAM_WEBHOOK_SECRET` (recommended; if omitted, an ephemeral secret is generated at startup)
- `PUBLIC_BASE_URL` (for example `https://your-app.up.railway.app`)
- `BROWSER_USE_HEADLESS` (defaults to `true`)
- `BROWSER_USE_MODEL` (optional Browser Use model override)
- `BROWSER_USE_MAX_CONCURRENT_TASKS` (defaults to `2`)
- `BROWSER_USE_TASK_RETENTION_SECONDS` (defaults to `1800`)

Railway note:
- If `PUBLIC_BASE_URL` is empty and Railway provides `RAILWAY_PUBLIC_DOMAIN`, startup auto-registers Telegram webhook to `https://$RAILWAY_PUBLIC_DOMAIN/webhook/telegram`.

Browser Use local note:
- Ensure Playwright Chromium is installed in the image/runtime (`uv run playwright install --with-deps chromium` in Docker).

## What startup configures automatically

- App binds to `HOST=0.0.0.0`, `PORT` from env (default `8000`).
- Telegram webhook is auto-configured when:
  - `TELEGRAM_BOT_TOKEN` exists, and
  - `PUBLIC_BASE_URL` or `RAILWAY_PUBLIC_DOMAIN` exists.
- Webhook URL is set to `<public_base_url>/webhook/telegram`.
- `secret_token` is sent in `setWebhook` using `TELEGRAM_WEBHOOK_SECRET`.

## Railway quick setup

1. Create a new Railway project from this repo.
2. Railway detects the `Dockerfile` and builds automatically.
3. Set env vars in Railway:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_BASE_URL` if you are not using OpenRouter
   - `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_WEBHOOK_SECRET` (recommended)
   - `PUBLIC_BASE_URL` (optional when `RAILWAY_PUBLIC_DOMAIN` is available)
4. Deploy.

## Persistence (recommended)

Railway services expose a single mounted volume path, so OpenTulpa supports a single data root for cloud deploys.

Recommended Railway setup:

1. Mount one volume at `/app/opentulpa_data`.
2. Set `OPENTULPA_DATA_ROOT=/app/opentulpa_data`.
3. Startup will alias both `/app/.opentulpa` and `/app/tulpa_stuff` into that mounted directory.

This keeps the existing runtime paths stable while making both durable state and generated sandbox files persist across redeploys.
