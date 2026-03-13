# Deployment Guide

This guide covers production deployment for OpenTulpa using Docker and Railway.

## Docker / Railway (Env-Only)

The repo includes a production `Dockerfile` so Railway can deploy directly.

### Required env vars

- `OPENROUTER_API_KEY`
- `TELEGRAM_BOT_TOKEN`

### Optional env vars

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
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_WEBHOOK_SECRET` (recommended)
   - `PUBLIC_BASE_URL` (optional when `RAILWAY_PUBLIC_DOMAIN` is available)
4. Deploy.

## Persistence (recommended)

Mount a volume for `/app/.opentulpa` so memory, skills, approvals, and checkpoints survive redeploys.
