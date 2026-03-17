<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa Logo" />
</p>

# OpenTulpa

OpenTulpa is a self-hosted personal AI agent that remembers your context, executes real tasks, and keeps getting better as it learns from you.

It is built for people who want more than chat: integrations, automations, and durable memory, all running on their own infrastructure.

## Why It Feels Different

- **Context that compounds:** it remembers preferences, files, directives, and prior decisions across sessions.
- **Action, not just answers:** it can research, write code, run commands, and execute routines.
- **Integration-native:** built-in web/Slack/Telegram hooks plus generated integrations for external APIs.
- **Safety gate for side effects:** external-impact actions can require explicit approval before execution.

## 30-Second Start (Local API Mode)

Prereqs: Python `3.12+` and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...
uv run python -m opentulpa
```

Health checks:
- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/agent/healthz`

Send your first turn:

```bash
curl -s http://127.0.0.1:8000/internal/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id":"demo_user",
    "thread_id":"chat-demo_user",
    "text":"Find the top 3 trends in AI agents this week and summarize with sources."
  }'
```

Note: setup is ~30 seconds once prerequisites are installed. First dependency install can take longer.

## What You Can Connect

- **Telegram interface (optional):** chat, files, voice notes, approval buttons, `/setup`, `/fresh`, `/status`.
- **Slack integration (optional):** list channels, read history, post messages after user consent.
- **Web intelligence:** web search + URL/file fetching (HTML, PDF, DOCX, image analysis).
- **Browser automation (optional):** local Browser Use tasks for dynamic websites.
- **Skills:** reusable `SKILL.md` capabilities with user/global scope and persistence.
- **Routines:** cron or one-time scheduled automations with durable storage.

## How OpenTulpa Creates Value

1. You describe a workflow in plain language.
2. OpenTulpa plans and executes with tools (research, files, code, terminal, APIs).
3. It saves useful patterns as reusable skills and routines.
4. Future requests get faster and more personalized because memory and artifacts persist.

Generated scripts/artifacts are tracked under local storage (for example `tulpa_stuff/` and `.opentulpa/`), so your automation stack stays inspectable and editable.

## Safety and Control

- External-impact actions can be routed through an approval broker before execution.
- Approval records are durable, single-use, and time-limited.
- Public access is restricted to webhook + health endpoints; internal routes are server-local by design.
- No external database is required by default (SQLite + local vector storage).

## Optional: Telegram in 2 Minutes

1. Create a bot via `@BotFather`.
2. Add `TELEGRAM_BOT_TOKEN` to `.env`.
3. Run:

```bash
./start.sh
```

For cloud deploys with a public URL (`PUBLIC_BASE_URL` or `RAILWAY_PUBLIC_DOMAIN`), startup can auto-register the webhook.

## Deploy

- Dockerfile included.
- Railway-ready config included.
- For durability across redeploys, mount `/app/.opentulpa`.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Chat Cookbook](docs/CHAT_COOKBOOK.md)
- [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md)
