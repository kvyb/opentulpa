<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa Logo" />
</p>

# OpenTulpa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-yes-green.svg)]()

OpenTulpa is a personal AI agent you run on your own server, accessible through Telegram and a direct internal chat API.

It does two things most assistants don't:

- **It knows you.** It remembers facts you tell it, files you send it, preferences you express, and context from every past conversation — and uses all of that without being asked.
- **It builds its own tools and skills.** Describe what you want in chat, and it can write integrations, run them, schedule them, and save reusable skills to solve what you ask.

The longer you run it, the more personal and capable it gets.

> One key. One command. Add interfaces and browsing only when you need them.

## Start in 5 Minutes

1. Set `OPENROUTER_API_KEY` in `.env`.
2. Run `uv run python -m opentulpa`.
3. Start chatting via the internal chat API.

Full steps are in [Quick Start](#quick-start).

---

## What It Can Do

### It Learns Who You Are

OpenTulpa remembers everything you share with it — not as a search index, but as context it actively uses when responding:

- Tell it your timezone, work schedule, or preferred tone or persona once — it applies that everywhere.
- Send it a PDF, image, or voice note — it understands and stores it for later reference.
- Mention a preference, a constraint, or a fact about your life — it factors that in unprompted from then on.
- Share a document and say "keep this in mind" — it will, across future sessions.

Early conversations are generic. Later ones feel like talking to someone who actually knows your context and acts like it.

### It Builds Its Own Integrations

Describe a workflow and OpenTulpa writes the code, runs it, schedules it, and saves it as a reusable skill — entirely from inside the chat. No dev environment. No context switching.

```text
"Pull the top 5 trending GIFs from Giphy and send me one every morning."
→ Writes the Giphy API script, schedules the job, done.

"Here's my Alpaca key. Give me a markets overview every weekday at 7am —
top movers, my portfolio delta, any earnings today."
→ Stores the key, writes the integration, registers the recurring job.

"Build me a Slack bot that posts a daily standup prompt to #engineering at 9am."
→ Writes the full Slack integration from scratch, saves it as a reusable skill.

"Register me on Moltbook." (with Browser Use connected)
→ Opens a browser, fills the form, completes the flow autonomously.

"Here's my Notion token. Summarize everything updated this week into a digest."
→ Done. Say "schedule that" and it registers the recurring job immediately.

"Write me a GitHub webhook that posts a Slack message on every failed CI run."
→ Builds both ends of the integration, inside the chat, from a single message.
```

If there's a public API or a service with documentation, OpenTulpa can integrate it without you writing a single line of code. Hand it a key or describe a service → it figures out the API → writes working code → runs it → stores it so it never rebuilds from scratch.

### Everything Else It Can Do

- **Internet research:** browse URLs, read pages, summarize findings.
- **Multimodal input:** send text, files, images, or voice notes — it handles all of them.
- **Background automation:** scheduled tasks, recurring jobs, long-running routines.
- **Artifact storage:** generated scripts and outputs are saved and reused across sessions.
- **Skills:** recurring workflows become named capabilities it maintains, loads and applies automatically.

---

## Cost Profile (Approximate)

- Estimated cost per agentic loop: **~$0.011**
- Most tasks complete in **~3-4 loops**
- Estimated cost per task/message: **~$0.033-$0.044**
- Estimated cost for **100 chat messages** (at 3-4 loops each): **~$3.30-$4.40 total**
- Midpoint estimate (3.5 loops avg): **~$3.85**
- These estimates assume the current default model mix: main runtime `gemini-3-flash-preview`, guardrail classifier `minimax/minimax-m2.5`, and wake classifier `gemini-3-flash-preview` unless `WAKE_CLASSIFIER_MODEL` is set.

This makes the agent runtime inexpensive for day-to-day use, even with tool-driven multi-step reasoning.

---

## Prerequisites

- Python `3.10+`
- [`uv`](https://docs.astral.sh/uv/) installed
- Telegram bot token from `@BotFather` *(optional, Telegram interface only)*
- `cloudflared` installed *(optional, local Telegram webhook tunneling only)*

---

## Quick Start

**1. Configure:**

```bash
cp .env.example .env
```

```bash
# .env
OPENROUTER_API_KEY=your_key
```

Current runtime expects OpenRouter-compatible chat routing for the main agent path.

**2. Start (API mode):**

```bash
uv run python -m opentulpa
```

**3. Optional: Telegram interface**

Create your Telegram bot first:
- Chat with `@BotFather` → run `/newbot` → copy the token.
- Open your new bot and press `Start`.

Add this env var:

```bash
TELEGRAM_BOT_TOKEN=your_botfather_token
```

Then start Telegram + webhook manager:

```bash
./start.sh
```

`start.sh` will:
- Start FastAPI on `:8000`
- Launch a `cloudflared` quick tunnel
- Auto-register the Telegram webhook at `<public_url>/webhook/telegram`
- Auto-generate a Telegram webhook secret for that run when missing
- Default to `HOST=127.0.0.1` for local-only bind unless you override `HOST`

Set this env var in `.env` for local Telegram mode:

```bash
TELEGRAM_BOT_TOKEN=your_botfather_token
```

If you are exposing OpenTulpa on your own public domain (without the quick tunnel manager), set the webhook manually:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://yourdomain.com/webhook/telegram"
```

> Telegram requires a public URL. For local dev, `cloudflared` or `ngrok` both work.

**4. Optional: Browser Use capability**

Add this env var only if you want Browser Use browsing tools:

```bash
BROWSER_USE_API_KEY=your_key
```

**5. Start chatting.** Try:

```text
Create a daily 8:30am Gmail summary and post the top 5 action items here.
```

**Stop:** `Ctrl+C` in the terminal.

**Health checks:**
- `http://localhost:8000/healthz`
- `http://localhost:8000/agent/healthz`

## Deployment

Deployment is documented in:
- [Deployment Guide](docs/DEPLOYMENT.md)

## Direct API turn (non-Telegram)

You can run conversation turns directly through the internal chat route:

```bash
curl -s http://127.0.0.1:8000/internal/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id":"test_user",
    "thread_id":"chat-test_user",
    "text":"Search the web for recent Pacific storms and summarize the risks."
  }'
```

---

## Skills

Skills are `SKILL.md` files the agent writes, stores, and loads on demand:

- **Scopes:** `user` (personal) and `global` (shared) — `user` always takes priority.
- **CRUD:** `skill_list`, `skill_get`, `skill_upsert`, `skill_delete`.
- **Self-authoring:** OpenTulpa can write new skills directly from chat and reuse them in future sessions without being reminded.

```text
"Create a reusable skill called Customer Follow-up Writer that takes a thread
summary and outputs 3 concise follow-up drafts in my tone."
```

---

## Configuration

**Required (all modes):**

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM routing and embeddings |

**Optional (Telegram interface):**

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram interface |

**Optional (Browser Use capability):**

| Variable | Purpose |
|---|---|
| `BROWSER_USE_API_KEY` | Browser automation (form filling, web flows) |

**Core stack:** FastAPI · LangGraph · LangChain · mem0 · SQLite · APScheduler

| Component | Role |
|---|---|
| `mem0` | Memory layer — persists user context across sessions |
| `APScheduler` | Recurring jobs and background automation |
| `SQLite` | Local persistence, no external DB required |

**External services** (only active when you configure them):
- OpenRouter — LLM routing and embeddings
- Telegram Bot API
- Browser Use Cloud *(optional)*
- Any API you integrate yourself

**Runtime data:**
- `.opentulpa/` — memory, profiles, context
- `tulpa_stuff/` — generated scripts and artifacts *(mostly gitignored)*
- `.opentulpa/logs/agent_behavior.jsonl` — structured agent decision/execution trace

---

## Safety and Privacy

- External-impact actions (writes, sends, posts) require explicit per-action approval — single-use, expiring, scoped to the requesting user only.
- Telegram webhook requests support secret verification.
- Public internet requests are denied for all routes except `/webhook/*` and health checks (`/healthz`, `/agent/healthz`).
- `/webhook/telegram` requires Telegram secret auth (`x-telegram-bot-api-secret-token`).
- `/internal/*` is intended for server-local traffic only (`localhost`/private network).
- No built-in telemetry or user-tracking pipeline.
- Fully open source (MIT). Self-hosted by default.
- All runtime data stays local unless you explicitly configure an external service.

---

## Project Structure

```
src/opentulpa/
├── agent/         # LangGraph runtime, graph, tool orchestration, context policy
├── api/           # FastAPI composition and internal routes
├── application/   # Use-case orchestrators (turns, wakes, approval execution)
├── approvals/     # Guardrail broker, adapters, persistence
├── context/       # Profiles, event backlog, file vault, rollups
├── domain/        # Domain contracts (conversation request/response)
├── interfaces/    # Telegram transport and streaming relay
├── integrations/  # External clients (web/slack/etc.)
├── memory/        # mem0 integration layer
├── policy/        # Approval intent/policy evaluation
├── scheduler/     # Routine scheduling and persistence
├── tasks/         # Task worker + sandbox execution
└── tools/         # Local tool modules
tulpa_stuff/      # Generated scripts and runtime artifacts
```

Reference docs:
- [Architecture](docs/ARCHITECTURE.md)
- [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md)

---

*If this is useful to you, consider starring the repo — it helps others find it.*
