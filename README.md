<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa Logo" />
</p>

<h1 align="center">OpenTulpa</h1>

<p align="center">
  <strong>Your agent shouldn't forget everything the moment a conversation ends.</strong><br/>
  A self-hosted persistent agent runtime for developers who build workflows, not just chatbots.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#how-it-works">How It Works</a> · <a href="#why-opentulpa">Why OpenTulpa</a> · <a href="docs/ARCHITECTURE.md">Architecture</a> · <a href="docs/DEPLOYMENT.md">Deploy</a> · <a href="docs/CHAT_COOKBOOK.md">Cookbook</a>
</p>

---

## The Problem

Every agent framework demos beautifully on the first request. Then the session ends, the state evaporates, and the next run starts from scratch. Context is gone. Decisions are forgotten. That workflow you just painstakingly walked through? You get to do it again.

**OpenTulpa doesn't reset at the prompt boundary.**

It persists context, artifacts, skills, routines, approvals, and thread rollups across sessions — so your agent gets *better* over time instead of starting over every time.

---

## What You Get

🧠 **Memory that actually persists** — context, preferences, and prior decisions survive across sessions and restarts

⚙️ **Real execution, not just text generation** — web retrieval, browser automation, file handling, generated scripts, scheduled routines

🔁 **Skills that compound** — repeated workflows become reusable capabilities your agent can call on forever

🛡️ **Guardrails where it matters** — external-impact actions go through an approval broker with durable, single-use, time-limited gates

🔍 **Fully inspectable, fully local** — everything lives on your machine, in readable files and local storage. No black boxes.

📡 **Telegram-native with an API escape hatch** — chat with it naturally through Telegram, or hit the internal API for programmatic control

---

## See It In Action

> *"Monitor this market every morning, summarize changes, and send me a brief."*

Here's what happens:

1. **Fetches** relevant sources and prior context from durable state
2. **Extracts** and summarizes the important changes
3. **Stores** the brief as a durable artifact you can reference later
4. **Saves** the entire workflow as a reusable routine
5. **Improves** on the next run — reusing your context, preferences, and feedback

No re-prompting. No copy-pasting old outputs. It just picks up where it left off.

---

## Quick Start

### Prerequisites

- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI-compatible API key

### 30-Second Setup

```bash
git clone <repo-url>
cd opentulpa
cp .env.example .env
```

Add your key to `.env`:

```bash
OPENAI_COMPATIBLE_API_KEY=...
```

Run it:

```bash
./start.sh --app
```

That's it. Health checks at `http://127.0.0.1:8000/healthz` and `http://127.0.0.1:8000/agent/healthz`.

### Connect Telegram (Recommended)

Telegram is the primary interface and where OpenTulpa really shines.

1. Create a bot via `@BotFather`
2. Add `TELEGRAM_BOT_TOKEN` to `.env`
3. Install `cloudflared`
4. Run:

```bash
./start.sh
```

`start.sh` handles Python deps, Playwright Chromium, and `cloudflared` tunnel setup automatically.

### Docker

```bash
docker build -t opentulpa .
docker run --rm -p 8000:8000 --env-file .env opentulpa
```

The image comes with Python dependencies, Node.js/npm, and Playwright pre-installed.

### Railway

1. Create a Railway project from this repo
2. Add one volume at `/app/opentulpa_data`
3. Set `OPENAI_COMPATIBLE_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `OPENTULPA_DATA_ROOT=/app/opentulpa_data`
4. Optionally set `TELEGRAM_WEBHOOK_SECRET` and `PUBLIC_BASE_URL`
5. Deploy

Full checklist in [Deployment docs](docs/DEPLOYMENT.md).

<details>
<summary><strong>More setup options</strong></summary>

#### Browser Automation

Installed by default. Skip it with:

```bash
./start.sh --no-browser-use
```

Runs locally inside OpenTulpa — no Browser Use Cloud required.

#### Composio (Optional)

Connect to external services through Composio by adding:

```bash
COMPOSIO_API_KEY=...
```

If not set, the Composio SDK never loads. When configured, OpenTulpa can authenticate against supported third-party services and use integrations on behalf of the active user.

OpenTulpa computes the OAuth callback URL automatically. Override only if needed:

```bash
COMPOSIO_DEFAULT_CALLBACK_URL=https://your-public-base/webhook/composio/callback
```

#### Script Modes

| Command | What it does |
|---|---|
| `./start.sh` | Install + run in quick-tunnel manager mode |
| `./start.sh --app` | Install + run in direct app mode |
| `./start.sh install` | Install/setup only |
| `./start.sh run --app` | Run only, skip install |

Control via `.env`: `START_MODE`, `INSTALL_BROWSER_USE`, `INSTALL_CLOUDFLARED`

</details>

---

## Why OpenTulpa

### Most agent frameworks throw away the most valuable part of every interaction.

They discard the operational state — the preferences learned, the decisions made, the context built up — that would make the next run faster, cheaper, and more accurate. You're left re-explaining things you already covered three sessions ago.

OpenTulpa was built around a different assumption: **the reusable parts of work should persist.**

| | Typical Agent | OpenTulpa |
|---|---|---|
| Context between sessions | ❌ Gone | ✅ Persisted and retrieved automatically |
| Repeated workflows | Manual re-prompting | Saved as reusable skills and routines |
| File outputs and artifacts | Lost in chat history | Stored locally, inspectable, referenceable |
| Side effects | YOLO | Gated behind approval with audit trail |
| Prompt costs | Full resend every turn | Provider-aware caching for stable prefixes |
| Hosting | Someone else's servers | Your machine. Your data. |

---

## How It Works

```
capture context → retrieve durable state → plan → execute with tools → gate side effects → persist outputs
```

```text
Telegram / Internal API / Events
              │
           FastAPI
              │
  capture context + retrieve state
              │
       LangGraph runtime
              │
 plan + tool execution + validation
              │
 approval gate for external actions
              │
 persist artifacts / skills / routines / rollups
              │
   local durable state (.opentulpa/, tulpa_stuff/)
```

**Under the hood:**

- **FastAPI** for webhook and internal routes
- **LangGraph** runtime for turn execution, validation, guardrails, and claim checking
- **Context services** for profiles, files, event backlog, thread rollups, and aliases
- **Skill store** for reusable capabilities
- **Scheduler** for one-off and recurring routines
- **Approval broker** for external-impact actions
- **Local storage** using SQLite + embedded Qdrant for vector search

No external database required by default. Everything lives on disk.

---

## What You Can Build

OpenTulpa is a strong fit for work that repeats, compounds, or has real-world consequences.

### Core Workflows

| Use Case | Example Request |
|---|---|
| **Market monitoring** | *"Monitor this market every morning and send me a concise brief."* |
| **Inbox management** | *"Summarize the most important unread items and draft replies."* |
| **Document intelligence** | *"Read this PDF, extract the decisions, and remember them for future work."* |
| **API automation** | *"Build an integration for this API, save it as a reusable skill, and schedule it."* |
| **Status reporting** | *"Check what changed in this project since yesterday and draft a status update."* |
| **Developer assistants** | Any self-hosted agent that needs memory, tools, execution, and guardrails in one runtime |

### With Composio Enabled

When you connect Composio, OpenTulpa can authenticate against third-party services and run automations that actually touch your real accounts. This is where it starts to feel less like a chatbot and more like a digital employee.

| Use Case | Example Request |
|---|---|
| **Social media management** | *"Check my Instagram every 5 minutes and reply to DMs about business collaborations or partnership inquiries on my behalf."* |
| **CRM automation** | *"When a new lead comes in on HubSpot, research their company, score the lead, and draft a personalized outreach email."* |
| **Calendar orchestration** | *"Look at my Google Calendar every morning, flag conflicts, and send me a Telegram summary with suggested reschedules."* |
| **GitHub ops** | *"Watch this repo for new issues labeled 'bug', reproduce them if possible, and post a triage comment with severity and suggested fix."* |
| **Slack delegation** | *"Monitor my Slack channels, summarize anything I'm tagged in, and draft responses I can approve before sending."* |
| **Cross-platform reporting** | *"Pull this week's analytics from Google Analytics and Stripe, combine them into a brief, and push it to a Notion page every Monday."* |
| **Email workflow** | *"Watch for emails from this client, extract any action items, add them to my Todoist, and send me a digest at end of day."* |

These aren't theoretical — OpenTulpa persists the routine, remembers your preferences from last time, and improves with each run. The approval gate ensures nothing gets sent, posted, or purchased without your sign-off when you want it.

---

## What You Can Connect

- **Telegram** — chat, files, voice notes, approval buttons, `/setup`, `/fresh`, `/status`
- **Internal API** — programmatic access to the runtime for custom integrations
- **Web intelligence** — search, URL/file fetching (HTML, PDF, DOCX), image analysis
- **Browser automation** — local Playwright sessions for dynamic websites
- **Composio** — OAuth-based connections to third-party apps (optional)
- **Skills** — reusable `SKILL.md` capabilities with user/global scope
- **Routines** — cron or one-time scheduled automations with durable storage

All generated scripts and artifacts are tracked under local storage (`tulpa_stuff/`, `.opentulpa/`) — inspectable and editable at all times.

---

## Safety Model

OpenTulpa doesn't assume every action is safe to execute blindly.

- **Read-only and internal actions** proceed directly
- **External writes, purchases, or costly actions** require approval
- **Unknown scope** defaults to requiring approval
- **Approvals** are durable, single-use, and time-limited
- **Public exposure** is limited to webhook and health routes; internal routes stay local/private
- **Memory vectors** are stored in an embedded local Qdrant instance — nothing leaves your infrastructure

For durable deploys, mount `/app/.opentulpa` so skills, approvals, checkpoints, and memory survive redeploys.

---

## Provider-Aware Prompt Caching

OpenTulpa separates stable prompt prefixes from turn-specific context so supported providers can reuse cached segments instead of re-billing the same instructions every turn.

| Provider | Caching Strategy |
|---|---|
| Anthropic/Claude | Request-level cache control |
| Gemini | Per-message cache breakpoints on stable prefix |
| OpenAI-compatible | Automatic caching (no explicit markers needed) |

Controlled via `AGENT_PROMPT_CACHING_ENABLED`.

---

## Built To Be Hacked On

OpenTulpa ships as a ready-to-run agent **and** a reference architecture you can extend.

- **Add tools** in the tool registry
- **Add routes** under `src/opentulpa/api/routes`
- **Add interfaces** under `src/opentulpa/interfaces`
- **Extend approval logic** through the policy and broker layers
- **Add durable skills** instead of hardcoding workflows into prompts

If you've been looking for a persistent, guarded, tool-using agent runtime that you can actually own and modify — this is it.

---

## Docs

| | |
|---|---|
| 📐 [Architecture](docs/ARCHITECTURE.md) | How the internals fit together |
| 🚀 [Deployment](docs/DEPLOYMENT.md) | Production deploy checklist |
| 💬 [Chat Cookbook](docs/CHAT_COOKBOOK.md) | Example conversations and patterns |
| 🛡️ [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md) | Guidelines for connecting external tools safely |

---

<p align="center">
  <strong>Stop re-explaining. Start compounding.</strong><br/>
  <a href="#quick-start">Get started in 30 seconds →</a>
</p>