<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa Logo" />
</p>

<h1 align="center">OpenTulpa</h1>

<p align="center">
  <strong>A digital employee that lives on your infrastructure.</strong><br/>
  Self-hosted. Persistent memory. Real execution. Gets better at your job the longer it works for you.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#how-it-works">How It Works</a> · <a href="#why-opentulpa">Why OpenTulpa</a> · <a href="docs/ARCHITECTURE.md">Architecture</a> · <a href="docs/DEPLOYMENT.md">Deploy</a> · <a href="docs/CHAT_COOKBOOK.md">Cookbook</a>
</p>

---

## Not Another Chatbot

Chatbots answer questions. Employees get things done.

OpenTulpa is closer to the second thing. It remembers what you told it last week. It learns how you like things done. It logs into your tools, runs work on a schedule, writes files, builds integrations, and asks for your approval before doing anything risky. And it runs entirely on your infrastructure — no SaaS middleman sitting between your agent and your data.

The longer it runs, the more it knows, the less you have to explain, and the more it handles on its own.

**Think of it as hiring someone who never sleeps, never forgets a briefing, and never sends that email without checking with you first.**

---

## What Your Digital Employee Can Do

🧠 **Remembers everything** — context, preferences, decisions, and past work persist across sessions and restarts. You brief it once.

⚙️ **Actually executes** — browses websites, fetches documents, writes files, generates scripts, calls APIs. Not just text — real output.

🔁 **Learns on the job** — repeated workflows get saved as reusable skills and scheduled routines. Your employee builds its own playbook.

🔌 **Connects to your stack** — Telegram, email, calendars, CRMs, GitHub, Slack, Notion, and hundreds more through Composio. It works where you work.

📅 **Works while you sleep** — cron-based routines and one-off scheduled tasks run in the background. Morning briefs, nightly reports, continuous monitoring.

🛡️ **Asks before acting** — external writes, purchases, and anything with real-world consequences go through an approval gate. Durable, single-use, time-limited.

🔍 **Fully transparent** — every artifact, skill, routine, and decision lives in readable files on your disk. Inspect anything. Edit anything. No black boxes.

---

## See It In Action

> *"Monitor this market every morning, summarize changes, and send me a brief."*

Here's what your digital employee does:

1. **Fetches** relevant sources and pulls in prior context it already knows about
2. **Extracts** and summarizes what actually changed
3. **Stores** the brief as a durable artifact you can reference later
4. **Saves** the entire workflow as a reusable routine
5. **Runs it again tomorrow** — better, because it remembers your feedback and preferences

You don't re-explain. You don't re-prompt. You just get the brief.

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

Recommended model stack:

```bash
LLM_MODEL=z-ai/glm-5.1:nitro
WAKE_EXECUTION_MODEL=z-ai/glm-5.1:nitro
MEMORY_LLM_MODEL=google/gemini-3-flash-preview
MULTIMODAL_LLM=google/gemini-3-flash-preview
GUARDRAIL_CLASSIFIER_MODEL=google/gemini-3-flash-preview
```

This is the current recommended production split in this repo:
- `GLM 5.1` for main chat and wake execution
- `Gemini Flash` for memory extraction, multimodal understanding, and guardrail classification
- If your main chat model is not multimodal, set `MULTIMODAL_LLM` so media and browser screenshot handling keeps working

Run it:

```bash
./start.sh --app
```

That's it. Health checks at `http://127.0.0.1:8000/healthz` and `http://127.0.0.1:8000/agent/healthz`.

### Connect Telegram (Recommended)

Telegram is the primary interface — think of it as your employee's desk where you walk up and give instructions.

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

Give your digital employee access to real-world services:

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

### You wouldn't hire someone who forgets everything at the end of each day.

That's what most agent frameworks do. They answer a request, maybe call a tool, then throw away every bit of operational state that would have made tomorrow's work easier. You're stuck re-briefing, re-prompting, re-explaining context that should have been obvious.

OpenTulpa was built around a different assumption: **a useful employee retains what they learn and gets better over time.**

| | Typical Agent | OpenTulpa |
|---|---|---|
| Memory | ❌ Gone after the session | ✅ Persisted and retrieved automatically |
| Repeated work | You re-prompt every time | Saved as reusable skills and scheduled routines |
| Output files and artifacts | Lost in chat history | Stored locally, inspectable, referenceable |
| Side effects | Fires and forgets | Gated behind approval with audit trail |
| Prompt costs | Full resend every turn | Provider-aware caching for stable prefixes |
| Your data | On someone else's servers | On your machine. Period. |
| Availability | When you prompt it | Runs routines on schedule, even while you're offline |

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

## What You Can Delegate

The whole point is to hand off real work — not just ask questions.

### Day-to-Day Operations

| Use Case | What You Say |
|---|---|
| **Morning brief** | *"Every morning at 8am, check my calendar, flag conflicts, summarize what's on my plate, and send it to me on Telegram."* |
| **Market monitoring** | *"Monitor these 5 competitors' pricing pages daily. Summarize anything that changed and keep a running log."* |
| **Inbox triage** | *"Summarize the most important unread items from my inbox and draft replies in my tone."* |
| **Document review** | *"Read this PDF, extract the key decisions, and remember them for future reference."* |
| **Status reporting** | *"Check what changed in this project since yesterday and draft a status update for the team."* |

### With Composio — Your Employee Gets Access to Your Stack

Connect Composio and OpenTulpa stops being a local sandbox. It becomes an employee with real logins that can operate across your tools, on your behalf, on a schedule.

| Use Case | What You Say |
|---|---|
| **Social media management** | *"Check my Instagram every 5 minutes. If someone DMs me about a collaboration or business inquiry, reply professionally and schedule a follow-up."* |
| **Lead handling** | *"When a new lead lands in HubSpot, research their company, score them, and draft a personalized outreach email for my review."* |
| **GitHub triage** | *"Watch this repo for new issues labeled 'bug'. Try to reproduce them, then post a triage comment with severity and a suggested fix."* |
| **Slack delegation** | *"Monitor my Slack channels for anything I'm tagged in. Summarize the threads and draft responses I can approve before sending."* |
| **Cross-platform reporting** | *"Every Monday, pull analytics from Google Analytics and Stripe, combine them into a brief, and push it to our Notion workspace."* |
| **Email-to-task pipeline** | *"Watch for emails from this client. Extract action items, add them to my Todoist, and send me a digest at end of day."* |
| **API integration work** | *"Build an integration for this API, save it as a reusable skill, and schedule it to run nightly."* |

These aren't demos — they're routines. OpenTulpa saves the workflow, remembers your preferences from last run, and improves over time. And the approval gate ensures nothing gets sent, posted, or purchased without your go-ahead when it matters.

---

## What You Can Connect

- **Telegram** — chat, files, voice notes, approval buttons, `/setup`, `/fresh`, `/status`
- **Internal API** — programmatic access to the runtime for custom integrations
- **Web intelligence** — search, URL/file fetching (HTML, PDF, DOCX), image analysis
- **Browser automation** — local Playwright sessions for dynamic websites
- **Composio** — OAuth-based connections to hundreds of third-party apps (optional)
- **Skills** — reusable `SKILL.md` capabilities with user/global scope
- **Routines** — cron or one-time scheduled automations with durable storage

All generated scripts and artifacts are tracked under local storage (`tulpa_stuff/`, `.opentulpa/`) — inspectable and editable at all times.

---

## Safety Model

A digital employee with no guardrails is a liability. OpenTulpa takes this seriously.

- **Read-only and internal actions** proceed directly — no friction where there's no risk
- **External writes, purchases, or costly actions** require your explicit approval
- **Unknown scope** defaults to asking first
- **Approvals** are durable, single-use, and time-limited — no stale blanket permissions
- **Public exposure** is limited to webhook and health routes; internal routes stay local/private
- **All data** stays on your infrastructure — memory vectors in embedded local Qdrant, state in SQLite

For durable deploys, mount `/app/.opentulpa` so skills, approvals, checkpoints, and memory survive redeploys.

---

## Provider-Aware Prompt Caching

Your digital employee shouldn't cost more to run each day than it saves you. OpenTulpa separates stable prompt prefixes from turn-specific context so supported providers can reuse cached segments instead of re-billing the same instructions every turn.

| Provider | Caching Strategy |
|---|---|
| Anthropic/Claude | Request-level cache control |
| Gemini | Per-message cache breakpoints on stable prefix |
| OpenAI-compatible | Automatic caching (no explicit markers needed) |

Controlled via `AGENT_PROMPT_CACHING_ENABLED`.

---

## Built To Be Hacked On

OpenTulpa ships as a ready-to-run digital employee **and** a reference architecture you can reshape to fit your operation.

- **Add tools** in the tool registry
- **Add routes** under `src/opentulpa/api/routes`
- **Add interfaces** under `src/opentulpa/interfaces`
- **Extend approval logic** through the policy and broker layers
- **Add durable skills** instead of hardcoding workflows into prompts

If you've been looking for a persistent, guarded, tool-using agent runtime that you actually own and can modify — one that works more like an employee than a chatbot — this is it.

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
  <strong>Stop re-explaining. Start delegating.</strong><br/>
  <a href="#quick-start">Hire your digital employee in 30 seconds →</a>
</p>
