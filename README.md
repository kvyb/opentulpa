<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa" width="180"/>
</p>

<h1 align="center">OpenTulpa</h1>

<p align="center">
  <strong>A self-hosted digital employee you brief, equip, and delegate to, in chat.</strong><br/>
  Persistent memory, durable workflow state, human-in-the-loop approvals, and native Telegram &amp; Instagram inbox handling. Runs on your infrastructure.
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#what-you-can-delegate">Delegate</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="docs/DEPLOYMENT.md">Deploy</a> ·
  <a href="docs/CHAT_COOKBOOK.md">Cookbook</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"/>
  <img src="https://img.shields.io/badge/self--hosted-yes-success.svg" alt="Self-hosted"/>
  <img src="https://img.shields.io/badge/status-actively%20developed-brightgreen.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/stack-LangGraph%20%7C%20FastAPI%20%7C%20Mem0%20%7C%20Qdrant%20%7C%20Composio%20%7C%20Playwright-informational" alt="Stack"/>
</p>

<p align="center">
  <sub>Targets OpenAI-compatible providers · App integrations via Composio · No external database required</sub>
</p>

---

## Why OpenTulpa

OpenTulpa is a **self-hosted agent runtime** built for work that repeats. You brief it in chat (goals, tools, source material, escalation rules) and it keeps working across sessions: saving skills, running scheduled routines, handling inbound customer DMs on Telegram and Instagram, and writing outcomes back into your systems.

It works as a personal operator on day one, and becomes a durable workflow employee the moment the work starts repeating. Most "AI agents" forget the job between sessions. OpenTulpa is built the other way around: **brief it once, and it keeps working**, on a runtime you own and can inspect.

|  | Typical agent app | **OpenTulpa** |
|---|---|---|
| Context | Session-bound | Persistent memory, files, checkpoints, workflow state |
| Setup | Prompt every time | Brief once, saved skills, routines, intake workflows |
| Knowledge | Pasted into prompts | Prepared knowledge packs bound to each worker |
| Execution | One-off | Real tools, browser, scripts, APIs, sink writes |
| Customer DMs | Separate bot code | Telegram Business + Instagram configured in chat |
| Integrations | Hand-rolled per tool | App connectors via Composio (Google, Slack, Notion, HubSpot...) |
| Side effects | Fire and forget | Approval gate: durable, single-use, time-limited |
| Ownership | Vendor black box | Local SQLite + embedded Qdrant, yours to inspect |

---

## What You Can Delegate

### Owner-facing: your personal operator

- **Research** topics, files, and links; produce reports and summaries with citations
- **Write, execute, and debug** Python/shell scripts in a sandboxed workspace, with automatic retry on failure
- **Monitor** dashboards, competitors, inboxes, or error signals and ping you only on exceptions
- **Scheduled routines** that run while you sleep. For example, a 7am brief that scrapes your dashboards, summarizes overnight errors, and DMs you the top three
- **Remember** preferences, decisions, and project context across sessions, not just within one chat

### Customer-facing: runs inbound DMs end to end

- **Qualify** inbound leads on Telegram Business or Instagram
- **Answer** pricing and service questions from approved source material only
- **Collect** appointment or intake fields across multiple messages, tolerating typos and reorderings
- **Book, update, or cancel** records inside allowed edit windows, writing to Google Sheets, Calendar, or any Composio-connected system
- **Escalate** anything outside the workflow to you instead of guessing

The best workflows are **narrow and operational**. The clearer you define the job, tools, source material, required fields, and escalation boundary, the more employee-like the result.

> **Example brief, pasted into chat:**
> *"Handle incoming Telegram Business messages for my car wash. Answer pricing from the attached sheet, collect name / phone / vehicle / date / time, write completed bookings to this Google Sheet. Redirect anything outside this workflow to me. Confirm the workflow before activating."*

<p align="center">
  <img src="docs/assets/opentulpa-conversation-insta.jpg" alt="Instagram conversation handled by OpenTulpa" width="360"/>
</p>

---

## Quick Start

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./start.sh
```

`start.sh` bootstraps `uv`, installs dependencies, creates `.env`, launches the app, opens a Cloudflare tunnel, and syncs the Telegram webhook.

Set these when prompted, or add them to `.env`:

```env
OPENAI_COMPATIBLE_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USERNAMES=your_handle
COMPOSIO_API_KEY=...
```

**Composio is strongly recommended.** It unlocks app connectors for Google Workspace, Slack, Notion, Linear, HubSpot, Gmail, Instagram, and more without writing custom integration code.

Then message your bot on Telegram. Health check: `http://127.0.0.1:8000/healthz`.

OpenTulpa targets OpenAI-compatible providers such as OpenAI-compatible proxies, OpenRouter, Groq, local vLLM, and similar runtimes. Specific model, multimodal, and tool-calling behavior depends on the provider and model you choose. Defaults live in `opentulpa.config.yaml`.

---

## How It Works

```text
incoming message or event
        |
  load durable context: workflow state, files, memory, checkpoints
        |
  plan and call tools via LangGraph
        |
  validate tool calls and gate external side effects
        |
  reply, write outputs, or schedule follow-up
        |
  persist state, logs, artifacts, and traces
```

Core pieces: **FastAPI** for webhooks, **LangGraph** for orchestration, **SQLite** for checkpoints and workflow state, **Mem0 + embedded Qdrant** for memory, **Composio** for third-party connectors, and **Playwright** for browser automation. No external database required.

The runtime is modular around models and tools. Bring an OpenAI-compatible model provider, use Composio-backed connectors, or add your own LangGraph tool definitions where the workflow needs custom actions.

**Inspectable by design.** Everything the employee does lands on disk under `.opentulpa/` (checkpoints, context, logs, databases, knowledge packs) and `tulpa_stuff/` (generated artifacts). Back it up, mount it as a volume, or read it directly. You always know what's happening.

---

## Docs

| Doc | Why you'd read it |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Runtime layout, request flows, safety controls, extension points |
| [Deployment](docs/DEPLOYMENT.md) | Local, Docker, and Railway setup |
| [E2E Testing](docs/E2E_TESTING.md) | Realistic workflow and intake validation |
| [Chat Cookbook](docs/CHAT_COOKBOOK.md) | Concrete prompt patterns and use cases |
| [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md) | Rules for connecting high-impact tools safely |

---

<p align="center">
  <strong>Stop re-explaining. Start delegating.</strong><br/>
  <em>Run your first self-hosted digital employee.</em>
</p>

<p align="center">
  <sub>MIT licensed</sub>
</p>
