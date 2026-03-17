<p align="center">
  <img src="docs/assets/opentulpa-logo.png" alt="OpenTulpa Logo" />
</p>

# OpenTulpa

**A self-hosted persistent agent runtime for developers.**

OpenTulpa is a self-hosted runtime for agents that need durable context, real execution, and reusable operational memory. It does not reset at the prompt boundary: it persists context, artifacts, skills, routines, approvals, and thread rollups so workflows become faster, safer, and more reusable over time.

It is built for developers who want an agent that can:

- remember context across sessions
- work through workflows, not just answer prompts
- turn repeated tasks into reusable skills and routines
- stay inspectable, editable, and local-first
- operate with guardrails when actions have real-world side effects

## How It Works

- ingest context from chat, files, and events
- retrieve durable state such as profiles, rollups, artifacts, skills, and routines
- plan and execute with tools
- gate external side effects behind approval
- persist outputs as artifacts, skills, routines, approvals, and updated thread context

## Walkthrough

Request: "Monitor this market every morning, summarize changes, and send me a brief."

OpenTulpa:

1. fetches the relevant sources and prior context
2. extracts and summarizes the important changes
3. stores the brief as a durable artifact
4. saves the workflow as a routine
5. reuses prior context and preferences on the next run

## Why OpenTulpa

Most agent demos stop at the prompt boundary. They can answer a request, maybe call a tool, and then discard the operational state that would make the next run easier. OpenTulpa persists the reusable parts of work: context, artifacts, skills, routines, approvals, and thread rollups.

It exposes a direct internal chat API for programmatic use, supports Telegram as a natural interface, and can be extended with Slack, browser automation, web retrieval, and generated task code.

That makes it useful for workflows developers actually care about:

- research that should persist beyond one chat
- repetitive operations that should become reusable automations
- assistants that need memory, tools, and execution in one runtime
- personal or team agents that must stay self-hosted and inspectable

## 30-Second Start (Local API Mode)

Prereqs: Python `3.12+`, [`uv`](https://docs.astral.sh/uv/), and an OpenRouter API key.

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

## Optional: Telegram in 2 Minutes

1. Create a bot via `@BotFather`.
2. Add `TELEGRAM_BOT_TOKEN` to `.env`.
3. Run:

```bash
./start.sh
```

For cloud deploys with a public URL (`PUBLIC_BASE_URL` or `RAILWAY_PUBLIC_DOMAIN`), startup can auto-register the webhook.

## What Makes It Different

1. **Durable operational state, not just chat history**

   OpenTulpa stores and reuses the things that usually get lost between sessions: preferences, directives, files, prior decisions, context events, artifacts, skills, routines, thread rollups, approvals, and link aliases.

2. **Execution, not just generation**

   It is designed to act through tools: web retrieval, files, browser sessions, Slack, internal APIs, generated scripts, and scheduled routines. Artifacts are saved locally so the system stays inspectable instead of disappearing into prompts.

3. **Skills that compound**

   When a workflow repeats, OpenTulpa can save reusable capabilities as skills and routines. Your runtime becomes a growing library of working behavior instead of rediscovering the same solution each time.

4. **Guardrails around side effects**

   Read-oriented and internal actions can proceed directly. External-impact actions can be routed through an approval broker with durable, single-use, time-limited approvals.

## Good Use Cases

OpenTulpa is a strong fit for:

- recurring market and competitive monitoring
- Slack or inbox triage with draft generation
- document review that extracts decisions and remembers them
- API integration scaffolding and scheduled automation
- recurring project, status, or executive briefs
- self-hosted developer assistants with guarded actions

Example requests:

- "Summarize the most important unread items from Slack and draft replies."
- "Monitor this market every morning and send me a concise brief."
- "Read this PDF, extract the decisions, and remember them for future work."
- "Build an integration for this API, save it as a reusable skill, and schedule it."
- "Check what changed in this project since yesterday and draft a status update."

## Architecture At A Glance

Core loop: capture context -> retrieve durable state -> plan -> act with tools -> gate side effects -> persist outputs.

```text
Telegram / Internal API / Events
              |
           FastAPI
              |
  capture context + retrieve state
              |
       LangGraph runtime
              |
 plan + tool execution + validation
              |
 approval gate for external actions
              |
 persist artifacts / skills / routines / rollups
              |
   local durable state (.opentulpa/, tulpa_stuff/)
```

Core pieces:

- FastAPI app for webhook and internal routes
- LangGraph runtime for turn execution, validation, guardrails, and claim checking
- Context services for profiles, files, event backlog, thread rollups, and aliases
- Skill store for reusable capabilities
- Scheduler/task service for one-off and recurring routines
- Approval broker for external-impact actions
- Local durable storage using SQLite plus embedded vector storage

## What You Can Connect

- **Telegram interface (optional):** chat, files, voice notes, approval buttons, `/setup`, `/fresh`, `/status`
- **Slack integration (optional):** list channels, read history, post messages after user consent
- **Web intelligence:** web search plus URL/file fetching for HTML, PDF, DOCX, and image analysis
- **Browser automation (optional):** local Browser Use tasks for dynamic websites
- **Skills:** reusable `SKILL.md` capabilities with user/global scope and persistence
- **Routines:** cron or one-time scheduled automations with durable storage

Generated scripts/artifacts are tracked under local storage (for example `tulpa_stuff/` and `.opentulpa/`), so your automation stack stays inspectable and editable.

## Safety And Storage

- Internal and read-oriented actions can be allowed directly.
- External writes, purchases, or costly actions can require approval.
- Unknown recipient scope fails toward approval.
- Approval records are durable, single-use, and time-limited.
- Public exposure is limited to webhook and health routes; internal routes are intended for local or private traffic.
- By default, OpenTulpa does not require an external database. It persists runtime state locally, and memory vectors are stored in an embedded local Qdrant setup.
- For durable deploys, mount `/app/.opentulpa` so skills, approvals, checkpoints, and memory survive redeploys.

## Developer Experience

OpenTulpa is a runnable reference architecture for persistent, guarded, tool-using agents. It is also meant to be hacked on.

- add tools in the tool registry
- add new internal routes under `src/opentulpa/api/routes`
- add interface adapters under `src/opentulpa/interfaces`
- extend approval behavior through the policy and broker layers
- add durable skills instead of hardcoding every workflow into prompts

Use it as a ready-to-run agent or as a reference architecture you can extend.

## Deploy

- Dockerfile included.
- Railway-ready config included.
- For durability across redeploys, mount `/app/.opentulpa`.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Chat Cookbook](docs/CHAT_COOKBOOK.md)
- [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md)
