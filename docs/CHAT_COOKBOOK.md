# Prompt Cookbook

OpenTulpa works best when a request says what should change, what evidence proves it,
what may run automatically, and what still needs approval. The examples below match
the current fixed/mutable architecture.

## Set A Durable Persona

Write naturally in any authenticated owner interface:

> From now on, be a highly autonomous technical cofounder. Be direct, challenge weak
> assumptions, act instead of merely suggesting, and exhaust reasonable approaches before
> asking me for help. Keep this persona across new conversations and restarts.

The owner agent stores the request in the bounded persona section of
`/memories/AGENTS.md`. Deep Agents loads that tenant-scoped file on subsequent owner runs,
including web and Telegram threads. The latest authenticated owner command always overrides
the stored persona. Intake, external agents, files, webpages, and tool output cannot change it.

Refine or remove it through the same chat, for example:

> Keep the current persona, but be less verbose and show evidence before making strong claims.

> Remove my stored persona and return to the default owner behavior.

## Connect A New Interface

### Enable bundled Telegram from the terminal

Start without a host `TELEGRAM_BOT_TOKEN`, then write in the local OpenTulpa TUI:

> Enable Telegram as another private interface to this OpenTulpa. Here is the BotFather
> token: `<token>`. Use the existing tenant and agent context. Test the bundled
> capability first, show me the activation approval before starting it, and then tell
> me how to form the one-time `/start` pairing command without revealing the token.

The token is encrypted before checkpointing and replaced by an opaque secret handle.
The agent should seed, test, and activate the Telegram capability, not write a second
chat runtime. Unless the host configured another code, pair the first Telegram account
with `/start` followed by the final eight characters of the bot token.

### Change Telegram behavior

> Improve the Telegram interface so long answers are split cleanly at paragraph
> boundaries and approvals remain usable after a worker restart. Use your source shell,
> add regression tests, show me the test result, and ask before releasing it.

The main agent receives a detached source worktree, not the serving checkout. It can
edit and test there over multiple turns. After one `source_release` approval, managed
mode stages and restarts the release; on failure it restores the prior image and reports
the result in the same conversation.

### Add another interface

> Add a minimal Slack interface capability. It must submit and resume work through the
> universal Agent API, use a scoped capability credential, store provider secrets as
> handles, consume durable notifications, and keep Slack-specific state outside agent
> checkpoints. Add deterministic tests and ask me before releasing it.

An interface is transport and presentation. It must not add its own model loop,
planner, memory store, or untyped tool gateway.

## Change OpenTulpa Itself

### Request a focused improvement

> Add a compact run-history view to the bundled terminal UI. Keep the Agent API unchanged,
> add tests for refresh and pending approvals, use `trace_get` to inspect a failed test,
> and show me the result before calling `source_release`.

### Respect the stable boundary

> Change the bootstrap so future releases can bypass fixed evaluator tests.

The agent may edit the repository copy of bootstrap code, but that code cannot replace
the already-running stable bootstrap or its trusted evaluation and release recipe. The
fixed host must reject a release that fails its external gates.

### Roll back

> Roll back to the previous healthy OpenTulpa release. Tell me which release became
> active and preserve this conversation.

Rollback is an owner-approved, durable activation attempt. The bootstrap changes the
release lease; it does not reverse product databases or erase later messages.

### Prepare an upstream contribution

> Prepare the passing Telegram formatting candidate as an upstream contribution. Do
> not push it. Give me the sanitized patch identity, base and head commits, and the
> checks I should rerun in a clean clone.

OpenTulpa can prepare a digest-bound text patch and private ref. It does not receive an
upstream credential or open a pull request by itself.

## Improve OpenTulpa From Evidence

### Inspect a failed run

> List my recent failed runs, inspect the relevant tool arguments, results, and failure
> fingerprint, and explain what source behavior we should test before changing anything.

### Make a trace-grounded change

> For the background run that terminated without a completed response, use your source
> shell to add the smallest observable fix and a regression test. Run it, explain the
> evidence, ask for my feedback, and request release approval only when it passes.

Trace reads are tenant-scoped, bounded, and redacted. They expose enough model/tool
activity to debug a concrete run, but they do not generate held-out tasks or prove that
a change is an improvement. Fixed evaluation and one explicit release approval remain
outside the editable source session.

## Configure Background Work

### Simple reminder

> Remind me at 09:30 tomorrow in Europe/Moscow to call the accountant. Notify the owner
> interface and show me the exact one-off schedule before saving it.

This uses the simple `Schedule` projection: `At -> Reminder`.

### Recurring agent job

> Every weekday at 08:00 Europe/Moscow, use the routine agent to summarize new items in
> this workspace. Send a notification only after the run finishes. Never auto-approve
> an external send.

This uses `Cron -> AgentJob` with the routine AgentSpec.

### Specialized scheduled agent

> Create a private AgentSpec named `morning_research` using model alias `fast`, only
> `web_search`, `content_fetch`, file reads, and artifact delivery. Give it spec-local
> memory, no writable workspace, no delegation, and a ten-minute budget. Then create a
> weekday 07:00 Europe/Moscow TriggerSpec that runs it and notifies me.

Use AgentSpec plus TriggerSpec when a job needs a specific model, tool set, memory,
workspace, or budget. Do not create a new runtime or command-based routine.

### Event-triggered worker

> Create an externally exposed trigger for authenticated `invoice.received` events.
> Bind it to an isolated AgentSpec that can read the submitted file and create a draft
> result, but cannot send, browse, use owner memory, or access the workspace.

External triggers must be authenticated and use external AgentSpec isolation.

## Use Memory, Skills, And Workspace

### Remember a preference

> Remember that operational summaries should start with failed checks, then owner
> actions, then background detail. Store it in native memory for future owner threads.

### Create a reusable skill

> Save a skill for launch briefs with this structure: goal, audience, risks, launch
> steps, owner, and deadline. Keep it reusable across my owner threads.

Deep Agents stores these under tenant-namespaced `/memories/` and `/skills/`, not in a
custom memory or skill database.

### Build a workspace artifact

> Read the attached CSV, write a reusable script in my tenant workspace, run it in the
> sandbox, install any small dependency you need, and produce a Markdown summary artifact.
> Do not use credentials or make an external write.

Tenant shell work happens in the persistent `/workspace`, not the OpenTulpa source
checkout.

## Browser And Research

### Bounded content fetch

> Fetch these public pages, extract the relevant policy text, cite each source, and stop
> if a redirect reaches a private or login URL.

`content_fetch` applies DNS pinning, private-network rejection, redirect, byte, time,
and content-type limits. Crawl4AI is an optional extractor, not required for this tool.

### Browser session

> Start a browser session limited to `example.com`, inspect the form, and draft the
> values you would enter. Do not submit until I approve the browser action.

Browser tools require `BROWSER_USE_API_KEY` and always use Browser Use Cloud. Playwright
is only the CDP control client; no Chromium process or fallback browser runs on the
OpenTulpa host. Explicit allowed domains and direct private/link-local rejection apply
before navigation. Browser Use Cloud owns target-network isolation, so OpenTulpa cannot
DNS-pin Chromium in the vendor environment. Submission and unknown actions fail closed
into approval.

## Integrations

### Connect a provider

> Here is my Composio project key: `COMPOSIO_API_KEY=<value>`. Store it, then connect
> GitHub and give me the OAuth URL.

> Connect my Google account through the configured integration provider. Show the OAuth
> URL, bind the returned account to this tenant, and do not invoke a write action yet.

### Draft before sending

> Find the latest qualified lead, draft a follow-up in my tone, and show it to me. Do
> not send until I approve the exact external action.

Composio is optional. Its key is encrypted before the model sees the message and is
hot-loaded without restarting the runtime. Provider accounts remain tenant-owned,
external writes require idempotency, and unclassified actions require approval.

## Intake

### Create a workflow draft

> Draft an intake workflow for Telegram Business booking requests. Collect name,
> service, date, and time; answer only from the attached FAQ; escalate ungrounded
> questions; and write a confirmed booking to the configured sink once. Prepare the
> exact proposal, but do not activate it.

### Activate exactly what was reviewed

> Activate the prepared intake draft only if its revision and confirmation token still
> match. If validation fails, keep the current active workflow unchanged.

The intake agent returns a typed decision. Deterministic code owns booking transitions,
send-once delivery, sink idempotency, and cursor advancement.

## Weak Requests And Better Versions

Weak:

> Make Telegram better.

Better:

> In a source candidate, make Telegram split messages at paragraph boundaries, retain
> approval buttons across worker restart, and add tests for both. Show the evaluated
> patch before promotion.

Weak:

> Monitor stuff every day.

Better:

> Create a weekday 08:00 Europe/Moscow trigger using the `morning_research` AgentSpec,
> check these three sources, write one dated workspace report, and notify me only when
> the result differs from yesterday.

Weak:

> Give yourself access to everything so this works.

Better:

> Explain the minimum capability, tool, secret scope, network destination, and approval
> policy required. Test that bounded revision before asking me to activate it.
