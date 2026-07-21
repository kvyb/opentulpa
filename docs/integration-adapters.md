# Integration Adapters

External providers are injected application adapters, not agent-runtime components.
FastAPI is the fixed Agent API surface. Telegram interface workers, Browser Use Cloud,
Composio, search providers, Crawl4AI, file parsers, APScheduler, and Langfuse sit behind
product ports or the universal interface protocol. Playwright is the Browser Use Cloud
CDP control client, not a local browser runtime.

The lean base install does not require Browser Use, Playwright, Composio, Crawl4AI, or
document-parser packages. Those are the `browser`, `integrations`, `research`, and
`documents` extras. Missing optional packages disable their adapter at use time rather
than adding behavior to the Deep Agents loop. `content_fetch` retains a bounded built-in
extractor without Crawl4AI.

Browser tools fail closed without Browser Use Cloud. Every session receives an explicit
domain allowlist and rejects direct private and link-local targets. Chromium and target
network access run in the vendor's isolation boundary. OpenTulpa can preflight and
intercept requests through CDP, but it cannot pin the DNS resolution used by remote
Chromium; deployment policy must not claim otherwise.

An adapter must:

- accept trusted tenant and actor context from its application service;
- validate provider account, session, file, and artifact ownership;
- keep credentials and provider identifiers out of model-visible schemas and results;
- separate reads from writes and classify unknown actions as approval-required;
- use durable idempotency for external writes;
- return sanitized typed errors and bounded results;
- expose health, timeout, and retry behavior without adding an agent loop.

An interface adapter must submit `RunSubmission` through the universal Agent API,
resume native approvals through the run API, and consume the durable notification
stream. It must not own another model loop, planner, checkpoint store, or owner bearer
token. Capability workers receive a revocable credential limited to their manifest
scopes. An Agent API credential also pins the exact tenant AgentSpec revision, run
kind, and trust class declared by the reviewed worker manifest. These values are never
accepted in the run request body. Owner API requests resolve the owner binding inside
the trusted API, while the bundled Telegram manifest is the only seed interface that
explicitly requests the same owner authority.

Composio is one integration provider, not the integration architecture. Native adapters
are appropriate when a provider lacks the required webhook, identity, ownership, or
delivery guarantees. Intake messaging adapters additionally normalize conversation
listing, conversation loading, and send-once reply delivery while webhook
authentication and durable ingestion stay in the channel layer.

The built-in in-process Browser Use and Composio ports remain reviewed base code. An
instance can add or revise an alternative adapter as a capability worker and MCP
manifest in the mutable overlay. If its exported intent matches a fixed tool, the
runtime exposes it as `<capability>__<tool>` instead of shadowing the kernel tool; an
AgentSpec can then select that reviewed alternative. Moving the built-in default itself
still requires normal upstream/base review. Changing trusted identity injection, tool
approval policy, the Agent API, or bootstrap activation is always a fixed-kernel change
and is rejected by candidate commit validation.
