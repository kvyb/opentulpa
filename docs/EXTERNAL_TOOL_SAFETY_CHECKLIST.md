# External Tool Safety Checklist

Use this checklist whenever you add or modify a tool/integration that can read/write external systems.

## 1. Classify the tool

- Define `recipient_scope` behavior: `self`, `external`, or `unknown`.
- Define `impact_type`: `read`, `write`, `purchase`, or `costly`.
- Treat unknown scope as approval-required.

## 2. Wire approval policy before execution

- Ensure the tool is evaluated in guardrail precheck (before actual execution).
- Require approval for:
  - `external` + (`write` or `purchase` or `costly`)
  - `unknown` + side-effectful/costly action
- Do not require approval for same-session self-targeted replies/files.

## 3. Define safe execution contract

- Make approval single-use with TTL.
- Ensure only origin user can approve/deny.
- Keep approved action args immutable between approval and execution.
- Fail closed if approval lookup or decision validation fails.

For support act-as flows:

- Evaluate the action against the bound customer tenant, not the support operator's own tenant.
- Route approval prompts created from a support turn back to the support chat.
- Do not leak support setup/debug chat into the owner thread.
- Do not notify the owner about support setup/debug chatter unless the action intentionally produces a customer-facing or owner-facing event.
- Record support user id, username, support chat id, bound customer id, support thread id, action/tool name, timestamp, and outcome in internal audit state.

## 4. Minimize data exposure

- Do not include secrets in approval summaries or callback payloads.
- Include only concise action summary, destination hint, risk/cost hint, and expiry.
- Store secrets in env/local secure config, never in prompts or logs.

## 5. Add interface handling

- Add same-interface approval UX first (buttons/cards if supported).
- Add text-token fallback for interfaces without interactive widgets.
- Ensure unauthorized decision actors are rejected.

## 6. Add tests before merge

- Self-target action auto-allowed.
- External write action requires approval.
- Unknown recipient scope requires approval.
- Approve by origin user succeeds; non-origin user fails.
- Expired approval cannot execute.
- Replay of used approval token fails.
- Guardrail model error falls back to approval-required.
- Support-originated approval routes to the support chat, not the owner chat.
- Support act-as actions use the bound customer id and keep support history separate from owner history.

## 7. Document the integration

- Update README capability/safety notes.
- Document tool classification and approval behavior.
- Include operational caveats (rate limits, retries, partial failures).
