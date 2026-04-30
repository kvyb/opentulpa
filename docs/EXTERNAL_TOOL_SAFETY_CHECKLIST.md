# External Tool Safety Checklist

Use this checklist whenever you add or modify a tool/integration that can read/write external systems.

## 1. Classify the tool

- Define `recipient_scope` behavior: `self`, `external`, or `unknown`.
- Define `impact_type`: `read`, `write`, `purchase`, or `costly`.
- Treat unknown scope as higher-risk and make the model spell out what it is touching.

## 2. Define the execution contract before execution

- Keep read operations separate from write operations where practical.
- Make side effects explicit in tool inputs and results.
- Return enough evidence for the assistant to know whether the requested action actually happened.
- Prefer idempotent writes or explicit duplicate detection when the target system allows it.

## 3. Keep support act-as execution tenant-correct

For support act-as flows:

- Execute the action against the bound customer tenant, not the support operator's own tenant.
- Do not leak support setup/debug chat into the owner thread.
- Do not notify the owner about support setup/debug chatter unless the action intentionally produces a customer-facing or owner-facing event.
- Record support user id, username, support chat id, bound customer id, support thread id, action/tool name, timestamp, and outcome in internal audit state.

## 4. Minimize data exposure

- Store secrets in env/local secure config, never in prompts or logs.
- Keep tool args/results redacted when they can contain tokens, cookies, headers, or uploaded media.
- Summarize side effects instead of dumping raw external payloads into chat or traces.

## 5. Add interface handling

- Add same-interface status updates first when a tool can take more than a few seconds.
- Ensure the user sees when work is still running versus when the result is final.

## 6. Add tests before merge

- Self-target action auto-allowed.
- External write action returns concrete success/failure evidence.
- Unknown recipient scope does not produce a misleading success claim.
- Repeated execution does not duplicate side effects unexpectedly.
- Support act-as actions use the bound customer id and keep support history separate from owner history.

## 7. Document the integration

- Update README capability/safety notes.
- Document tool classification and the execution contract.
- Include operational caveats (rate limits, retries, partial failures).
