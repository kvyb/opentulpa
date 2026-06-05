# Intake Owner Handoffs

## First Runtime Pass

This branch adds the runtime/backend handoff foundation:

- Workflow config can store `handoff_rules`.
- Intake AI can request owner handoff only by referencing an enabled configured rule.
- Runtime creates or updates one durable non-terminal handoff per workflow conversation.
- Web API exposes handoff list/detail/respond routes.
- Owner advice is private guidance; runtime resumes the intake and asks the agent to write the customer-facing reply itself.
- If resume/reply fails, handoff moves to `failed_reply` and emits a web event with the failure state.

Website UI is still separate work.

## Runtime Flow

```text
source item -> normalized conversation -> intake decision
  -> no handoff: normal apply/reply/save
  -> request_owner: create/update durable handoff, optional wait reply, mark cursor, stop normal apply
owner response -> reload source conversation -> decide with private owner feedback -> apply/reply/save
```

Handoff logic sits between decision and apply in `workflow_runner.py`. Source adapters, sink writer, and `send_owner_update` do not own handoff behavior.

## Rule Shape

```json
{
  "handoff_rules": [
    {
      "id": "discount_approval",
      "label": "Discount approval",
      "condition": "Customer asks for a discount, price exception, or manager approval.",
      "owner_prompt": "Customer wants discount approval. Ask owner for approve/deny/counter-offer.",
      "customer_wait_reply": "Let me check that and get back to you.",
      "enabled": true
    }
  ]
}
```

`condition` is required. `id` is normalized if provided, otherwise generated deterministically.

## Web API

Routes require `Authorization: Bearer <OPENTULPA_WEB_TOKEN>` and `customer_id`.

```text
GET  /web/intake/handoffs?customer_id=...&status=awaiting_owner
GET  /web/intake/handoffs/{handoff_id}?customer_id=...
POST /web/intake/handoffs/{handoff_id}/respond?customer_id=...
```

Respond body:

```json
{
  "owner_feedback": "Approve 10%, not 20%."
}
```

Response includes lead identity and message context:

```json
{
  "handoff_id": "hnd_...",
  "status": "awaiting_owner",
  "lead": {
    "username": "alice",
    "display_name": "alice",
    "platform_user_id": "cust_1"
  },
  "messages": {
    "latest": [
      {
        "message_id": "msg_2",
        "direction": "inbound",
        "text": "Can you do 20% off?",
        "created_at": "2026-04-07T08:01:00+00:00"
      }
    ],
    "previous": []
  }
}
```

## State Rules

- New spam from same lead updates existing open handoff, not create duplicate.
- Wait reply is sent only on first create, not every update.
- Owner response is compare-and-swap: only `awaiting_owner` accepts response.
- Terminal statuses stop current handoff only; later customer messages can be evaluated fresh.

Implemented statuses:

- `awaiting_owner`
- `owner_responded`
- `resuming`
- `resolved`
- `resolved_no_reply`
- `failed_reply`
- `expired`
- `canceled_by_customer_update`

## Remaining Work

- Website handoff inbox/detail/respond UI.
- Better expiry/cleanup policy.
- Optional owner notification adapters beyond web events.
- Richer “latest spam batch” display if source adapters expose larger unread batches.
