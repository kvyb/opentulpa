# Integration Adapters

OpenTulpa should treat Composio as one integration provider, not as the integration
architecture. Composio remains the best default when it already supports a
customer app, but native adapters are needed when Composio lacks a channel,
cannot provide webhook delivery, or cannot expose the source identity OpenTulpa
needs for reliable intake.

The immediate change is deliberately narrow:

- Route existing Instagram DM intake through a Composio-backed messaging adapter.
- Route existing Telegram Business conversation reads and outbound replies
  through a Telegram Business messaging adapter.
- Keep workflow behavior, storage, and webhook entrypoints unchanged.
- Avoid dynamic plugin loading until OpenTulpa has a second native provider that
  proves the need.

The adapter boundary gives each messaging provider one canonical place to own:

- health and availability checks
- conversation listing
- detailed conversation loading
- outbound reply delivery
- provider-specific identity and source configuration

This lets future channels such as WhatsApp, Messenger, LINE, or unsupported CRMs
plug into the same intake execution path without adding more channel/provider
branches inside `IntakeWorkflowService`.

The bar for new adapters is production-shaped, not generic:

- take `customer_id` and `source_config` explicitly
- return normalized conversation summaries
- fail visibly when configured sources cannot be read
- keep webhook verification and raw event logging in the route that receives
  provider traffic
- avoid global fallback accounts or ambiguous customer scope
