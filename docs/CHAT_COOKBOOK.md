# OpenTulpa Prompt Cookbook

Use this document when you want examples of the kind of work OpenTulpa is meant to handle.

These are not meant as magic prompt templates. They are concrete examples of delegated work that benefits from persistence, tools, and repeatability.

One especially strong use case is structured lead intake: you describe the intake rules in one prompt, and OpenTulpa turns that into a reusable workflow that can keep talking to inbound leads on your behalf.

## Before you start

- OpenTulpa can remember files, prior decisions, and past outputs
- OpenTulpa stores generated scripts and artifacts in `tulpa_stuff`
- External write actions can still be gated behind approval
- If Composio is configured, OpenTulpa can connect to supported third-party services

## Good prompt pattern

The best prompts usually include:

- the goal
- the data source or tool to use
- the output format
- what to remember for next time
- whether the work should become a routine

Example:

> "Every weekday at 8am, check my calendar and unread priority messages, summarize what matters in five bullets, and send it on Telegram. Remember how I like the brief formatted."

## Daily operations

### Morning brief

> "Every morning at 8am, check my calendar, flag conflicts, summarize the day's priorities, and send me a short brief on Telegram."

### Inbox triage

> "Summarize the most important unread items from my inbox, group them by urgency, and draft replies in my tone for the top three."

### Project status

> "Check what changed in this project since yesterday, summarize the important diffs, and draft a team update I can send."

### Decision capture

> "Read this PDF, extract the decisions and deadlines, and remember them so I can reference them later."

## Monitoring and research

### Competitor monitoring

> "Monitor these five competitor pricing pages daily. Save anything that changed and send me a summary only when there is a meaningful difference."

### Market scan

> "Every morning, scan these sources for changes in this market, summarize what matters, and keep a running log I can refer back to."

### Incident support

> "Watch these logs for failures, summarize probable impact, and propose the next recovery actions when something new appears."

## Lead handling and intake

### One-prompt intake setup

For tightly scoped businesses, a good intake workflow can be configured directly through chat.

Example:

> "I want you to handle incoming booking requests in my Telegram. Ask for the missing details one by one, use my uploaded FAQ and pricing rules, make sure there is only one booking per hour between 08:00 and 23:00, and when everything is confirmed write the booking into my Google Sheet."

That kind of prompt is valuable because it defines:

- what channel the lead comes from
- what fields must be collected
- what business constraints must be enforced
- what source of truth to write to
- what should happen only after confirmation

### Telegram Business lead qualification

> "Handle inbound Telegram Business leads, ask for missing appointment details, use my FAQ and policy files when needed, and save completed bookings once the lead is fully qualified."

This is not just autoresponse behavior. The agent can continue the conversation across multiple messages, keep track of what is still missing, and finish the booking flow when the lead has provided enough information.

### Telegram Premium forwarding workflow

> "All incoming client messages are forwarded here. Treat them as booking requests, ask follow-up questions when details are missing, and complete the booking flow in my sheet once confirmed."

This is a useful low-friction setup when you want to stand up intake behavior quickly through chat instead of building a separate bot product first.

### Follow-up driven intake

> "When a new lead comes in, ask follow-up questions until you have name, company, budget, and desired timeline. Do not save the lead until all fields are complete."

### Sheet-backed booking flow

> "Use this uploaded booking policy and write confirmed appointments into my Google Sheet only after I approve the final write."

### Instagram DM intake

> "Handle inbound Instagram DMs for bookings. Ask for the missing date, time, service type, and car type, then confirm the booking once everything is collected."

This lets OpenTulpa act like a front-desk employee inside a customer-facing channel, not just an assistant talking back to you.

## Integrations and automation

### Slack summarizer

> "Monitor the Slack channels where I'm tagged, summarize the important threads, and draft responses I can approve before sending."

### GitHub triage

> "Watch this repo for new issues labeled bug, try to reproduce them, and draft a triage note with severity and the likely next step."

### CRM enrichment

> "When a new lead lands in HubSpot, research their company, summarize what they do, and draft a personalized outreach email for review."

### Cross-tool reporting

> "Every Monday, pull analytics from the configured sources, combine them into a one-page brief, and save it as a reusable weekly report."

## Building durable workflows

OpenTulpa is most valuable when a one-off task turns into a repeatable workflow.

Good examples:

- "Turn this reporting flow into a routine that runs every weekday"
- "Save this lead qualification behavior as a reusable skill"
- "Remember that I prefer concise summaries with action items first"
- "Use the same output format as last time unless I tell you otherwise"

## What usually works poorly

These prompts are weaker because they do not define a job clearly:

- "Help me with my business"
- "Do research"
- "Monitor stuff for me"
- "Handle my leads somehow"

Better versions make the task specific:

- what source to watch
- what output to produce
- what should be saved
- what should require approval
- whether the work repeats
