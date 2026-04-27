# E2E Testing

This repo has two different kinds of end-to-end tests:

- `tests/e2e/scenarios/`
  - realistic app-level scenarios that drive the FastAPI app, the LangGraph runtime, and the Telegram/webhook surfaces together
- `tests/e2e/live/`
  - heavier smoke tests for live external integrations such as Composio

The important distinction is that the `live_llm` scenarios use a real model. They are not mocked unit tests.

## How env loading works

OpenTulpa settings load `.env` automatically through `src/opentulpa/core/config.py`.

That means:

- you do **not** need to manually `export` the keys first if you run tests from the repo root
- `uv run pytest ...` will pick up `.env` through the app settings loader

Required minimum for the live-LLM scenario suite:

```env
OPENAI_COMPATIBLE_API_KEY=...
```

Accepted backward-compatible alias:

```env
OPENROUTER_API_KEY=...
```

The generic scenario harness accepts either name. Some standalone smokes under
`tests/e2e/live/` have extra opt-in gates:

- `test_guardrail_live_llm.py`: `OPENROUTER_API_KEY` plus `OPENTULPA_RUN_LIVE_OPENROUTER_E2E=1`
- `test_browser_use_google.py`: `OPENROUTER_API_KEY`, Playwright Chromium, plus `OPENTULPA_ENABLE_LIVE_BROWSER_USE_E2E=1`
- `test_intake_workflow_composio.py`: `COMPOSIO_API_KEY` plus `OPENTULPA_ENABLE_LIVE_COMPOSIO_INTAKE_E2E=1`; write coverage also requires `OPENTULPA_ENABLE_LIVE_COMPOSIO_INTAKE_WRITE_E2E=1` and the live sink mapping env values in that test file

Optional but common:

```env
OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=test-secret
COMPOSIO_API_KEY=...
```

## Fast commands

Run all scenario e2e tests:

```bash
uv run pytest tests/e2e/scenarios --run-e2e -q -rs
```

Run only Telegram scenario tests:

```bash
uv run pytest tests/e2e/scenarios -m telegram --run-e2e -q -rs
```

Run only the Telegram intake workflow real-chat scenarios with live LLM calls:

```bash
uv run pytest tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py --run-e2e --run-live-llm -s
```

Run all e2e tests, including `tests/e2e/live/`:

```bash
uv run pytest tests/e2e --run-e2e --run-live-llm -q -rs
```

Some standalone live smokes have their own env gates and are skipped unless those gates are set. See "How env loading works" above for the exact flags.

Run opted-in scenario-level real Composio tests only when you intentionally want real connected-account access:

```bash
uv run pytest tests/e2e --run-e2e --run-live-llm --run-real-composio -q -rs
```

Standalone live Composio smokes under `tests/e2e/live/` are controlled by their `OPENTULPA_ENABLE_LIVE_COMPOSIO_*` env flags instead of `--run-real-composio`.

## What the Telegram intake workflow e2e covers

`tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py` exercises realistic Telegram intake paths:

1. Owner Telegram chat creates a `telegram_business_dm` workflow through the actual Telegram webhook path
2. Owner Telegram chat deletes an existing workflow through the same path
3. A Telegram Business lead message hits an active workflow and the lead gets a real reply on the business connection
4. Owner Telegram chat creates a car-wash workflow, a lead completes the booking over multiple Telegram DM turns, and the completed booking is persisted to the configured sink
5. Owner Telegram chat uploads source material, OpenTulpa prepares scoped workflow knowledge, confirms setup, and then handles realistic Russian leads for `Мойка` and `Шиномонтаж`
6. Edge cases cover out-of-scope services, missing phone, ambiguous vehicle class, unavailable price, and update/cancel requests

These are intentionally close to real usage:

- owner messages go through `/webhook/telegram`
- interactive owner chat runs through `TelegramChatService`
- workflow creation/deletion goes through the actual tool/runtime flow
- uploaded files go through the file vault and prepared workflow knowledge path
- lead messages go through the `business_message` webhook path

`tests/e2e/scenarios/test_telegram_support_act_as.py` covers support operator act-as behavior: customer listing, binding, support thread isolation, owner invisibility, approval routing, and optional live-LLM setup under a bound customer.

## Recommended test order while iterating on intake

If you are changing Telegram intake behavior, use this order:

1. Fast local safety checks

```bash
uv run pytest \
  tests/test_intake_workflow_service.py \
  tests/test_intake_workflow_routes.py \
  tests/test_workflow_setup_service.py \
  tests/test_intake_tools.py \
  tests/test_runtime_thread_scope.py -q
```

2. Telegram surface checks

```bash
uv run pytest \
  tests/test_telegram_business_webhook.py \
  tests/test_telegram_interactive_mailbox.py \
  tests/test_telegram_fresh_command.py -q
```

3. Real Telegram intake e2e

```bash
uv run pytest tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py --run-e2e --run-live-llm -s
```

This catches the exact class of bugs we hit recently:

- workflow save/delete works, but the Telegram streamed turn crashes afterward
- wizard path works in isolation, but not when triggered from real Telegram chat
- workflow setup mode works through internal chat, but not through Telegram owner/support chat
- source files are uploaded, but the prepared workflow knowledge is missing or too broad
- Telegram Business workflows save correctly, but lead webhook execution breaks
- multi-turn lead collection appears to work manually, but the booking never reaches the final storage sink

## Recommended suite shape for intake flows

For business-intake regressions, split the suite into two lanes:

1. Stable PR-gating scenarios

- use the real app, real runtime, real Telegram webhook path, and real live LLM decisions
- keep external sinks fake or local so assertions stay exact
- assert business outcomes, not just replies:
  - workflow saved with the right channel/source
  - missing-field follow-up happened before save
  - booking reached `completed`
  - sink write succeeded
  - stored row contains the expected fields

2. Exploratory realism runs

- keep the same app-level harness, but replace the scripted lead with an LLM-driven lead simulator
- use these on demand or nightly, not as the only gating signal
- capture full artifacts so you can inspect failures:
  - owner transcript
  - lead inbound messages
  - assistant outbound messages
  - prepared workflow knowledge
  - workflow snapshot
  - booking snapshots
  - sink arguments / written rows
  - stage judgements
  - behavior log and LLM trace log

This gives you one lane that is consistent enough to block regressions, and another lane that is realistic enough to expose prompt and skill weaknesses.

## Model selection for e2e

By default, the e2e harness uses the same repo settings as normal runtime:

- `settings.llm_model`
- `settings.wake_classifier_model`
- `settings.guardrail_classifier_model`
- `google/gemini-3-flash-preview` for the optional lead simulator lane

You can still override them explicitly for e2e-only runs:

```bash
OPENTULPA_E2E_MODEL=...
OPENTULPA_E2E_WAKE_MODEL=...
OPENTULPA_E2E_GUARDRAIL_MODEL=...
OPENTULPA_E2E_LEAD_SIM_MODEL=google/gemini-3-flash-preview
uv run pytest tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py --run-e2e --run-live-llm -s
```

`OPENTULPA_E2E_LEAD_SIM_MODEL` controls the incoming-lead simulator used by the simulator-backed Telegram intake scenario.

## Reports and logs

The scenario harness writes structured artifacts under the pytest temp directory for each run:

- system events
- agent behavior log
- LLM trace log
- scenario status report
- owner and lead transcripts
- prepared workflow knowledge
- workflow snapshot
- sheet writes
- stage judgements where enabled

The status report includes the concrete file paths so you can inspect what happened after a failure.

When a scenario fails, rerun with:

```bash
uv run pytest tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py --run-e2e --run-live-llm -vv -s
```

## Common failure modes

### Test is skipped even though `.env` exists

Run pytest from the repo root:

```bash
cd /path/to/opentulpa
uv run pytest tests/e2e/scenarios/test_telegram_intake_workflow_real_chat.py --run-e2e --run-live-llm -q -rs
```

The skip gate uses the same settings loader as the app, so `.env` should count. You still need the opt-in flags: `--run-e2e` for all e2e tests and `--run-live-llm` for live model tests.

### Telegram owner chat returns a backend error

Check these first:

- `tests/test_runtime_thread_scope.py`
- `tests/test_telegram_interactive_mailbox.py`
- the scenario status report path from the failed test output

This class of bug is usually in the interactive streaming/runtime boundary, not in the workflow DB write itself.

### Lead webhook did not reply

Check:

- `tests/test_telegram_business_webhook.py`
- whether the created workflow is:
  - `channel=telegram_business_dm`
  - `provider=telegram_bot_api`
  - `enabled=true`
- whether the Telegram Business connection exists and matches the workflow source

## CI note

If you later wire these tests into CI, separate them by cost:

- fast unit/integration tests on every PR
- scenario `live_llm` e2e on demand or protected branches
- heavier `tests/e2e/live/` integration smokes only when the required external credentials are intentionally available
- real Composio scenario tests only behind explicit `--run-real-composio`
- standalone live Composio smokes only behind their `OPENTULPA_ENABLE_LIVE_COMPOSIO_*` env flags
