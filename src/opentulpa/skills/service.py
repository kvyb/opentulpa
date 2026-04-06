"""Persistent user/global skill storage and retrieval."""

from __future__ import annotations

import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_SKILL_CREATOR_DESCRIPTION = (
    "Use this skill when the user asks for recurring behavior/capabilities so the "
    "assistant can create or update reusable skills."
)
_LEGACY_SKILL_CREATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Turn repeated user requests into durable reusable skills.\n\n"
    "## Workflow\n"
    "1. Detect recurring requests (style, reporting format, parser behavior, domain workflow).\n"
    "2. Ask concise clarifying questions if requirements are ambiguous.\n"
    "3. Create or update a user skill with durable instructions.\n"
    "4. Confirm what was stored and when it will be reused.\n\n"
    "## Storage Rule\n"
    "Store user-specific skills in user scope by default.\n"
    "Use global scope only for universally applicable capabilities."
)
_DEFAULT_SKILL_CREATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Turn repeated user requests into durable reusable skills that stay concise, valid, and reusable.\n\n"
    "## When to create or update\n"
    "1. Use this for recurring asks: style, reporting format, parser behavior, domain workflow, or tool procedure.\n"
    "2. Update an existing skill instead of creating a near-duplicate when the capability is the same.\n\n"
    "## Authoring rules\n"
    "1. Keep SKILL.md lean; store only durable instructions the assistant is unlikely to infer reliably.\n"
    "2. Start with YAML frontmatter and include the exact skill `name` plus a clear non-empty `description`.\n"
    "3. In the body, focus on trigger conditions, workflow, guardrails, and expected output.\n"
    "4. Prefer short examples over long explanation.\n"
    "5. Put large or variant-specific detail in supporting files like `references/`, `scripts/`, or `assets/` only when needed.\n"
    "6. Do not add extra docs like README or CHANGELOG just for the skill.\n\n"
    "## Workflow\n"
    "1. Detect the recurring request and the durable behavior worth storing.\n"
    "2. Ask concise clarifying questions only if ambiguity would make the stored behavior wrong.\n"
    "3. Choose scope: user by default, global only for broadly useful capabilities.\n"
    "4. Create or update the skill with concise durable instructions and only the supporting files that are actually needed.\n"
    "5. Confirm what was stored and when it will be reused.\n\n"
    "## Validation\n"
    "1. Ensure the frontmatter name matches the requested skill name.\n"
    "2. Ensure the description is specific enough to trigger later.\n"
    "3. Ensure the instructions are durable, not tied to a single conversation.\n"
)
_DEFAULT_BROWSER_USE_OPERATOR_DESCRIPTION = (
    "Use this skill for interactive browser tasks that require real page navigation, "
    "JavaScript rendering, or multi-step website workflows."
)
_DEFAULT_SIGNAL_INTEGRATION_OPERATOR_DESCRIPTION = (
    "Use this skill when setting up inbound webhooks/signals and outbound reply adapters "
    "for channels that should feed events into OpenTulpa."
)
_DEFAULT_INCOMING_SIGNAL_HANDLER_DESIGNER_DESCRIPTION = (
    "Use this skill when the OpenTulpa owner wants to define or amend how incoming "
    "contact messages from signal/webhook channels should be handled."
)
_LEGACY_BROWSER_USE_OPERATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Use Browser Use tools safely and cost-effectively for tasks normal link fetch/search "
    "cannot complete reliably.\n\n"
    "## When to use\n"
    "1. Dynamic websites where static fetching is insufficient.\n"
    "2. Multi-step navigation/extraction across pages.\n"
    "3. Tasks requiring browser state and real interactions.\n\n"
    "## Workflow\n"
    "1. Clarify task objective and exact deliverable.\n"
    "2. Set tight scope first: allowed domains and low max_steps.\n"
    "3. Call browser_use_run.\n"
    "4. If timed out/in progress, call browser_use_task_get.\n"
    "5. If needed, call browser_use_task_control to stop/pause.\n"
    "6. Return concise results, confidence, and any unresolved gaps.\n\n"
    "## Safety & cost guardrails\n"
    "- Start with conservative defaults (max_steps around 10-25).\n"
    "- Restrict domains whenever possible.\n"
    "- Avoid autonomous long runs without explicit user request.\n"
    "- Prefer ordinary web tools for simple fetch/search tasks."
)
_DEFAULT_BROWSER_USE_OPERATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Use Browser Use tools safely and cost-effectively for tasks normal link fetch/search "
    "cannot complete reliably.\n\n"
    "## When to use\n"
    "1. Dynamic websites where static fetching is insufficient.\n"
    "2. Multi-step navigation/extraction across pages.\n"
    "3. Tasks requiring browser state and real interactions.\n\n"
    "## Workflow\n"
    "1. Clarify task objective and exact deliverable.\n"
    "2. If continuing the same site/workflow, reuse an existing idle browser session instead of creating a fresh one.\n"
    "3. Reuse the latest known `session_id` when available; if unsure, call `browser_use_session_list` and choose an idle reusable session.\n"
    "4. Call `browser_use_run`, passing `session_id` when continuing work in the same browser session.\n"
    "5. If the task is still running or paused, call `browser_use_task_get` instead of starting another run on the same session.\n"
    "6. If the user needs a screenshot artifact, call `browser_use_task_screenshot` and then `tulpa_file_send` with the returned `path`.\n"
    "7. When the session is no longer needed, call `browser_use_task_control` to stop it; otherwise idle sessions auto-expire after about 1 hour.\n"
    "8. Return concise results, confidence, and any unresolved gaps.\n\n"
    "## Safety & cost guardrails\n"
    "- Start with conservative defaults (max_steps around 10-25).\n"
    "- Restrict domains whenever possible.\n"
    "- Avoid autonomous long runs without explicit user request.\n"
    "- Prefer ordinary web tools for simple fetch/search tasks.\n"
    "- Reuse idle sessions to avoid spawning unnecessary browsers and wasting RAM."
)
_LEGACY_SIGNAL_INTEGRATION_OPERATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Create channel-specific webhook/reply glue without rebuilding queueing, wake rules, or turn orchestration.\n\n"
    "## Core boundary\n"
    "1. Keep channel glue in `tulpa_stuff/`.\n"
    "2. Use OpenTulpa core routes for signal ingestion, wake rules, and outbound replies.\n"
    "3. Do not reimplement queueing, batching, wake classification, or conversation orchestration inside the connector.\n\n"
    "## Connector contract\n"
    "1. Public webhook routes should live in a tulpa module `public_router` and are mounted at `/webhook/tulpa/<module_name>/...`.\n"
    "2. Internal helper routes can stay in the normal `router` mounted at `/tulpa/<module_name>/...`.\n"
    "3. Verify webhook secrets before accepting payloads.\n"
    "4. By default, keep `customer_id` as the OpenTulpa owner/operator identity, not the external sender identity.\n"
    "5. When using `/internal/signals/ingest`, pass `owner_customer_id` and optional `owner_thread_id`; do not derive `customer_id` from the external sender.\n"
    "6. Use generic metadata such as `external_subject_id` and `external_conversation_id`, not provider-specific field names.\n"
    "7. Treat the incoming webhook JSON as generic. Keep only the small routing envelope explicit; arbitrary extra fields can pass through as payload metadata.\n"
    "8. Prefer `await request.app.state.signal_ingest({...})` inside public tulpa routes; it wires owner identity, generic external IDs, wake enqueueing, and stores non-envelope fields as payload metadata automatically.\n"
    "9. If you cannot use `request.app.state.signal_ingest`, send normalized events to `/internal/signals/ingest`; it applies the same wiring.\n"
    "10. Read pending outbound replies from `/internal/signals/outbox` and mark sent with `/internal/signals/outbox/{id}/sent` after successful delivery.\n\n"
    "## Wake rules\n"
    "1. Configure behavior through `/internal/signals/rules/upsert` or the `signal_rule_upsert` tool instead of hardcoding per-connector logic.\n"
    "2. Use `wake_mode=always` for immediate processing, `classifier` for model-decided wakeups, and `never` for backlog-only collection.\n"
    "3. Use `batch_window_seconds` to debounce bursty channels.\n"
    "4. Put durable answering guidance in rule `guidance_text` or normal OpenTulpa context/skills, not in ad-hoc connector prompts.\n\n"
    "## Workflow\n"
    "1. Clarify the external channel, auth method, webhook payload shape, and outbound API.\n"
    "2. Write or update one tulpa module for the adapter.\n"
    "3. Validate the file, reload tulpa routes, and test with a sample payload.\n"
    "4. Create or update the matching signal wake rule, then verify it with `/internal/signals/rules` or the `signal_rule_list` tool before claiming success.\n"
    "5. Keep the connector thin and channel-specific."
)
_DEFAULT_SIGNAL_INTEGRATION_OPERATOR_INSTRUCTIONS = (
    "## Purpose\n"
    "Create channel-specific webhook/reply glue without rebuilding queueing, wake rules, turn orchestration, or owner-defined handling policy.\n\n"
    "## Identify the upstream provider first\n"
    "1. When the owner asks to process inbound Instagram, Facebook, WhatsApp, Telegram, or other social messages, first determine which upstream webhook provider will actually deliver those messages into OpenTulpa.\n"
    "2. If the owner has not chosen a provider yet and they want social-channel automation for Instagram, Facebook Messenger, WhatsApp, or Telegram, suggest ManyChat as the default option to evaluate first.\n"
    "3. If the owner already has a different provider or a direct webhook integration, use that instead of forcing ManyChat.\n"
    "4. Do not design the connector until the actual upstream provider, webhook path, auth method, and payload shape are clear.\n\n"
    "## Owner guidance first\n"
    "1. Before enabling a new inbound signal channel for live contact handling, ask the OpenTulpa owner how incoming contacts should be handled.\n"
    "2. Clarify goals such as qualification, FAQ answering, booking, escalation, follow-up, and what data should be written to external systems.\n"
    "3. If this handling policy is missing or ambiguous, do not claim the inbound connection is fully ready for autonomous replies.\n"
    "4. After clarifying the policy, use the `incoming-signal-handler-designer` skill to create or update the owner's durable handling skill before finalizing the connector.\n\n"
    "## Core boundary\n"
    "1. Keep channel glue in `tulpa_stuff/`.\n"
    "2. Use OpenTulpa core routes for signal ingestion, wake rules, and outbound replies.\n"
    "3. Do not reimplement queueing, batching, wake classification, conversation orchestration, or owner policy inside the connector.\n\n"
    "## Connector contract\n"
    "1. Public webhook routes should live in a tulpa module `public_router` and are mounted at `/webhook/tulpa/<module_name>/...`.\n"
    "2. Internal helper routes can stay in the normal `router` mounted at `/tulpa/<module_name>/...`.\n"
    "3. Verify webhook secrets before accepting payloads.\n"
    "4. By default, keep `customer_id` as the OpenTulpa owner/operator identity, not the external sender identity.\n"
    "5. When using `/internal/signals/ingest`, pass `owner_customer_id` and optional `owner_thread_id`; do not derive `customer_id` from the external sender.\n"
    "6. Use generic metadata such as `external_subject_id` and `external_conversation_id`, not provider-specific field names.\n"
    "7. Treat the incoming webhook JSON as generic. Keep only the small routing envelope explicit; arbitrary extra fields can pass through as payload metadata.\n"
    "8. Prefer `await request.app.state.signal_ingest({...})` inside public tulpa routes; it wires owner identity, generic external IDs, wake enqueueing, and stores non-envelope fields as payload metadata automatically.\n"
    "9. If you cannot use `request.app.state.signal_ingest`, send normalized events to `/internal/signals/ingest`; it applies the same wiring.\n"
    "10. Read pending outbound replies from `/internal/signals/outbox` and mark sent with `/internal/signals/outbox/{id}/sent` after successful delivery.\n\n"
    "## Wake rules\n"
    "1. Configure behavior through `/internal/signals/rules/upsert` or the `signal_rule_upsert` tool instead of hardcoding per-connector logic.\n"
    "2. Use `wake_mode=always` for immediate processing, `classifier` for model-decided wakeups, and `never` for backlog-only collection.\n"
    "3. Use `batch_window_seconds` to debounce bursty channels.\n"
    "4. Set `handler_skill_name` in the signal rule to the saved incoming-handler playbook; this wiring is required for live signal handling.\n"
    "5. Put durable answering guidance in rule `guidance_text` or in the owner's saved incoming-handler skill, not in ad-hoc connector prompts.\n\n"
    "## Workflow\n"
    "1. Identify the actual upstream webhook provider for the requested channel.\n"
    "2. Clarify the auth method, webhook payload shape, and outbound API.\n"
    "3. Confirm the owner-facing handling policy exists; if not, gather it and save it first.\n"
    "4. Write or update one tulpa module for the adapter.\n"
    "5. Validate the file, reload tulpa routes, and test with a sample payload.\n"
    "6. Create or update the matching signal wake rule, including `handler_skill_name`, then verify it with `/internal/signals/rules` or the `signal_rule_list` tool before claiming success.\n"
    "7. Keep the connector thin and channel-specific."
)
_DEFAULT_INCOMING_SIGNAL_HANDLER_DESIGNER_INSTRUCTIONS = (
    "## Purpose\n"
    "Capture and maintain the OpenTulpa owner's durable policy for how incoming contact messages from signal/webhook channels should be handled.\n\n"
    "## When to use\n"
    "1. Before setting up a new inbound signal connector that will talk to external contacts.\n"
    "2. When the owner asks to change how incoming Instagram/ManyChat/WhatsApp/etc. conversations should be handled.\n"
    "3. When the owner wants OpenTulpa to perform business operations in response to inbound messages, such as qualification, booking appointments, logging lead data, or updating tables/docs.\n\n"
    "## Setup dependency\n"
    "1. This playbook is required for live signal handling; the signal rule should wire it through `handler_skill_name`.\n"
    "2. If the upstream provider is not chosen yet, coordinate with the signal integration flow first so the source and connector plan are clear.\n\n"
    "## Required owner guidance\n"
    "1. Ask the owner what outcome the assistant should drive for incoming contacts.\n"
    "2. Clarify what questions OpenTulpa should ask contacts, in what order, and when to stop.\n"
    "3. Clarify what data should be captured and where it should be written: custom fields, local files, Google Sheets, Google Docs tables, APIs, CRM rows, etc.\n"
    "4. Clarify which operations are allowed during handling: terminal commands, scripts, browser tasks, API calls, file writes, scheduling, or human escalation.\n"
    "5. Clarify tone, boundaries, disallowed claims, handoff rules, and when to notify the owner instead of replying automatically.\n\n"
    "## Storage rule\n"
    "1. Save the handling policy as a user-scoped skill so it persists and can be amended later through chat.\n"
    "2. Prefer a stable source-specific name like `<source>-incoming-handler` such as `manychat-incoming-handler`.\n"
    "3. Update the existing handler skill instead of creating duplicates when the owner is refining the same workflow.\n"
    "4. Keep the skill focused on durable behavior, not one-off conversation context.\n"
    "5. After saving or updating the skill, wire the exact skill name into the signal rule via `handler_skill_name`.\n\n"
    "## What the saved handler skill should contain\n"
    "1. Trigger condition: which source/channel and which kinds of contacts/messages it applies to.\n"
    "2. Conversation workflow: what to ask, what to confirm, and when to escalate or stop.\n"
    "3. Operational actions: which scripts, sandbox actions, APIs, docs, or tables may be used while handling the contact.\n"
    "4. Data contract: what fields or rows to write and the expected success/failure behavior.\n"
    "5. Reply policy: tone, brevity, language, and what must never be claimed without tool evidence.\n\n"
    "## Example outcomes\n"
    "1. Qualify an Instagram lead, ask scheduling questions, then append an appointment row into a Google Sheet via an approved script.\n"
    "2. Answer FAQs from business context, collect contact details, and escalate complex requests to the owner.\n"
    "3. Gather booking details, create/update a local queue or external table, then send a confirmation message.\n\n"
    "## Workflow\n"
    "1. Ask concise owner questions until the handling policy is operationally specific.\n"
    "2. Create or update the user skill with `skill_upsert`.\n"
    "3. Confirm the saved skill name and summarize how future incoming contacts will be handled.\n"
    "4. If the owner later amends the behavior through chat, update the same user skill rather than inventing a new policy."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_skill_name(name: str) -> str:
    value = str(name or "").strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        raise ValueError("skill name is required")
    if len(value) > 64:
        raise ValueError("skill name too long (max 64 chars)")
    return value


def _sanitize_customer_segment(customer_id: str) -> str:
    value = str(customer_id or "").strip()
    if not value:
        raise ValueError("customer_id is required for user skills")
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _strip_quotes(text: str) -> str:
    raw = str(text or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1].strip()
    return raw


def parse_skill_frontmatter(skill_markdown: str) -> tuple[str, str]:
    text = str(skill_markdown or "")
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("SKILL.md frontmatter terminator not found")
    frontmatter = text[4:end]
    data: dict[str, str] = {}
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip().lower()] = _strip_quotes(value)
    name = _normalize_skill_name(data.get("name", ""))
    description = str(data.get("description", "")).strip()
    if not description:
        raise ValueError("skill frontmatter requires non-empty description")
    if len(description) > 1024:
        description = description[:1024]
    return name, description


def build_skill_markdown(*, name: str, description: str, instructions: str) -> str:
    normalized = _normalize_skill_name(name)
    desc = str(description or "").strip()
    body = str(instructions or "").strip()
    if not desc:
        raise ValueError("description is required")
    if not body:
        raise ValueError("instructions are required")
    return (
        f"---\n"
        f"name: {normalized}\n"
        f"description: {desc}\n"
        f"---\n\n"
        f"# {normalized}\n\n"
        f"{body}\n"
    )


class SkillStoreService:
    """Store and resolve skills with user-overrides-global precedence."""

    def __init__(self, *, db_path: Path, root_dir: Path) -> None:
        self.db_path = db_path.resolve()
        self.root_dir = root_dir.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    scope TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    skill_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (scope, customer_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_skills_customer
                    ON skills(customer_id, updated_at DESC);
                """
            )

    def _validate_scope(self, scope: str) -> str:
        s = str(scope or "user").strip().lower()
        if s not in {"user", "global"}:
            raise ValueError("scope must be 'user' or 'global'")
        return s

    def _scope_customer(self, *, scope: str, customer_id: str) -> str:
        if scope == "global":
            return ""
        return str(customer_id or "").strip()

    def _skill_dir(self, *, scope: str, customer_id: str, name: str) -> Path:
        if scope == "global":
            return (self.root_dir / "global" / name).resolve()
        customer_segment = _sanitize_customer_segment(customer_id)
        return (self.root_dir / "users" / customer_segment / name).resolve()

    @staticmethod
    def _validate_supporting_files(files: dict[str, str] | None) -> dict[str, str]:
        if files is None:
            return {}
        if not isinstance(files, dict):
            raise ValueError("supporting_files must be an object mapping relative paths to text")
        out: dict[str, str] = {}
        total_bytes = 0
        for raw_path, raw_content in files.items():
            rel = str(raw_path or "").strip()
            if not rel:
                raise ValueError("supporting_files contains empty path")
            p = Path(rel)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("supporting_files paths must be relative and cannot use '..'")
            content = str(raw_content or "")
            encoded = content.encode("utf-8")
            total_bytes += len(encoded)
            if len(encoded) > 2_000_000:
                raise ValueError(f"supporting file too large: {rel}")
            out[str(p)] = content
        if total_bytes > 10_000_000:
            raise ValueError("supporting_files total payload too large (>10MB)")
        return out

    def upsert_skill(
        self,
        *,
        scope: str,
        customer_id: str,
        name: str,
        skill_markdown: str,
        source: str = "agent",
        enabled: bool = True,
        supporting_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        safe_scope = self._validate_scope(scope)
        safe_customer = self._scope_customer(scope=safe_scope, customer_id=customer_id)
        safe_name = _normalize_skill_name(name)
        markdown = str(skill_markdown or "")
        if len(markdown.encode("utf-8")) > 10_000_000:
            raise ValueError("SKILL.md exceeds 10MB limit")
        parsed_name, description = parse_skill_frontmatter(markdown)
        if parsed_name != safe_name:
            raise ValueError("frontmatter name must match requested skill name")
        files = self._validate_supporting_files(supporting_files)

        skill_dir = self._skill_dir(scope=safe_scope, customer_id=safe_customer, name=safe_name)
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md_path = (skill_dir / "SKILL.md").resolve()
        skill_md_path.write_text(markdown, encoding="utf-8")
        for rel_path, content in files.items():
            path = (skill_dir / rel_path).resolve()
            if skill_dir not in path.parents:
                raise ValueError("supporting file path escapes skill directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        now = _utc_now()
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT created_at FROM skills
                WHERE scope=? AND customer_id=? AND name=?
                """,
                (safe_scope, safe_customer, safe_name),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO skills
                    (scope, customer_id, name, description, source, enabled, skill_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, customer_id, name)
                DO UPDATE SET
                    description=excluded.description,
                    source=excluded.source,
                    enabled=excluded.enabled,
                    skill_path=excluded.skill_path,
                    updated_at=excluded.updated_at
                """,
                (
                    safe_scope,
                    safe_customer,
                    safe_name,
                    description,
                    str(source or "agent"),
                    1 if enabled else 0,
                    str(skill_md_path),
                    created_at,
                    now,
                ),
            )
            conn.commit()
        return self.get_skill(
            customer_id=customer_id,
            name=safe_name,
            include_files=False,
            include_global=True,
        ) or {
            "name": safe_name,
            "description": description,
            "scope": safe_scope,
            "customer_id": safe_customer,
        }

    def list_skills(
        self,
        *,
        customer_id: str,
        include_global: bool = True,
        include_disabled: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        safe_customer = str(customer_id or "").strip()
        safe_limit = max(1, min(int(limit), 500))
        rows: list[sqlite3.Row] = []
        with self._conn() as conn:
            if include_global:
                rows.extend(
                    conn.execute(
                        """
                        SELECT scope, customer_id, name, description, source, enabled, skill_path, created_at, updated_at
                        FROM skills
                        WHERE scope='global'
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (safe_limit,),
                    ).fetchall()
                )
            if safe_customer:
                rows.extend(
                    conn.execute(
                        """
                        SELECT scope, customer_id, name, description, source, enabled, skill_path, created_at, updated_at
                        FROM skills
                        WHERE scope='user' AND customer_id=?
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (safe_customer, safe_limit),
                    ).fetchall()
                )
        # precedence: user skill overrides global with same name
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = self._row_to_item(row, include_paths=False)
            if not include_disabled and not item["enabled"]:
                continue
            name = item["name"]
            current = merged.get(name)
            if current is None:
                merged[name] = item
                continue
            if (item["scope"] == "user" and current["scope"] == "global") or (
                item["updated_at"] > current["updated_at"]
            ):
                merged[name] = item
        out = sorted(merged.values(), key=lambda x: x["updated_at"], reverse=True)
        return out[:safe_limit]

    def get_skill(
        self,
        *,
        customer_id: str,
        name: str,
        include_files: bool = True,
        include_global: bool = True,
    ) -> dict[str, Any] | None:
        safe_name = _normalize_skill_name(name)
        safe_customer = str(customer_id or "").strip()
        with self._conn() as conn:
            row = None
            if safe_customer:
                row = conn.execute(
                    """
                    SELECT scope, customer_id, name, description, source, enabled, skill_path, created_at, updated_at
                    FROM skills
                    WHERE scope='user' AND customer_id=? AND name=?
                    """,
                    (safe_customer, safe_name),
                ).fetchone()
            if row is None and include_global:
                row = conn.execute(
                    """
                    SELECT scope, customer_id, name, description, source, enabled, skill_path, created_at, updated_at
                    FROM skills
                    WHERE scope='global' AND customer_id='' AND name=?
                    """,
                    (safe_name,),
                ).fetchone()
        if row is None:
            return None
        item = self._row_to_item(row, include_paths=True)
        skill_path = Path(item["skill_path"])
        if not skill_path.exists():
            return None
        item["skill_markdown"] = skill_path.read_text(encoding="utf-8", errors="replace")
        if include_files:
            item["supporting_files"] = self._load_supporting_files(skill_path.parent)
        return item

    def delete_skill(
        self,
        *,
        scope: str,
        customer_id: str,
        name: str,
    ) -> bool:
        safe_scope = self._validate_scope(scope)
        safe_customer = self._scope_customer(scope=safe_scope, customer_id=customer_id)
        safe_name = _normalize_skill_name(name)
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT skill_path FROM skills
                WHERE scope=? AND customer_id=? AND name=?
                """,
                (safe_scope, safe_customer, safe_name),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                DELETE FROM skills
                WHERE scope=? AND customer_id=? AND name=?
                """,
                (safe_scope, safe_customer, safe_name),
            )
            conn.commit()
        skill_md = Path(str(row["skill_path"]))
        skill_dir = skill_md.parent
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
        return True

    @staticmethod
    def _load_supporting_files(skill_dir: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        max_files = 12
        max_chars = 12000
        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "SKILL.md":
                continue
            if len(out) >= max_files:
                break
            rel = str(path.relative_to(skill_dir))
            out[rel] = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return out

    @staticmethod
    def _row_to_item(row: sqlite3.Row, *, include_paths: bool) -> dict[str, Any]:
        item = {
            "scope": str(row["scope"]),
            "customer_id": str(row["customer_id"]),
            "name": str(row["name"]),
            "description": str(row["description"]),
            "source": str(row["source"]),
            "enabled": bool(int(row["enabled"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        if include_paths:
            item["skill_path"] = str(row["skill_path"])
        return item

    def _ensure_global_skill(self, *, name: str, description: str, instructions: str) -> None:
        if self.get_skill(customer_id="", name=name, include_files=False, include_global=True):
            return
        markdown = build_skill_markdown(
            name=name,
            description=description,
            instructions=instructions,
        )
        self.upsert_skill(
            scope="global",
            customer_id="",
            name=name,
            skill_markdown=markdown,
            source="system_bootstrap",
            enabled=True,
            supporting_files=None,
        )

    def ensure_default_skill(self) -> None:
        existing_skill_creator = self.get_skill(
            customer_id="",
            name="skill-creator",
            include_files=False,
            include_global=True,
        )
        desired_skill_creator = build_skill_markdown(
            name="skill-creator",
            description=_DEFAULT_SKILL_CREATOR_DESCRIPTION,
            instructions=_DEFAULT_SKILL_CREATOR_INSTRUCTIONS,
        )
        legacy_skill_creator = build_skill_markdown(
            name="skill-creator",
            description=_DEFAULT_SKILL_CREATOR_DESCRIPTION,
            instructions=_LEGACY_SKILL_CREATOR_INSTRUCTIONS,
        )
        if existing_skill_creator is None or (
            existing_skill_creator["source"] == "system_bootstrap"
            and existing_skill_creator["skill_markdown"] in {legacy_skill_creator, desired_skill_creator}
        ):
            self.upsert_skill(
                scope="global",
                customer_id="",
                name="skill-creator",
                skill_markdown=desired_skill_creator,
                source="system_bootstrap",
                enabled=True,
                supporting_files=None,
            )
        existing_browser_use_operator = self.get_skill(
            customer_id="",
            name="browser-use-operator",
            include_files=False,
            include_global=True,
        )
        desired_browser_use_operator = build_skill_markdown(
            name="browser-use-operator",
            description=_DEFAULT_BROWSER_USE_OPERATOR_DESCRIPTION,
            instructions=_DEFAULT_BROWSER_USE_OPERATOR_INSTRUCTIONS,
        )
        legacy_browser_use_operator = build_skill_markdown(
            name="browser-use-operator",
            description=_DEFAULT_BROWSER_USE_OPERATOR_DESCRIPTION,
            instructions=_LEGACY_BROWSER_USE_OPERATOR_INSTRUCTIONS,
        )
        if existing_browser_use_operator is None or (
            existing_browser_use_operator["source"] == "system_bootstrap"
            and existing_browser_use_operator["skill_markdown"]
            in {legacy_browser_use_operator, desired_browser_use_operator}
        ):
            self.upsert_skill(
                scope="global",
                customer_id="",
                name="browser-use-operator",
                skill_markdown=desired_browser_use_operator,
                source="system_bootstrap",
                enabled=True,
                supporting_files=None,
            )
        existing_signal_integration_operator = self.get_skill(
            customer_id="",
            name="signal-integration-operator",
            include_files=False,
            include_global=True,
        )
        desired_signal_integration_operator = build_skill_markdown(
            name="signal-integration-operator",
            description=_DEFAULT_SIGNAL_INTEGRATION_OPERATOR_DESCRIPTION,
            instructions=_DEFAULT_SIGNAL_INTEGRATION_OPERATOR_INSTRUCTIONS,
        )
        legacy_signal_integration_operator = build_skill_markdown(
            name="signal-integration-operator",
            description=_DEFAULT_SIGNAL_INTEGRATION_OPERATOR_DESCRIPTION,
            instructions=_LEGACY_SIGNAL_INTEGRATION_OPERATOR_INSTRUCTIONS,
        )
        if existing_signal_integration_operator is None or (
            existing_signal_integration_operator["source"] == "system_bootstrap"
            and existing_signal_integration_operator["skill_markdown"]
            in {legacy_signal_integration_operator, desired_signal_integration_operator}
        ):
            self.upsert_skill(
                scope="global",
                customer_id="",
                name="signal-integration-operator",
                skill_markdown=desired_signal_integration_operator,
                source="system_bootstrap",
                enabled=True,
                supporting_files=None,
            )
        self._ensure_global_skill(
            name="routine-schedule-composer",
            description=(
                "Use this skill when creating or updating reminders/scheduled routines with "
                "routine_create, especially when you need clear schedule-time instructions that "
                "capture scripts, files, and required resources."
            ),
            instructions=(
                "## Purpose\n"
                "Compose routine_create payloads so schedule-time behavior is explicit and deterministic.\n\n"
                "## Field mapping\n"
                "1. instruction: schedule-time scratchpad (what to run, files to read/write, expected output).\n"
                "2. implementation_command: concrete shell/script command for scheduled execution and guardrail evaluation.\n\n"
                "3. implementation_command path style: keep script/file arguments relative to working_dir.\n"
                "   Example with default working_dir=tulpa_stuff: use `python3 tg_login.py`, not `python3 tulpa_stuff/tg_login.py`.\n\n"
                "## Instruction style\n"
                "1. Write instruction in second-person imperative voice: start with 'You must ...'.\n"
                "2. Include concrete steps, required scripts/files/keys source, and expected result.\n"
                "3. Include failure/reporting behavior (what to return/log if blocked).\n\n"
                "## Execution claim policy\n"
                "1. If user asked for immediate bootstrap/initialization, execute now and verify before claiming success.\n"
                "2. If only scheduling was done, state clearly that future runs are scheduled but bootstrap was not executed.\n"
                "3. Never include concrete fetched facts (headlines/metrics) unless they came from tool output in this run.\n\n"
                "## Defaults\n"
                "1. Set notify_user=true unless user explicitly asks for silent runs.\n"
                "2. For one-time reminders from relative time phrases, use local ISO datetime schedule.\n"
                "3. For recurring jobs, use cron schedule.\n\n"
                "## Quality checks before calling routine_create\n"
                "1. Ensure instruction describes the actual work output (file/API/update).\n"
                "2. Ensure instruction references required scripts/files/keys source as needed.\n"
                "3. Ensure implementation_command is concrete (executable + args), not natural language.\n"
            ),
        )
        self._ensure_global_skill(
            name="incoming-signal-handler-designer",
            description=_DEFAULT_INCOMING_SIGNAL_HANDLER_DESIGNER_DESCRIPTION,
            instructions=_DEFAULT_INCOMING_SIGNAL_HANDLER_DESIGNER_INSTRUCTIONS,
        )
