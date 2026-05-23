"""OpenAI-compatible oracle client for business knowledge queries."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from opentulpa.business_knowledge.extraction import content_hash

DEFAULT_ORACLE_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_ORACLE_MAX_OUTPUT_TOKENS = 1000
DEFAULT_OPENROUTER_APP_REFERER = "https://github.com/kvyb/opentulpa"
DEFAULT_OPENROUTER_APP_TITLE = "OpenTulpa"


class OpenAICompatibleKnowledgeOracleClient:
    """Small OpenAI-compatible chat client for the business knowledge oracle."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = DEFAULT_ORACLE_MODEL,
        timeout_seconds: float = 45.0,
        trace_path: Path | None = None,
        langfuse_tracer: Any | None = None,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "").strip() or DEFAULT_ORACLE_MODEL
        self.timeout_seconds = float(timeout_seconds)
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.base_url:
            raise ValueError("base_url is required")
        self.trace_path = trace_path.resolve() if isinstance(trace_path, Path) else None
        self.langfuse_tracer = langfuse_tracer
        self._trace_lock = threading.Lock()

    def answer(
        self,
        *,
        source_pack: str,
        query: str,
        workflow_context: dict[str, Any] | None = None,
        max_output_tokens: int = DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ) -> str:
        started = time.monotonic()
        request_body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max(1, int(max_output_tokens)),
            "messages": [
                {"role": "system", "content": oracle_system_prompt()},
                {
                    "role": "user",
                    "content": oracle_user_prompt(
                        source_pack=source_pack,
                        query=query,
                        workflow_context=workflow_context,
                    ),
                },
            ],
        }
        request_body.update(oracle_reasoning_control(self.base_url))
        response_payload: dict[str, Any] | None = None
        answer_text = ""
        error_text: str | None = None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **openrouter_app_headers(self.base_url),
                },
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw_payload = response.json()
            response_payload = raw_payload if isinstance(raw_payload, dict) else {}
            answer_text = extract_chat_text(response_payload)
            return answer_text
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_oracle_trace(
                source_pack=source_pack,
                query=query,
                workflow_context=workflow_context,
                max_output_tokens=int(str(request_body.get("max_tokens") or "0")),
                response_payload=response_payload,
                response_text=answer_text,
                error=error_text,
                elapsed_ms=_elapsed_ms(started),
            )

    def extract_intent(self, *, query: str) -> dict[str, Any]:
        started = time.monotonic()
        request_body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": oracle_intent_system_prompt()},
                {"role": "user", "content": str(query or "").strip()},
            ],
        }
        request_body.update(oracle_reasoning_control(self.base_url))
        response_payload: dict[str, Any] | None = None
        response_text = ""
        error_text: str | None = None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **openrouter_app_headers(self.base_url),
                },
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw_payload = response.json()
            response_payload = raw_payload if isinstance(raw_payload, dict) else {}
            response_text = extract_chat_text(response_payload)
            return clean_query_intent(parse_json_object(response_text), str(query or ""))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_oracle_trace(
                source_pack="",
                query=query,
                workflow_context={},
                max_output_tokens=int(str(request_body.get("max_tokens") or "0")),
                response_payload=response_payload,
                response_text=response_text,
                error=error_text,
                elapsed_ms=_elapsed_ms(started),
                call_site="knowledge_oracle_intent",
            )

    def _record_oracle_trace(
        self,
        *,
        source_pack: str,
        query: str,
        workflow_context: dict[str, Any] | None,
        max_output_tokens: int,
        response_payload: dict[str, Any] | None,
        response_text: str,
        error: str | None,
        elapsed_ms: int,
        call_site: str = "knowledge_oracle",
    ) -> None:
        source_text = str(source_pack or "")
        payload = _oracle_trace_payload(
            model=self.model,
            source_text=source_text,
            query=query,
            workflow_context=workflow_context,
            max_output_tokens=max_output_tokens,
            response_payload=response_payload,
            response_text=response_text,
            error=error,
            elapsed_ms=elapsed_ms,
            call_site=call_site,
        )
        tracer = getattr(self, "langfuse_tracer", None)
        record_generation = getattr(tracer, "record_generation", None)
        if callable(record_generation):
            with suppress(Exception):
                record_generation(payload)
        path = self.trace_path
        if path is None:
            return
        serialized = json.dumps(payload, ensure_ascii=False, default=str)

        def _commit() -> None:
            existing: list[str] = []
            with suppress(Exception):
                existing = [
                    line.rstrip("\n")
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            kept = existing[-499:]
            kept.append(serialized)
            with path.open("w", encoding="utf-8") as f:
                f.write("\n".join(kept) + "\n")

        with suppress(Exception):
            path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), self._trace_lock:
            _commit()


def _oracle_trace_payload(
    *,
    model: str,
    source_text: str,
    query: str,
    workflow_context: dict[str, Any] | None,
    max_output_tokens: int,
    response_payload: dict[str, Any] | None,
    response_text: str,
    error: str | None,
    elapsed_ms: int,
    call_site: str,
) -> dict[str, Any]:
    source_hash = content_hash(source_text.encode("utf-8", errors="replace"))
    clean_response = str(response_text or "").strip()
    return {
        "ts": datetime.now(UTC).isoformat(),
        "model_name": model,
        "stable_prefix_count": 0,
        "prompt_messages": _oracle_trace_messages(
            source_text=source_text,
            source_hash=source_hash,
            query=query,
            workflow_context=workflow_context,
        ),
        "prompt_message_count": 2,
        "response_type": "KnowledgeOracleAnswer",
        "response_message": None,
        "response_text": clean_response,
        "response_content": clean_response,
        "response_tool_calls": None,
        "error": str(error or "").strip() or None,
        "call_site": str(call_site or "knowledge_oracle"),
        "source_pack_chars": len(source_text),
        "source_pack_sha256": source_hash,
        "max_output_tokens": int(max_output_tokens),
        "elapsed_ms": int(elapsed_ms),
        "usage": (response_payload or {}).get("usage") if isinstance(response_payload, dict) else None,
    }


def _oracle_trace_messages(
    *,
    source_text: str,
    source_hash: str,
    query: str,
    workflow_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "type": "SystemMessage", "text": oracle_system_prompt()},
        {
            "role": "user",
            "type": "HumanMessage",
            "text": (
                f"QUERY:\n{str(query or '').strip()}\n\n"
                f"WORKFLOW_CONTEXT_JSON:\n{_json_dumps(workflow_context or {})}\n\n"
                f"SOURCE_PACK_CHARS: {len(source_text)}\n"
                f"SOURCE_PACK_SHA256: {source_hash}"
            ),
        },
    ]


def oracle_system_prompt() -> str:
    return (
        "You are a workflow business knowledge oracle. Answer only from the SOURCE PACK. "
        "The SOURCE PACK is normalized evidence from user-uploaded files. It may be compact TOON table evidence. "
        "For TOON evidence, row_label is the original row text and cells contain column/header/value pairs. "
        "When cells include header_group N, the group numbers are distinct column/header groups from left to right; "
        "if the query names class, tier, category, option, or group N, use that matching header_group N. "
        "Duplicate header suffixes like [1] and [2] mean repeated adjacent source columns in spreadsheet order; "
        "when row_label clearly lists ordered variants, map earlier/later variants to earlier/later duplicate columns. "
        "If SOURCE PACK mode is overview, summarize available matching services/categories and ask a concise clarifying "
        "question when an exact price requires a service, variant, class, size, or other qualifier. "
        "For capability questions, answer yes only if a matching source row supports that capability; otherwise say you need to confirm. "
        "If the query or workflow context says the latest customer request is outside the configured workflow scope, "
        "do not answer source facts for that out-of-scope category; return exactly NO_SOURCE. "
        "If the query is only about cancelling, rescheduling, or correcting an existing booking and no new source-backed business fact is needed, return exactly NO_SOURCE. "
        "Do not guess, infer unsupported facts, or use outside knowledge. "
        "If multiple rows support the same requested fact with different values, state the concise distinction instead of choosing silently. "
        "Return a plain string only: concise but informative facts the intake agent needs, with source refs when useful. "
        "For broad pricing questions, include the relevant ranges/options and the missing qualifiers needed for an exact answer. "
        "If the source does not answer the query, return exactly NO_SOURCE. "
        "If the source is ambiguous, return AMBIGUOUS: followed by one concise clarifying question. "
        "Stay under 1000 tokens, and prefer much shorter when the answer is narrow."
    )


def oracle_intent_system_prompt() -> str:
    return (
        "Extract structured search intent for local matching over arbitrary business files. "
        "Return raw JSON only, no markdown. Keys: mode, target_terms, qualifier_terms, ignore_terms. "
        "mode is one of specific_fact, category_overview, corpus_overview, capability_check. "
        "target_terms are service/product/item/action names likely found in row labels, headings, or column groups. "
        "qualifier_terms are variants/classes/sizes/dimensions/locations likely found in headers, row labels, or nearby context. "
        "Use category_overview or corpus_overview for broad questions asking what exists, what services are offered, "
        "or price lists for a whole category. Use capability_check for yes/no service availability questions. "
        "ignore_terms are generic request words that should not drive retrieval. Preserve user wording. Do not answer."
    )


def openrouter_app_headers(base_url: str) -> dict[str, str]:
    if "openrouter.ai" not in str(base_url or "").casefold():
        return {}
    title = str(os.environ.get("OPENROUTER_APP_TITLE", "")).strip() or DEFAULT_OPENROUTER_APP_TITLE
    headers = {"HTTP-Referer": DEFAULT_OPENROUTER_APP_REFERER}
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def oracle_reasoning_control(base_url: str) -> dict[str, Any]:
    if "openrouter.ai" not in str(base_url or "").casefold():
        return {}
    return {"reasoning": {"effort": "none", "exclude": True}}


def oracle_user_prompt(
    *,
    source_pack: str,
    query: str,
    workflow_context: dict[str, Any] | None,
) -> str:
    context = _json_dumps(workflow_context or {})
    return (
        f"QUERY:\n{query.strip()}\n\n"
        f"WORKFLOW_CONTEXT_JSON:\n{context}\n\n"
        "SOURCE PACK:\n"
        f"{source_pack}"
    )


def extract_chat_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            return "\n".join(parts)
    text = first.get("text")
    return str(text or "")


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    with suppress(Exception):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        with suppress(Exception):
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
    return {}


def clean_query_intent(value: Any, query: str) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else {}
    mode = str(parsed.get("mode", "") or "").strip().lower()
    if mode not in {"specific_fact", "category_overview", "corpus_overview", "capability_check"}:
        mode = "specific_fact"
    intent = {
        "mode": mode,
        "target_terms": safe_text_list(parsed.get("target_terms")),
        "qualifier_terms": safe_text_list(parsed.get("qualifier_terms")),
        "ignore_terms": safe_text_list(parsed.get("ignore_terms")),
    }
    if not intent["target_terms"]:
        intent["target_terms"] = [str(query or "").strip()]
    return intent


def safe_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return unique_strings([str(item).strip() for item in value if str(item).strip()])
    text = str(value or "").strip()
    return [text] if text else []


def unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
