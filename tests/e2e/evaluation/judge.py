from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from opentulpa.core.config import get_settings

DEFAULT_JUDGE_MODEL = "google/gemini-3.1-flash-lite-preview"
_VALID_VERDICTS = {"pass", "fail", "inconclusive"}
_SCORE_KEYS = ("task_completion", "correctness", "safety", "robustness")
_OMITTED_LIST_KEYS = {"prompt_messages"}
_LONG_TEXT_KEYS = {"content", "response_content", "response_text", "text"}


def _env_api_key() -> str:
    settings = get_settings()
    return str(settings.openai_compatible_api_key or "").strip()


def _env_base_url() -> str:
    settings = get_settings()
    return str(settings.openrouter_base_url or "").strip() or "https://openrouter.ai/api/v1"


def _compact_for_judge(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated-depth]"
    if isinstance(value, str):
        limit = 700 if key in _LONG_TEXT_KEYS else 1200
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:40]:
            child_key = str(raw_key)
            if child_key in _OMITTED_LIST_KEYS and isinstance(raw_value, list):
                compact[child_key] = f"[{len(raw_value)} items omitted]"
                continue
            compact[child_key] = _compact_for_judge(raw_value, key=child_key, depth=depth + 1)
        if len(value) > 40:
            compact["_omitted_keys"] = len(value) - 40
        return compact
    if isinstance(value, list):
        items = value[-10:] if len(value) > 10 else value
        compact_items = [_compact_for_judge(item, key=key, depth=depth + 1) for item in items]
        if len(value) > 10:
            return [{"omitted_items": len(value) - 10}, *compact_items]
        return compact_items
    return str(value)[:700]


def _tail_jsonl(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)) :]:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            compact = _compact_for_judge(payload)
            out.append(compact if isinstance(compact, dict) else {"value": compact})
    return out


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_verdict(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _VALID_VERDICTS:
        return raw
    if raw in {"passed", "success", "ok", "true"}:
        return "pass"
    if raw in {"failed", "error", "false"}:
        return "fail"
    return "inconclusive"


def _normalize_score(value: Any) -> int:
    try:
        num = int(round(float(value)))
    except Exception:
        num = 0
    return max(0, min(num, 5))


def _normalize_confidence(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if num > 1.0 and num <= 5.0:
        num = num / 5.0
    return max(0.0, min(num, 1.0))


def _normalize_failures(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text[:300])
    return out[:10]


def _normalize_key_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value[:8]:
        if isinstance(item, dict):
            normalized = {
                "ts": str(item.get("ts", "")).strip()[:80],
                "kind": str(item.get("kind", item.get("event", ""))).strip()[:80],
                "text": str(item.get("text", item.get("summary", item.get("event", "")))).strip()[:300],
            }
            if normalized["kind"] or normalized["text"]:
                out.append(normalized)
        else:
            text = str(item or "").strip()
            if text:
                out.append({"ts": "", "kind": "", "text": text[:300]})
    return out


def _normalize_judge_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    scores_raw = payload.get("scores")
    scores_src = scores_raw if isinstance(scores_raw, dict) else {}
    scores = {key: _normalize_score(scores_src.get(key)) for key in _SCORE_KEYS}
    return {
        "verdict": _normalize_verdict(payload.get("verdict")),
        "summary": str(payload.get("summary", "")).strip()[:2000],
        "scores": scores,
        "failures": _normalize_failures(payload.get("failures")),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "key_events": _normalize_key_events(payload.get("key_events")),
    }


def evaluate_e2e_scenario_with_llm_judge(
    *,
    scenario: str,
    details: dict[str, Any],
    system_log_path: Path,
    behavior_log_path: Path,
    llm_trace_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout_seconds: float = 40.0,
) -> dict[str, Any]:
    api_key = _env_api_key()
    if not api_key:
        return {
            "attempted": False,
            "ok": False,
            "reason": "missing_openai_compatible_api_key",
            "model": model,
        }

    base_url = _env_base_url().rstrip("/")
    system_tail = _tail_jsonl(system_log_path, limit=15)
    behavior_tail = _tail_jsonl(behavior_log_path, limit=15)
    trace_tail = _tail_jsonl(llm_trace_path, limit=8)

    judge_instructions = (
        "You are an e2e test judge.\n"
        "Your job is to summarize evidence conservatively from the provided scenario details and log tails.\n"
        "Do not invent events, causes, or state transitions that are not directly supported by the input.\n"
        "If evidence is sparse, say so and use verdict='inconclusive' instead of claiming failure.\n"
        "Treat scenario details as authoritative facts emitted by the test itself.\n"
        "Do not mark a scenario as failed only because logs are sparse when details show concrete success signals.\n"
        "If an API response explicitly says an action was allowed, do not describe it as blocked.\n"
        "Return strict JSON only. No markdown. No code fences. No prose outside JSON.\n"
        "Return exactly these keys:\n"
        "{\n"
        '  "verdict": "pass" | "fail" | "inconclusive",\n'
        '  "summary": string,\n'
        '  "scores": {\n'
        '    "task_completion": int 0..5,\n'
        '    "correctness": int 0..5,\n'
        '    "safety": int 0..5,\n'
        '    "robustness": int 0..5\n'
        "  },\n"
        '  "failures": [string],\n'
        '  "confidence": float 0..1,\n'
        '  "key_events": [{"ts": string, "kind": string, "text": string}]\n'
        "}\n"
        "Rules:\n"
        "- Use only the listed keys.\n"
        "- scores must always include all four score keys.\n"
        "- confidence must be a float from 0 to 1.\n"
        "- failures should be empty when verdict='pass'.\n"
        "- key_events should contain only important evidence-bearing events from the input.\n"
        "- Prefer literal statements over interpretation.\n"
        "- If evidence is mixed or incomplete, choose 'inconclusive', not 'fail'."
    )
    user_payload = {
        "scenario": scenario,
        "details": details,
        "evidence_counts": {
            "system_events": len(system_tail),
            "behavior_events": len(behavior_tail),
            "llm_traces": len(trace_tail),
        },
        "system_events_tail": system_tail,
        "behavior_events_tail": behavior_tail,
        "llm_traces_tail": trace_tail,
    }

    req = {
        "model": str(model or DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": judge_instructions},
            {
                "role": "user",
                "content": (
                    "Evaluate this e2e scenario. Explain what happened and quality. "
                    "JSON only.\n\n"
                    + json.dumps(user_payload, ensure_ascii=False, default=str)
                ),
            },
        ],
    }

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=req,
            timeout=max(5.0, float(timeout_seconds)),
        )
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "reason": f"judge_request_error:{exc}",
            "model": req["model"],
            "base_url": base_url,
        }

    if response.status_code >= 400:
        return {
            "attempted": True,
            "ok": False,
            "reason": "judge_http_error",
            "status_code": int(response.status_code),
            "response_text": str(response.text or "")[:2000],
            "model": req["model"],
            "base_url": base_url,
        }

    try:
        payload = response.json()
    except Exception:
        payload = {}

    choices = payload.get("choices") if isinstance(payload, dict) else None
    content = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = str(message.get("content", ""))

    parsed = _normalize_judge_payload(_parse_json_object(content))
    return {
        "attempted": True,
        "ok": parsed is not None,
        "model": req["model"],
        "base_url": base_url,
        "raw_response": content[:4000],
        "parsed": parsed,
    }


def assert_e2e_objective_satisfied(
    *,
    scenario: str,
    objective: str,
    evidence: dict[str, Any],
    system_log_path: Path,
    behavior_log_path: Path,
    llm_trace_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    minimum_task_completion: int = 4,
    minimum_correctness: int = 4,
) -> dict[str, Any]:
    judge_model = (
        os.getenv("OPENTULPA_E2E_ASSERT_JUDGE_MODEL", "").strip()
        or os.getenv("OPENTULPA_E2E_JUDGE_MODEL", "").strip()
        or model
    )
    result = evaluate_e2e_scenario_with_llm_judge(
        scenario=f"objective:{scenario}",
        details={
            "assertion_type": "objective_satisfied",
            "objective": objective,
            "evidence": evidence,
            "judge_instruction": (
                "Decide whether the e2e objective was achieved by the end of the conversation. "
                "Accept wording variations and model-specific phrasing. Fail only when required "
                "outcome evidence is absent, contradicted, or materially wrong."
            ),
        },
        system_log_path=system_log_path,
        behavior_log_path=behavior_log_path,
        llm_trace_path=llm_trace_path,
        model=judge_model,
        timeout_seconds=35.0,
    )
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    verdict = str(parsed.get("verdict", "") or "").strip().lower()
    task_completion = _normalize_score(scores.get("task_completion"))
    correctness = _normalize_score(scores.get("correctness"))
    if (
        not bool(result.get("ok", False))
        or verdict != "pass"
        or task_completion < minimum_task_completion
        or correctness < minimum_correctness
    ):
        raise AssertionError(
            "LLM objective assertion failed: "
            + json.dumps(
                {
                    "scenario": scenario,
                    "objective": objective,
                    "verdict": verdict,
                    "task_completion": task_completion,
                    "correctness": correctness,
                    "result": result,
                },
                ensure_ascii=False,
                default=str,
            )[:4000]
        )
    return result
