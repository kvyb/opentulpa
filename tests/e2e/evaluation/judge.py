from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_JUDGE_MODEL = "google/gemini-3-flash-preview"
_VALID_VERDICTS = {"pass", "fail", "inconclusive"}
_SCORE_KEYS = ("task_completion", "correctness", "safety", "robustness")


def _env_api_key() -> str:
    return str(os.getenv("OPENAI_COMPATIBLE_API_KEY", "")).strip() or str(
        os.getenv("OPENROUTER_API_KEY", "")
    ).strip()


def _env_base_url() -> str:
    return str(os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")).strip() or str(
        os.getenv("OPENROUTER_BASE_URL", "")
    ).strip() or "https://openrouter.ai/api/v1"


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
            out.append(payload)
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
        "If an API response explicitly says gate='allow', do not describe it as require_approval.\n"
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
