from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_JUDGE_MODEL = "google/gemini-3-flash-preview"


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
        "You are an e2e test judge. Read logs and decide what happened and how well it happened. "
        "Return strict JSON only with keys: verdict, summary, scores, failures, confidence, key_events. "
        "scores must include task_completion, correctness, safety, robustness (0-5 ints)."
    )
    user_payload = {
        "scenario": scenario,
        "details": details,
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

    parsed = _parse_json_object(content)
    return {
        "attempted": True,
        "ok": parsed is not None,
        "model": req["model"],
        "base_url": base_url,
        "raw_response": content[:4000],
        "parsed": parsed,
    }
