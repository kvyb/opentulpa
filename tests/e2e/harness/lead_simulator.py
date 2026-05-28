from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from harness.llm_json import extract_chat_completion_text, normalize_bool, parse_json_object
from harness.logging import JsonlRecorder

DEFAULT_LEAD_SIMULATOR_MODEL = os.getenv(
    "OPENTULPA_E2E_LEAD_SIM_MODEL",
    "google/gemini-3-flash-preview",
)


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text[:120])
    return out[:12]


def _booking_completed(booking_state: dict[str, Any] | None) -> bool:
    if not isinstance(booking_state, dict):
        return False
    return str(booking_state.get("status", "")).strip().lower() == "completed"


@dataclass
class LeadProfile:
    objective: str
    initial_message: str
    known_facts: dict[str, str]
    persona: str = ""
    disclosure_style: str = (
        "Be cooperative, realistic, concise, and do not volunteer every fact at once unless asked."
    )
    rules: list[str] = field(default_factory=list)
    max_turns: int = 6

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "objective": str(self.objective or "").strip(),
            "initial_message": str(self.initial_message or "").strip(),
            "known_facts": {
                str(key or "").strip(): str(value or "").strip()
                for key, value in dict(self.known_facts or {}).items()
                if str(key or "").strip() and str(value or "").strip()
            },
            "persona": str(self.persona or "").strip(),
            "disclosure_style": str(self.disclosure_style or "").strip(),
            "rules": [str(item or "").strip() for item in list(self.rules or []) if str(item or "").strip()],
            "max_turns": max(1, int(self.max_turns or 6)),
        }


@dataclass
class LeadTurnPlan:
    done: bool
    message: str
    reason: str
    shared_fact_keys: list[str]
    raw_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "done": bool(self.done),
            "message": str(self.message or "").strip(),
            "reason": str(self.reason or "").strip(),
            "shared_fact_keys": list(self.shared_fact_keys),
            "raw_text": str(self.raw_text or "").strip(),
        }


class LeadSimulator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        recorder: JsonlRecorder | None = None,
        model: str = DEFAULT_LEAD_SIMULATOR_MODEL,
        timeout_seconds: float = 40.0,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._recorder = recorder
        self._model = str(model or DEFAULT_LEAD_SIMULATOR_MODEL).strip() or DEFAULT_LEAD_SIMULATOR_MODEL
        self._timeout_seconds = max(5.0, float(timeout_seconds))

    @property
    def model(self) -> str:
        return self._model

    def plan_next_turn(
        self,
        *,
        profile: LeadProfile,
        transcript: list[dict[str, Any]],
        booking_state: dict[str, Any] | None = None,
    ) -> LeadTurnPlan:
        if _booking_completed(booking_state):
            return LeadTurnPlan(
                done=True,
                message="",
                reason="booking_state_completed",
                shared_fact_keys=[],
            )

        payload = {
            "lead_profile": profile.as_prompt_payload(),
            "transcript": [
                {
                    "role": str(item.get("role", "")).strip()[:30],
                    "text": str(item.get("text", "")).strip()[:1000],
                }
                for item in transcript[-14:]
            ],
            "booking_state": dict(booking_state or {}),
        }
        if self._recorder is not None:
            self._recorder.add(
                "lead_simulator_prompt",
                model=self._model,
                payload=payload,
            )

        system_prompt = (
            "You simulate a realistic incoming Telegram DM lead for an e2e test.\n"
            "Stay in character as the lead only.\n"
            "Use the hidden profile as ground truth and remain internally consistent.\n"
            "The lead is cooperative and genuinely wants to complete the booking.\n"
            "Be concise and natural, like a real Telegram DM.\n"
            "Do not reveal every hidden fact at once unless the assistant explicitly asks for them in the same turn.\n"
            "Answer direct questions first, then provide any requested missing details.\n"
            "Never mention being a simulator, test, prompt, JSON, or model.\n"
            "Set done=true only when booking_state.status is completed.\n"
            "Do not treat phrases like 'I'll get that confirmed for you' or pricing confirmation as a completed booking.\n"
            "Return strict JSON only with exactly these keys:\n"
            "{\n"
            '  "done": boolean,\n'
            '  "message": string,\n'
            '  "reason": string,\n'
            '  "shared_fact_keys": [string]\n'
            "}\n"
            "Rules:\n"
            "- If done=false, message must be a non-empty lead reply.\n"
            "- If done=true, message should usually be empty.\n"
            "- shared_fact_keys should list the hidden fact keys revealed in this turn.\n"
            "- Do not invent facts outside the hidden profile."
        )
        user_prompt = (
            "Plan the lead's next message.\n\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
        )

        raw_text = ""
        error_text = ""
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                response_payload = response.json()
                raw_text = extract_chat_completion_text(response_payload)
                parsed = parse_json_object(raw_text)
                if isinstance(parsed, dict):
                    plan = LeadTurnPlan(
                        done=normalize_bool(parsed.get("done")),
                        message=str(parsed.get("message", "") or "").strip(),
                        reason=str(parsed.get("reason", "") or "").strip()[:400],
                        shared_fact_keys=_normalize_text_list(parsed.get("shared_fact_keys")),
                        raw_text=raw_text,
                    )
                    if plan.done and not _booking_completed(booking_state):
                        error_text = "premature_done_without_completed_booking"
                        continue
                    if plan.done or plan.message:
                        if self._recorder is not None:
                            self._recorder.add(
                                "lead_simulator_response",
                                model=self._model,
                                attempt=attempt + 1,
                                payload=plan.as_dict(),
                            )
                        return plan
                error_text = "invalid_json_output"
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"

        fallback = self._fallback_turn(
            profile=profile,
            transcript=transcript,
            reason=error_text or "lead_simulator_fallback",
            raw_text=raw_text,
        )
        if self._recorder is not None:
            self._recorder.add(
                "lead_simulator_response",
                model=self._model,
                attempt=0,
                payload=fallback.as_dict(),
            )
        return fallback

    def _fallback_turn(
        self,
        *,
        profile: LeadProfile,
        transcript: list[dict[str, Any]],
        reason: str,
        raw_text: str,
    ) -> LeadTurnPlan:
        known_facts = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in dict(profile.known_facts or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        disclosed_keys = self._disclosed_fact_keys(profile=profile, transcript=transcript)
        last_assistant = ""
        for item in reversed(transcript):
            if str(item.get("role", "")).strip().lower() == "assistant":
                last_assistant = str(item.get("text", "")).strip().lower()
                break
        prioritized_keys: list[str] = []
        keyword_map = {
            "car_model": ("model", "car", "vehicle"),
            "car_type": ("type", "suv", "sedan", "hatchback", "car"),
            "wash_type": ("wash", "service", "package", "detail"),
            "date": ("date", "day", "tomorrow", "when"),
            "time": ("time", "slot", "hour", "am", "pm"),
        }
        for key, tokens in keyword_map.items():
            if key in known_facts and any(token in last_assistant for token in tokens):
                prioritized_keys.append(key)
        for key in known_facts:
            if key not in prioritized_keys:
                prioritized_keys.append(key)
        selected_keys = [key for key in prioritized_keys if key not in disclosed_keys][:2]
        if not selected_keys and prioritized_keys:
            selected_keys = prioritized_keys[:2]
        parts = [known_facts[key] for key in selected_keys if key in known_facts]
        message = ". ".join(part for part in parts if part).strip()
        if not message:
            message = "Yes, that works for me."
        return LeadTurnPlan(
            done=False,
            message=message,
            reason=str(reason or "fallback").strip()[:400],
            shared_fact_keys=selected_keys,
            raw_text=raw_text,
        )

    @staticmethod
    def _disclosed_fact_keys(
        *,
        profile: LeadProfile,
        transcript: list[dict[str, Any]],
    ) -> set[str]:
        lead_text = " ".join(
            [
                str(item.get("text", "") or "").strip().lower()
                for item in transcript
                if str(item.get("role", "")).strip().lower() == "lead"
            ]
        )
        disclosed: set[str] = set()
        for key, value in dict(profile.known_facts or {}).items():
            safe_key = str(key or "").strip()
            safe_value = str(value or "").strip().lower()
            if safe_key and safe_value and safe_value in lead_text:
                disclosed.add(safe_key)
        return disclosed
