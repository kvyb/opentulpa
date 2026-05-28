from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from harness.lead_simulator import DEFAULT_LEAD_SIMULATOR_MODEL
from harness.llm_json import extract_chat_completion_text, normalize_bool, parse_json_object
from harness.logging import JsonlRecorder

DEFAULT_OWNER_SIMULATOR_MODEL = os.getenv(
    "OPENTULPA_E2E_OWNER_SIM_MODEL",
    DEFAULT_LEAD_SIMULATOR_MODEL,
)


@dataclass
class OwnerProfile:
    objective: str
    initial_message: str
    known_facts: dict[str, str]
    persona: str = "Practical business owner using Telegram."
    rules: list[str] = field(default_factory=list)
    max_turns: int = 7

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
            "rules": [str(item or "").strip() for item in self.rules if str(item or "").strip()],
            "max_turns": max(1, int(self.max_turns or 7)),
        }


@dataclass
class OwnerTurnPlan:
    done: bool
    message: str
    reason: str
    raw_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "done": bool(self.done),
            "message": str(self.message or "").strip(),
            "reason": str(self.reason or "").strip(),
            "raw_text": str(self.raw_text or "").strip(),
        }


class OwnerSimulator:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        recorder: JsonlRecorder | None = None,
        model: str = DEFAULT_OWNER_SIMULATOR_MODEL,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._recorder = recorder
        self._model = str(model or DEFAULT_OWNER_SIMULATOR_MODEL).strip() or DEFAULT_OWNER_SIMULATOR_MODEL
        self._timeout_seconds = max(5.0, float(timeout_seconds))

    @property
    def model(self) -> str:
        return self._model

    def plan_next_turn(
        self,
        *,
        profile: OwnerProfile,
        transcript: list[dict[str, Any]],
        workflow_state: dict[str, Any],
    ) -> OwnerTurnPlan:
        workflow_count = int(workflow_state.get("workflow_count") or 0)
        if workflow_count > 0:
            return OwnerTurnPlan(done=True, message="", reason="workflow_created")

        payload = {
            "owner_profile": profile.as_prompt_payload(),
            "workflow_state": dict(workflow_state),
            "transcript": [
                {
                    "role": str(item.get("role", "")).strip()[:30],
                    "text": str(item.get("text", "")).strip()[:1200],
                }
                for item in transcript[-16:]
            ],
        }
        if self._recorder is not None:
            self._recorder.add("owner_simulator_prompt", model=self._model, payload=payload)

        system_prompt = (
            "You simulate a business owner using OpenTulpa in Telegram for a live e2e test.\n"
            "Stay in character as the owner only.\n"
            "Your hidden objective is to create and activate the configured intake workflow.\n"
            "Use the hidden facts as ground truth. Answer OpenTulpa's questions directly.\n"
            "If OpenTulpa proposes a workflow or asks for confirmation, confirm saving and activation.\n"
            "If OpenTulpa asks for missing configuration, provide the relevant hidden facts.\n"
            "Do not mention being a simulator, model, JSON, prompt, or test.\n"
            "Set done=true only when workflow_state.workflow_count is greater than zero.\n"
            "Return strict JSON only with exactly these keys:\n"
            '{"done": boolean, "message": string, "reason": string}'
        )
        user_prompt = "Plan the owner's next Telegram message.\n\n" + json.dumps(
            payload, ensure_ascii=False, default=str
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
                raw_text = extract_chat_completion_text(response.json())
                parsed = parse_json_object(raw_text)
                if isinstance(parsed, dict):
                    plan = OwnerTurnPlan(
                        done=normalize_bool(parsed.get("done")),
                        message=str(parsed.get("message", "") or "").strip(),
                        reason=str(parsed.get("reason", "") or "").strip()[:500],
                        raw_text=raw_text,
                    )
                    if plan.done and workflow_count <= 0:
                        error_text = "premature_done_without_workflow"
                        continue
                    if plan.done or plan.message:
                        if self._recorder is not None:
                            self._recorder.add(
                                "owner_simulator_response",
                                model=self._model,
                                attempt=attempt + 1,
                                payload=plan.as_dict(),
                            )
                        return plan
                error_text = "invalid_json_output"
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"

        fallback = self._fallback_turn(profile=profile, transcript=transcript, reason=error_text, raw_text=raw_text)
        if self._recorder is not None:
            self._recorder.add(
                "owner_simulator_response",
                model=self._model,
                attempt=0,
                payload=fallback.as_dict(),
            )
        return fallback

    def _fallback_turn(
        self,
        *,
        profile: OwnerProfile,
        transcript: list[dict[str, Any]],
        reason: str,
        raw_text: str,
    ) -> OwnerTurnPlan:
        facts = profile.as_prompt_payload()["known_facts"]
        disclosed = " ".join(str(item.get("text", "")) for item in transcript).lower()
        undisclosed = [
            f"{key}: {value}"
            for key, value in facts.items()
            if str(value).strip().lower() not in disclosed
        ]
        message = "Please use this configuration: " + "; ".join(undisclosed[:6])
        if not undisclosed:
            message = "Looks correct. Please save and activate this workflow now."
        return OwnerTurnPlan(
            done=False,
            message=message,
            reason=reason or "owner_simulator_fallback",
            raw_text=raw_text,
        )
