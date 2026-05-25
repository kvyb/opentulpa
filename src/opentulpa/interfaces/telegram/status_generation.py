"""LLM-generated Telegram status updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class _StatusMessageDecision(BaseModel):
    ok: bool = False
    text: str = ""


def _status_model(runtime: Any) -> tuple[Any, str]:
    for model_attr, name_attr in (
        ("_workflow_setup_input_classifier_model", "_workflow_setup_input_classifier_model_name"),
        ("_wake_classifier_model", "_wake_classifier_model_name"),
        ("_model", "model_name"),
    ):
        model = getattr(runtime, model_attr, None)
        if model is None:
            continue
        return model, str(getattr(runtime, name_attr, "") or "").strip()
    return None, ""


def _clean_status_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > 160:
        text = text[:157].rstrip() + "..."
    return text


async def generate_llm_status_message(
    *,
    runtime: Any,
    customer_id: str,
    thread_id: str,
    context: dict[str, Any],
    language: str = "Russian",
    timeout_seconds: float = 8.0,
) -> str | None:
    """Return LLM-generated visible status text, or None on any failure."""

    direct = getattr(runtime, "generate_status_message", None)
    if callable(direct):
        try:
            result = await direct(
                customer_id=customer_id,
                thread_id=thread_id,
                context=dict(context),
                language=language,
            )
        except Exception:
            logger.exception("telegram.status_generation direct generator failed")
            return None
        if isinstance(result, dict) and not bool(result.get("ok", False)):
            return None
        text = _clean_status_text(result.get("text") if isinstance(result, dict) else result)
        return text or None

    invoke_structured = getattr(runtime, "_invoke_structured_model", None)
    if not callable(invoke_structured):
        return None
    model, model_name = _status_model(runtime)
    if model is None:
        return None
    safe_context = json.dumps(dict(context), ensure_ascii=False)[:4000]
    try:
        decision, error = await asyncio.wait_for(
            invoke_structured(
                model=model,
                model_name=model_name or None,
                schema=_StatusMessageDecision,
                messages=[
                    SystemMessage(
                        content=(
                            "Generate one short visible Telegram status update while the assistant is still working.\n"
                            "Return strict JSON only with keys: ok, text.\n"
                            f"Write text in {language}. One sentence, under 90 characters.\n"
                            "Do not answer the user's business question. Do not invent prices, availability, "
                            "booking facts, or completion. Only say that the answer is being checked or prepared.\n"
                            "The context may include the user's latest message so you understand why the turn "
                            "is taking time, but it is background-only.\n"
                            "Do not mention the user's specific task, object names, message text, or table contents. "
                            "Do not repeat, translate, summarize, or paraphrase the user's wording.\n"
                            'If a useful status update is not appropriate, set ok=false and text="".'
                        )
                    ),
                    HumanMessage(content=f"context={safe_context}"),
                ],
                call_context={
                    "call_site": "telegram_status_generation",
                    "customer_id": str(customer_id or "").strip(),
                    "thread_id": str(thread_id or "").strip(),
                },
            ),
            timeout=max(0.1, float(timeout_seconds)),
        )
    except TimeoutError:
        logger.warning(
            "telegram.status_generation structured generator timed out timeout_seconds=%s",
            max(0.1, float(timeout_seconds)),
        )
        return None
    except Exception:
        logger.exception("telegram.status_generation structured generator failed")
        return None
    if error or decision is None or not isinstance(decision, _StatusMessageDecision):
        return None
    if not bool(decision.ok):
        return None
    text = _clean_status_text(decision.text)
    return text or None
