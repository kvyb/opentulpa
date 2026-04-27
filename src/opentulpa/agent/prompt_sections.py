"""Sectioned prompt builders for the OpenTulpa agent."""

from __future__ import annotations

from typing import Literal

from opentulpa.agent.lc_messages import SystemMessage

PromptMode = Literal["literal_chat", "task_chat", "execution", "workflow_setup"]

# Placed between stable policy and per-turn injected context. Keeps the prefix
# before this marker byte-stable for provider prompt caching (OpenAI/Gemini
# implicit; Anthropic explicit / automatic via OpenRouter).
PROMPT_DYNAMIC_BOUNDARY = (
    "[OPENTULPA_PROMPT_DYNAMIC_BOUNDARY]\n"
    "Below this marker, injected context may change every turn (modes, time, "
    "retrieval, skills, aliases)."
)


def build_prompt_mode_message(prompt_mode: PromptMode) -> SystemMessage:
    if prompt_mode == "literal_chat":
        text = (
            "Prompt mode: literal_chat.\n"
            "Treat this as a local conversational turn.\n"
            "Answer the visible user question directly.\n"
            "Do not pull in hidden project context, thread summaries, or matched skills unless the user explicitly references them.\n"
            "If the user asks a greeting or how-you-are question, answer it plainly and warmly.\n"
            "Do not pivot into a new topic or end with a follow-up question unless the user asked for help beyond the greeting."
        )
    elif prompt_mode == "execution":
        text = (
            "Prompt mode: execution.\n"
            "This turn likely needs tools, fresh state, or side effects.\n"
            "Use relevant retrieved context when it improves execution reliability.\n"
            "Prefer concrete status reporting over broad planning language."
        )
    elif prompt_mode == "workflow_setup":
        text = (
            "Prompt mode: workflow_setup.\n"
            "This is a collaborative intake workflow setup session.\n"
            "Treat the stored draft as the source of truth for the in-progress workflow configuration.\n"
            "Prefer concise setup questions, draft updates, and proposal summaries over generic chat.\n"
            "Do not ask for Telegram Business DM polling/schedule intervals; those workflows run from inbound messages.\n"
            "Before showing the final proposal, run the setup preflight once; if it returns a focused follow-up, ask that instead of proposing.\n"
            "When enough required fields are known, propose the workflow with stated assumptions instead of continuing optional clarification.\n"
            "When uploaded files are used, preserve source file ids in the scratchpad and bind only prepared knowledge files to the final workflow.\n"
            "Only commit the workflow after explicit user confirmation."
        )
    else:
        text = (
            "Prompt mode: task_chat.\n"
            "This is an interactive task discussion.\n"
            "Use only the minimum retrieved context needed to stay coherent and useful.\n"
            "Answer directly before branching into optional extra work."
        )
    return SystemMessage(content=text)


def build_retrieved_context_message(*, title: str, body: str) -> SystemMessage | None:
    safe_title = str(title or "").strip()
    safe_body = str(body or "").strip()
    if not safe_title or not safe_body:
        return None
    return SystemMessage(content=f"{safe_title}\n{safe_body}")
