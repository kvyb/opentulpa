from __future__ import annotations

import json

from opentulpa.intake.workflow_skill import build_intake_workflow_skill


def test_build_intake_workflow_skill_uses_shared_template_for_instagram_and_telegram() -> None:
    base = {
        "workflow_id": "iwf_123",
        "name": "Booking Intake",
        "provider": "composio",
        "source_config": {},
        "intent_description": "Handle customer appointment requests.",
        "required_fields": ["name", "time"],
        "field_guidance": {"time": "Confirm the final appointment time explicitly."},
        "assistant_instructions": "Be concise and do not promise unavailable slots.",
        "business_facts": {
            "prices": {"small_wash": "1000 RUB"},
            "hours": "Daily 10:00-20:00",
        },
        "knowledge_file_ids": [],
        "sink_type": "local_csv",
        "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
    }

    instagram = build_intake_workflow_skill({**base, "channel": "instagram_dm"})
    telegram = build_intake_workflow_skill(
        {
            **base,
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": "bc_123"},
        }
    )

    for skill in (instagram, telegram):
        markdown = str(skill["skill_markdown"])
        assert "## Workflow Goal" in markdown
        assert "## Operating Context" in markdown
        assert "## Execution Strategy" in markdown
        assert "## Save Behavior" in markdown
        assert "## Safety" in markdown
        payload = json.loads(skill["supporting_files"]["workflow.json"])
        assert payload["intent_description"] == "Handle customer appointment requests."
        assert payload["required_fields"] == ["name", "time"]
        assert payload["business_facts"]["prices"]["small_wash"] == "1000 RUB"
        assert "## Owner-Provided Business Facts" in markdown
        assert "1000 RUB" in markdown

    assert "Instagram DMs" in str(instagram["skill_markdown"])
    assert "Telegram Business DMs" in str(telegram["skill_markdown"])
    assert "single durable intake policy" in str(telegram["skill_markdown"])
    assert "cannot be edited in place" in str(telegram["skill_markdown"])
    assert "Do not use the workflow intent as a front-door filter" in str(telegram["skill_markdown"])
    assert "backend can reuse the single connected Telegram Business account automatically" in str(
        telegram["skill_markdown"]
    )

    strict = build_intake_workflow_skill(
        {
            **base,
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": "bc_123", "intent_match_required": True},
        }
    )
    assert "Strictly match only conversations that fit this intent" in str(strict["skill_markdown"])
