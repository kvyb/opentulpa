from __future__ import annotations

from pathlib import Path

from opentulpa.skills.service import SkillStoreService, build_skill_markdown


def _mk_service(tmp_path: Path) -> SkillStoreService:
    return SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )


def test_skill_store_default_skill_and_user_override(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    store.ensure_default_skill()

    all_global = store.list_skills(customer_id="user_1", include_global=True)
    names = {s["name"] for s in all_global}
    assert "skill-creator" in names
    assert "browser-use-operator" in names
    assert "composio-operator" in names
    assert "routine-schedule-composer" in names

    global_md = build_skill_markdown(
        name="weather-report",
        description="Generate weather summaries.",
        instructions="Always return concise weather summaries.",
    )
    store.upsert_skill(
        scope="global",
        customer_id="",
        name="weather-report",
        skill_markdown=global_md,
        source="test",
        enabled=True,
    )
    user_md = build_skill_markdown(
        name="weather-report",
        description="Generate weather summaries with humidity and wind.",
        instructions="Include humidity and wind in all weather answers.",
    )
    store.upsert_skill(
        scope="user",
        customer_id="user_1",
        name="weather-report",
        skill_markdown=user_md,
        source="test",
        enabled=True,
    )

    listed = store.list_skills(customer_id="user_1", include_global=True)
    weather = next(s for s in listed if s["name"] == "weather-report")
    assert weather["scope"] == "user"
    assert "humidity" in weather["description"]


def test_skill_store_supporting_files_roundtrip(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    md = build_skill_markdown(
        name="csv-parser",
        description="Parse CSV with custom formatting.",
        instructions="Use delimiter detection and normalize headers.",
    )
    store.upsert_skill(
        scope="user",
        customer_id="user_2",
        name="csv-parser",
        skill_markdown=md,
        source="test",
        enabled=True,
        supporting_files={
            "references/rules.md": "# Rules\n\n- Normalize headers\n",
            "scripts/transform.py": "def run():\n    return 'ok'\n",
        },
    )
    fetched = store.get_skill(customer_id="user_2", name="csv-parser", include_files=True)
    assert fetched is not None
    files = fetched.get("supporting_files", {})
    assert "references/rules.md" in files
    assert "scripts/transform.py" in files


def test_skill_creator_default_upgrades_legacy_bootstrap_copy(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    legacy_md = build_skill_markdown(
        name="skill-creator",
        description=(
            "Use this skill when the user asks for recurring behavior/capabilities so the "
            "assistant can create or update reusable skills."
        ),
        instructions=(
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
        ),
    )
    store.upsert_skill(
        scope="global",
        customer_id="",
        name="skill-creator",
        skill_markdown=legacy_md,
        source="system_bootstrap",
        enabled=True,
    )

    store.ensure_default_skill()

    skill = store.get_skill(customer_id="", name="skill-creator", include_files=False)
    assert skill is not None
    assert "## Authoring rules" in skill["skill_markdown"]
    assert "references/`, `scripts/`, or `assets/`" in skill["skill_markdown"]


def test_browser_use_operator_default_mentions_session_reuse(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    store.ensure_default_skill()

    skill = store.get_skill(customer_id="", name="browser-use-operator", include_files=False)
    assert skill is not None
    assert "browser_use_session_list" in skill["skill_markdown"]
    assert "browser_use_task_screenshot" in skill["skill_markdown"]


def test_composio_operator_default_mentions_auth_and_schema_flow(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    store.ensure_default_skill()

    skill = store.get_skill(customer_id="", name="composio-operator", include_files=False)
    assert skill is not None
    assert "composio_authorize_toolkit" in skill["skill_markdown"]
    assert "composio_wait_for_connection" in skill["skill_markdown"]
    assert "composio_tool_search" in skill["skill_markdown"]
    assert "composio_tool_schema" in skill["skill_markdown"]
    assert "composio_tool_execute" in skill["skill_markdown"]
    assert "redirect_url" in skill["skill_markdown"]


def test_list_skills_auto_heals_stale_rows(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    md = build_skill_markdown(
        name="ghost-skill",
        description="Temporary skill.",
        instructions="Do a thing.",
    )
    store.upsert_skill(
        scope="user",
        customer_id="user_1",
        name="ghost-skill",
        skill_markdown=md,
        source="test",
        enabled=True,
    )
    skill_dir = tmp_path / "skills" / "users" / "user_1" / "ghost-skill"
    assert skill_dir.exists()
    for path in sorted(skill_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    skill_dir.rmdir()

    listed = store.list_skills(customer_id="user_1", include_global=False)
    assert all(item["name"] != "ghost-skill" for item in listed)
    assert store.get_skill(customer_id="user_1", name="ghost-skill", include_files=False) is None


def test_get_skill_auto_heals_stale_row(tmp_path: Path) -> None:
    store = _mk_service(tmp_path)
    md = build_skill_markdown(
        name="ghost-fetch",
        description="Temporary skill.",
        instructions="Do a thing.",
    )
    store.upsert_skill(
        scope="user",
        customer_id="user_2",
        name="ghost-fetch",
        skill_markdown=md,
        source="test",
        enabled=True,
    )
    skill_md = tmp_path / "skills" / "users" / "user_2" / "ghost-fetch" / "SKILL.md"
    assert skill_md.exists()
    skill_md.unlink()

    assert store.get_skill(customer_id="user_2", name="ghost-fetch", include_files=False) is None
    listed = store.list_skills(customer_id="user_2", include_global=False)
    assert all(item["name"] != "ghost-fetch" for item in listed)
