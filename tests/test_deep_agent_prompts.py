from opentulpa.deep_agent.prompts import OWNER_PROMPT


def test_owner_prompt_uses_authenticated_chat_for_secret_ingress() -> None:
    assert "ask the owner to paste the credential in their next message" in OWNER_PROMPT
    assert "never send them to a separate\nhost UI, CLI" in OWNER_PROMPT
    assert "`secret://<handle_id>`" in OWNER_PROMPT
    assert "use its handle ID in capability tools" in OWNER_PROMPT
    assert "earlier conversation message" in OWNER_PROMPT
    assert "is obsolete and must be corrected" in OWNER_PROMPT
