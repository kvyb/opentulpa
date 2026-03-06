from opentulpa.interfaces.telegram.chat_service import _start_help_text


def test_start_help_text_includes_capabilities_and_onboarding_questions() -> None:
    text = _start_help_text()
    assert "What I can do:" in text
    assert "Web + links" in text
    assert "Interactive browsing" in text
    assert "To personalize quickly, answer these:" in text
    assert "What are you struggling with right now?" in text
    assert "Which repetitive task should I automate first?" in text
    assert "/fresh" in text
