from opentulpa.interfaces.telegram.chat_service import _format_agent_error_for_user


def test_format_agent_error_authentication() -> None:
    msg = _format_agent_error_for_user(
        RuntimeError("openai.AuthenticationError: Error code: 401 - User not found.")
    )
    assert "Model authentication failed" in msg
    assert "OPENAI_COMPATIBLE_API_KEY" in msg


def test_format_agent_error_rate_limit() -> None:
    msg = _format_agent_error_for_user(RuntimeError("429 Too Many Requests"))
    assert "rate-limiting" in msg


def test_format_agent_error_generic() -> None:
    msg = _format_agent_error_for_user(RuntimeError("something else"))
    assert msg == "I hit a backend error while generating a reply. Please try again."
