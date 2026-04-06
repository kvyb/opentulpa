from __future__ import annotations

from opentulpa.core.config import Settings, get_openai_compatible_api_key_from_env


def test_settings_accepts_primary_openai_compatible_api_key_name() -> None:
    settings = Settings(OPENAI_COMPATIBLE_API_KEY="primary-key")
    assert settings.openai_compatible_api_key == "primary-key"
    assert settings.openrouter_api_key == "primary-key"


def test_settings_accepts_legacy_openrouter_api_key_alias() -> None:
    settings = Settings(OPENROUTER_API_KEY="legacy-key")
    assert settings.openai_compatible_api_key == "legacy-key"
    assert settings.openrouter_api_key == "legacy-key"


def test_env_helper_prefers_primary_name(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "primary-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")
    assert get_openai_compatible_api_key_from_env() == "primary-key"


def test_env_helper_falls_back_to_legacy_name(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")
    assert get_openai_compatible_api_key_from_env() == "legacy-key"
