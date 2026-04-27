from __future__ import annotations

from pathlib import Path

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


def test_settings_accepts_primary_openai_compatible_base_url_name() -> None:
    settings = Settings(OPENAI_COMPATIBLE_BASE_URL="https://example.com/v1")
    assert settings.openrouter_base_url == "https://example.com/v1"
    assert settings.openai_compatible_base_url == "https://example.com/v1"


def test_settings_accepts_legacy_openrouter_base_url_alias() -> None:
    settings = Settings(OPENROUTER_BASE_URL="https://legacy.example/v1")
    assert settings.openrouter_base_url == "https://legacy.example/v1"


def test_settings_accepts_primary_openai_compatible_embedding_model_name() -> None:
    settings = Settings(OPENAI_COMPATIBLE_EMBEDDING_MODEL="text-embedding-x")
    assert settings.openrouter_embedding_model == "text-embedding-x"
    assert settings.openai_compatible_embedding_model == "text-embedding-x"


def test_settings_accepts_legacy_openrouter_embedding_model_alias() -> None:
    settings = Settings(OPENROUTER_EMBEDDING_MODEL="legacy-embedding-model")
    assert settings.openrouter_embedding_model == "legacy-embedding-model"


def test_settings_accepts_primary_multimodal_llm_name() -> None:
    settings = Settings(MULTIMODAL_LLM="google/gemini-3-flash-preview")
    assert settings.multimodal_llm == "google/gemini-3-flash-preview"


def test_settings_accepts_legacy_telegram_media_model_alias() -> None:
    settings = Settings(TELEGRAM_MEDIA_MODEL="google/gemini-3-flash-preview")
    assert settings.multimodal_llm == "google/gemini-3-flash-preview"


def test_settings_default_agent_models_use_kimi(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("WAKE_EXECUTION_MODEL", raising=False)

    settings = Settings()

    assert settings.llm_model == "moonshotai/kimi-k2.6"
    assert settings.wake_execution_model == "moonshotai/kimi-k2.6"


def test_settings_loads_runtime_defaults_from_yaml(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text(
        "llm_model: from-yaml\nagent_recursion_limit: 42\n"
        "openai_compatible_base_url: https://yaml.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.llm_model == "from-yaml"
    assert settings.agent_recursion_limit == 42
    assert settings.openai_compatible_base_url == "https://yaml.example/v1"


def test_dotenv_overrides_yaml_runtime_defaults(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text("llm_model: from-yaml\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=str(env_file))

    assert settings.llm_model == "from-dotenv"


def test_settings_discovers_yaml_by_walking_parent_directories(
    monkeypatch, tmp_path: Path
) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text("llm_model: from-parent\n", encoding="utf-8")
    nested_dir = tmp_path / "nested" / "deeper"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    settings = Settings()

    assert settings.llm_model == "from-parent"
