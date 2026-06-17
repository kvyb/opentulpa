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


def test_settings_default_agent_models_use_glm52(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("WAKE_EXECUTION_MODEL", raising=False)
    monkeypatch.delenv("BUSINESS_KNOWLEDGE_ORACLE_MODEL", raising=False)

    settings = Settings()

    assert settings.llm_model == "z-ai/glm-5.2"
    assert settings.wake_execution_model == "z-ai/glm-5.2"
    assert settings.workflow_setup_input_classifier_model == "z-ai/glm-5.2"
    assert settings.business_knowledge_oracle_model == "google/gemini-3.1-flash-lite-preview"


def test_settings_accepts_business_knowledge_oracle_model_env(monkeypatch) -> None:
    monkeypatch.setenv("BUSINESS_KNOWLEDGE_ORACLE_MODEL", "provider/oracle-model")

    settings = Settings()

    assert settings.business_knowledge_oracle_model == "provider/oracle-model"


def test_settings_accepts_capsolver_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("CAPSOLVER_API_KEY", "cap-key")

    settings = Settings()

    assert settings.capsolver_api_key == "cap-key"


def test_settings_accepts_browser_use_user_data_dir_env(monkeypatch) -> None:
    monkeypatch.setenv("BROWSER_USE_USER_DATA_DIR", "/tmp/opentulpa-browser-profiles")

    settings = Settings()

    assert settings.browser_use_user_data_dir == "/tmp/opentulpa-browser-profiles"


def test_settings_accepts_langfuse_base_url_or_host_alias(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_DEPLOYMENT_TAG", "carwash-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "carwash-test")

    settings = Settings()

    assert settings.langfuse_public_key == "pk"
    assert settings.langfuse_secret_key == "sk"
    assert settings.langfuse_base_url == "https://us.cloud.langfuse.com"
    assert settings.langfuse_deployment_tag == "carwash-test"
    assert settings.langfuse_environment == "carwash-test"


def test_settings_accepts_langfuse_environment_alias(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("LANGFUSE_ENVIRONMENT", "staging")

    settings = Settings()

    assert settings.langfuse_environment == "staging"


def test_settings_defaults_langfuse_base_url_to_us_cloud(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    settings = Settings()

    assert settings.langfuse_base_url == "https://us.cloud.langfuse.com"


def test_settings_loads_runtime_defaults_from_yaml(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text(
        "llm_model: from-yaml\nagent_recursion_limit: 42\n"
        "openai_compatible_base_url: https://yaml.example/v1\n"
        "business_knowledge_oracle_model: oracle-from-yaml\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.llm_model == "from-yaml"
    assert settings.agent_recursion_limit == 42
    assert settings.openai_compatible_base_url == "https://yaml.example/v1"
    assert settings.business_knowledge_oracle_model == "oracle-from-yaml"


def test_settings_accepts_agent_recursion_limit_250() -> None:
    settings = Settings(agent_recursion_limit=250)

    assert settings.agent_recursion_limit == 250


def test_dotenv_overrides_yaml_runtime_defaults(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text("llm_model: from-yaml\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=str(env_file))

    assert settings.llm_model == "from-dotenv"


def test_settings_discovers_yaml_by_walking_parent_directories(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "opentulpa.config.yaml"
    config_file.write_text("llm_model: from-parent\n", encoding="utf-8")
    nested_dir = tmp_path / "nested" / "deeper"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    settings = Settings()

    assert settings.llm_model == "from-parent"
