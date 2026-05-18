"""Configuration from environment + YAML defaults."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"
LEGACY_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
PRIMARY_OPENAI_COMPATIBLE_BASE_URL_ENV = "OPENAI_COMPATIBLE_BASE_URL"
LEGACY_OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
PRIMARY_OPENAI_COMPATIBLE_EMBEDDING_MODEL_ENV = "OPENAI_COMPATIBLE_EMBEDDING_MODEL"
LEGACY_OPENROUTER_EMBEDDING_MODEL_ENV = "OPENROUTER_EMBEDDING_MODEL"
DEFAULT_CONFIG_FILENAME = "opentulpa.config.yaml"


def get_openai_compatible_api_key_from_env() -> str | None:
    value = (
        os.environ.get(PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV)
        or os.environ.get(LEGACY_OPENROUTER_API_KEY_ENV)
        or ""
    )
    text = str(value).strip()
    return text or None


class Settings(BaseSettings):
    """App settings from env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load defaults from YAML, but allow env/.env overrides."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlRuntimeDefaultsSource(settings_cls),
            file_secret_settings,
        )

    # Host
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="FastAPI port")
    agent_checkpoint_db_path: str = Field(
        default=".opentulpa/langgraph_checkpoints.sqlite",
        description="SQLite path for LangGraph thread checkpoints.",
    )
    agent_recursion_limit: int = Field(
        default=80,
        ge=5,
        le=200,
        description="Maximum LangGraph steps per turn.",
    )
    agent_max_completion_tokens: int = Field(
        default=4096,
        ge=128,
        le=32768,
        description="Maximum model completion tokens per agent turn.",
    )
    agent_max_user_reply_chars: int = Field(
        default=4000,
        ge=500,
        le=20000,
        description="Hard cap for any single user-visible assistant reply before truncation.",
    )
    agent_context_token_limit: int = Field(
        default=12000,
        ge=10000,
        le=1000000,
        description="Short-term high-watermark (estimated tokens) before thread context compaction.",
    )
    agent_context_recent_tokens: int = Field(
        default=3500,
        ge=1000,
        le=1000000,
        description="Short-term low-watermark target (estimated tokens) after compaction.",
    )
    agent_context_rollup_tokens: int = Field(
        default=2200,
        ge=500,
        le=300000,
        description="Estimated token budget for compressed older-context rollup.",
    )
    agent_context_compaction_source_tokens: int = Field(
        default=100000,
        ge=1000,
        le=500000,
        description="Max oldest-token span folded into rollup in one compaction pass.",
    )
    link_alias_db_path: str = Field(
        default=".opentulpa/link_aliases.db",
        description="SQLite path for customer-scoped long-link alias registry.",
    )
    # Telegram
    telegram_bot_token: str | None = Field(default=None, description="Telegram bot token")
    telegram_allowed_usernames: str | None = Field(
        default=None,
        description="Optional CSV allowlist of Telegram usernames (without @).",
    )
    telegram_allowed_user_ids: str | None = Field(
        default=None,
        description="Optional CSV allowlist of Telegram numeric user IDs.",
    )
    telegram_support_user_ids: str | None = Field(
        default=None,
        description="Optional CSV allowlist of trusted Telegram support operator numeric user IDs.",
    )
    telegram_support_usernames: str | None = Field(
        default=None,
        description="Optional CSV allowlist of trusted Telegram support operator usernames.",
    )

    telegram_webhook_secret: str | None = Field(
        default=None,
        description="Optional secret for webhook path",
    )
    opentulpa_owner_customer_id: str | None = Field(
        default=None,
        description=(
            "Optional canonical owner customer id for generic-first deployments. "
            "When set to a non-telegram id, an allowed Telegram username can bootstrap "
            "a numeric Telegram id binding on first message."
        ),
    )
    opentulpa_web_token: str | None = Field(
        default=None,
        description="Bearer token required for dashboard web operations against this deployment.",
    )

    # Memory (mem0)
    mem0_user_id: str = Field(default="default", description="Default user id for mem0")
    mem0_qdrant_path: str = Field(
        default=".opentulpa/qdrant",
        description="Local path for embedded Qdrant vector store used by mem0.",
    )
    mem0_qdrant_on_disk: bool = Field(
        default=True,
        description="Persist Qdrant vectors on disk (recommended true for durability).",
    )

    # LLM: single OpenAI-compatible model backend used for agent calls and mem0.
    openai_compatible_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV,
            LEGACY_OPENROUTER_API_KEY_ENV,
        ),
        description=(
            "API key for the configured OpenAI-compatible model endpoint "
            f"(loaded from {PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV} in env/.env; "
            f"{LEGACY_OPENROUTER_API_KEY_ENV} is accepted as a backward-compatible alias)."
        ),
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices(
            "openai_compatible_base_url",
            "openrouter_base_url",
            PRIMARY_OPENAI_COMPATIBLE_BASE_URL_ENV,
            LEGACY_OPENROUTER_BASE_URL_ENV,
        ),
        description=(
            "Base URL for the configured OpenAI-compatible model endpoint. "
            "Defaults to OpenRouter. "
            f"{PRIMARY_OPENAI_COMPATIBLE_BASE_URL_ENV} is the preferred env name; "
            f"{LEGACY_OPENROUTER_BASE_URL_ENV} is accepted as a backward-compatible alias."
        ),
    )
    llm_model: str = Field(
        default="z-ai/glm-5.1",
        description=(
            "Model identifier accepted by the configured provider. "
            "Recommended default is the OpenRouter slug z-ai/glm-5.1 for main chat turns."
        ),
    )
    memory_llm_model: str = Field(
        default="google/gemini-3-flash-preview",
        description=(
            "Model used by mem0 for background memory extraction. "
            "Defaults to google/gemini-3-flash-preview so chat can use a different main model "
            "without making memory inference expensive or brittle."
        ),
    )
    llm_reasoning_effort: str | None = Field(
        default="medium",
        description=(
            "Optional reasoning effort for providers/models that support it "
            "(for example: low, medium, high). Defaults to medium for agent-owned "
            "LLM calls; set empty/null to avoid sending reasoning_effort."
        ),
    )
    wake_classifier_model: str | None = Field(
        default=None,
        description=(
            "Optional cheaper model for wake/heartbeat notify decisions. If unset, uses LLM_MODEL."
        ),
    )
    wake_execution_model: str | None = Field(
        default="z-ai/glm-5.1",
        description=(
            "Model used for wake/routine execution turns that need stronger reasoning "
            "and tool use. Recommended default aligns this with the main chat model: z-ai/glm-5.1."
        ),
    )
    multimodal_llm: str = Field(
        default="google/gemini-3.1-flash-lite-preview",
        validation_alias=AliasChoices(
            "MULTIMODAL_LLM",
            "TELEGRAM_MEDIA_MODEL",
        ),
        description=(
            "Model used for multimodal understanding of non-text inputs such as Telegram "
            "attachments, browser screenshots, voice notes, and audio/video files before "
            "passing text summaries into the main chat model. "
            "Recommended default is google/gemini-3.1-flash-lite-preview."
        ),
    )
    workflow_setup_input_classifier_model: str = Field(
        default="z-ai/glm-5.1",
        description=(
            "Model used to classify messages sent while workflow setup is already running "
            "as status nudges versus real setup edits."
        ),
    )
    business_knowledge_oracle_model: str = Field(
        default="google/gemini-3.1-flash-lite-preview",
        description=(
            "Model used by the workflow business knowledge oracle for source-grounded "
            "answers over normalized uploaded files."
        ),
    )
    proactive_heartbeat_default_hours: int = Field(
        default=3,
        ge=1,
        le=24,
        description="Default heartbeat interval (hours) when proactive mode auto-enables.",
    )
    agent_behavior_log_enabled: bool = Field(
        default=True,
        description="Enable structured JSONL behavior logging for agent execution flow.",
    )
    agent_behavior_log_path: str = Field(
        default=".opentulpa/logs/agent_behavior.jsonl",
        description="Path for structured JSONL behavior logs.",
    )
    openrouter_embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        validation_alias=AliasChoices(
            "openai_compatible_embedding_model",
            "openrouter_embedding_model",
            PRIMARY_OPENAI_COMPATIBLE_EMBEDDING_MODEL_ENV,
            LEGACY_OPENROUTER_EMBEDDING_MODEL_ENV,
        ),
        description=(
            "Embedding model identifier for mem0 via the configured "
            "OpenAI-compatible embeddings endpoint. "
            f"{PRIMARY_OPENAI_COMPATIBLE_EMBEDDING_MODEL_ENV} is the preferred env name; "
            f"{LEGACY_OPENROUTER_EMBEDDING_MODEL_ENV} is accepted as a backward-compatible alias."
        ),
    )
    browser_use_headless: bool = Field(
        default=True,
        description="Run local Browser Use sessions in headless mode by default.",
    )
    browser_use_model: str | None = Field(
        default=None,
        description=(
            "Optional Browser Use model override. If unset, Browser Use reuses "
            "MULTIMODAL_LLM so browser steps keep a multimodal-capable model."
        ),
    )
    browser_use_max_concurrent_tasks: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum concurrent local Browser Use tasks.",
    )
    browser_use_task_retention_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        description="How long completed local Browser Use task records remain queryable in memory.",
    )
    browser_use_user_data_dir: str = Field(
        default=".opentulpa/browser_use_profiles",
        description=(
            "Directory for persistent Browser Use profile storage. Each Browser Use "
            "session_id gets its own subdirectory here so cookies/localStorage can "
            "survive process restarts when local/container storage persists."
        ),
    )
    browser_use_api_key: str | None = Field(
        default=None,
        description=(
            "Optional Browser Use Cloud API key. When set, OpenTulpa drives a hosted "
            "Browser Use Cloud browser session via CDP, with a per-owner cloud profile "
            "for cookies."
        ),
    )
    browser_use_cloud_proxy_country_code: str | None = Field(
        default="us",
        description="Optional Browser Use Cloud proxy country code for hosted browser sessions.",
    )
    browser_use_cloud_timeout_minutes: int = Field(
        default=15,
        ge=1,
        le=240,
        description="Browser Use Cloud hosted browser session timeout in minutes.",
    )
    capsolver_api_key: str | None = Field(
        default=None,
        description=(
            "Optional CapSolver API key. When set, local Browser Use tasks get an explicit "
            "CAPTCHA-solving action for supported reCAPTCHA v2 and Cloudflare Turnstile pages."
        ),
    )
    composio_api_key: str | None = Field(
        default=None,
        description="Composio API key used for Tool Router sessions and auth flows.",
    )
    composio_default_callback_url: str | None = Field(
        default=None,
        description=(
            "Optional override callback URL used when starting Composio auth flows. "
            "If unset, OpenTulpa derives it automatically from the public base URL."
        ),
    )
    langfuse_public_key: str | None = Field(
        default=None,
        description="Optional Langfuse public key. Langfuse stays disabled unless public key, secret key, and base URL are set.",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Optional Langfuse secret key. Langfuse stays disabled unless public key, secret key, and base URL are set.",
    )
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
        description="Langfuse base URL. Defaults to https://us.cloud.langfuse.com.",
    )
    langfuse_deployment_tag: str | None = Field(
        default=None,
        description="Optional deployment tag added to Langfuse trace metadata and tags.",
    )
    langfuse_environment: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_TRACING_ENVIRONMENT", "LANGFUSE_ENVIRONMENT"),
        description=(
            "Optional Langfuse tracing environment override. If unset, OpenTulpa derives it "
            "from the deployment tag or Railway service/environment metadata."
        ),
    )
    langfuse_content_level: str = Field(
        default="full_debug",
        description="Langfuse capture mode for OpenTulpa payloads. Defaults to full_debug with redaction.",
    )
    agent_prompt_caching_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "AGENT_PROMPT_CACHING_ENABLED",
            "AGENT_ANTHROPIC_PROMPT_CACHING",
        ),
        description=(
            "When True, enable provider-specific request prompt caching when supported by "
            "the current model/provider. Unsupported models silently no-op. "
            "See https://openrouter.ai/docs/guides/best-practices/prompt-caching"
        ),
    )
    agent_prompt_cache_ttl_1h: bool = Field(
        default=False,
        description=(
            "When prompt caching is enabled, request 1-hour cache TTL instead of the "
            "default 5-minute TTL where supported (higher cache write cost, better for "
            "long sessions)."
        ),
    )

    # The OPENROUTER_* env names are kept for compatibility even when pointing at
    # another OpenAI-compatible endpoint.

    @property
    def openrouter_api_key(self) -> str | None:
        """Backward-compatible alias for older callers."""
        return self.openai_compatible_api_key

    @property
    def openai_compatible_base_url(self) -> str:
        """Preferred neutral provider naming for base URL."""
        return self.openrouter_base_url

    @property
    def openai_compatible_embedding_model(self) -> str:
        """Preferred neutral provider naming for embedding model."""
        return self.openrouter_embedding_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


class _YamlRuntimeDefaultsSource(PydanticBaseSettingsSource):
    """Optional repository-level YAML defaults source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._delegate = self._build_delegate(settings_cls)

    def _candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def _add_path(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(path)

        for base in [Path.cwd(), Path(__file__).resolve().parents[3]]:
            _add_path(base / DEFAULT_CONFIG_FILENAME)
            for parent in base.parents:
                _add_path(parent / DEFAULT_CONFIG_FILENAME)

        return candidates

    def _build_delegate(
        self, settings_cls: type[BaseSettings]
    ) -> PydanticBaseSettingsSource | None:
        for candidate in self._candidate_paths():
            if candidate.exists():
                return YamlConfigSettingsSource(settings_cls, yaml_file=candidate)
        return None

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        if self._delegate is None:
            return None, field_name, False
        return self._delegate.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        if self._delegate is None:
            return {}
        return self._delegate()
