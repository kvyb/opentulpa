"""Configuration from environment + YAML defaults."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PRIMARY_OPENAI_COMPATIBLE_API_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"
LEGACY_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
PRIMARY_OPENAI_COMPATIBLE_BASE_URL_ENV = "OPENAI_COMPATIBLE_BASE_URL"
LEGACY_OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
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
        env_ignore_empty=True,
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
    deepagents_checkpoint_db_path: str = Field(
        default=".opentulpa/deepagents/checkpoints.db",
        description="Fresh Deep Agents checkpoint database; legacy chats are not imported.",
    )
    deepagents_store_db_path: str = Field(
        default=".opentulpa/deepagents/store.db",
        description="Tenant-namespaced native memory and skills store.",
    )
    deepagents_runs_db_path: str = Field(
        default=".opentulpa/deepagents/runs.db",
        description="Durable agent run and approval state.",
    )
    deepagents_workspaces_root: str = Field(
        default=".opentulpa/deepagents/workspaces",
        description="Persistent tenant workspace roots mounted into disposable sandboxes.",
    )
    evolution_enabled: bool = Field(
        default=True,
        description=(
            "Enable source editing and self-release through the immutable managed bootstrap."
        ),
    )
    evolution_source_repository: str | None = Field(
        default=None,
        description=(
            "Canonical Git checkout used by the managed bootstrap to create disposable "
            "source candidates. Defaults to the bootstrap project root."
        ),
    )
    evolution_sandbox_image: str = Field(
        default="opentulpa-evolution:0.1.0",
        description="Locally built, reviewed OCI image used by the main agent's source shell.",
    )
    evolution_evaluator_image: str = Field(
        default="opentulpa-evolution:0.1.0",
        description="Locally built OCI image containing trusted candidate evaluation dependencies.",
    )
    intake_drafts_db_path: str = Field(
        default=".opentulpa/deepagents/intake_drafts.db",
        description="Revisioned intake workflow drafts migrated from setup sessions.",
    )
    agent_max_completion_tokens: int = Field(
        default=4096,
        ge=128,
        le=32768,
        description="Maximum model completion tokens per agent turn.",
    )
    sandbox_container_cli: str = Field(
        default="docker",
        validation_alias=AliasChoices(
            "sandbox_container_cli",
            "SANDBOX_CONTAINER_CLI",
            "OPENTULPA_CONTAINER_CLI",
        ),
    )
    sandbox_image: str = Field(
        default="opentulpa-tenant-sandbox:0.1.0",
        description=(
            "Reviewed local tenant workspace image. Direct mode resolves this tag to its "
            "immutable local image ID; managed mode resolves and owns it in the stable host."
        ),
    )
    sandbox_cpu_limit: str = Field(default="1")
    sandbox_memory_limit: str = Field(default="512m")
    sandbox_pid_limit: int = Field(default=128, ge=16, le=4096)
    sandbox_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    sandbox_max_output_bytes: int = Field(default=512_000, ge=1024, le=10_000_000)
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

    # LLM: one OpenRouter-compatible model used by Deep Agents.
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
        default="moonshotai/kimi-k3",
        description=(
            "Model identifier accepted by the configured provider. "
            "Recommended default is the OpenRouter slug moonshotai/kimi-k3 for main chat turns."
        ),
    )
    llm_fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "z-ai/glm-5.2",
            "google/gemini-3.1-pro-preview",
        ],
        validation_alias=AliasChoices(
            "llm_fallback_models",
            "LLM_FALLBACK_MODELS",
            "llm_provider_rejection_fallback_model",
            "LLM_PROVIDER_REJECTION_FALLBACK_MODEL",
        ),
        description=(
            "Ordered fallback models for provider failures during a model call. "
            "Environment values may be a JSON array or comma-separated model identifiers."
        ),
    )

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def validate_llm_fallback_models(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError("LLM_FALLBACK_MODELS must be valid JSON or CSV") from exc
            else:
                value = raw.split(",")
        if not isinstance(value, list | tuple):
            raise ValueError("LLM_FALLBACK_MODELS must be a list")
        models: list[str] = []
        for raw_model in value:
            model = str(raw_model or "").strip()
            if not model:
                continue
            if len(model) > 300 or any(ord(char) < 32 for char in model):
                raise ValueError("fallback model identifier is invalid")
            if model not in models:
                models.append(model)
        if len(models) > 8:
            raise ValueError("at most 8 fallback models may be configured")
        return models

    llm_provider_order: Annotated[dict[str, list[str]], NoDecode] = Field(
        default_factory=lambda: {
            "z-ai/glm-5.2": ["z-ai/fp8", "fireworks", "deepinfra/fp4"],
        },
        validation_alias=AliasChoices("llm_provider_order", "LLM_PROVIDER_ORDER"),
        description=(
            "Optional ordered OpenRouter provider slugs per model. Only the listed providers "
            "are used for that model before OpenTulpa advances to the next model."
        ),
    )

    @field_validator("llm_provider_order", mode="before")
    @classmethod
    def validate_llm_provider_order(cls, value: Any) -> dict[str, list[str]]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("LLM_PROVIDER_ORDER must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("LLM_PROVIDER_ORDER must be an object")
        result: dict[str, list[str]] = {}
        for raw_model, raw_providers in value.items():
            model = str(raw_model or "").strip()
            if not model or len(model) > 300 or any(ord(char) < 32 for char in model):
                raise ValueError("provider-order model identifier is invalid")
            if not isinstance(raw_providers, list | tuple):
                raise ValueError("provider order must be a list")
            providers: list[str] = []
            for raw_provider in raw_providers:
                provider = str(raw_provider or "").strip()
                if not provider or len(provider) > 100 or any(ord(char) < 32 for char in provider):
                    raise ValueError("provider-order provider identifier is invalid")
                if provider not in providers:
                    providers.append(provider)
            if len(providers) > 8:
                raise ValueError("at most 8 providers may be configured per model")
            if providers:
                result[model] = providers
        return result

    model_aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Trusted AgentSpec model aliases mapped to provider model identifiers. "
            "Environment values use a JSON object."
        ),
    )

    @field_validator("model_aliases")
    @classmethod
    def validate_model_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for raw_alias, raw_model in value.items():
            alias = str(raw_alias or "").strip()
            model = str(raw_model or "").strip()
            if (
                not alias
                or len(alias) > 100
                or not alias[0].isalnum()
                or any(
                    char
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for char in alias
                )
            ):
                raise ValueError("model alias is invalid")
            if not model or len(model) > 300 or any(ord(char) < 32 for char in model):
                raise ValueError("model identifier is invalid")
            aliases[alias] = model
        return aliases
    llm_reasoning_effort: str | None = Field(
        default="medium",
        description=(
            "Optional reasoning effort for providers/models that support it "
            "(for example: low, medium, high). Defaults to medium for agent-owned "
            "LLM calls; set empty/null to avoid sending reasoning_effort."
        ),
    )
    business_knowledge_oracle_model: str = Field(
        default="google/gemini-3.1-flash-lite-preview",
        description=(
            "Model used by the workflow business knowledge oracle for source-grounded "
            "answers over normalized uploaded files."
        ),
    )
    browser_use_user_data_dir: str = Field(
        default=".opentulpa/browser_use_profiles",
        description=(
            "Directory for tenant-scoped Browser Use Cloud profile metadata. Browser "
            "cookies remain in the vendor profile, not on the OpenTulpa host."
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
