"""Configuration from environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings from env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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
    approvals_db_path: str = Field(
        default=".opentulpa/pending_approvals.db",
        description="SQLite path for external-impact pending approvals.",
    )
    approvals_ttl_minutes: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Default expiration window (minutes) for approval challenges.",
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

    # Slack (for Slack skill: list channels, read history, post)
    slack_bot_token: str | None = Field(
        default=None, description="Slack Bot OAuth token (xoxb-...)"
    )
    telegram_webhook_secret: str | None = Field(
        default=None,
        description="Optional secret for webhook path",
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

    # LLM: used for Parlant (OpenRouter) and mem0 (Gemini). Single model for all agent/LLM calls.
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (loaded from OPENROUTER_API_KEY in env/.env).",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL for OpenAI-compatible clients (mem0).",
    )
    llm_model: str = Field(
        default="google/gemini-3-flash-preview",
        description="OpenRouter model id, including provider slug (for example: google/gemini-3-flash-preview).",
    )
    wake_classifier_model: str | None = Field(
        default=None,
        description=(
            "Optional cheaper model for wake/heartbeat notify decisions. "
            "If unset, uses LLM_MODEL."
        ),
    )
    guardrail_classifier_model: str = Field(
        default="minimax/minimax-m2.5",
        description=(
            "Model used by guardrail intent classification for approval decisions. "
            "Defaults to minimax/minimax-m2.5."
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
        description="Embedding model id for mem0 via OpenRouter embeddings API.",
    )
    browser_use_headless: bool = Field(
        default=True,
        description="Run local Browser Use sessions in headless mode by default.",
    )
    browser_use_model: str | None = Field(
        default=None,
        description=(
            "Optional Browser Use model override. If unset, browser tasks use tool arg llm "
            "(when explicit) or LLM_MODEL."
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

    # OpenRouter: OPENROUTER_API_KEY required. OPENROUTER_MODEL uses llm_model as-is.
    # mem0 is configured via OpenRouter base URL + key (OpenAI-compatible endpoints).

@lru_cache
def get_settings() -> Settings:
    return Settings()
