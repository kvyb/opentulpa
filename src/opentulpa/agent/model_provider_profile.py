"""Provider-specific model behavior used by runtime model calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelProviderProfile:
    model_name: str
    cache_strategy: str
    supports_top_level_cache: bool = False
    supports_cache_breakpoints: bool = False
    openrouter_session_sticky: bool = False
    stream_chunk_timeout_env_names: tuple[str, ...] = (
        "OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS",
        "OPENTULPA_MODEL_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS",
    )
    stream_chunk_timeout_default_seconds: float = 25.0

    def cache_control_payload(self, *, ttl_1h: bool) -> dict[str, Any]:
        if not (self.supports_top_level_cache or self.supports_cache_breakpoints):
            return {}
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if ttl_1h:
            cache_control["ttl"] = "1h"
        return cache_control

    def prompt_cache_profile(self, *, enabled: bool, ttl_1h: bool) -> dict[str, Any]:
        if not enabled:
            return {
                "enabled": False,
                "strategy": "disabled",
                "supports_top_level": False,
                "supports_breakpoints": False,
                "cache_control": {},
                "model_name": self.model_name,
            }
        return {
            "enabled": True,
            "strategy": self.cache_strategy,
            "supports_top_level": self.supports_top_level_cache,
            "supports_breakpoints": self.supports_cache_breakpoints,
            "cache_control": self.cache_control_payload(ttl_1h=ttl_1h),
            "model_name": self.model_name,
        }

    def stream_chunk_timeout_seconds(self) -> float:
        default = float(self.stream_chunk_timeout_default_seconds)
        raw = str(default)
        for env_name in self.stream_chunk_timeout_env_names:
            value = str(os.getenv(env_name, "") or "").strip()
            if value:
                raw = value
                break
        try:
            return max(0.05, min(180.0, float(raw)))
        except ValueError:
            return default


def model_provider_profile(model_name: str | None) -> ModelProviderProfile:
    safe_model_name = str(model_name or "").strip()
    slug = safe_model_name.lower()
    if "anthropic/" in slug or "claude" in slug:
        return ModelProviderProfile(
            model_name=safe_model_name,
            cache_strategy="top_level",
            supports_top_level_cache=True,
            supports_cache_breakpoints=True,
        )
    if "gemini" in slug or slug.startswith("google/"):
        return ModelProviderProfile(
            model_name=safe_model_name,
            cache_strategy="breakpoint",
            supports_cache_breakpoints=True,
        )
    if slug.startswith("qwen/") or "qwen" in slug:
        return ModelProviderProfile(
            model_name=safe_model_name,
            cache_strategy="implicit_stable_prefix",
            openrouter_session_sticky=True,
        )
    if slug.startswith("minimax/") or "minimax" in slug:
        return ModelProviderProfile(
            model_name=safe_model_name,
            cache_strategy="implicit_stable_prefix",
            openrouter_session_sticky=True,
            stream_chunk_timeout_env_names=(
                "OPENTULPA_MINIMAX_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS",
                "OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS",
                "OPENTULPA_MODEL_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS",
            ),
            stream_chunk_timeout_default_seconds=75.0,
        )
    if slug.startswith("z-ai/") or slug.startswith("zai/") or "glm" in slug:
        return ModelProviderProfile(
            model_name=safe_model_name,
            cache_strategy="automatic",
            stream_chunk_timeout_env_names=(
                "OPENTULPA_ZAI_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS",
                "OPENTULPA_MODEL_STREAM_CHUNK_TIMEOUT_SECONDS",
                "OPENTULPA_MODEL_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS",
            ),
            stream_chunk_timeout_default_seconds=75.0,
        )
    automatic_markers = (
        "openai/",
        "gpt-",
        "o1",
        "o3",
        "o4",
        "deepseek",
        "grok",
        "x-ai/",
        "moonshot",
        "kimi",
        "groq/",
    )
    if any(marker in slug for marker in automatic_markers):
        return ModelProviderProfile(model_name=safe_model_name, cache_strategy="automatic")
    return ModelProviderProfile(model_name=safe_model_name, cache_strategy="unknown")


def provider_prompt_cache_profile(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    return model_provider_profile(model_name).prompt_cache_profile(
        enabled=enabled,
        ttl_1h=ttl_1h,
    )


def provider_prompt_cache_invoke_extras(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    profile = provider_prompt_cache_profile(
        enabled=enabled,
        model_name=model_name,
        ttl_1h=ttl_1h,
    )
    if profile.get("strategy") != "top_level":
        return {}
    return {"extra_body": {"cache_control": dict(profile.get("cache_control") or {})}}
