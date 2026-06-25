"""Runtime configuration for cobaiter.

Settings are loaded from environment variables (prefix ``COBAITER_``) and an
optional ``.env`` file. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COBAITER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Downstream LiteLLM gateway ---
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = ""

    # --- Valkey (mapping table / conversation state) ---
    valkey_url: str = "redis://localhost:6379/0"

    # --- Model registry ---
    # Path to an externally-managed registry file (YAML/JSON) that maps each model
    # to its routing capabilities + policy (tier / fallback_chain / context window).
    # Empty = use the built-in default seed. The file is the source of truth: on
    # startup the registry is reconciled to exactly match it.
    models_config: str = ""

    # --- Routing ---
    # Virtual model name the agents call. Requests addressed to this model are routed.
    virtual_model: str = "cobaiter-auto"
    # Lightweight model used to score candidate models when more than one remains.
    classifier_model: str = "claude-haiku-4-5"
    # Max completion tokens for a classifier call. Must be large enough for
    # "thinking" classifier models to finish reasoning AND emit the JSON verdict
    # (a too-small value truncates before the JSON, forcing the heuristic fallback).
    classifier_max_tokens: int = 2048
    # Safe fallback when no candidate satisfies the constraints.
    default_model: str = "claude-haiku-4-5"

    # --- Conversation stickiness / hysteresis ---
    conv_ttl_seconds: int = 60 * 60 * 24 * 7  # 1 week
    # Minimum turns to stay on a model before a *soft* (quality-driven) switch is allowed.
    min_dwell_turns: int = 3
    # Required score advantage of the best candidate over the pinned model to switch.
    switch_margin: float = 0.15
    # Re-run the classifier for soft re-evaluation at most every N turns.
    soft_recheck_every: int = 4
    # EMA smoothing factor applied to classifier scores (0..1; higher = more reactive).
    score_ema_alpha: float = 0.5

    # --- Credit / availability ---
    # A model whose remaining credit headroom (USD, from LiteLLM budget/spend) drops
    # below this floor is treated as unavailable and filtered out.
    credit_floor: float = 0.0
    # Cache TTL (seconds) for LiteLLM budget/spend lookups.
    credit_cache_ttl: int = 30

    # --- HTTP server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- HTTP client ---
    request_timeout: float = 600.0


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
