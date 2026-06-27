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
    # Max completion tokens for a classifier call. The classifier emits a tiny JSON
    # verdict ({"d":<float>,"r":[<float>,...]}), so this only needs slack for that.
    # Keep reasoning DISABLED on the classifier model (e.g. enable_thinking:false):
    # a "thinking" model spends tokens (and seconds) reasoning before the JSON, which
    # both slows the call and can truncate the verdict.
    classifier_max_tokens: int = 512
    # Max characters of recent conversation shown to the classifier. The classifier
    # only needs the latest request to judge difficulty + domain, so this is kept
    # small: the conversation digest is the dominant contributor to classifier INPUT
    # tokens (prefill), and a large value makes every classifier call slower for no
    # routing gain.
    classifier_digest_chars: int = 800
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

    # --- Cost / tier aware selection ---
    # The classifier returns a use-case *relevance* (0..1) plus one task *difficulty*.
    # The router first folds difficulty + tier into a capability-fit (penalising only
    # UNDER-powered models), then re-ranks deterministically with two PENALTIES:
    #     effective = suitability - cost_bias*(cost/maxCost) - tier_bias*(tier/maxTier)
    # Both favour the cheapest, *lightest* model that is still suitable. "High tier
    # wins on hard tasks" is already handled by capability-fit, so tier here is a
    # penalty, NOT a bonus: its job is to avoid over-provisioning on easy tasks (do
    # not pick a heavyweight when a lighter model is equally suitable). ``cost_bias``
    # should dominate ``tier_bias`` ("decide on cost, then weight").
    cost_bias: float = 0.4
    tier_bias: float = 0.1
    # Capability-fit normalises a candidate's tier against the MAX tier — but only
    # among candidates that are actually in contention, i.e. whose relevance is at
    # least this fraction of the top relevance. Out-of-domain models (relevance ~0)
    # are excluded so an unrelated heavyweight (e.g. a tier-6 coding model on a non-
    # coding task) cannot inflate maxTier and deflate every in-domain model's fit
    # (which would push the no-think -> think boundary far too low).
    capability_rel_fraction: float = 0.5

    # --- Credit / availability ---
    # A model whose remaining credit headroom (USD, from LiteLLM budget/spend) drops
    # below this floor is treated as unavailable and filtered out.
    credit_floor: float = 0.0
    # Cache TTL (seconds) for LiteLLM budget/spend lookups.
    credit_cache_ttl: int = 30

    # --- Logging ---
    # Logger level for the "cobaiter" logger (DEBUG/INFO/WARNING/ERROR).
    log_level: str = "DEBUG"

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
