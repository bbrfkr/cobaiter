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

    # Path to an externally-managed file (YAML/JSON) holding the difficulty
    # exemplars — the small ``easy:`` / ``hard:`` task-phrase sets the classifier
    # anchors task difficulty against (see cobaiter.classifier). Like the model
    # registry and the difficulty anchors, these are deployment- and embedding-
    # model-specific TUNING data (the classifier docstring notes they must be
    # re-measured when the embedding model changes), so they belong in config, not
    # code. Empty = use the built-in defaults baked into cobaiter.classifier.
    difficulty_exemplars_config: str = ""

    # --- Routing ---
    # Virtual model name the agents call. Requests addressed to this model are routed.
    virtual_model: str = "cobaiter-auto"
    # Embedding model (served through the LiteLLM gateway's /v1/embeddings) used
    # to score each candidate's use-case relevance: cosine similarity between the
    # conversation digest and the candidate's registry ``description``. Replaces
    # the old synchronous LLM classifier (~1s per decision) with one small
    # embedding call — description vectors are cached in-process, so only the
    # digest is embedded at steady state.
    embedding_model: str = "text-embedding-3-small"
    # Relevance contrast band. Raw cosine similarity sits in a model-dependent
    # compressed range (unrelated texts rarely score near 0), so relevance is
    # anchored on the best candidate: the top similarity maps to 1.0, and a
    # candidate whose similarity falls this far below the top scores 0.0.
    # Smaller = sharper domain separation; larger = softer.
    embedding_rel_band: float = 0.10
    # Max characters of recent conversation folded into the task digest that gets
    # embedded for relevance scoring. Small is enough to capture the topic, and
    # keeps the embedding call fast.
    classifier_digest_chars: int = 400
    # Difficulty is estimated by where the task digest's embedding falls between
    # two small fixed exemplar sets ("easy": greetings/simple lookups, "hard":
    # domain-diverse expert tasks — math proofs, debugging, legal analysis, ...;
    # see classifier.py) rather than a per-domain keyword list, so it
    # generalises to domains no one wrote a keyword for. The raw signal is
    # ``ratio = sim_hard / (sim_hard + sim_easy)`` (max cosine similarity to
    # each set); these two anchors are the ratio values *measured* on the
    # calibration set for the embedding model in use (Qwen3-Embedding-0.6B) —
    # ratio<=easy_anchor maps to difficulty 0.15, ratio>=hard_anchor maps to
    # 0.85, linear in between. Re-measure and adjust both if the embedding
    # model changes (like ``embedding_rel_band``, this is model-dependent).
    difficulty_easy_anchor: float = 0.22
    difficulty_hard_anchor: float = 0.70
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
    # UNDER-powered models), then re-ranks deterministically with two PENALTIES that
    # are themselves relaxed on hard tasks:
    #     effective = suitability
    #               - (cost_bias*(cost/maxCost) + tier_bias*(tier/maxTier)) * (1 - difficulty)
    # Both favour the cheapest, *lightest* model that is still suitable. "High tier
    # wins on hard tasks" is already handled by capability-fit, so tier here is a
    # penalty, NOT a bonus: its job is to avoid over-provisioning on easy tasks (do
    # not pick a heavyweight when a lighter model is equally suitable). The
    # ``(1 - difficulty)`` factor makes difficulty the single knob trading cost for
    # capability: easy tasks pay full cost/tier penalty (cheap light model wins),
    # hard tasks relax it (a clearly-more-capable premium model may win). Scaling by
    # difficulty rather than by each model's own suitability is deliberate — the
    # latter zeroes the penalty for any perfect-fit model, letting an expensive cloud
    # model always beat an equally-suitable free local one. ``cost_bias`` should
    # dominate ``tier_bias`` ("decide on cost, then weight").
    cost_bias: float = 0.4
    tier_bias: float = 0.1
    # Capability-fit normalises a candidate's tier against the MAX tier — but only
    # among candidates that are actually in contention, i.e. whose relevance is at
    # least this fraction of the top relevance. Out-of-domain models (relevance ~0)
    # are excluded so an unrelated heavyweight (e.g. a tier-6 coding model on a non-
    # coding task) cannot inflate maxTier and deflate every in-domain model's fit
    # (which would push the no-think -> think boundary far too low).
    capability_rel_fraction: float = 0.5
    # Exponent applied to ``difficulty`` before the capability-fit comparison:
    # ``capability_fit = 1 - max(0, difficulty**capability_curve - tier/maxTier)``.
    # With curve=1 (linear) a WIDE tier ladder in one domain (e.g. a local
    # no-think/think pair plus a much higher-tier cloud escalation) compresses the
    # "safe" difficulty range for the low-tier model: maxTier is set by the
    # farthest-away escalation target, so even a trivial task can look
    # under-powered relative to it, wrongly favouring the mid-tier model over the
    # lightest one. curve > 1 delays the onset of the capability penalty at
    # low/mid difficulty (difficulty**curve < difficulty) while curve=1 behaviour
    # is preserved at difficulty=1 (1**curve == 1, so full escalation to the top
    # tier for the hardest tasks is unaffected). Once capability_fit no longer
    # over-penalises the lightest sufficient model, ``tier_bias`` below is what
    # decides the local within-domain preference (lighter/faster wins when
    # equally capable) — this is deliberately a single knob, not a second
    # tier-like axis: `tier` keeps its one meaning (capability ceiling), and
    # speed preference among "sufficient" candidates stays entirely in the
    # existing cost/tier re-ranking.
    capability_curve: float = 2.0

    # --- Decision logging (offline recalibration input) ---
    # Persist one DecisionLogEntry (task text + raw classifier signals + the
    # eventual routing decision) to Valkey for every classifier-driven decision
    # (routes ``classifier-select``/``context-switch``). Used offline by
    # ``cobaiter.calibrate`` to re-derive ``difficulty_easy_anchor``/
    # ``hard_anchor``/``embedding_rel_band`` from real traffic instead of the
    # one-off manual calibration set. Logging is best-effort: a failure here
    # never blocks or fails a routing decision.
    decision_log_enabled: bool = True
    # Cap on the Valkey stream length (approximate trim via XADD MAXLEN ~).
    decision_log_maxlen: int = 20_000

    # --- Offline recalibration (cobaiter.calibrate) ---
    # Judge model (routed through the same LiteLLM gateway) used to produce gold
    # difficulty/relevance labels for a sample of logged decisions. Deliberately
    # separate from ``default_model``: judge quality directly determines
    # calibration quality, so it should be set explicitly rather than silently
    # defaulting to a cheap routing fallback.
    calibration_judge_model: str = ""
    # Max number of logged decisions sent to the judge per calibration run.
    calibration_sample_size: int = 200

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
