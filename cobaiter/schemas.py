"""Pydantic models shared across cobaiter.

Only the parts of the OpenAI Chat Completions API that cobaiter needs to inspect
are modelled strictly; everything else is preserved verbatim and forwarded
downstream via ``ChatCompletionRequest.raw``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelSpec(BaseModel):
    """Capabilities and routing metadata for one downstream model."""

    model: str
    context_window: int = 8192
    multimodal: bool = False
    supports_tools: bool = False
    is_local: bool = False
    # Relative monetary cost (e.g. USD per 1M tokens). Local/free models = 0.0.
    # Drives the router's deterministic cost-aware re-ranking; NOT shown to the
    # classifier (which scores use-case suitability only).
    cost: float = 0.0
    # Capability tier as an integer: larger = more capable / slower / heavier.
    # Used by the router deterministically: it matches a candidate's tier against
    # the task difficulty (capability fit) and adds a mild capability bonus. NOT
    # part of the relevance scoring, which judges use-case domain only — the tier
    # vs difficulty maths stays in code so the score is calibrated and reproducible.
    tier: int = 1
    # Free-text description of the model's USE-CASE DOMAIN only (e.g. software
    # development vs non-coding general use). Embedded and compared against the
    # conversation digest to score TOPIC relevance — so it must NOT contain
    # difficulty/capability, cost, speed, or input-length wording; difficulty is a
    # separate per-task axis matched against ``tier`` in the router. Models sharing
    # a domain should share the SAME description (they then tie on relevance) and
    # be distinguished by ``tier``/``cost``.
    description: str = ""
    # Short, representative task phrases for this model's use-case domain (same
    # content rules as ``description``: positively stated, domain-only, shared
    # across a domain's tier variants). When non-empty, relevance is scored as
    # the top-2-mean cosine similarity between the task text and EACH example
    # (multi-prototype matching) instead of a single description vector — this
    # damps the per-example embedding brittleness already observed in this
    # codebase (see classifier.py's ``_EDGE_PUNCT`` handling) and avoids letting
    # a domain win purely by having more examples than a rival domain. When
    # empty, relevance falls back to treating ``description`` as a single
    # one-item example list (byte-for-byte the old single-vector behaviour).
    task_examples: list[str] = Field(default_factory=list)
    # Ordered list of alternates to try when this model becomes unavailable.
    fallback_chain: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Conversation state (the "mapping table" entry, persisted in Valkey)
# --------------------------------------------------------------------------- #
class ConversationState(BaseModel):
    """Sticky binding of a conversation to a concrete model."""

    model: str
    # How far down the fallback chain (of the *originally chosen* model) we are.
    fallback_step: int = 0
    # 1-based *user-turn* counter (advances once per genuine new user message, NOT
    # per downstream API call). Dwell / recheck hysteresis is measured against this
    # so a single agentic instruction's tool round-trips never burn the windows.
    turn: int = 0
    # Number of user-role messages seen at the last routing decision. A request
    # whose user-message count exceeds this is a new user turn; an equal count is a
    # mid-instruction agentic round-trip (tool call / retry) and stays pinned.
    user_msgs: int = 0
    # Turn index at which the model last changed (for dwell / hysteresis).
    last_switch_turn: int = 0
    # EMA-smoothed score of the currently pinned model.
    score_ema: float | None = None
    # Origin model that owns the fallback chain currently in use.
    chain_origin: str | None = None
    # --- stage-1 soft-gate signals (cheap change detectors) ---
    sig_code_blocks: int = 0
    sig_constraints: str = ""
    sig_token_band: int = 0
    # Turn index of the last classifier re-evaluation.
    last_recheck_turn: int = 0


# --------------------------------------------------------------------------- #
# Hard constraints derived from a request
# --------------------------------------------------------------------------- #
class Constraints(BaseModel):
    needs_multimodal: bool = False
    needs_tools: bool = False
    needs_local: bool = False
    # Estimated prompt tokens; a model's context_window must exceed this.
    estimated_tokens: int = 0


# --------------------------------------------------------------------------- #
# Routing decision
# --------------------------------------------------------------------------- #
class Route(str, Enum):
    PINNED = "pinned"
    RULE = "rule"  # single candidate after constraint filter (deterministic)
    CLASSIFIER_SELECT = "classifier-select"  # multiple candidates -> classifier chose
    CONTEXT_SWITCH = "context-switch"  # soft re-route past hysteresis gate
    FAILOVER = "failover"  # constraint violation / unavailable / out of credit
    DEFAULT = "default"  # no candidate satisfied constraints -> default_model


class RouteDecision(BaseModel):
    model: str
    route: Route
    conversation_key: str
    state: ConversationState


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions request (loose)
# --------------------------------------------------------------------------- #
class ChatCompletionRequest(BaseModel):
    """Loose model: validate what we read, forward the rest untouched."""

    model_config = {"extra": "allow"}

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    functions: list[dict[str, Any]] | None = None
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None

    def to_downstream(self, target_model: str) -> dict[str, Any]:
        """Return the raw payload with ``model`` rewritten to ``target_model``."""
        payload = self.model_dump(exclude_none=True)
        payload["model"] = target_model
        return payload


# --------------------------------------------------------------------------- #
# Classifier output
# --------------------------------------------------------------------------- #
class CandidateScore(BaseModel):
    model: str
    # Use-case suitability in 0..1. As emitted by the classifier this is pure
    # *relevance* (does the description cover the topic); the router then folds in
    # capability-fit (tier vs difficulty) and cost to derive the effective score.
    score: float


class ClassifierDiagnostics(BaseModel):
    """Raw, pre-adjustment classifier signals — the inputs the tunable anchors
    (``embedding_rel_band``, ``difficulty_easy_anchor``/``hard_anchor``) turn into
    scores. Logged alongside the eventual routing decision so an offline job can
    later re-derive better anchors from real traffic without re-embedding."""

    task_text: str | None = None
    # model -> raw cosine similarity to that model's registry description.
    candidate_sims: dict[str, float] = Field(default_factory=dict)
    # model -> the RESOLVED reference texts actually compared at decision time
    # (``task_examples`` if set, else ``[description]``; see ``ModelSpec``),
    # truncated per-string for log-size safety (see classifier.py's
    # ``_REF_LOG_CHARS``). This is registry metadata, not user conversation
    # content, so — unlike ``task_text`` — it is NEVER redacted for privacy
    # (``needs_local``) conversations. Consumed by ``cobaiter.calibrate`` to
    # group candidates sharing a domain and ask a judge which domain was
    # actually correct, for automatic ``embedding_rel_band`` recalibration.
    candidate_refs: dict[str, list[str]] = Field(default_factory=dict)
    # Max cosine similarity of the task text to the easy/hard difficulty exemplar
    # sets (see classifier.py); ``None`` when the exemplar-based estimate wasn't
    # used (e.g. the low-intent shortcut or token-count fallback fired instead).
    sim_easy: float | None = None
    sim_hard: float | None = None


class ClassifierResult(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)
    # Task difficulty estimate in 0..1 for the whole conversation (capability-fit
    # input). ``None`` when no usable estimate exists, in which case the router
    # skips the capability-fit step and applies the cost/tier penalties in full.
    difficulty: float | None = None
    # Pre-adjustment diagnostics (task text, raw per-candidate cosine similarity,
    # difficulty exemplar similarities). Opaque to the router's scoring pipeline —
    # carried through ``_apply_difficulty``/``_adjust_scores`` unchanged and only
    # consumed by decision logging (see ``DecisionLogEntry``) for later offline
    # recalibration of the anchors/band that turn these raw signals into scores.
    # ``None`` when the embedding call failed (nothing informative to log).
    raw: ClassifierDiagnostics | None = None

    def best(self) -> CandidateScore | None:
        return max(self.scores, key=lambda s: s.score) if self.scores else None

    def score_for(self, model: str) -> float | None:
        for s in self.scores:
            if s.model == model:
                return s.score
        return None


# --------------------------------------------------------------------------- #
# Decision logging (offline recalibration input)
# --------------------------------------------------------------------------- #
class DecisionLogEntry(BaseModel):
    """One classifier-driven routing decision, persisted for later offline
    recalibration of the difficulty/relevance heuristics against real traffic.

    Only emitted for routes where the classifier actually ran (``classifier-
    select`` / ``context-switch``) — ``pinned``/``rule``/``failover``/``default``
    carry no raw signal worth logging. ``task_text`` is omitted (redacted) for
    conversations flagged privacy-sensitive (``needs_local``).
    """

    ts: float
    conversation_key: str
    turn: int
    route: str
    chosen_model: str
    best_model: str | None = None
    difficulty: float | None = None
    diagnostics: ClassifierDiagnostics | None = None


# Availability classification of a downstream error.
AvailabilityError = Literal[
    "context_length", "rate_limit", "quota", "overloaded", "none"
]
