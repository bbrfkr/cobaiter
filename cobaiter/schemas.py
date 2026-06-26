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
    # shown to the classifier, which judges use-case *relevance* only — the tier vs
    # difficulty maths stays in code so the score is calibrated and reproducible.
    tier: int = 1
    # Free-text description of the model's USE-CASE DOMAIN only (e.g. software
    # development vs non-coding general use). Surfaced to the classifier, which
    # scores TOPIC relevance from it — so it must NOT contain difficulty/capability,
    # cost, speed, or input-length wording; difficulty is a separate per-task axis
    # matched against ``tier`` in the router. Models sharing a domain should share
    # the same description and be distinguished by ``tier``/``cost``.
    description: str = ""
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


class ClassifierResult(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)
    # Task difficulty estimate in 0..1 for the whole conversation (capability-fit
    # input). ``None`` when the classifier gave no usable estimate (parse failure /
    # heuristic fallback), in which case the router skips the capability-fit step.
    difficulty: float | None = None

    def best(self) -> CandidateScore | None:
        return max(self.scores, key=lambda s: s.score) if self.scores else None

    def score_for(self, model: str) -> float | None:
        for s in self.scores:
            if s.model == model:
                return s.score
        return None


# Availability classification of a downstream error.
AvailabilityError = Literal[
    "context_length", "rate_limit", "quota", "overloaded", "none"
]
