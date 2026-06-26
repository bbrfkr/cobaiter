"""RouteEngine — the per-turn unified routing brain.

Combines, in one pass each turn:
* sticky reuse of the per-conversation pinned model,
* immediate failover on hard-constraint violation / unavailability / credit exhaustion,
* hysteresis-gated *soft* re-routing when the context materially changes.

See the implementation plan for the staged design. The engine only *decides*; the
app layer performs the downstream call and may report a failover back in (see
``failover_to``) when a request fails mid-flight.
"""

from __future__ import annotations

import logging

from .classifier import Classifier
from .config import Settings
from .features import (
    conversation_key,
    count_code_blocks,
    count_user_messages,
    extract_constraints,
)
from .litellm_client import LiteLLMClient
from .schemas import (
    CandidateScore,
    ChatCompletionRequest,
    ClassifierResult,
    Constraints,
    ConversationState,
    ModelSpec,
    Route,
    RouteDecision,
)
from .store import Store

log = logging.getLogger("cobaiter")

# Output headroom required on top of the prompt estimate.
_OUTPUT_MARGIN = 1024
_TOKEN_BAND = 4000  # bucket size for the stage-1 "size jumped" trigger


class RouteEngine:
    def __init__(
        self,
        store: Store,
        client: LiteLLMClient,
        classifier: Classifier,
        settings: Settings,
    ) -> None:
        self._store = store
        self._client = client
        self._classifier = classifier
        self._s = settings

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def decide(
        self,
        req: ChatCompletionRequest,
        *,
        header_id: str | None = None,
        privacy_header: str | None = None,
    ) -> RouteDecision:
        key = conversation_key(req, header_id)
        constraints = extract_constraints(req, privacy_header=privacy_header)
        eligible = await self._eligible_models(constraints)

        state = await self._store.get_conversation(key)
        user_msgs = count_user_messages(req.messages)
        if state is None:
            decision = await self._initial(key, req, constraints, eligible)
            decision.state.user_msgs = user_msgs
        elif user_msgs > state.user_msgs:
            # Genuine new user turn: advance the turn counter and re-evaluate.
            state.turn += 1
            state.user_msgs = user_msgs
            decision = await self._continue(key, req, constraints, eligible, state)
        else:
            # Mid-instruction agentic round-trip (tool call / retry): keep the
            # conversation pinned so one user instruction never switches models.
            # Only a forced failover may still move (see ``_continue_locked``).
            decision = await self._continue_locked(key, constraints, eligible, state)

        await self._store.set_conversation(key, decision.state)
        score = decision.state.score_ema
        log.info(
            "decision: conv=%s model=%s route=%s score=%s",
            key, decision.model, decision.route.value,
            f"{score:.3f}" if score is not None else "n/a",
        )
        return decision

    async def failover_to(
        self,
        key: str,
        constraints: Constraints,
    ) -> RouteDecision | None:
        """Advance a conversation to its next available model after a live failure.

        Called by the app layer when a downstream request raised an availability
        error. Returns None if the conversation is unknown.
        """
        state = await self._store.get_conversation(key)
        if state is None:
            return None
        self._client.invalidate_credit(state.model)
        eligible = await self._eligible_models(constraints, exclude={state.model})
        decision = await self._do_failover(key, state, eligible, failed=state.model)
        await self._store.set_conversation(key, decision.state)
        log.info(
            "decision: conv=%s model=%s route=%s score=n/a",
            key, decision.model, decision.route.value,
        )
        return decision

    # ------------------------------------------------------------------ #
    # New conversation
    # ------------------------------------------------------------------ #
    async def _initial(
        self,
        key: str,
        req: ChatCompletionRequest,
        constraints: Constraints,
        eligible: dict[str, ModelSpec],
    ) -> RouteDecision:
        candidates = list(eligible.values())

        if not candidates:
            state = self._fresh_state(self._s.default_model, req, constraints)
            return RouteDecision(
                model=self._s.default_model, route=Route.DEFAULT,
                conversation_key=key, state=state,
            )

        if len(candidates) == 1:
            spec = candidates[0]
            state = self._fresh_state(spec.model, req, constraints)
            return RouteDecision(
                model=spec.model, route=Route.RULE,
                conversation_key=key, state=state,
            )

        result = await self._rank(req, candidates)
        best = self._select_best(result, candidates)
        chosen = next((c for c in candidates if c.model == best.model), candidates[0])
        log.debug(
            "rank(initial): conv=%s effective=%s selected=%s",
            key, _fmt_scores(result), chosen.model,
        )
        state = self._fresh_state(chosen.model, req, constraints)
        state.score_ema = best.score
        state.last_recheck_turn = state.turn
        return RouteDecision(
            model=chosen.model, route=Route.CLASSIFIER_SELECT,
            conversation_key=key, state=state,
        )

    # ------------------------------------------------------------------ #
    # Continuing conversation
    # ------------------------------------------------------------------ #
    async def _continue(
        self,
        key: str,
        req: ChatCompletionRequest,
        constraints: Constraints,
        eligible: dict[str, ModelSpec],
        state: ConversationState,
    ) -> RouteDecision:
        pinned_spec = eligible.get(state.model)

        # (a) pinned model no longer valid/available -> immediate failover.
        if pinned_spec is None:
            ex = {state.model}
            return await self._do_failover(
                key, state, await self._eligible_models(constraints, exclude=ex),
                failed=state.model,
            )

        # (b) pinned still valid: decide whether a soft re-evaluation is warranted.
        if self._should_recheck(state, req, constraints):
            return await self._soft_reeval(key, req, constraints, eligible, state)

        # (c) stay pinned.
        self._refresh_signals(state, req, constraints)
        return RouteDecision(
            model=state.model, route=Route.PINNED, conversation_key=key, state=state,
        )

    # ------------------------------------------------------------------ #
    # Mid-instruction round-trip (no new user turn)
    # ------------------------------------------------------------------ #
    async def _continue_locked(
        self,
        key: str,
        constraints: Constraints,
        eligible: dict[str, ModelSpec],
        state: ConversationState,
    ) -> RouteDecision:
        """Route a request that is *not* a new user turn (agentic tool round-trip
        or a retry of the same instruction).

        No soft re-evaluation and no turn advance: a single user instruction must
        stay on one model for consistency. The only move permitted is a *forced*
        failover when the pinned model is no longer eligible (constraint violation
        / unavailable / out of credit) — staying put there is not an option.

        Signals are intentionally left untouched so the next genuine user turn
        compares against the baseline captured at the last routing decision.
        """
        if eligible.get(state.model) is not None:
            return RouteDecision(
                model=state.model, route=Route.PINNED,
                conversation_key=key, state=state,
            )
        return await self._do_failover(
            key, state,
            await self._eligible_models(constraints, exclude={state.model}),
            failed=state.model,
        )

    # ------------------------------------------------------------------ #
    # Soft re-evaluation (stage 2 + stage 3)
    # ------------------------------------------------------------------ #
    async def _soft_reeval(
        self,
        key: str,
        req: ChatCompletionRequest,
        constraints: Constraints,
        eligible: dict[str, ModelSpec],
        state: ConversationState,
    ) -> RouteDecision:
        candidates = list(eligible.values())
        result = await self._rank(req, candidates)
        state.last_recheck_turn = state.turn
        log.info("classify: conv=%s candidates=%s", key, [c.model for c in candidates])

        best = self._select_best(result, candidates)
        pinned_raw = result.score_for(state.model)
        if best is None or pinned_raw is None:
            self._refresh_signals(state, req, constraints)
            return RouteDecision(
                model=state.model, route=Route.CLASSIFIER_SELECT,
                conversation_key=key, state=state,
            )

        # EMA-smooth the pinned model's score to suppress flapping.
        pinned_ema = _ema(state.score_ema, pinned_raw, self._s.score_ema_alpha)
        state.score_ema = pinned_ema

        margin = best.score - pinned_ema
        switching = best.model != state.model and margin > self._s.switch_margin
        log.debug(
            "rank(recheck): conv=%s effective=%s best=%s pinned=%s(ema=%.3f) "
            "margin=%.3f -> selected=%s",
            key, _fmt_scores(result), best.model, state.model, pinned_ema, margin,
            best.model if switching else state.model,
        )
        if switching:
            spec = eligible[best.model]
            self._switch_state(state, spec, best.score, req, constraints)
            return RouteDecision(
                model=spec.model, route=Route.CONTEXT_SWITCH,
                conversation_key=key, state=state,
            )

        self._refresh_signals(state, req, constraints)
        return RouteDecision(
            model=state.model, route=Route.CLASSIFIER_SELECT,
            conversation_key=key, state=state,
        )

    # ------------------------------------------------------------------ #
    # Failover mechanics
    # ------------------------------------------------------------------ #
    async def _do_failover(
        self,
        key: str,
        state: ConversationState,
        eligible: dict[str, ModelSpec],
        *,
        failed: str,
    ) -> RouteDecision:
        origin = state.chain_origin or failed
        # Fallback chains are static registry metadata; fetch directly so an
        # exhausted/excluded origin still contributes its chain.
        origin_spec = await self._store.get_model(origin)
        target = self._failover_target(origin, origin_spec, eligible, failed)
        if target is None:
            # Last resort: default_model even if marginal; app maps None-eligibility to 503.
            target = self._s.default_model
        state.model = target
        state.last_switch_turn = state.turn
        state.fallback_step += 1
        state.score_ema = None
        return RouteDecision(
            model=target, route=Route.FAILOVER, conversation_key=key, state=state,
        )

    def _failover_target(
        self,
        origin: str,
        origin_spec: ModelSpec | None,
        eligible: dict[str, ModelSpec],
        failed: str,
    ) -> str | None:
        order = [origin]
        # The origin's fallback chain may include models not in ``eligible``; that's fine.
        chain = origin_spec.fallback_chain if origin_spec else []
        order += [m for m in chain if m not in order]
        # Walk the chain starting after the failed model.
        try:
            start = order.index(failed) + 1
        except ValueError:
            start = 0
        for name in order[start:]:
            if name in eligible:
                return name
        if self._s.default_model in eligible:
            return self._s.default_model
        # Nothing in our preferred order is eligible; take any eligible model.
        return next(iter(eligible), None)

    # ------------------------------------------------------------------ #
    # Stage-1 cheap gate
    # ------------------------------------------------------------------ #
    def _should_recheck(
        self,
        state: ConversationState,
        req: ChatCompletionRequest,
        constraints: Constraints,
    ) -> bool:
        # dwell pre-gate: never soft-switch too soon after the last change.
        if state.turn - state.last_switch_turn < self._s.min_dwell_turns:
            return False

        # periodic check
        if state.turn - state.last_recheck_turn >= self._s.soft_recheck_every:
            return True
        # constraint-set change
        if _constraint_sig(constraints) != state.sig_constraints:
            return True
        # new code blocks appeared
        if count_code_blocks(req.messages) > state.sig_code_blocks:
            return True
        # token band jumped
        if _token_band(constraints.estimated_tokens) > state.sig_token_band:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Eligibility (constraint filter + availability + credit)
    # ------------------------------------------------------------------ #
    async def _eligible_models(
        self, constraints: Constraints, *, exclude: set[str] | None = None
    ) -> dict[str, ModelSpec]:
        exclude = exclude or set()
        out: dict[str, ModelSpec] = {}
        for spec in await self._store.list_models():
            if spec.model in exclude:
                continue
            if not _satisfies(spec, constraints):
                continue
            credit = await self._client.credit_remaining(spec.model)
            if credit is not None and credit < self._s.credit_floor:
                continue
            out[spec.model] = spec
        return out

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def _fresh_state(
        self,
        model: str,
        req: ChatCompletionRequest,
        constraints: Constraints,
    ) -> ConversationState:
        state = ConversationState(
            model=model, turn=1, last_switch_turn=1, chain_origin=model,
        )
        self._refresh_signals(state, req, constraints)
        return state

    def _switch_state(
        self,
        state: ConversationState,
        spec: ModelSpec,
        score: float,
        req: ChatCompletionRequest,
        constraints: Constraints,
    ) -> None:
        state.model = spec.model
        state.last_switch_turn = state.turn
        state.chain_origin = spec.model
        state.fallback_step = 0
        state.score_ema = score
        self._refresh_signals(state, req, constraints)

    def _refresh_signals(
        self,
        state: ConversationState,
        req: ChatCompletionRequest,
        constraints: Constraints,
    ) -> None:
        state.sig_code_blocks = count_code_blocks(req.messages)
        state.sig_constraints = _constraint_sig(constraints)
        state.sig_token_band = _token_band(constraints.estimated_tokens)

    # ------------------------------------------------------------------ #
    # Scoring pipeline: classifier -> capability-fit -> cost/tier re-ranking
    # ------------------------------------------------------------------ #
    async def _rank(
        self, req: ChatCompletionRequest, candidates: list[ModelSpec]
    ) -> ClassifierResult:
        """Run the classifier and fold its semantic judgement into effective scores.

        Three stages: the classifier emits per-candidate *relevance* + one task
        *difficulty*; ``_apply_difficulty`` turns those into use-case suitability via
        tier-vs-difficulty capability-fit; ``_adjust_scores`` then applies the
        deterministic cost penalty / tier bonus.
        """
        raw = await self._classifier.score(req, candidates)
        return self._adjust_scores(self._apply_difficulty(raw, candidates), candidates)

    def _apply_difficulty(
        self, result: ClassifierResult, candidates: list[ModelSpec]
    ) -> ClassifierResult:
        """Fold task difficulty + capability tier into per-candidate relevance.

        ``suitability = relevance * capability_fit`` where
        ``capability_fit = 1 - max(0, difficulty - tier/maxTier)``.

        A model is penalised only when *under-powered* for the task (its normalised
        tier falls short of the difficulty); ample capability is never punished.

        ``maxTier`` is taken over the candidates actually IN CONTENTION — those whose
        relevance is at least ``capability_rel_fraction`` of the top relevance — not
        the whole set. Otherwise an out-of-domain heavyweight (e.g. a tier-6 coding
        model on a non-coding task, relevance ~0) would inflate maxTier and deflate
        every in-domain model's fit, pushing the no-think -> think boundary far too
        low (a trivial title-generation task would then upgrade to a think model).

        When the classifier gave no difficulty estimate (parse failure / heuristic
        fallback) relevance passes through unchanged.
        """
        if result.difficulty is None:
            return result
        specs = {c.model: c for c in candidates}
        # Size capability against the in-domain contenders only.
        max_rel = max((s.score for s in result.scores), default=0.0)
        threshold = max_rel * self._s.capability_rel_fraction
        contenders = [
            specs[s.model].tier
            for s in result.scores
            if s.model in specs and s.score >= threshold
        ]
        max_tier = max(contenders, default=0) or 1
        difficulty = result.difficulty
        adjusted: list[CandidateScore] = []
        for s in result.scores:
            spec = specs.get(s.model)
            if spec is None:
                adjusted.append(s)
                continue
            capability_fit = 1.0 - max(0.0, difficulty - spec.tier / max_tier)
            adjusted.append(
                CandidateScore(model=s.model, score=s.score * capability_fit)
            )
        return ClassifierResult(scores=adjusted)

    # ------------------------------------------------------------------ #
    # Cost / tier aware re-ranking of pure suitability scores
    # ------------------------------------------------------------------ #
    def _adjust_scores(
        self, result: ClassifierResult, candidates: list[ModelSpec]
    ) -> ClassifierResult:
        """Fold cost + tier into the classifier's pure suitability scores.

        ``effective = suitability - cost_bias*(cost/maxCost) - tier_bias*(tier/maxTier)``

        Both terms are PENALTIES: cheaper wins, and — all else equal — the *lightest*
        model wins. "Is the model capable enough for a hard task?" is already handled
        upstream by ``_apply_difficulty`` (capability-fit), which penalises only
        under-powered models; this tier penalty does the complementary job of NOT
        over-provisioning on easy tasks (don't pick a heavyweight when a lighter
        model is equally suitable). Costs/tiers are normalised across the current
        candidate set so the biases stay in suitability units. The result drives
        selection AND the hysteresis (EMA / margin) downstream.
        """
        specs = {c.model: c for c in candidates}
        max_cost = max((c.cost for c in candidates), default=0.0) or 1.0
        max_tier = max((c.tier for c in candidates), default=0) or 1
        adjusted: list[CandidateScore] = []
        for s in result.scores:
            spec = specs.get(s.model)
            if spec is None:
                adjusted.append(s)
                continue
            effective = (
                s.score
                - self._s.cost_bias * (spec.cost / max_cost)
                - self._s.tier_bias * (spec.tier / max_tier)
            )
            adjusted.append(CandidateScore(model=s.model, score=effective))
        return ClassifierResult(scores=adjusted)

    def _select_best(
        self, result: ClassifierResult, candidates: list[ModelSpec]
    ) -> CandidateScore | None:
        """Pick the highest effective score, breaking exact ties deterministically
        in favour of local, then cheaper, then lower-tier (lighter) models —
        consistent with the cost/tier penalties: prefer the lightest model that is
        equally suitable."""
        specs = {c.model: c for c in candidates}

        def key(s: CandidateScore) -> tuple:
            spec = specs.get(s.model)
            return (
                s.score,
                1 if (spec and spec.is_local) else 0,
                -(spec.cost if spec else 0.0),
                -(spec.tier if spec else 0),
            )

        return max(result.scores, key=key) if result.scores else None


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _fmt_scores(result: ClassifierResult) -> str:
    """Format effective (cost/tier-adjusted) scores, highest first, for logging."""
    return ", ".join(
        f"{s.model}={s.score:.3f}"
        for s in sorted(result.scores, key=lambda s: s.score, reverse=True)
    )


def _satisfies(spec: ModelSpec, c: Constraints) -> bool:
    if c.needs_multimodal and not spec.multimodal:
        return False
    if c.needs_tools and not spec.supports_tools:
        return False
    if c.needs_local and not spec.is_local:
        return False
    if c.estimated_tokens and spec.context_window <= c.estimated_tokens + _OUTPUT_MARGIN:
        return False
    return True


def _constraint_sig(c: Constraints) -> str:
    return f"{int(c.needs_multimodal)}{int(c.needs_tools)}{int(c.needs_local)}"


def _token_band(tokens: int) -> int:
    return tokens // _TOKEN_BAND


def _ema(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev
