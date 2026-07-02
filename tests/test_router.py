"""Initial routing: constraint filter + classifier branch + stickiness."""

from __future__ import annotations

from cobaiter.schemas import (
    CandidateScore,
    ChatCompletionRequest,
    ClassifierResult,
    ModelSpec,
    Route,
)

from conftest import user_req


def _general_pair_plus_coding() -> list[ModelSpec]:
    """A think/no-think general pair plus an unrelated high-tier coding model."""
    return [
        ModelSpec(model="gen-think", tier=3, description="general"),
        ModelSpec(model="gen-no-think", tier=1, description="general"),
        ModelSpec(model="coding", tier=6, description="coding"),
    ]


def test_capability_fit_normalises_within_relevant_candidates(engine):
    """An out-of-domain heavyweight (coding tier 6, relevance 0) must NOT inflate
    maxTier: on an easy general task the no-think model stays un-penalised."""
    candidates = _general_pair_plus_coding()
    # Easy general task: both general models fully relevant, coding irrelevant.
    result = ClassifierResult(
        scores=[
            CandidateScore(model="gen-think", score=1.0),
            CandidateScore(model="gen-no-think", score=1.0),
            CandidateScore(model="coding", score=0.0),
        ],
        difficulty=0.3,
    )
    out = {s.model: s.score for s in engine._apply_difficulty(result, candidates).scores}
    # maxTier is taken over {gen-think:3, gen-no-think:1}, not the coding tier 6,
    # so no-think (tier1 -> 1/3 >= 0.3) keeps capability_fit 1.0.
    assert out["gen-no-think"] == 1.0
    assert out["gen-think"] == 1.0


def test_capability_fit_penalises_underpowered_on_hard_task(engine):
    """A hard task still penalises the no-think model (within-domain maxTier=3)."""
    candidates = _general_pair_plus_coding()
    result = ClassifierResult(
        scores=[
            CandidateScore(model="gen-think", score=1.0),
            CandidateScore(model="gen-no-think", score=1.0),
            CandidateScore(model="coding", score=0.0),
        ],
        difficulty=1.0,
    )
    out = {s.model: s.score for s in engine._apply_difficulty(result, candidates).scores}
    # gen-no-think: 1 - (1.0**curve - 1/3) = 0.333...; gen-think (tier3==maxTier): 1.0.
    # difficulty=1.0 is the curve's fixed point (1**curve == 1), so this boundary
    # value is unaffected by capability_curve.
    assert out["gen-think"] == 1.0
    assert abs(out["gen-no-think"] - 1 / 3) < 1e-9


def _general_trio_with_distant_cloud() -> list[ModelSpec]:
    """A same-DOMAIN no-think/think pair plus a much higher-tier cloud escalation
    target (unlike ``_general_pair_plus_coding``, the tier-6 model here is fully
    relevant, not out-of-domain — it legitimately sets maxTier)."""
    return [
        ModelSpec(model="gen-no-think", tier=1, description="general"),
        ModelSpec(model="gen-think", tier=3, description="general"),
        ModelSpec(model="gen-cloud", tier=6, description="general"),
    ]


def test_capability_curve_keeps_no_think_unpenalised_at_low_difficulty(engine):
    """Regression: a distant in-domain escalation target (tier 6) must not hijack
    the LOCAL no-think(1)/think(3) comparison at low difficulty. With the linear
    (curve=1) formula, maxTier=6 makes even a trivial task (difficulty 0.25) look
    like it needs more than tier 1 (1/6 ~= 0.167 < 0.25). capability_curve=2 delays
    that onset (0.25**2 = 0.0625 < 0.167), leaving no-think fully un-penalised so
    the cost/tier re-ranking's existing lighter-wins preference can pick it."""
    candidates = _general_trio_with_distant_cloud()
    result = ClassifierResult(
        scores=[CandidateScore(model=c.model, score=1.0) for c in candidates],
        difficulty=0.25,
    )
    out = {s.model: s.score for s in engine._apply_difficulty(result, candidates).scores}
    assert out["gen-no-think"] == 1.0
    assert out["gen-think"] == 1.0


def test_capability_curve_still_escalates_to_cloud_on_hard_task(engine):
    """The curve must not weaken escalation for genuinely hard tasks: at
    difficulty=1.0 the exponent is a no-op (1**curve == 1), so the distant cloud
    tier still dominates and the local models are penalised same as before."""
    candidates = _general_trio_with_distant_cloud()
    result = ClassifierResult(
        scores=[CandidateScore(model=c.model, score=1.0) for c in candidates],
        difficulty=1.0,
    )
    out = {s.model: s.score for s in engine._apply_difficulty(result, candidates).scores}
    assert out["gen-cloud"] == 1.0
    assert abs(out["gen-think"] - (1 - (1.0 - 3 / 6))) < 1e-9
    assert abs(out["gen-no-think"] - (1 - (1.0 - 1 / 6))) < 1e-9
    assert out["gen-no-think"] < out["gen-think"] < out["gen-cloud"]


async def test_trivial_greeting_prefers_lightest_local_model(engine, classifier):
    """End-to-end: with a distant cloud escalation target sharing the domain, a
    plain greeting must still route to the lightest (no-think) local model, not
    the mid-tier think model — the original bug report this fix addresses.
    ``difficulty=0.25`` mirrors what the heuristic assigns a short non-meta
    message like "こんにちは" (see ``classifier.EmbeddingClassifier._difficulty``,
    fallback path)."""
    candidates = [
        ModelSpec(model="cloud-general", tier=6, cost=5.0, description="general"),
        ModelSpec(model="local-think", tier=3, cost=0.0, is_local=True, description="general"),
        ModelSpec(model="local-no-think", tier=1, cost=0.0, is_local=True, description="general"),
    ]
    for spec in candidates:
        await engine._store.put_model(spec)
    classifier.table = {c.model: 1.0 for c in candidates}
    classifier.difficulty = 0.25
    d = await engine.decide(user_req("こんにちは"), header_id="greet-1")
    assert d.model == "local-no-think"


def test_cost_penalty_applies_even_at_full_suitability(engine):
    """Regression: a perfect-fit (suitability 1.0) expensive cloud model must still
    pay a cost penalty, so an equally-suitable free local model wins. The old
    ``* (1 - suitability)`` scaling zeroed the penalty at suitability 1.0, letting
    the pricey model always win."""
    candidates = [
        ModelSpec(model="cloud", tier=3, cost=5.0, is_local=False, description="general"),
        ModelSpec(model="local", tier=3, cost=0.0, is_local=True, description="general"),
    ]
    result = ClassifierResult(
        scores=[
            CandidateScore(model="cloud", score=1.0),
            CandidateScore(model="local", score=1.0),
        ],
        difficulty=0.3,
    )
    out = {s.model: s.score for s in engine._adjust_scores(result, candidates).scores}
    assert out["cloud"] < 1.0  # penalised despite the perfect fit
    assert out["local"] > out["cloud"]  # free local wins the tie on cost


def test_high_difficulty_relaxes_cost_penalty(engine):
    """The cost/tier penalty shrinks as difficulty rises, so a premium model keeps
    more of its suitability on hard tasks (where paying for capability is justified)."""
    candidates = [
        ModelSpec(model="cloud", tier=3, cost=5.0, is_local=False, description="general"),
    ]
    easy = engine._adjust_scores(
        ClassifierResult(scores=[CandidateScore(model="cloud", score=1.0)], difficulty=0.2),
        candidates,
    ).scores[0].score
    hard = engine._adjust_scores(
        ClassifierResult(scores=[CandidateScore(model="cloud", score=1.0)], difficulty=0.9),
        candidates,
    ).scores[0].score
    assert hard > easy


async def test_single_candidate_is_deterministic_rule(engine, classifier):
    """Privacy constraint leaves only the local model -> RULE, no classifier call."""
    req = user_req("secret stuff", metadata={"privacy": True})
    d = await engine.decide(req, header_id="c1")
    assert d.route is Route.RULE
    assert d.model == "qwen2.5"
    assert classifier.calls == 0


async def test_multiple_candidates_calls_classifier(engine, classifier):
    classifier.table = {"claude-opus-4-8": 0.9, "claude-sonnet-4-6": 0.7}
    d = await engine.decide(user_req("write a poem"), header_id="c1")
    assert d.route is Route.CLASSIFIER_SELECT
    assert d.model == "claude-opus-4-8"
    assert classifier.calls == 1


async def test_no_candidate_falls_back_to_default(engine, settings):
    """Token estimate beyond every model's window -> DEFAULT."""
    huge = "word " * 300_000
    d = await engine.decide(user_req(huge), header_id="c1")
    assert d.route is Route.DEFAULT
    assert d.model == settings.default_model


async def test_conversation_is_sticky(engine, classifier):
    classifier.table = {"claude-opus-4-8": 0.9}
    first = await engine.decide(user_req("q1"), header_id="c1")
    calls_after_first = classifier.calls
    second = await engine.decide(user_req("q2"), header_id="c1")
    assert second.route is Route.PINNED
    assert second.model == first.model
    # No extra classifier call on the sticky turn.
    assert classifier.calls == calls_after_first


async def test_different_conversations_can_differ(engine, classifier):
    # Unconstrained + opus clearly most suitable -> opus wins despite its cost.
    # A privacy-constrained conversation is forced onto the only local model.
    classifier.table = {"claude-opus-4-8": 0.9, "qwen2.5": 0.2}
    a = await engine.decide(user_req("hard reasoning task"), header_id="a")
    b = await engine.decide(user_req("hello", metadata={"privacy": True}), header_id="b")
    assert a.model == "claude-opus-4-8"
    assert b.model == "qwen2.5"


async def test_multimodal_excludes_non_multimodal(engine):
    req = ChatCompletionRequest(
        model="cobaiter-auto",
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    )
    d = await engine.decide(req, header_id="c1")
    assert d.model != "qwen2.5"  # qwen2.5 is not multimodal in the seed registry
