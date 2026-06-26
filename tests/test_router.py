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
    # gen-no-think: 1 - (1.0 - 1/3) = 0.333...; gen-think (tier3 == maxTier): 1.0.
    assert out["gen-think"] == 1.0
    assert abs(out["gen-no-think"] - 1 / 3) < 1e-9


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
