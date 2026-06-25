"""Failover on constraint violation / unavailability / credit exhaustion."""

from __future__ import annotations

from cobaiter.litellm_client import DownstreamError, classify_error
from cobaiter.schemas import Route

from conftest import user_req


async def test_credit_exhaustion_walks_fallback_chain(engine, classifier, client):
    classifier.table = {"claude-opus-4-8": 0.95}
    d = await engine.decide(user_req("hard reasoning"), header_id="c1")
    assert d.model == "claude-opus-4-8"
    # Exhaust opus; next turn must fail over to the chain's next entry (sonnet).
    client.credit["claude-opus-4-8"] = -1.0
    d2 = await engine.decide(user_req("more"), header_id="c1")
    assert d2.route is Route.FAILOVER
    assert d2.model == "claude-sonnet-4-6"


async def test_failover_to_returns_next_model(engine, classifier, client):
    from cobaiter.features import extract_constraints

    classifier.table = {"claude-opus-4-8": 0.95}
    req = user_req("hard")
    d = await engine.decide(req, header_id="c1")
    assert d.model == "claude-opus-4-8"
    constraints = extract_constraints(req)
    nd = await engine.failover_to("id:c1", constraints)
    assert nd is not None
    assert nd.route is Route.FAILOVER
    assert nd.model == "claude-sonnet-4-6"


async def test_failover_to_unknown_conversation_is_none(engine):
    from cobaiter.schemas import Constraints

    assert await engine.failover_to("id:nope", Constraints()) is None


def test_classify_error_categories():
    assert classify_error(429, "rate limit") == "rate_limit"
    assert classify_error(429, "exceeded your quota") == "quota"
    assert classify_error(400, "maximum context length is 8192") == "context_length"
    assert classify_error(503, "overloaded") == "overloaded"
    assert classify_error(400, "invalid field") == "none"


def test_downstream_error_carries_kind():
    exc = DownstreamError(429, "quota exceeded", classify_error(429, "quota exceeded"))
    assert exc.kind == "quota"
    assert exc.status == 429
