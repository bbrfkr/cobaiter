"""Hysteresis-gated soft re-routing (context-switch) and per-instruction locking.

Note on turns: a "turn" is a genuine *user* message, not a downstream API call.
Multi-turn conversations are simulated by re-sending with an incremented user
count via ``convo_req``; mid-instruction agentic round-trips (no new user
message) via ``agentic_followup_req``.
"""

from __future__ import annotations

from cobaiter.schemas import Route

from conftest import agentic_followup_req, convo_req


async def test_no_switch_within_dwell(engine, classifier):
    classifier.table = {"claude-haiku-4-5": 0.9, "claude-opus-4-8": 0.5}
    d0 = await engine.decide(convo_req(1, last="hi"), header_id="c1")
    start = d0.model
    # Make opus look great immediately, but we are within dwell.
    classifier.table = {"claude-haiku-4-5": 0.4, "claude-opus-4-8": 0.99}
    d1 = await engine.decide(convo_req(2, last="hi"), header_id="c1")
    d2 = await engine.decide(convo_req(3, last="hi"), header_id="c1")
    assert d1.route is Route.PINNED and d1.model == start
    assert d2.route is Route.PINNED and d2.model == start


async def test_soft_switch_after_dwell_with_trigger(engine, classifier):
    classifier.table = {"claude-haiku-4-5": 0.9, "claude-opus-4-8": 0.5}
    first = await engine.decide(convo_req(1, last="hi"), header_id="c1")
    assert first.model == "claude-haiku-4-5"
    # Burn dwell turns with identical content (no trigger fires).
    await engine.decide(convo_req(2, last="hi"), header_id="c1")
    await engine.decide(convo_req(3, last="hi"), header_id="c1")
    # Now opus is clearly better AND a cheap trigger fires (new code block).
    classifier.table = {"claude-haiku-4-5": 0.5, "claude-opus-4-8": 0.95}
    d = await engine.decide(
        convo_req(4, last="refactor ```py\nx=1\n```"), header_id="c1"
    )
    assert d.route is Route.CONTEXT_SWITCH
    assert d.model == "claude-opus-4-8"


async def test_margin_blocks_marginal_switch(engine, classifier):
    classifier.table = {"claude-haiku-4-5": 0.9, "claude-opus-4-8": 0.5}
    await engine.decide(convo_req(1, last="hi"), header_id="c1")
    await engine.decide(convo_req(2, last="hi"), header_id="c1")
    await engine.decide(convo_req(3, last="hi"), header_id="c1")
    # opus only slightly better than pinned (ema ~0.9) -> below switch_margin.
    classifier.table = {"claude-haiku-4-5": 0.7, "claude-opus-4-8": 0.75}
    d = await engine.decide(
        convo_req(4, last="now a ```py\ny=2\n``` block"), header_id="c1"
    )
    assert d.route is Route.PINNED
    assert d.model == "claude-haiku-4-5"


async def test_periodic_recheck_triggers_classifier(engine, classifier, settings):
    classifier.table = {"claude-haiku-4-5": 0.9, "claude-opus-4-8": 0.5}
    await engine.decide(convo_req(1, last="hi"), header_id="c1")  # classifier-select
    calls = classifier.calls
    # Identical content; classifier should only re-run on the periodic boundary.
    for t in range(settings.soft_recheck_every):
        await engine.decide(convo_req(2 + t, last="hi"), header_id="c1")
    assert classifier.calls > calls


# --------------------------------------------------------------------------- #
# A single user instruction must not switch models across its agentic loop.
# --------------------------------------------------------------------------- #
async def test_agentic_roundtrip_stays_pinned_despite_better_candidate(
    engine, classifier
):
    """The crux: within ONE user instruction, tool round-trips never re-route,
    even past the dwell window and even when the classifier now strongly prefers
    another model and a cheap trigger (new code block) would otherwise fire."""
    classifier.table = {"claude-haiku-4-5": 0.9, "claude-opus-4-8": 0.5}
    first = await engine.decide(convo_req(1, last="solve this"), header_id="c1")
    assert first.model == "claude-haiku-4-5"

    # Make opus look dramatically better; if the loop counted as turns this would
    # blow past dwell and switch. It must NOT: no new user message arrived.
    classifier.table = {"claude-haiku-4-5": 0.1, "claude-opus-4-8": 0.99}
    for _ in range(6):
        d = await engine.decide(
            agentic_followup_req(1, tool_payload="result ```py\nx=1\n```"),
            header_id="c1",
        )
        assert d.route is Route.PINNED
        assert d.model == "claude-haiku-4-5"
    # The classifier was never consulted during the loop.
    assert classifier.calls == 1
    # The turn counter advanced only for the single real user turn.
    state = await engine._store.get_conversation("id:c1")
    assert state.turn == 1


async def test_forced_failover_still_works_mid_instruction(engine, classifier, client):
    """A mid-instruction round-trip may still failover when the pinned model
    becomes unavailable — staying put is not an option there."""
    classifier.table = {"claude-opus-4-8": 0.95}
    d = await engine.decide(convo_req(1, last="hard reasoning"), header_id="c1")
    assert d.model == "claude-opus-4-8"
    # Opus runs out of credit; a tool round-trip (no new user turn) must still move.
    client.credit["claude-opus-4-8"] = -1.0
    d2 = await engine.decide(agentic_followup_req(1), header_id="c1")
    assert d2.route is Route.FAILOVER
    assert d2.model == "claude-sonnet-4-6"
