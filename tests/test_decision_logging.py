"""RouteEngine persists a DecisionLogEntry for classifier-driven routes only,
redacting task text for privacy-flagged conversations. See cobaiter.calibrate
for the offline job that consumes this log."""

from __future__ import annotations

from cobaiter.schemas import ClassifierDiagnostics, ModelSpec, Route

from conftest import convo_req, user_req


def _diagnostics(task_text: str = "write a poem") -> ClassifierDiagnostics:
    return ClassifierDiagnostics(
        task_text=task_text,
        candidate_sims={"claude-opus-4-8": 0.8, "claude-sonnet-4-6": 0.3},
        sim_easy=0.2,
        sim_hard=0.6,
    )


async def test_classifier_select_logs_a_decision(engine, classifier):
    classifier.table = {"claude-opus-4-8": 0.9, "claude-sonnet-4-6": 0.7}
    classifier.raw = _diagnostics()

    d = await engine.decide(user_req("write a poem"), header_id="c1")
    assert d.route is Route.CLASSIFIER_SELECT

    entries = await engine._store.read_decisions(count=10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.route == "classifier-select"
    assert entry.chosen_model == "claude-opus-4-8"
    assert entry.best_model == "claude-opus-4-8"
    assert entry.conversation_key == d.conversation_key
    assert entry.diagnostics is not None
    assert entry.diagnostics.task_text == "write a poem"
    assert entry.diagnostics.candidate_sims == {
        "claude-opus-4-8": 0.8, "claude-sonnet-4-6": 0.3,
    }


async def test_rule_route_does_not_log(engine, classifier):
    """Single eligible candidate -> RULE, no classifier call, nothing to log."""
    req = user_req("secret stuff", metadata={"privacy": True})
    d = await engine.decide(req, header_id="c1")
    assert d.route is Route.RULE
    assert await engine._store.read_decisions(count=10) == []


async def test_pinned_route_does_not_log(engine, classifier):
    classifier.table = {"claude-opus-4-8": 0.9}
    classifier.raw = _diagnostics()
    await engine.decide(user_req("q1"), header_id="c1")
    # Sticky continuation -> PINNED, classifier not re-consulted.
    await engine.decide(user_req("q2"), header_id="c1")

    entries = await engine._store.read_decisions(count=10)
    # Only the initial CLASSIFIER_SELECT decision was logged.
    assert len(entries) == 1
    assert entries[0].route == "classifier-select"


async def test_decision_log_disabled_setting_suppresses_logging(engine, classifier, settings):
    settings.decision_log_enabled = False
    classifier.table = {"claude-opus-4-8": 0.9}
    classifier.raw = _diagnostics()

    await engine.decide(user_req("write a poem"), header_id="c1")

    assert await engine._store.read_decisions(count=10) == []


async def test_no_raw_diagnostics_means_nothing_logged(engine, classifier):
    """Mirrors an embedding-call failure: classifier.raw stays None (see
    EmbeddingClassifier.score), so there is no signal worth persisting."""
    classifier.table = {"claude-opus-4-8": 0.9}
    classifier.raw = None

    await engine.decide(user_req("write a poem"), header_id="c1")

    assert await engine._store.read_decisions(count=10) == []


async def test_privacy_conversation_redacts_task_text(engine, classifier):
    """needs_local exists specifically to keep sensitive text off of anything but
    the local model — the decision log must honour that even though it only
    persists to Valkey, not a third party."""
    local_a = ModelSpec(model="local-a", tier=1, is_local=True, description="general A")
    local_b = ModelSpec(model="local-b", tier=1, is_local=True, description="general B")
    await engine._store.put_model(local_a)
    await engine._store.put_model(local_b)
    classifier.table = {"local-a": 0.9, "local-b": 0.5}
    classifier.raw = _diagnostics("this is sensitive task text")

    req = user_req("sensitive stuff", metadata={"privacy": True})
    d = await engine.decide(req, header_id="c1")
    assert d.route is Route.CLASSIFIER_SELECT  # 2 local candidates -> classifier ran

    entries = await engine._store.read_decisions(count=10)
    assert len(entries) == 1
    assert entries[0].diagnostics is not None
    assert entries[0].diagnostics.task_text is None  # redacted
    # Non-text signal (needed for calibration's OLS fit) is still logged.
    assert entries[0].diagnostics.sim_easy == 0.2
    assert entries[0].diagnostics.sim_hard == 0.6


async def test_context_switch_logs_a_decision(engine, classifier):
    classifier.table = {"claude-haiku-4-5": 0.9}
    classifier.raw = _diagnostics("hi")
    await engine.decide(convo_req(1, last="hi"), header_id="c1")
    await engine.decide(convo_req(2, last="hi"), header_id="c1")
    await engine.decide(convo_req(3, last="hi"), header_id="c1")

    classifier.table = {"claude-haiku-4-5": 0.5, "qwen2.5": 0.95}
    classifier.raw = _diagnostics("refactor this")
    d = await engine.decide(
        convo_req(4, last="refactor ```py\nx=1\n```"), header_id="c1"
    )
    assert d.route is Route.CONTEXT_SWITCH
    assert d.model == "qwen2.5"

    entries = await engine._store.read_decisions(count=10)
    switch_entries = [e for e in entries if e.route == "context-switch"]
    assert len(switch_entries) == 1
    assert switch_entries[0].chosen_model == "qwen2.5"
    assert switch_entries[0].diagnostics is not None
    assert switch_entries[0].diagnostics.task_text == "refactor this"
