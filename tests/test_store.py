"""Store: decision log persistence (Valkey stream) for offline recalibration."""

from __future__ import annotations

from cobaiter.schemas import ClassifierDiagnostics, DecisionLogEntry


def _entry(turn: int, *, task_text: str = "hi") -> DecisionLogEntry:
    return DecisionLogEntry(
        ts=1.0,
        conversation_key="conv-1",
        turn=turn,
        route="classifier-select",
        chosen_model="m-a",
        best_model="m-a",
        difficulty=0.4,
        diagnostics=ClassifierDiagnostics(
            task_text=task_text,
            candidate_sims={"m-a": 0.9, "m-b": 0.3},
            candidate_refs={"m-a": ["fix this bug"], "m-b": ["plan a trip"]},
            sim_easy=0.2,
            sim_hard=0.6,
        ),
    )


async def test_log_decision_round_trips_via_read_decisions(store):
    await store.log_decision(_entry(1))
    await store.log_decision(_entry(2))

    entries = await store.read_decisions(count=10)

    assert [e.turn for e in entries] == [1, 2]  # oldest first
    assert entries[0].conversation_key == "conv-1"
    assert entries[0].chosen_model == "m-a"
    assert entries[0].diagnostics is not None
    assert entries[0].diagnostics.task_text == "hi"
    assert entries[0].diagnostics.candidate_sims == {"m-a": 0.9, "m-b": 0.3}
    assert entries[0].diagnostics.candidate_refs == {
        "m-a": ["fix this bug"], "m-b": ["plan a trip"],
    }
    assert entries[0].diagnostics.sim_easy == 0.2
    assert entries[0].diagnostics.sim_hard == 0.6


async def test_read_decisions_returns_most_recent_count(store):
    for i in range(5):
        await store.log_decision(_entry(i))

    entries = await store.read_decisions(count=2)

    # Most recent 2, oldest-first within that window.
    assert [e.turn for e in entries] == [3, 4]


async def test_read_decisions_empty_stream_returns_empty_list(store):
    assert await store.read_decisions(count=10) == []


async def test_log_decision_trims_to_maxlen(store, settings):
    settings.decision_log_maxlen = 3
    for i in range(20):
        await store.log_decision(_entry(i))

    entries = await store.read_decisions(count=1000)

    # XADD MAXLEN ~ trims approximately, not exactly — just assert it stayed
    # small instead of growing unbounded to 20.
    assert len(entries) <= 20
    assert len(entries) < 20
