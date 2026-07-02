"""Offline recalibration: OLS anchor fit + ambiguous-relevance flagging.

Uses a FakeJudgeClient (a stand-in for LiteLLMClient.chat) that returns a
scripted difficulty per call, so the OLS math is exercised against a KNOWN
ground truth relationship instead of a real judge model.
"""

from __future__ import annotations

import json

from cobaiter.calibrate import (
    _find_ambiguous_relevance,
    _invert_fit,
    _judge_difficulty,
    _ols,
    format_report,
    run_calibration,
)
from cobaiter.classifier import difficulty_from_ratio
from cobaiter.config import Settings
from cobaiter.schemas import ClassifierDiagnostics, DecisionLogEntry


class FakeJudgeClient:
    """Returns ``responses[call_index]`` as the judge's raw message content."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def chat(self, payload):
        content = self.responses[self.calls]
        self.calls += 1
        return {"choices": [{"message": {"content": content}}]}


def _entry(ratio_pair: tuple[float, float], task_text: str = "task") -> DecisionLogEntry:
    sim_easy, sim_hard = ratio_pair
    return DecisionLogEntry(
        ts=1.0, conversation_key="c", turn=1, route="classifier-select",
        chosen_model="m", best_model="m", difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text=task_text, candidate_sims={"m": 0.9, "n": 0.4},
            sim_easy=sim_easy, sim_hard=sim_hard,
        ),
    )


# --- OLS / anchor inversion -------------------------------------------------- #
def test_ols_recovers_known_line():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 1 + 2x
    intercept, slope = _ols(xs, ys)
    assert abs(intercept - 1.0) < 1e-9
    assert abs(slope - 2.0) < 1e-9


def test_ols_none_when_x_has_no_variance():
    assert _ols([0.5, 0.5, 0.5], [0.1, 0.9, 0.4]) is None


def test_ols_none_with_fewer_than_two_points():
    assert _ols([0.5], [0.5]) is None


def test_invert_fit_round_trips_difficulty_from_ratio():
    """If the judge's labels exactly matched difficulty_from_ratio for some
    (lo, hi), fitting a line through them and inverting must recover (lo, hi)."""
    lo, hi = 0.25, 0.65
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    judged = [difficulty_from_ratio(r, lo, hi) for r in ratios]
    intercept, slope = _ols(ratios, judged)
    got = _invert_fit(intercept, slope)
    assert got is not None
    got_lo, got_hi = got
    assert abs(got_lo - lo) < 1e-6
    assert abs(got_hi - hi) < 1e-6


def test_invert_fit_none_on_non_positive_slope():
    assert _invert_fit(intercept=0.5, slope=0.0) is None
    assert _invert_fit(intercept=0.5, slope=-1.0) is None


# --- judge parsing ------------------------------------------------------------ #
async def test_judge_difficulty_parses_strict_json():
    client = FakeJudgeClient([json.dumps({"difficulty": 0.7})])
    got = await _judge_difficulty(client, "judge-model", "some task")
    assert got == 0.7


async def test_judge_difficulty_extracts_json_from_surrounding_prose():
    client = FakeJudgeClient(['Sure, here it is: {"difficulty": 0.42} thanks!'])
    got = await _judge_difficulty(client, "judge-model", "some task")
    assert got == 0.42


async def test_judge_difficulty_clamps_out_of_range_values():
    client = FakeJudgeClient([json.dumps({"difficulty": 1.5})])
    got = await _judge_difficulty(client, "judge-model", "some task")
    assert got == 1.0


async def test_judge_difficulty_none_on_unparseable_response():
    client = FakeJudgeClient(["not json at all"])
    got = await _judge_difficulty(client, "judge-model", "some task")
    assert got is None


# --- ambiguous relevance flagging -------------------------------------------- #
def test_find_ambiguous_relevance_flags_close_margins():
    settings = Settings(_env_file=None, embedding_rel_band=0.10)
    close = DecisionLogEntry(
        ts=1.0, conversation_key="close", turn=1, route="classifier-select",
        chosen_model="m", best_model="m", difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text="t", candidate_sims={"m": 0.85, "n": 0.80}, sim_easy=0.1, sim_hard=0.1,
        ),
    )
    clear = DecisionLogEntry(
        ts=1.0, conversation_key="clear", turn=1, route="classifier-select",
        chosen_model="m", best_model="m", difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text="t", candidate_sims={"m": 0.9, "n": 0.1}, sim_easy=0.1, sim_hard=0.1,
        ),
    )
    flags = _find_ambiguous_relevance([close, clear], settings)
    assert [f.conversation_key for f in flags] == ["close"]


def test_find_ambiguous_relevance_skips_entries_with_fewer_than_two_candidates():
    settings = Settings(_env_file=None, embedding_rel_band=0.10)
    entry = DecisionLogEntry(
        ts=1.0, conversation_key="c", turn=1, route="classifier-select",
        chosen_model="m", best_model="m", difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text="t", candidate_sims={"m": 0.9}, sim_easy=0.1, sim_hard=0.1,
        ),
    )
    assert _find_ambiguous_relevance([entry], settings) == []


# --- end-to-end run_calibration ------------------------------------------------ #
async def test_run_calibration_suggests_anchors_matching_judge_labels(store, settings):
    """Log decisions whose ratios span a known (lo, hi), script the fake judge to
    reply with exactly difficulty_from_ratio(ratio, lo, hi), and check the fitted
    anchors converge back to (lo, hi). Ratios are kept INSIDE [lo, hi] so the
    resulting judge label always lands in [0.15, 0.85] — outside that range
    ``_judge_difficulty`` clamps to [0, 1], which would (correctly) distort a
    fit that assumes an unclamped linear judge signal."""
    lo, hi = 0.20, 0.70
    ratio_pairs = [(1 - r, r) for r in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)] * 4
    for pair in ratio_pairs:
        await store.log_decision(_entry(pair))

    judged_values = []
    for sim_easy, sim_hard in ratio_pairs:
        ratio = sim_hard / (sim_hard + sim_easy) if (sim_hard + sim_easy) > 0 else 0.5
        judged_values.append(json.dumps({"difficulty": difficulty_from_ratio(ratio, lo, hi)}))
    client = FakeJudgeClient(judged_values)
    settings.calibration_sample_size = len(ratio_pairs)

    report = await run_calibration(store, client, settings)

    assert report.judged == len(ratio_pairs)
    assert report.difficulty.suggested_easy_anchor is not None
    assert abs(report.difficulty.suggested_easy_anchor - lo) < 1e-6
    assert abs(report.difficulty.suggested_hard_anchor - hi) < 1e-6
    assert report.difficulty.suggested_rmse < 1e-6

    text = format_report(report)
    assert "suggested COBAITER_DIFFICULTY_EASY_ANCHOR" in text


async def test_run_calibration_reports_insufficient_samples(store, settings):
    await store.log_decision(_entry((0.5, 0.5)))
    client = FakeJudgeClient([json.dumps({"difficulty": 0.5})])
    settings.calibration_sample_size = 1

    report = await run_calibration(store, client, settings)

    assert report.difficulty.suggested_easy_anchor is None
    assert "need at least" in (report.difficulty.note or "")
