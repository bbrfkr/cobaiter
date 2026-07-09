"""Offline recalibration: OLS anchor fit + ambiguous-relevance flagging.

Uses a FakeJudgeClient (a stand-in for LiteLLMClient.chat) that returns a
scripted difficulty per call, so the OLS math is exercised against a KNOWN
ground truth relationship instead of a real judge model.
"""

from __future__ import annotations

import json

from cobaiter.calibrate import (
    _BAND_GRID,
    _calibrate_relevance,
    _find_ambiguous_relevance,
    _group_candidates,
    _invert_fit,
    _judge_difficulty,
    _judge_relevance,
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


# --- relevance: domain grouping ---------------------------------------------- #
def test_group_candidates_groups_by_identical_refs():
    diag = ClassifierDiagnostics(
        task_text="t",
        candidate_sims={"think": 0.9, "no-think": 0.85, "general": 0.3},
        candidate_refs={
            "think": ["coding ex"], "no-think": ["coding ex"], "general": ["chat ex"],
        },
    )
    groups = _group_candidates(diag)
    by_refs = {tuple(refs): sorted(members) for members, refs in groups}
    assert by_refs[("coding ex",)] == ["no-think", "think"]
    assert by_refs[("chat ex",)] == ["general"]


# --- relevance: judge parsing ------------------------------------------------- #
async def test_judge_relevance_parses_domain_index():
    client = FakeJudgeClient([json.dumps({"domain_index": 1})])
    got = await _judge_relevance(client, "judge-model", "task", [["a ex"], ["b ex"]])
    assert got == 1


async def test_judge_relevance_extracts_json_from_surrounding_prose():
    client = FakeJudgeClient(['I think it is: {"domain_index": 0} for sure'])
    got = await _judge_relevance(client, "judge-model", "task", [["a ex"], ["b ex"]])
    assert got == 0


async def test_judge_relevance_accepts_no_match_sentinel():
    client = FakeJudgeClient([json.dumps({"domain_index": -1})])
    got = await _judge_relevance(client, "judge-model", "task", [["a ex"], ["b ex"]])
    assert got == -1


async def test_judge_relevance_none_on_out_of_range_index():
    client = FakeJudgeClient([json.dumps({"domain_index": 5})])
    got = await _judge_relevance(client, "judge-model", "task", [["a ex"], ["b ex"]])
    assert got is None


async def test_judge_relevance_none_on_unparseable_response():
    client = FakeJudgeClient(["not json at all"])
    got = await _judge_relevance(client, "judge-model", "task", [["a ex"], ["b ex"]])
    assert got is None


# --- relevance: band calibration (grid search over "safe suppression") ------ #
_AB_REFS = {"a": ["A ex"], "b": ["B ex"]}


def _relevance_entry(
    turn: int, sims: dict[str, float], refs: dict[str, list[str]],
) -> DecisionLogEntry:
    return DecisionLogEntry(
        ts=1.0, conversation_key="conv", turn=turn, route="classifier-select",
        chosen_model=max(sims, key=lambda m: sims[m]), best_model=None, difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text=f"task {turn}", candidate_sims=sims, candidate_refs=refs,
        ),
    )


async def test_calibrate_relevance_recovers_largest_safe_band(settings):
    """19 "easy" entries where the wrong domain ("b") trails by 0.5 raw cosine
    (safe up to band=1.0, way past the grid), plus 1 "tight" entry where it
    trails by exactly 0.05 (safe boundary == 2*0.05 == 0.10, exactly on the
    grid). The largest band that is STILL safe on every judged entry is 0.10;
    anything above lets the tight entry's wrong domain exceed the 0.5
    suitability threshold."""
    entries = [_relevance_entry(i, {"a": 0.9, "b": 0.4}, _AB_REFS) for i in range(19)]
    entries.append(_relevance_entry(19, {"a": 0.9, "b": 0.85}, _AB_REFS))
    client = FakeJudgeClient([json.dumps({"domain_index": 0})] * 20)
    settings.embedding_rel_band = 0.25  # deliberately unsafe for the tight entry

    relevance, judged = await _calibrate_relevance(entries, client, settings)

    assert relevance.n == 20
    assert relevance.raw_top_accuracy == 1.0
    assert relevance.current_band == 0.25
    assert relevance.current_accuracy == 19 / 20  # tight entry unsafe at 0.25
    assert relevance.suggested_band == 0.10
    assert relevance.suggested_accuracy == 1.0
    assert len(judged) == 20


async def test_calibrate_relevance_prefers_largest_band_among_safe_ties(settings):
    """Every entry has a huge deficit, so the WHOLE grid is safe. The
    suggested band must be the LARGEST grid point, not the smallest — unlike
    difficulty's anchor fit, "smaller is always safer" here so the useful
    number is how generous the band can be made without losing safety."""
    entries = [_relevance_entry(i, {"a": 0.99, "b": 0.01}, _AB_REFS) for i in range(20)]
    client = FakeJudgeClient([json.dumps({"domain_index": 0})] * 20)

    relevance, _judged = await _calibrate_relevance(entries, client, settings)

    assert relevance.suggested_band == _BAND_GRID[-1]
    assert relevance.suggested_accuracy == 1.0


async def test_calibrate_relevance_reports_raw_top_accuracy_below_one_on_misfires(settings):
    """10 entries where "a" (judge-correct) is genuinely the raw top, and 10
    where "b" wins on raw similarity despite the judge saying "a" is correct
    — a real relevance misfire no band can fix. Only the correctly-classified
    entries feed the band search; raw_top_accuracy surfaces the misfire rate."""
    correct = [_relevance_entry(i, {"a": 0.9, "b": 0.4}, _AB_REFS) for i in range(10)]
    misfires = [_relevance_entry(10 + i, {"a": 0.3, "b": 0.9}, _AB_REFS) for i in range(10)]
    client = FakeJudgeClient([json.dumps({"domain_index": 0})] * 20)

    relevance, _judged = await _calibrate_relevance(correct + misfires, client, settings)

    assert relevance.n == 20
    assert relevance.raw_top_accuracy == 0.5
    assert relevance.suggested_band is not None


async def test_calibrate_relevance_reports_insufficient_samples(settings):
    entries = [_relevance_entry(0, {"a": 0.9, "b": 0.4}, _AB_REFS)]
    client = FakeJudgeClient([json.dumps({"domain_index": 0})])

    relevance, judged = await _calibrate_relevance(entries, client, settings)

    assert relevance.suggested_band is None
    assert "need at least" in (relevance.note or "")


async def test_run_calibration_includes_relevance_band_section(store, settings):
    """End-to-end (mirrors test_run_calibration_suggests_anchors_matching_
    judge_labels): logs BOTH difficulty-shaped and relevance-shaped decisions,
    scripts one FakeJudgeClient covering both judge passes in call order, and
    checks the relevance section lands in the final report."""
    lo, hi = 0.20, 0.70
    ratio_pairs = [(1 - r, r) for r in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)] * 4
    for pair in ratio_pairs:
        await store.log_decision(_entry(pair))
    for i in range(19):
        await store.log_decision(_relevance_entry(100 + i, {"a": 0.9, "b": 0.4}, _AB_REFS))
    await store.log_decision(_relevance_entry(119, {"a": 0.9, "b": 0.85}, _AB_REFS))

    judged_difficulty = [
        json.dumps({"difficulty": difficulty_from_ratio(
            sim_hard / (sim_hard + sim_easy), lo, hi
        )})
        for sim_easy, sim_hard in ratio_pairs
    ]
    judged_relevance = [json.dumps({"domain_index": 0})] * 20
    client = FakeJudgeClient(judged_difficulty + judged_relevance)
    settings.calibration_sample_size = max(len(ratio_pairs), 20)

    report = await run_calibration(store, client, settings)

    assert report.relevance.n == 20
    assert report.relevance.suggested_band == 0.10
    text = format_report(report)
    assert "COBAITER_EMBEDDING_REL_BAND" in text


# --- relevance: ambiguous-flag judge annotation ------------------------------ #
def _ab_flag_entry(turn: int = 3) -> DecisionLogEntry:
    return DecisionLogEntry(
        ts=1.0, conversation_key="c1", turn=turn, route="classifier-select",
        chosen_model="a", best_model="a", difficulty=0.5,
        diagnostics=ClassifierDiagnostics(
            task_text="t", candidate_sims={"a": 0.85, "b": 0.80}, candidate_refs=_AB_REFS,
        ),
    )


def test_find_ambiguous_relevance_annotates_agreement_when_judged():
    settings = Settings(_env_file=None, embedding_rel_band=0.10)
    entry = _ab_flag_entry()
    flags = _find_ambiguous_relevance([entry], settings, {("c1", 3): 0})
    assert len(flags) == 1
    assert flags[0].judge_domain == 0
    assert flags[0].agreed_with_router is True


def test_find_ambiguous_relevance_annotates_disagreement_when_judged():
    settings = Settings(_env_file=None, embedding_rel_band=0.10)
    entry = _ab_flag_entry()
    flags = _find_ambiguous_relevance([entry], settings, {("c1", 3): 1})
    assert len(flags) == 1
    assert flags[0].agreed_with_router is False


def test_find_ambiguous_relevance_no_annotation_when_not_judged():
    settings = Settings(_env_file=None, embedding_rel_band=0.10)
    entry = _ab_flag_entry()
    flags = _find_ambiguous_relevance([entry], settings)
    assert len(flags) == 1
    assert flags[0].judge_domain is None
    assert flags[0].agreed_with_router is None
