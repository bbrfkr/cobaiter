"""Offline recalibration of the classifier's difficulty anchors from real traffic.

The embedding classifier (see ``classifier.py``) turns raw signal — an exemplar
similarity ratio for difficulty, a raw cosine similarity per candidate for
relevance — into scores via a few hand-picked constants (``difficulty_easy_anchor``
/ ``difficulty_hard_anchor``, ``embedding_rel_band``). Those were set once from a
small manual calibration set. This module closes the loop: it reads the
``DecisionLogEntry`` records the router persists for every classifier-driven
decision (see ``router.RouteEngine._log_decision``), asks a judge LLM to rate the
*actual* difficulty of a sample of real task text, and fits new anchors against
that gold data via ordinary least squares.

Deliberately NOT a live/automatic loop:
* The judge call is a full generative LLM request — exactly the ~1s synchronous
  cost the embedding classifier replaced (see classifier.py's module docstring).
  It only ever runs here, offline, batched, never on the request hot path.
* This tool only PRINTS a report with suggested ``.env`` values; it never writes
  config itself. Feeding a router's own routing mistakes back in as unreviewed
  "ground truth" would let errors compound instead of correct themselves, so a
  human is expected to look at the report (and the flagged ambiguous relevance
  calls) before touching ``difficulty_easy_anchor``/``hard_anchor``.

Relevance recalibration is intentionally out of scope for the automatic fit: the
raw signal alone (a candidate's cosine similarity) doesn't tell you WHICH domain
was actually correct without a judge that also sees each candidate's
description text, which the decision log does not currently persist. Instead,
this reports the routing decisions where the top and runner-up candidate's raw
similarity were within ``embedding_rel_band`` of each other — i.e. calls the
current band treats as close enough to matter — for manual review.

Usage::

    COBAITER_CALIBRATION_JUDGE_MODEL=claude-sonnet-4-6 uv run python -m cobaiter.calibrate
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field

from .classifier import difficulty_from_ratio
from .config import Settings, get_settings
from .litellm_client import DownstreamError, LiteLLMClient
from .schemas import DecisionLogEntry
from .store import Store

log = logging.getLogger("cobaiter")

# Below this many judge-labelled samples, an OLS fit is too noisy to trust.
_MIN_CALIBRATION_SAMPLES = 20
_MAX_RELEVANCE_FLAGS = 20

_JUDGE_SYSTEM = (
    "You are calibrating an LLM router's task-difficulty heuristic. Given a "
    "single user task/message (any language), rate how difficult it would be "
    "for a capable AI assistant to handle well, on a continuous scale from 0.0 "
    "(trivial: greetings, simple lookups, one-line factual answers) to 1.0 "
    "(expert-level: deep technical/scientific/legal reasoning, non-trivial "
    "debugging, multi-step analysis or design). Judge the task itself, not its "
    "length. Respond with ONLY a JSON object of the exact form "
    '{"difficulty": <float between 0 and 1>} and nothing else.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# Report data
# --------------------------------------------------------------------------- #
@dataclass
class DifficultyCalibration:
    n: int
    current_rmse: float | None = None
    suggested_easy_anchor: float | None = None
    suggested_hard_anchor: float | None = None
    suggested_rmse: float | None = None
    note: str | None = None


@dataclass
class RelevanceFlag:
    conversation_key: str
    turn: int
    route: str
    chosen_model: str
    best_model: str | None
    margin: float
    task_text: str | None


@dataclass
class CalibrationReport:
    sampled: int
    judged: int
    difficulty: DifficultyCalibration
    relevance_flags: list[RelevanceFlag] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
async def run_calibration(
    store: Store, client: LiteLLMClient, settings: Settings
) -> CalibrationReport:
    # Over-fetch: not every logged decision has a usable (non-redacted,
    # exemplar-scored) task_text, so pull more than the sample target and take
    # the most recent N that qualify.
    entries = await store.read_decisions(count=max(settings.calibration_sample_size * 5, 500))

    usable = [
        e for e in entries
        if e.diagnostics
        and e.diagnostics.task_text
        and e.diagnostics.sim_easy is not None
        and e.diagnostics.sim_hard is not None
    ]
    sample = usable[-settings.calibration_sample_size :]

    ratios: list[float] = []
    judged: list[float] = []
    for entry in sample:
        d = entry.diagnostics
        assert d is not None and d.task_text is not None  # filtered above
        assert d.sim_easy is not None and d.sim_hard is not None
        gold = await _judge_difficulty(client, settings.calibration_judge_model, d.task_text)
        if gold is None:
            continue
        denom = d.sim_hard + d.sim_easy
        ratios.append(d.sim_hard / denom if denom > 0 else 0.5)
        judged.append(gold)

    difficulty = _calibrate_difficulty(ratios, judged, settings)
    relevance_flags = _find_ambiguous_relevance(entries, settings)
    return CalibrationReport(
        sampled=len(sample), judged=len(judged),
        difficulty=difficulty, relevance_flags=relevance_flags,
    )


async def _judge_difficulty(client: LiteLLMClient, model: str, task_text: str) -> float | None:
    try:
        resp = await client.chat({
            "model": model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": task_text},
            ],
            "temperature": 0,
        })
        content = resp["choices"][0]["message"]["content"]
    except (DownstreamError, KeyError, IndexError, TypeError) as exc:
        log.warning("calibrate: judge call failed: %s", exc)
        return None
    match = _JSON_OBJECT.search(content)
    if not match:
        log.warning("calibrate: judge response had no JSON object: %r", content[:200])
        return None
    try:
        value = float(json.loads(match.group(0))["difficulty"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        log.warning("calibrate: judge response unparseable: %r", content[:200])
        return None
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- #
# Difficulty anchor fit (ordinary least squares, no numpy dependency)
# --------------------------------------------------------------------------- #
def _calibrate_difficulty(
    ratios: list[float], judged: list[float], settings: Settings
) -> DifficultyCalibration:
    n = len(ratios)
    if n < _MIN_CALIBRATION_SAMPLES:
        return DifficultyCalibration(
            n=n,
            note=f"need at least {_MIN_CALIBRATION_SAMPLES} judged samples, got {n}",
        )
    current_preds = [
        difficulty_from_ratio(r, settings.difficulty_easy_anchor, settings.difficulty_hard_anchor)
        for r in ratios
    ]
    current_rmse = _rmse(current_preds, judged)

    fit = _ols(ratios, judged)
    if fit is None:
        return DifficultyCalibration(
            n=n, current_rmse=current_rmse,
            note="ratio has ~zero variance in this sample; cannot fit a line",
        )
    intercept, slope = fit
    anchors = _invert_fit(intercept, slope)
    if anchors is None:
        return DifficultyCalibration(
            n=n, current_rmse=current_rmse,
            note="fit slope is non-positive; judge difficulty doesn't track the "
                 "embedding ratio in this sample (bad judge model? bad sample?)",
        )
    lo, hi = anchors
    suggested_preds = [difficulty_from_ratio(r, lo, hi) for r in ratios]
    suggested_rmse = _rmse(suggested_preds, judged)
    note = None
    if not (0.0 <= lo < hi <= 1.0):
        note = "suggested anchors fall outside [0,1] — treat with caution, gather more samples"
    return DifficultyCalibration(
        n=n, current_rmse=current_rmse,
        suggested_easy_anchor=lo, suggested_hard_anchor=hi,
        suggested_rmse=suggested_rmse, note=note,
    )


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Ordinary least squares fit ``y = intercept + slope*x``. None if x has ~no
    variance (degenerate — e.g. every sampled task landed at the same ratio)."""
    n = len(xs)
    if n < 2:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    var_x = sum((x - xbar) ** 2 for x in xs)
    if var_x <= 1e-9:
        return None
    cov_xy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = ybar - slope * xbar
    return intercept, slope


def _invert_fit(intercept: float, slope: float) -> tuple[float, float] | None:
    """Invert ``judged = intercept + slope*ratio`` into ``(easy_anchor, hard_anchor)``
    such that ``difficulty_from_ratio(ratio, lo, hi) ≈ judged`` (least squares).

    ``difficulty_from_ratio`` is affine: ``y = 0.15 + (x-lo)/(hi-lo) * 0.7``, i.e.
    ``slope = 0.7/(hi-lo)`` and ``intercept = 0.15 - lo*slope``. A non-positive
    slope means "harder tasks got a LOWER embedding ratio" in this sample, which
    would invert the anchors — a sign the judge/sample is unreliable rather than
    a real anchor to adopt, so this returns ``None`` for the caller to report.
    """
    if slope <= 1e-6:
        return None
    span = 0.7 / slope
    lo = (0.15 - intercept) / slope
    hi = lo + span
    return lo, hi


def _rmse(preds: list[float], actual: list[float]) -> float:
    n = len(preds)
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actual)) / n) if n else 0.0


# --------------------------------------------------------------------------- #
# Relevance: flag close calls for manual review (see module docstring)
# --------------------------------------------------------------------------- #
def _find_ambiguous_relevance(
    entries: list[DecisionLogEntry], settings: Settings
) -> list[RelevanceFlag]:
    flags: list[RelevanceFlag] = []
    for e in entries:
        sims = e.diagnostics.candidate_sims if e.diagnostics else {}
        if len(sims) < 2:
            continue
        top, runner_up = sorted(sims.values(), reverse=True)[:2]
        margin = top - runner_up
        if margin >= settings.embedding_rel_band:
            continue
        flags.append(RelevanceFlag(
            conversation_key=e.conversation_key, turn=e.turn, route=e.route,
            chosen_model=e.chosen_model, best_model=e.best_model,
            margin=margin, task_text=e.diagnostics.task_text if e.diagnostics else None,
        ))
    flags.sort(key=lambda f: f.margin)
    return flags[:_MAX_RELEVANCE_FLAGS]


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #
def format_report(report: CalibrationReport) -> str:
    lines = [
        f"decisions sampled: {report.sampled}  (judge-labelled: {report.judged})",
        "",
        "== difficulty anchors (COBAITER_DIFFICULTY_EASY_ANCHOR / _HARD_ANCHOR) ==",
        f"samples used: {report.difficulty.n}",
    ]
    d = report.difficulty
    if d.current_rmse is not None:
        lines.append(f"current anchors RMSE vs judge: {d.current_rmse:.3f}")
    if d.suggested_easy_anchor is not None and d.suggested_hard_anchor is not None:
        lines.append(f"suggested COBAITER_DIFFICULTY_EASY_ANCHOR={d.suggested_easy_anchor:.3f}")
        lines.append(f"suggested COBAITER_DIFFICULTY_HARD_ANCHOR={d.suggested_hard_anchor:.3f}")
        lines.append(f"suggested anchors RMSE vs judge: {d.suggested_rmse:.3f}")
    if d.note:
        lines.append(f"note: {d.note}")
    lines += [
        "",
        "== ambiguous relevance calls (top-2 raw similarity margin below the "
        f"configured COBAITER_EMBEDDING_REL_BAND, showing up to {_MAX_RELEVANCE_FLAGS}) ==",
    ]
    if not report.relevance_flags:
        lines.append("(none found)")
    for f in report.relevance_flags:
        snippet = (f.task_text or "<redacted>")[:80]
        lines.append(
            f"  conv={f.conversation_key} turn={f.turn} route={f.route} "
            f"chosen={f.chosen_model} best={f.best_model} margin={f.margin:.3f}  {snippet!r}"
        )
    lines += ["", "This is a REPORT only — no config is written; review before editing .env."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
async def _amain() -> None:
    settings = get_settings()
    if not settings.calibration_judge_model:
        raise SystemExit(
            "COBAITER_CALIBRATION_JUDGE_MODEL must be set to a judge model routed "
            "through the LiteLLM gateway before running calibration."
        )
    store = Store.from_url(settings)
    client = LiteLLMClient.create(settings)
    try:
        report = await run_calibration(store, client, settings)
    finally:
        await store.close()
        await client.close()
    print(format_report(report))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
