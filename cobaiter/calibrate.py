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

Relevance (``embedding_rel_band``) is also auto-calibrated: the decision log now
carries each candidate's RESOLVED reference texts (``candidate_refs`` — its
``task_examples``, or ``description`` as a fallback; see ``ModelSpec``), so a
judge can be shown "task text + each candidate's example tasks, grouped by
domain" and asked which domain is actually correct.

Unlike difficulty, ``embedding_rel_band`` is NOT something an OLS-style fit (or
even an argmax-accuracy grid search) can meaningfully calibrate: by
construction, ``relevance_from_sims`` always maps the raw-top-similarity
candidate to exactly 1.0 for any band, so band can never change WHICH group
has the highest relevance — a genuine relevance misfire (wrong domain has the
higher raw similarity) is a ``description``/``task_examples`` wording problem,
not a band problem, and is reported separately as ``raw_top_accuracy``. What
band DOES control is how strongly the OTHER (wrong-domain) groups get
suppressed once the top pick is already correct — see ``_calibrate_relevance``
for the "largest band that keeps every wrong-domain group's suitability below
the classifier's own neutral value" search this module runs instead. The judge
pass is reused to annotate the routing decisions where the top and runner-up
candidate's raw similarity were within ``embedding_rel_band`` of each other —
i.e. calls the current band treats as close enough to matter — for manual
review.

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

from .classifier import difficulty_from_ratio, relevance_from_sims
from .config import Settings, get_settings
from .litellm_client import DownstreamError, LiteLLMClient
from .schemas import ClassifierDiagnostics, DecisionLogEntry
from .store import Store

log = logging.getLogger("cobaiter")

# Below this many judge-labelled samples, an OLS fit / accuracy estimate is too
# noisy to trust.
_MIN_CALIBRATION_SAMPLES = 20
_MIN_RELEVANCE_SAMPLES = 20
_MAX_RELEVANCE_FLAGS = 20
# Grid of embedding_rel_band candidates simulated against already-embedded raw
# similarities (see _calibrate_relevance) — every point is a free re-run of
# relevance_from_sims(), no LLM call, so a fine 0.01 step is cheap.
_BAND_GRID = [round(0.01 * i, 2) for i in range(1, 41)]
# relevance_from_sims always maps the raw-top-similarity candidate to EXACTLY
# 1.0 for any band > 0 (top - top == 0), so band can never change WHICH group
# has the highest relevance — only how far OTHER groups are suppressed below
# it. 0.5 is the threshold below which a wrong-domain candidate's eventual
# ``suitability`` (relevance * capability_fit, see router.py) is no better
# than the classifier's own "no signal" neutral value, i.e. cost/tier can no
# longer plausibly make it win. This is what _band_accuracy checks for.
_SAFE_RELEVANCE_THRESHOLD = 0.5

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

_JUDGE_RELEVANCE_SYSTEM = (
    "You are calibrating an LLM router's domain-relevance heuristic. Given a "
    "single user task/message (any language) and a numbered list of candidate "
    "domains (each described by a few short example tasks it is meant to "
    "handle), pick the ONE domain that best fits the task. Respond with ONLY a "
    'JSON object of the exact form {"domain_index": <int>} where the integer '
    "is the 0-based index of the best-fitting domain, or -1 if none of the "
    "listed domains fit the task at all. Nothing else."
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
class RelevanceCalibration:
    n: int  # total judged entries (judge returned a valid, non-"-1" domain index)
    # Band-INDEPENDENT: fraction of judged entries where the raw top-similarity
    # group already matched the judge's chosen domain. Low values point at a
    # description/task_examples wording problem, not a band problem (see
    # module docstring) — band tuning below only ever applies to the
    # ALREADY-correct subset.
    raw_top_accuracy: float | None = None
    current_band: float | None = None
    current_accuracy: float | None = None
    suggested_band: float | None = None
    suggested_accuracy: float | None = None
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
    # The judge's chosen domain-group index for this entry, and whether the
    # router's raw top pick landed in that same group — populated only when
    # this entry happened to be part of the relevance judge sample this run
    # (see _calibrate_relevance); ``None`` otherwise ("not judged this run").
    judge_domain: int | None = None
    agreed_with_router: bool | None = None


@dataclass
class CalibrationReport:
    sampled: int
    judged: int
    difficulty: DifficultyCalibration
    relevance: RelevanceCalibration
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
    relevance, relevance_judged = await _calibrate_relevance(entries, client, settings)
    relevance_flags = _find_ambiguous_relevance(entries, settings, relevance_judged)
    return CalibrationReport(
        sampled=len(sample), judged=len(judged),
        difficulty=difficulty, relevance=relevance, relevance_flags=relevance_flags,
    )


async def _judge_difficulty(client: LiteLLMClient, model: str, task_text: str) -> float | None:
    try:
        resp = await client.chat({
            "model": model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": task_text},
            ],
            # No explicit temperature: some judge models (e.g. reasoning-style
            # models like gpt-5.5) reject any non-default value outright
            # (litellm's drop_params only strips UNSUPPORTED params, not
            # unsupported VALUES of an otherwise-supported one, so this must
            # be omitted rather than set to 0). Determinism isn't critical
            # here — this is an offline batch job aggregating over many
            # samples, not a single decisive call.
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
# Relevance: automatic embedding_rel_band calibration (grid search + accuracy)
# --------------------------------------------------------------------------- #
def _group_candidates(diag: ClassifierDiagnostics) -> list[tuple[list[str], list[str]]]:
    """Group candidate models that share IDENTICAL resolved reference texts
    (``candidate_refs``) into one "domain option". Tier variants sharing a
    domain (e.g. a think/no-think pair) must not be asked of the judge as
    separate choices — the judge only needs to pick the right DOMAIN, exactly
    like relevance's own job. Returns ``[(member_models, shared_refs), ...]``,
    ordered by each group's minimum model name for a stable, reproducible
    index across the judge prompt and the safety-check group lookup."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for model in diag.candidate_sims:
        key = tuple(diag.candidate_refs.get(model, []))
        groups.setdefault(key, []).append(model)
    ordered = sorted(groups.items(), key=lambda kv: min(kv[1]))
    return [(members, list(key)) for key, members in ordered]


async def _judge_relevance(
    client: LiteLLMClient, model: str, task_text: str, groups: list[list[str]]
) -> int | None:
    domains = "\n".join(f"{i}: " + " / ".join(refs) for i, refs in enumerate(groups))
    try:
        resp = await client.chat({
            "model": model,
            "messages": [
                {"role": "system", "content": _JUDGE_RELEVANCE_SYSTEM},
                {"role": "user", "content": f"task:\n{task_text}\n\ndomains:\n{domains}"},
            ],
            # See _judge_difficulty: no explicit temperature, some judge
            # models reject any non-default value.
        })
        content = resp["choices"][0]["message"]["content"]
    except (DownstreamError, KeyError, IndexError, TypeError) as exc:
        log.warning("calibrate: relevance judge call failed: %s", exc)
        return None
    match = _JSON_OBJECT.search(content)
    if not match:
        log.warning("calibrate: relevance judge response had no JSON object: %r", content[:200])
        return None
    try:
        value = int(json.loads(match.group(0))["domain_index"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        log.warning("calibrate: relevance judge response unparseable: %r", content[:200])
        return None
    if value < -1 or value >= len(groups):
        log.warning("calibrate: relevance judge domain_index out of range: %r", value)
        return None
    return value


def _band_accuracy(
    rows: list[tuple[dict[str, float], list[str]]], band: float
) -> float:
    """Fraction of ``rows`` (each: per-model raw sims, judge-correct-group
    member models) where every WRONG-group candidate's relevance, re-simulated
    at ``band``, stays at/below ``_SAFE_RELEVANCE_THRESHOLD``.

    ``relevance_from_sims`` maps the raw-top-similarity candidate to EXACTLY
    1.0 for any band (top - top == 0), so band can never change WHICH group
    wins on relevance alone — this is why ``rows`` only ever contains entries
    where the raw top ALREADY agrees with the judge (see
    ``_calibrate_relevance``): band cannot rescue an already-wrong top pick,
    so including those here would make the metric meaningless. What band DOES
    control, and what this checks, is how confidently the OTHER (wrong-domain)
    groups get suppressed once the top pick is correct."""
    if not rows:
        return 0.0
    safe = 0
    for sims_by_model, correct_members in rows:
        models = list(sims_by_model)
        relevance = dict(zip(
            models, relevance_from_sims([sims_by_model[m] for m in models], band)
        ))
        if all(
            relevance[m] <= _SAFE_RELEVANCE_THRESHOLD
            for m in models if m not in correct_members
        ):
            safe += 1
    return safe / len(rows)


async def _calibrate_relevance(
    entries: list[DecisionLogEntry], client: LiteLLMClient, settings: Settings
) -> tuple[RelevanceCalibration, dict[tuple[str, int], int]]:
    """Judge a sample of logged decisions for which DOMAIN was actually
    correct, then (for the subset where relevance's raw top pick already
    agreed with the judge) search ``embedding_rel_band`` for the LARGEST value
    that still keeps every wrong-domain group safely suppressed in every one
    of those rows (see ``_band_accuracy``). "Largest safe value" — not
    "highest accuracy with ties broken toward smaller" — because the safety
    rate is monotonically non-increasing in band, so the smallest grid point
    is trivially always "safest"; the useful number to surface is how
    generous ``embedding_rel_band`` can be made without losing that safety
    margin on real traffic, which needs the judge-labelled sample to answer.

    Returns the calibration report AND the ``(conversation_key, turn) ->
    judge_domain_index`` map, so the caller can also annotate the ambiguous-
    margin flag list (see _find_ambiguous_relevance) without a second judge
    pass over the same entries.
    """
    usable = [
        e for e in entries
        if e.diagnostics
        and e.diagnostics.task_text
        and len(e.diagnostics.candidate_sims) >= 2
        and e.diagnostics.candidate_refs
    ]
    sample = usable[-settings.calibration_sample_size :]

    judged: dict[tuple[str, int], int] = {}
    safety_rows: list[tuple[dict[str, float], list[str]]] = []
    n_judged = 0
    n_correct_top = 0
    for entry in sample:
        d = entry.diagnostics
        assert d is not None and d.task_text is not None  # filtered above
        groups = _group_candidates(d)
        if len(groups) < 2:
            continue  # every candidate shares one domain -> nothing to judge
        domain_index = await _judge_relevance(
            client, settings.calibration_judge_model, d.task_text,
            [refs for _members, refs in groups],
        )
        if domain_index is None or domain_index == -1:
            continue
        n_judged += 1
        judged[(entry.conversation_key, entry.turn)] = domain_index
        correct_members, _refs = groups[domain_index]
        top_model = max(d.candidate_sims, key=lambda m: d.candidate_sims[m])
        if top_model in correct_members:
            n_correct_top += 1
            safety_rows.append((d.candidate_sims, correct_members))

    if n_judged < _MIN_RELEVANCE_SAMPLES:
        return RelevanceCalibration(
            n=n_judged,
            note=f"need at least {_MIN_RELEVANCE_SAMPLES} judged samples, got {n_judged}",
        ), judged

    raw_top_accuracy = n_correct_top / n_judged
    if not safety_rows:
        return RelevanceCalibration(
            n=n_judged, raw_top_accuracy=raw_top_accuracy,
            note="relevance's raw top pick never matched the judge in this sample — "
                 "this is a domain-separation problem (description/task_examples "
                 "wording), not something embedding_rel_band can fix",
        ), judged

    current_band = settings.embedding_rel_band
    current_accuracy = _band_accuracy(safety_rows, current_band)

    # Scan DESCENDING so the first band to reach the eventual max (safety is
    # monotonically non-increasing as band grows) is the LARGEST one — the
    # most generous band that is still fully safe on this sample.
    best_band, best_accuracy = _BAND_GRID[0], -1.0
    for band in reversed(_BAND_GRID):
        acc = _band_accuracy(safety_rows, band)
        if acc > best_accuracy:
            best_band, best_accuracy = band, acc

    return RelevanceCalibration(
        n=n_judged, raw_top_accuracy=raw_top_accuracy,
        current_band=current_band, current_accuracy=current_accuracy,
        suggested_band=best_band, suggested_accuracy=best_accuracy,
    ), judged


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
    entries: list[DecisionLogEntry],
    settings: Settings,
    judged: dict[tuple[str, int], int] | None = None,
) -> list[RelevanceFlag]:
    """Flag close-margin decisions for manual review, annotated (when
    available) with the relevance judge's verdict from ``_calibrate_relevance``
    — reuses that same judge pass rather than calling the judge a second time
    for entries that happen to also be close-margin."""
    judged = judged or {}
    flags: list[RelevanceFlag] = []
    for e in entries:
        sims = e.diagnostics.candidate_sims if e.diagnostics else {}
        if len(sims) < 2:
            continue
        top, runner_up = sorted(sims.values(), reverse=True)[:2]
        margin = top - runner_up
        if margin >= settings.embedding_rel_band:
            continue
        judge_domain = judged.get((e.conversation_key, e.turn))
        agreed = None
        if judge_domain is not None and e.diagnostics is not None:
            groups = _group_candidates(e.diagnostics)
            best_model = max(sims, key=lambda m: sims[m])
            winning_group = next(
                (i for i, (members, _refs) in enumerate(groups) if best_model in members),
                None,
            )
            agreed = winning_group == judge_domain
        flags.append(RelevanceFlag(
            conversation_key=e.conversation_key, turn=e.turn, route=e.route,
            chosen_model=e.chosen_model, best_model=e.best_model,
            margin=margin, task_text=e.diagnostics.task_text if e.diagnostics else None,
            judge_domain=judge_domain, agreed_with_router=agreed,
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
        "== relevance band (COBAITER_EMBEDDING_REL_BAND) ==",
        f"judged samples: {report.relevance.n}",
    ]
    r = report.relevance
    if r.raw_top_accuracy is not None:
        lines.append(
            f"raw top-pick accuracy vs judge (band-independent): {r.raw_top_accuracy:.3f}"
        )
    if r.current_accuracy is not None:
        lines.append(
            f"current band={r.current_band:.2f} safe wrong-domain suppression rate: "
            f"{r.current_accuracy:.3f}"
        )
    if r.suggested_band is not None:
        lines.append(f"suggested COBAITER_EMBEDDING_REL_BAND={r.suggested_band:.2f}")
        lines.append(
            f"suggested band safe wrong-domain suppression rate: {r.suggested_accuracy:.3f}"
        )
    if r.note:
        lines.append(f"note: {r.note}")

    lines += [
        "",
        "== ambiguous relevance calls (top-2 raw similarity margin below the "
        f"configured COBAITER_EMBEDDING_REL_BAND, showing up to {_MAX_RELEVANCE_FLAGS}) ==",
    ]
    if not report.relevance_flags:
        lines.append("(none found)")
    for f in report.relevance_flags:
        snippet = (f.task_text or "<redacted>")[:80]
        judge_str = f.judge_domain if f.judge_domain is not None else "n/a"
        agreed_str = f.agreed_with_router if f.agreed_with_router is not None else "n/a"
        lines.append(
            f"  conv={f.conversation_key} turn={f.turn} route={f.route} "
            f"chosen={f.chosen_model} best={f.best_model} margin={f.margin:.3f} "
            f"judge={judge_str} agreed={agreed_str}  {snippet!r}"
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
