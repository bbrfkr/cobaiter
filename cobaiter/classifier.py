"""Lightweight classifier: score candidate models for the current context.

The classifier is only invoked when the constraint filter leaves *more than one*
candidate (initial routing) or when the stage-1 soft-gate fires (re-evaluation).
It returns a 0..1 use-case relevance per candidate plus one task-difficulty
scalar; the router applies the capability-fit / cost-tier / hysteresis logic on
top.

The previous implementation obtained both axes from a synchronous LLM call,
which put ~1s of fixed latency on every routing decision — painful for agentic
callers that fan one user instruction into many API calls. This version keeps
the relevance/difficulty two-axis design but computes both without a generative
LLM, from embeddings served through the LiteLLM gateway:

* ``relevance``   — top-2-mean cosine similarity between the task text and
  each candidate's use-case reference texts (``ModelSpec.task_examples``, or
  its ``description`` as a single-item fallback). Reference vectors are
  cached in-process.
* ``difficulty``  — where the task text sits, by cosine similarity, between
  two small fixed sets of "easy" and "hard" example tasks spanning MULTIPLE
  domains (math, coding, science, law, ...), not just coding. A keyword list
  (like the old ``_HIGH_INTENT`` regex this replaced) only ever covers the
  domains someone thought to enumerate; embedding similarity generalises to
  domains no one wrote a keyword for (e.g. "prove Gödel's incompleteness
  theorem" scores as hard without any math-specific keyword list).

Both axes are judged from the SAME text: the LATEST user message only (see
``_task_text``) — not the system message (agent scaffolding), not assistant
replies, and not earlier user turns. Conversation-level topic stability is
already handled by sticky pinning (see router.py), not by blending prior
turns into every embedding comparison; blending would let a long-running
conversation's difficulty/relevance drift based on how many turns it has had,
independent of what is actually being asked right now (measured: a two-turn
"よろしく" + "こんにちは" scored measurably harder than "こんにちは" alone
when earlier turns were folded in). Both axes reuse the SAME embedding, so the
steady-state cost stays at ONE small embedding call per decision (description
and exemplar vectors are cached after their first use).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import httpx

from .config import Settings
from .features import count_code_blocks, estimate_tokens, message_text
from .litellm_client import DownstreamError, LiteLLMClient
from .schemas import (
    CandidateScore,
    ChatCompletionRequest,
    ClassifierDiagnostics,
    ClassifierResult,
    ModelSpec,
)

log = logging.getLogger("cobaiter")


class EmbeddingClassifier:
    """Relevance + difficulty from embedding similarity.

    Same ``score(req, candidates) -> ClassifierResult`` contract as the old LLM
    classifier, so the router is unchanged. On any embedding failure it falls
    back to neutral relevance and a token-count-only difficulty estimate (the
    router's deterministic cost/tier re-ranking then decides) — routing never
    hard-fails on the classifier.
    """

    def __init__(
        self,
        client: LiteLLMClient,
        settings: Settings,
        *,
        easy_exemplars: list[str] | None = None,
        hard_exemplars: list[str] | None = None,
    ) -> None:
        self._client = client
        self._s = settings
        # Difficulty exemplars: injected (from Settings.difficulty_exemplars_config
        # via the app layer) or, when not configured, the built-in defaults below.
        # Defaults resolve at call time — not as argument defaults — because the
        # module-level constants are defined further down this file.
        self._easy_exemplars = (
            list(easy_exemplars) if easy_exemplars is not None else list(_EASY_EXEMPLARS)
        )
        self._hard_exemplars = (
            list(hard_exemplars) if hard_exemplars is not None else list(_HARD_EXEMPLARS)
        )
        self._difficulty_exemplars = self._easy_exemplars + self._hard_exemplars
        # reference text (a model's task_examples, or its description as a
        # single-item fallback) / exemplar text -> embedding vector. Both sets
        # are small and stable per process (registry data, fixed exemplar
        # list), so a plain dict cache never needs eviction.
        self._ref_vecs: dict[str, list[float]] = {}
        self._exemplar_vecs: dict[str, list[float]] = {}

    async def score(
        self,
        req: ChatCompletionRequest,
        candidates: list[ModelSpec],
    ) -> ClassifierResult:
        if not candidates:
            return ClassifierResult(scores=[])
        latest_user_msg = _latest_user_message(req.messages)
        latest_user = message_text(latest_user_msg) if latest_user_msg else ""
        latest_user = _strip_edge_punct(latest_user)
        task_text = _task_text(latest_user, self._s.classifier_digest_chars)

        task_vec, rels, raw_sims, refs_per_candidate = await self._embed_and_score_relevance(
            task_text, candidates
        )
        if rels is None:
            # No embedding signal: every candidate is equally relevant. The
            # router's cost/tier re-ranking then picks the cheapest, lightest
            # model.
            rels = [_NEUTRAL_SUITABILITY] * len(candidates)

        difficulty, sim_easy, sim_hard = self._difficulty(
            latest_user, task_vec, latest_user_msg
        )

        scores = [
            CandidateScore(model=c.model, score=r)
            for c, r in zip(candidates, rels)
        ]
        # No diagnostics worth logging when the embedding call itself never
        # produced a task vector (empty digest / gateway failure) — there is no
        # raw signal for cobaiter.calibrate to regress against, only the neutral/
        # fallback values already reflected in ``scores``/``difficulty``.
        diagnostics = None
        if task_vec is not None:
            raw_sims = raw_sims or []
            refs_per_candidate = refs_per_candidate or []
            diagnostics = ClassifierDiagnostics(
                task_text=task_text or None,
                candidate_sims={
                    c.model: s for c, s in zip(candidates, raw_sims) if s is not None
                },
                candidate_refs={
                    c.model: [_truncate_ref(r) for r in refs]
                    for c, refs, s in zip(candidates, refs_per_candidate, raw_sims)
                    if s is not None
                },
                sim_easy=sim_easy,
                sim_hard=sim_hard,
            )
        result = ClassifierResult(scores=scores, difficulty=difficulty, raw=diagnostics)
        log.debug(
            "classify: embedding_model=%s difficulty=%.2f relevance=%s",
            self._s.embedding_model, difficulty, _fmt_scores(result),
        )
        return result

    # ------------------------------------------------------------------ #
    async def _embed_and_score_relevance(
        self, task_text: str, candidates: list[ModelSpec]
    ) -> tuple[
        list[float] | None,
        list[float] | None,
        list[float | None] | None,
        list[list[str]] | None,
    ]:
        """Embed the task text (+ any not-yet-cached reference texts/exemplars)
        in ONE batched call. Returns ``(task_vec, relevance, raw_sims,
        refs_per_candidate)``; the first three may be ``None`` on failure or
        empty input — the caller degrades each independently (difficulty still
        works without reference texts, relevance just falls back to neutral).
        ``raw_sims`` is the pre-``relevance_from_sims`` top-2-mean cosine
        similarity per candidate (diagnostics only, logged for offline
        recalibration of ``embedding_rel_band``); ``refs_per_candidate`` is the
        RESOLVED reference texts (see ``_resolve_refs``) used for each
        candidate, also logged (as ``candidate_refs``) for the same offline
        recalibration to know which domain each candidate actually represented."""
        if not task_text.strip():
            return None, None, None, None
        refs_per_candidate = [_resolve_refs(c) for c in candidates]
        all_refs = [r for refs in refs_per_candidate for r in refs]
        missing_refs = [r for r in dict.fromkeys(all_refs) if r not in self._ref_vecs]
        missing_exemplars = [
            e for e in self._difficulty_exemplars if e not in self._exemplar_vecs
        ]
        batch = [task_text] + missing_refs + missing_exemplars
        try:
            vecs = await self._client.embed(self._s.embedding_model, batch)
            if len(vecs) != len(batch):
                raise ValueError(f"expected {len(batch)} embeddings, got {len(vecs)}")
        except (
            DownstreamError, httpx.HTTPError,
            KeyError, IndexError, ValueError, TypeError,
        ) as exc:
            log.warning(
                "classify: embedding_model=%s failed (%s); using neutral relevance "
                "and fallback difficulty",
                self._s.embedding_model, exc,
            )
            return None, None, None, None
        task_vec = vecs[0]
        self._ref_vecs.update(zip(missing_refs, vecs[1 : 1 + len(missing_refs)]))
        self._exemplar_vecs.update(zip(missing_exemplars, vecs[1 + len(missing_refs) :]))

        if not all_refs:
            return task_vec, None, None, refs_per_candidate
        sims = [
            _multi_sim(task_vec, [self._ref_vecs[r] for r in refs]) if refs else None
            for refs in refs_per_candidate
        ]
        log.debug(
            "classify(raw): embedding_model=%s sims=%s",
            self._s.embedding_model,
            ", ".join(
                f"{c.model}={s:.3f}" if s is not None else f"{c.model}=n/a"
                for c, s in zip(candidates, sims)
            ),
        )
        return (
            task_vec,
            relevance_from_sims(sims, self._s.embedding_rel_band),
            sims,
            refs_per_candidate,
        )

    # ------------------------------------------------------------------ #
    def _difficulty(
        self,
        latest_user: str,
        task_vec: list[float] | None,
        latest_user_msg: dict[str, Any] | None,
    ) -> tuple[float, float | None, float | None]:
        """0..1 task difficulty, plus the raw (sim_easy, sim_hard) exemplar
        similarities behind it (``None`` when the exemplar-based estimate wasn't
        used — diagnostics only, for offline recalibration of the anchors). See
        module docstring for the embedding-vs-exemplars approach; falls back to a
        token-count-only estimate when no embedding is available (empty task text
        / embedding call failed)."""
        intent = _intent_slice(latest_user)
        # 0.15 keeps a tier-1 model un-penalised in the capability fit even when
        # a tier-6 candidate sets maxTier (1/6 ≈ 0.167 is the no-penalty
        # threshold), so meta-tasks stay on the lightest (no-think) model.
        # Checked before the embedding signal: it is a reliable, deterministic
        # override (a title-generation wrapper is trivial no matter how hard
        # the embedded material looks) that similarity-to-exemplars alone isn't
        # guaranteed to catch, since head+tail truncation of a long message can
        # still carry hard-looking embedded content alongside the trivial
        # instruction.
        if _LOW_INTENT.search(intent):
            return 0.15, None, None

        easy_vecs = [self._exemplar_vecs[e] for e in self._easy_exemplars if e in self._exemplar_vecs]
        hard_vecs = [self._exemplar_vecs[e] for e in self._hard_exemplars if e in self._exemplar_vecs]
        if task_vec is None or not easy_vecs or not hard_vecs:
            return _fallback_difficulty(latest_user_msg), None, None

        sim_easy = max(_cosine(task_vec, v) for v in easy_vecs)
        sim_hard = max(_cosine(task_vec, v) for v in hard_vecs)
        ratio = sim_hard / (sim_hard + sim_easy) if (sim_hard + sim_easy) > 0 else 0.5
        difficulty = difficulty_from_ratio(
            ratio, self._s.difficulty_easy_anchor, self._s.difficulty_hard_anchor
        )
        if _ERROR_MARKERS.search(latest_user):
            difficulty += 0.05
        scoped = [latest_user_msg] if latest_user_msg else []
        if count_code_blocks(scoped) > 0:
            difficulty += 0.03
        return _clamp(difficulty, 0.1, 0.9), sim_easy, sim_hard


# --------------------------------------------------------------------------- #
# Difficulty: embedding-anchored, domain-diverse exemplars
# --------------------------------------------------------------------------- #
# Small, hand-picked exemplar sets spanning MULTIPLE domains, so difficulty
# generalises the way relevance does — via meaning, not a per-domain keyword
# list that has to be extended every time a new domain (law, economics,
# science, ...) shows up. Kept short: each is embedded once and cached, so the
# set size only affects a one-time warm-up cost, not steady-state latency.
_EASY_EXEMPLARS = [
    "こんにちは",
    "ありがとうございます",
    "今日の天気は？",
    "元気ですか？",
    "了解しました",
    "1たす1は？",
    "フランスの首都はどこですか？",
    "この単語の意味を教えて",
    "おすすめのレシピを教えて",
    "今何時ですか？",
]
# Denser coverage of NON-coding expert domains (math / physics / law / economics /
# statistics-science) alongside the two software ones. Measured on the live
# embedding model: with only 2 of 6 hard exemplars being coding-flavoured, general
# hard tasks (proofs, physics derivations, legal/economic/statistical analysis)
# scored a much weaker sim_hard than coding hard tasks (~0.50 vs ~0.58), so their
# difficulty ratio never separated from general *medium* and they never escalated.
# Adding domain-archetype (not eval-copy) hard exemplars per expert domain lifted
# general-hard escalation from 0/7 to 5/7 on a 31-prompt eval while keeping general
# medium at 0/5 (no false escalation) and coding hard unchanged at 2/6. Phrased as
# generic domain archetypes (e.g. "ある数学の予想…") — NOT verbatim of any single
# task — to avoid the exemplar-verbatim overfitting the router already suffers from.
_HARD_EXEMPLARS = [
    # math
    "ある数学の予想がなぜ重要なのか、その理論的背景と証明の難しさを厳密に論じてください",
    "与えられた命題を厳密に証明し、各ステップの論理的正当性を示してください",
    # physics
    "特殊相対性理論における同時性の相対性を、ローレンツ変換を用いて厳密に導出してください",
    "量子もつれが超光速の情報伝達に使えない理由を物理学的に厳密に説明してください",
    # law
    "この契約に潜む法的リスクと、独占禁止法など関連法規に抵触する可能性を多角的に分析してください",
    # economics
    "金融政策が長期金利・為替・資産価格に波及する経路を、メカニズムに分解して論じてください",
    # statistics / science
    "統計データの有意性の解釈と、交絡因子やバイアスの可能性を批判的に検討してください",
    "科学的な現象の背後にあるメカニズムを、理論に基づいて厳密に説明してください",
    # software
    "このシステムのメモリリークの原因を調査してデバッグしてください",
    "この分散システムのアーキテクチャ設計をレビューし、ボトルネックを特定してください",
]
_DIFFICULTY_EXEMPLARS = _EASY_EXEMPLARS + _HARD_EXEMPLARS

# Intent keywords matched against the LATEST USER MESSAGE only (both ends —
# wrappers state the action up-front, Japanese prompts often put it at the
# end). The system message is deliberately NOT scanned: for agentic clients
# (opencode, Claude Code, ...) it is generic agent/tool scaffolding, not a
# per-task instruction, and routinely describes the AGENT's own capabilities
# ("helps debug, implement, and review code") — matching those words would
# misjudge every single request from that client as high-difficulty regardless
# of what was actually asked.
_LOW_INTENT = re.compile(
    r"タイトル|題名|要約|翻訳|訳して|整形|抽出|挨拶|"
    r"title|summar|translat|extract|reformat|greeting",
    re.IGNORECASE,
)
_ERROR_MARKERS = re.compile(
    r"traceback|exception|stack ?trace|スタックトレース|エラー|error[: ]|"
    r"diff --git|panic:",
    re.IGNORECASE,
)

# Portion of the latest user message scanned for intent keywords (chars).
_INTENT_USER_SLICE = 300

# Trailing/leading whitespace and sentence-terminal punctuation, stripped before
# embedding. Measured: the embedding model is brittle to this — "こんにちは"
# (an _EASY_EXEMPLARS entry verbatim) embeds near-identically to itself
# (difficulty ~0.26), but "こんにちは。" with just a trailing 句点 no longer
# matches and scores difficulty ~0.34, a swing large enough to matter downstream.
# Only the edges are stripped (not mid-text) so code/error text is untouched.
_EDGE_PUNCT = re.compile(r"^[\s。、！？!?]+|[\s。、！？!?]+$")


def _strip_edge_punct(text: str) -> str:
    return _EDGE_PUNCT.sub("", text)


def _intent_slice(latest_user: str) -> str:
    return latest_user[:_INTENT_USER_SLICE] + "\n" + latest_user[-_INTENT_USER_SLICE:]


def _latest_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((m for m in reversed(messages) if m.get("role") == "user"), None)


def _rescale(x: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    """Linearly map ``x`` from ``[lo, hi]`` to ``[out_lo, out_hi]`` (unclamped;
    the caller clamps the final difficulty after adding bumps)."""
    span = hi - lo
    if span <= 0:
        return (out_lo + out_hi) / 2
    return out_lo + (x - lo) / span * (out_hi - out_lo)


def difficulty_from_ratio(ratio: float, easy_anchor: float, hard_anchor: float) -> float:
    """Map the raw ``sim_hard / (sim_hard + sim_easy)`` exemplar-similarity ratio
    to the 0.15..0.85 difficulty range, given the two calibration anchors.

    Public (unlike the rest of this module's helpers) and deliberately free of
    the ``+0.05``/``+0.03`` bumps and final clamp applied in ``_difficulty`` —
    ``cobaiter.calibrate`` needs this exact, invertible core formula to fit new
    anchors from judge-labelled traffic without duplicating it.
    """
    return _rescale(ratio, easy_anchor, hard_anchor, 0.15, 0.85)


def _fallback_difficulty(latest_user_msg: dict[str, Any] | None) -> float:
    """Token-count-only estimate used only when no embedding signal is
    available at all (empty digest / embedding call failed). Coarser than the
    exemplar-similarity signal but needs no network."""
    scoped = [latest_user_msg] if latest_user_msg else []
    tokens = estimate_tokens(scoped)
    if tokens < 400:
        return 0.25
    if tokens < 2000:
        return 0.4
    if tokens < 8000:
        return 0.55
    return 0.65


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_NEUTRAL_SUITABILITY = 0.5
# Number of a candidate's reference texts (task_examples, or the 1-item
# description fallback) averaged for relevance. With exactly 1 reference (the
# fallback case) this reduces to a plain cosine similarity — byte-for-byte the
# old single-description behaviour. With more, top-2-mean damps the per-example
# embedding brittleness this module already documents (see ``_EDGE_PUNCT``)
# and avoids a candidate winning purely by having more task_examples than a
# rival (more prototypes -> a higher expected max by chance alone).
_RELEVANCE_TOP_K = 2
# Cap on each reference string logged into ClassifierDiagnostics.candidate_refs
# (decision log size safety; embedding itself uses the untruncated text).
_REF_LOG_CHARS = 200


def _resolve_refs(spec: ModelSpec) -> list[str]:
    """Reference texts used to score ``spec``'s relevance: its
    ``task_examples`` if set, else a single-item list wrapping ``description``
    (byte-for-byte the old single-vector behaviour when task_examples is
    empty)."""
    examples = [e.strip() for e in spec.task_examples if e.strip()]
    if examples:
        return examples
    desc = spec.description.strip()
    return [desc] if desc else []


def _multi_sim(task_vec: list[float], vecs: list[list[float]]) -> float:
    """Top-``_RELEVANCE_TOP_K``-mean cosine similarity between the task vector
    and a candidate's reference vectors."""
    sims = sorted((_cosine(task_vec, v) for v in vecs), reverse=True)
    top = sims[:_RELEVANCE_TOP_K]
    return sum(top) / len(top)


def _truncate_ref(text: str) -> str:
    return text if len(text) <= _REF_LOG_CHARS else text[:_REF_LOG_CHARS] + "…"


def _fmt_scores(result: ClassifierResult) -> str:
    return ", ".join(f"{s.model}={s.score:.3f}" for s in result.scores)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def relevance_from_sims(sims: list[float | None], band: float) -> list[float]:
    """Map cosine similarities to 0..1 relevance.

    Public (unlike the rest of this module's helpers) so ``cobaiter.calibrate``
    can re-simulate different ``embedding_rel_band`` values against already-
    logged raw similarities without duplicating this normalisation — mirrors
    why ``difficulty_from_ratio`` is public.

    Raw cosine values sit in a model-dependent compressed band (unrelated texts
    rarely score near 0), so absolute similarity is meaningless to the router.
    Anchor on the best candidate instead: the top similarity maps to relevance
    1.0 and a full ``band`` of similarity deficit costs the whole relevance
    range. Same-domain candidates share reference texts (identical similarity)
    and stay tied at 1.0, leaving tier/cost to decide; a different-domain
    candidate falls off fast. Candidates without any reference text score
    neutral.
    """
    band = max(band, 1e-6)
    top = max(s for s in sims if s is not None)
    return [
        _NEUTRAL_SUITABILITY if s is None else _clamp(1.0 - (top - s) / band)
        for s in sims
    ]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _task_text(latest_user: str, limit: int = 800) -> str:
    """Truncate the LATEST user message to ``limit`` chars for embedding.

    Scoped to that single message — not the system message (agent
    scaffolding), not assistant replies, and not earlier user turns — so both
    relevance and difficulty judge what's being asked NOW (see module
    docstring). Keeps BOTH ends when over budget rather than only the tail:
    the HEAD carries the task instruction that often sits up-front (e.g. a
    "generate a title for the following chat" wrapper whose body is the
    embedded conversation); keeping only the tail would show the embedded
    content and hide the actual, trivial action. The tail keeps the most
    recent context. Total stays within ``limit`` bar the short elision marker.
    """
    if len(latest_user) <= limit:
        return latest_user
    head = limit // 3
    tail = limit - head
    return latest_user[:head] + "\n…\n" + latest_user[-tail:]
