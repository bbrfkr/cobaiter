"""EmbeddingClassifier: relevance + difficulty, both from embedding similarity."""

from __future__ import annotations

import json
from typing import Any

import httpx

from cobaiter.classifier import (
    _EASY_EXEMPLARS,
    _HARD_EXEMPLARS,
    EmbeddingClassifier,
)
from cobaiter.config import Settings
from cobaiter.litellm_client import LiteLLMClient
from cobaiter.schemas import ChatCompletionRequest, ModelSpec

# The fake embedding space has two orthogonal axes: "coding" and "everything
# else". Any text containing a coding marker lands on the coding axis, so a
# coding digest is similarity-1.0 to a coding description and 0.0 to a general
# one — enough to exercise the spread/relevance maths deterministically.
_CODING_MARKERS = ("code", "コード", "デバッグ", "ソフトウェア", "バグ")
# Difficulty exemplars all collapse to one of two fixed points on a SEPARATE
# axis pair, so max-similarity-to-set reduces to a single known value. A test
# digest's position on that axis is controlled via marker substrings (see
# _vec) rather than real semantics — the real embedding model's behaviour is
# already validated empirically (see conversation record), this only exercises
# the ratio/rescale/clamp/bump wiring deterministically.
_EASY_POINT = [1.0, 0.0]
_HARD_POINT = [0.0, 1.0]
_MID_POINT = [0.70710678, 0.70710678]  # equidistant -> ratio == 0.5

# Extra fixed points (unit vectors at known angles from [1.0, 0.0], the fake
# space's "coding" axis) used ONLY by the multi-prototype relevance tests
# below, so a candidate's task_examples can be given precise, distinct raw
# cosine similarities to a coding-marked task digest — something the
# EASY/HARD/MID points (shared with the difficulty tests) don't offer enough
# distinct values for.
_REL_VECTORS = {
    "REL_MARKER_90": [0.9, 0.4358898943540674],   # cos to [1,0] == 0.9
    "REL_MARKER_60": [0.6, 0.8],                   # cos to [1,0] == 0.6
    "REL_MARKER_50": [0.5, 0.8660254037844386],    # cos to [1,0] == 0.5
    "REL_MARKER_00": [0.0, 1.0],                   # cos to [1,0] == 0.0
}


def _vec(text: str) -> list[float]:
    if text in _EASY_EXEMPLARS:
        return _EASY_POINT
    if text in _HARD_EXEMPLARS:
        return _HARD_POINT
    if "EASY_TASK_MARKER" in text:
        return _EASY_POINT
    if "HARD_TASK_MARKER" in text:
        return _HARD_POINT
    if "MID_TASK_MARKER" in text:
        return _MID_POINT
    for marker, vec in _REL_VECTORS.items():
        if marker in text:
            return vec
    t = text.lower()
    if any(k in t for k in _CODING_MARKERS):
        return [1.0, 0.0]
    return [0.0, 1.0]


def _make_classifier(
    seen: list[dict[str, Any]],
) -> tuple[EmbeddingClassifier, httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        data = [
            {"object": "embedding", "index": i, "embedding": _vec(t)}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"object": "list", "data": data})

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://litellm:4000"
    )
    settings = Settings(_env_file=None)
    return EmbeddingClassifier(LiteLLMClient(http, settings), settings), http


def _req(content: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="cobaiter-auto", messages=[{"role": "user", "content": content}]
    )


# --- relevance ------------------------------------------------------------- #
async def test_relevance_prefers_matching_domain():
    """A coding task scores the coding-domain description 1.0 and the general one
    ~0 (beyond the similarity band)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m-coding", tier=3, description="ソフトウェア開発・コードレビュー向け"),
        ModelSpec(model="m-general", tier=3, description="一般的な対話・文章作成向け"),
    ]
    result = await clf.score(_req("このコードのバグを直して"), candidates)
    by = {s.model: s.score for s in result.scores}
    assert by["m-coding"] == 1.0
    assert by["m-general"] == 0.0
    assert result.difficulty is not None
    await http.aclose()


async def test_relevance_ignores_system_prompt_scaffolding():
    """Regression: an agentic client's system prompt (tool defs, a capability
    blurb mentioning "デバッグ"/"コード") must not pollute the embedded task
    text — only the user's actual message should determine domain relevance.
    Before this fix, the embedded text joined every message including system,
    so any agentic client's boilerplate system prompt would skew relevance
    toward whatever domain that boilerplate happened to describe."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m-coding", tier=3, description="ソフトウェア開発向け"),
        ModelSpec(model="m-general", tier=3, description="一般対話向け"),
    ]
    req = ChatCompletionRequest(
        model="cobaiter-auto",
        messages=[
            {"role": "system", "content": "あなたはコーディング・デバッグを支援するエージェントです。"},
            {"role": "user", "content": "調子はどう？"},
        ],
    )
    result = await clf.score(req, candidates)
    # The text embedded must be the user's message alone ("調子はどう？", edge
    # punctuation stripped -> "調子はどう"), which has no coding marker, so it
    # must land on the general axis, not coding.
    assert seen[0]["input"][0] == "調子はどう"
    by = {s.model: s.score for s in result.scores}
    assert by["m-general"] == 1.0
    assert by["m-coding"] == 0.0
    await http.aclose()


async def test_relevance_ignores_assistant_replies_and_earlier_user_turns():
    """Relevance is scoped to the LATEST user message only — neither an
    assistant's prior reply nor an earlier user turn should factor into
    relevance for the current ask (conversation-level topic stability is
    handled by sticky pinning, not by blending prior turns into every
    re-evaluation; blending would let relevance drift with how long the
    conversation has run, independent of the current request)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m-general", tier=1, description="一般対話向け")]
    req = ChatCompletionRequest(
        model="cobaiter-auto",
        messages=[
            {"role": "user", "content": "このコードのバグを直して"},
            {"role": "assistant", "content": "コードのデバッグをお手伝いします。"},
            {"role": "user", "content": "調子はどう？"},
        ],
    )
    await clf.score(req, candidates)
    assert seen[0]["input"][0] == "調子はどう"
    await http.aclose()


async def test_edge_punctuation_stripped_before_embedding():
    """Regression: trailing/leading sentence punctuation must not change the
    embedded text. Before this fix, "こんにちは。" (with a trailing 句点) no
    longer matched the "こんにちは" easy-difficulty exemplar verbatim, and the
    embedding model turned out to be brittle enough to that single character
    that difficulty swung from ~0.26 to ~0.34 for an identical greeting."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m-general", tier=1, description="一般対話向け")]
    await clf.score(_req("こんにちは。"), candidates)
    assert seen[0]["input"][0] == "こんにちは"
    await http.aclose()


async def test_embed_payload_has_descriptions_but_no_model_ids_or_tier():
    """Only the digest + descriptions (+ difficulty exemplars) are embedded;
    model names and tier never leave the process (relevance must stay
    brand/capability neutral)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m-coding", tier=3, description="ソフトウェア開発向け"),
        ModelSpec(model="m-general", tier=1, description="一般対話向け"),
    ]
    await clf.score(_req("hi"), candidates)
    body = seen[0]
    assert "ソフトウェア開発向け" in body["input"]
    assert "一般対話向け" in body["input"]
    blob = json.dumps(body)
    assert "m-coding" not in blob and "m-general" not in blob
    assert "tier" not in blob.lower()
    await http.aclose()


async def test_shared_description_is_deduped_and_cached():
    """Same-domain models share one description: it is embedded once, both models
    tie at full relevance, and later calls embed only the digest (+ exemplars,
    once) — description vectors are cached from then on."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    desc = "ソフトウェア開発・コードレビュー向け"
    candidates = [
        ModelSpec(model="think", tier=3, description=desc),
        ModelSpec(model="no-think", tier=1, description=desc),
    ]
    exemplar_count = len(_EASY_EXEMPLARS) + len(_HARD_EXEMPLARS)
    first = await clf.score(_req("コードを書いて"), candidates)
    # digest + ONE deduped description + all (not-yet-cached) exemplars.
    assert len(seen[0]["input"]) == 2 + exemplar_count
    assert {s.score for s in first.scores} == {1.0}  # tie -> tier/cost decide

    second = await clf.score(_req("コードを書いて"), candidates)
    # description + exemplar vectors now cached; only the digest is embedded.
    assert len(seen[1]["input"]) == 1
    assert {s.score for s in second.scores} == {1.0}
    await http.aclose()


async def test_no_descriptions_means_neutral_relevance_but_difficulty_still_works():
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="a"), ModelSpec(model="b")]
    result = await clf.score(_req("hello"), candidates)
    assert {s.score for s in result.scores} == {0.5}
    assert result.difficulty is not None
    await http.aclose()


# --- relevance: task_examples multi-prototype scoring ----------------------- #
async def test_relevance_uses_top2_mean_over_task_examples():
    """A candidate with 3 task_examples at raw cosine [0.9, 0.5, 0.0] to the
    task digest scores top-2-mean = (0.9+0.5)/2 = 0.7 — not diluted by the
    weak third example (plain mean would give ~0.467), and not just the best
    match either (plain max would give 0.9)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(
            model="m-multi",
            task_examples=["REL_MARKER_90 ex", "REL_MARKER_50 ex", "REL_MARKER_00 ex"],
        ),
    ]
    result = await clf.score(_req("please look at this code"), candidates)
    assert result.raw is not None
    assert abs(result.raw.candidate_sims["m-multi"] - 0.7) < 1e-6
    await http.aclose()


async def test_relevance_falls_back_to_description_when_task_examples_empty():
    """Backward compatibility pin: with task_examples empty, relevance is
    scored from `description` alone (top-1-mean of a single item == a plain
    cosine similarity) — byte-for-byte the pre-task_examples behaviour."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m-desc-only", description="REL_MARKER_60 desc")]
    result = await clf.score(_req("please look at this code"), candidates)
    assert result.raw is not None
    assert abs(result.raw.candidate_sims["m-desc-only"] - 0.6) < 1e-6
    assert result.raw.candidate_refs["m-desc-only"] == ["REL_MARKER_60 desc"]
    await http.aclose()


async def test_relevance_multi_example_candidate_does_not_win_purely_on_example_count():
    """Regression for the item-1 bias concern: a candidate with 1 example that
    matches moderately well (cos=0.6) must beat a rival with 5 examples where
    only 1 matches equally well and the other 4 are irrelevant (cos=0.0) —
    with plain max() both would tie at 0.6; top-2-mean penalises the
    many-irrelevant-examples candidate (mean of its top 2 = 0.3)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m-focused", task_examples=["REL_MARKER_60 ex"]),
        ModelSpec(
            model="m-many",
            task_examples=[
                "REL_MARKER_60 ex",
                "REL_MARKER_00 ex a",
                "REL_MARKER_00 ex b",
                "REL_MARKER_00 ex c",
                "REL_MARKER_00 ex d",
            ],
        ),
    ]
    result = await clf.score(_req("please look at this code"), candidates)
    assert result.raw is not None
    sims = result.raw.candidate_sims
    assert abs(sims["m-focused"] - 0.6) < 1e-6
    assert abs(sims["m-many"] - 0.3) < 1e-6
    assert sims["m-focused"] > sims["m-many"]
    await http.aclose()


async def test_relevance_task_examples_are_embedded_individually_and_cached():
    """Mirrors test_shared_description_is_deduped_and_cached but for
    task_examples: each distinct example text is its own embedding entry on
    first use, and only the digest is re-embedded on a later call (examples
    cached)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m", task_examples=["REL_MARKER_90 ex1", "REL_MARKER_50 ex2"]),
    ]
    exemplar_count = len(_EASY_EXEMPLARS) + len(_HARD_EXEMPLARS)
    await clf.score(_req("please look at this code"), candidates)
    # digest + 2 distinct (not-yet-cached) examples + all exemplars.
    assert len(seen[0]["input"]) == 3 + exemplar_count

    await clf.score(_req("please look at this code"), candidates)
    # examples + exemplars now cached; only the digest is embedded.
    assert len(seen[1]["input"]) == 1
    await http.aclose()


# --- difficulty: LOW_INTENT fast path (no embedding needed) ---------------- #
def _classifier() -> EmbeddingClassifier:
    return EmbeddingClassifier(client=None, settings=Settings(_env_file=None))


def _user_msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def test_difficulty_meta_task_stays_low_over_hard_content():
    """A title-generation wrapper must stay LOW difficulty no matter how hard the
    embedded conversation looks (the action is trivial, not the material). This
    is a deterministic override checked before any embedding comparison."""
    clf = _classifier()
    content = "以下の会話のタイトルを生成してください。\n" + (
        "設計とリファクタリングと原因調査の議論。" * 100
    )
    d, sim_easy, sim_hard = clf._difficulty(content, task_vec=None, latest_user_msg=_user_msg(content))
    assert d == 0.15
    assert sim_easy is None and sim_hard is None


# --- difficulty: token-count fallback (no embedding available) ------------- #
def test_difficulty_fallback_short_chat_is_low():
    clf = _classifier()
    text = "こんにちは、調子はどう？"
    d, _, _ = clf._difficulty(text, task_vec=None, latest_user_msg=_user_msg(text))
    assert d == 0.25


def test_difficulty_fallback_ignores_agentic_system_prompt_scaffolding():
    """Regression: the fallback token count must come from the LATEST USER
    MESSAGE only. Before this fix, the heuristic scanned the whole message
    list, so a large agentic system prompt (tool defs, capability blurb) would
    inflate difficulty regardless of the actual user ask. ``_difficulty`` never
    even receives the system message now — only ``latest_user_msg`` — so this
    also guards against a future regression that widens its inputs again."""
    clf = _classifier()
    text = "こんにちは"
    d, _, _ = clf._difficulty(text, task_vec=None, latest_user_msg=_user_msg(text))
    assert d == 0.25


# --- difficulty: embedding-anchored semantic ratio -------------------------- #
async def test_difficulty_semantic_ratio_at_midpoint_matches_calibrated_rescale():
    """A digest equidistant from the easy/hard exemplar clusters (ratio=0.5)
    must map through the configured anchors exactly as documented:
    ``0.15 + (ratio - easy_anchor) / (hard_anchor - easy_anchor) * (0.85 - 0.15)``."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m", description="")]
    req = _req("MID_TASK_MARKER 何かの依頼")
    result = await clf.score(req, candidates)
    s = clf._s
    expected = 0.15 + (0.5 - s.difficulty_easy_anchor) / (
        s.difficulty_hard_anchor - s.difficulty_easy_anchor
    ) * (0.85 - 0.15)
    assert abs(result.difficulty - expected) < 1e-6
    await http.aclose()


async def test_difficulty_semantic_ratio_near_easy_exemplars_is_low():
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m", description="")]
    result = await clf.score(_req("EASY_TASK_MARKER"), candidates)
    assert result.difficulty <= 0.2


async def test_difficulty_semantic_ratio_near_hard_exemplars_is_high():
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m", description="")]
    result = await clf.score(_req("HARD_TASK_MARKER"), candidates)
    assert result.difficulty >= 0.8


async def test_difficulty_error_marker_and_code_fence_bump_the_semantic_score():
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m", description="")]
    base = (await clf.score(_req("MID_TASK_MARKER"), candidates)).difficulty
    with_error = (
        await clf.score(_req("MID_TASK_MARKER\nTraceback (most recent call last):"), candidates)
    ).difficulty
    with_code = (
        await clf.score(_req("MID_TASK_MARKER\n```\nx = 1\n```"), candidates)
    ).difficulty
    assert with_error > base
    assert with_code > base
    await http.aclose()


async def test_difficulty_uses_latest_user_message_only_not_accumulated_digest():
    """Regression: difficulty must reflect the LATEST user turn, not the
    multi-turn relevance digest. Before this fix, a growing conversation could
    drag a trivial current message's difficulty upward based on what an
    EARLIER turn discussed (e.g. a real "よろしく" + "こんにちは" two-turn
    conversation measurably scored higher than "こんにちは" alone, because
    difficulty and relevance shared one digest embedding spanning all user
    turns)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m", description="")]
    req = ChatCompletionRequest(
        model="cobaiter-auto",
        messages=[
            {"role": "user", "content": "HARD_TASK_MARKER 前のターン"},
            {"role": "assistant", "content": "了解しました"},
            {"role": "user", "content": "EASY_TASK_MARKER"},
        ],
    )
    result = await clf.score(req, candidates)
    assert result.difficulty <= 0.2  # reflects only the latest (easy) turn
    await http.aclose()


async def test_difficulty_falls_back_to_token_count_when_embedding_fails():
    """On an embedding failure, difficulty must not be lost entirely — it falls
    back to the token-count heuristic instead of e.g. raising or defaulting to
    a fixed neutral value regardless of input."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://litellm:4000"
    )
    settings = Settings(_env_file=None)
    clf = EmbeddingClassifier(LiteLLMClient(http, settings), settings)
    candidates = [ModelSpec(model="m", description="general")]
    result = await clf.score(_req("こんにちは"), candidates)
    assert result.difficulty == 0.25  # short-chat fallback band
    assert result.raw is None  # nothing informative to log for decision logging
    await http.aclose()


# --- raw diagnostics (decision-logging input, see cobaiter.calibrate) ------ #
async def test_raw_diagnostics_populated_on_success():
    """Coding-marker text lands exactly on the fake space's coding axis, which
    (see ``_vec``) coincides with the easy-difficulty axis point — so this also
    pins down the raw exemplar similarities, not just the raw candidate cosines."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [
        ModelSpec(model="m-coding", tier=3, description="ソフトウェア開発向け"),
        ModelSpec(model="m-general", tier=3, description="一般対話向け"),
    ]
    result = await clf.score(_req("このコードのバグを直して"), candidates)
    assert result.raw is not None
    assert result.raw.task_text == "このコードのバグを直して"
    assert result.raw.candidate_sims == {"m-coding": 1.0, "m-general": 0.0}
    assert result.raw.candidate_refs == {
        "m-coding": ["ソフトウェア開発向け"], "m-general": ["一般対話向け"],
    }
    assert result.raw.sim_easy == 1.0
    assert result.raw.sim_hard == 0.0
    await http.aclose()


async def test_raw_diagnostics_sim_easy_hard_none_for_low_intent_meta_task():
    """The LOW_INTENT fast path never touches the exemplar embeddings, so raw
    diagnostics must reflect that (no sim_easy/sim_hard to log/calibrate from),
    even though relevance still ran (task_text/candidate_sims are populated)."""
    seen: list[dict[str, Any]] = []
    clf, http = _make_classifier(seen)
    candidates = [ModelSpec(model="m-general", tier=1, description="一般対話向け")]
    result = await clf.score(_req("この会話のタイトルを生成してください"), candidates)
    assert result.difficulty == 0.15
    assert result.raw is not None
    assert result.raw.sim_easy is None and result.raw.sim_hard is None
    await http.aclose()
