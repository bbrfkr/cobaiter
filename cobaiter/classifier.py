"""Lightweight classifier: score candidate models for the current context.

The classifier is only invoked when the constraint filter leaves *more than one*
candidate (initial routing) or when the stage-1 soft-gate fires (re-evaluation).
It returns a 0..1 suitability score per candidate; the router applies the
selection / hysteresis logic on top.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import Settings
from .litellm_client import DownstreamError, LiteLLMClient
from .schemas import (
    CandidateScore,
    ChatCompletionRequest,
    ClassifierResult,
    ModelSpec,
)

log = logging.getLogger("cobaiter")

# The classifier judges only *semantics* — it does NOT emit the final calibrated
# score. It returns (a) one task-difficulty scalar and (b) a per-candidate use-case
# relevance; the router turns those into a suitability via tier-vs-difficulty
# capability-fit. Splitting the work this way keeps each LLM judgement simple (a
# relative call it is good at) and moves the float arithmetic into reproducible
# code, so scores spread across 0..1 instead of collapsing to {0, 1}. Cost and
# tier are deliberately NOT shown — relevance must stay capability/price neutral.
_SYSTEM = (
    "You score candidate models for a user's task. Judge each candidate ONLY by its "
    "described use-case DOMAIN; ignore cost, speed, capability, and any wording about "
    "how advanced/hard/complex a model is (a hard task does NOT make an 'advanced'-"
    "sounding model relevant).\n"
    "Return ONLY compact JSON: {\"d\":<float>,\"r\":[<float>,...]}\n"
    "- d = task difficulty 0.0-1.0 (0=trivial greeting/lookup, 0.5=moderate, "
    "1.0=deep/expert reasoning), judged from what is asked, not message length.\n"
    "- r = one relevance 0.0-1.0 PER candidate, in the SAME ORDER as the numbered "
    "list, for how well its domain matches the task topic (1.0=squarely covers it, "
    "0.0=clearly a different domain, e.g. a coding model on a translation task). If a "
    "description says the model is only for / unsuitable for a domain, score ~0.0 "
    "outside that domain. Spread the values; do NOT make everything 0 or 1."
)


class Classifier:
    def __init__(self, client: LiteLLMClient, settings: Settings) -> None:
        self._client = client
        self._s = settings

    async def score(
        self,
        req: ChatCompletionRequest,
        candidates: list[ModelSpec],
    ) -> ClassifierResult:
        """Return suitability scores for ``candidates``.

        On any downstream/parse failure, falls back to a uniform-suitability
        heuristic so routing never hard-fails on the classifier; the router's
        deterministic cost/tier re-ranking then picks the cheapest, lightest model.
        """
        if not candidates:
            return ClassifierResult(scores=[])
        names = [c.model for c in candidates]
        try:
            payload = self._build_payload(req, candidates)
            data = await self._client.chat(payload)
            text = _message_text(data)
            # Diagnostic: the position->real-model mapping (the order the numbered
            # catalog was sent in, i.e. how the classifier's ``r`` array maps back to
            # models) plus the classifier's verbatim reply. Exposes degenerate
            # winner-take-all output from a weak classifier model.
            alias_map = {a: c.model for a, c in zip(_aliases(candidates), candidates)}
            log.debug(
                "classify(raw): classifier_model=%s alias_map=%s response=%r",
                self._s.classifier_model, alias_map, text,
            )
            result = self._parse(text, candidates)
            log.debug(
                "classify: classifier_model=%s candidates=%s scores=%s",
                self._s.classifier_model, names, _fmt_scores(result),
            )
            return result
        except (DownstreamError, httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            result = self._heuristic(candidates)
            log.warning(
                "classify: classifier_model=%s failed (%s); using heuristic scores=%s",
                self._s.classifier_model, exc, _fmt_scores(result),
            )
            return result

    # ------------------------------------------------------------------ #
    def _build_payload(
        self, req: ChatCompletionRequest, candidates: list[ModelSpec]
    ) -> dict[str, Any]:
        # Anonymise model names: the classifier is itself an LLM and would otherwise
        # let brand priors (e.g. a famous model name) override the neutral
        # description. Show only a numbered list of descriptions — no model ids — and
        # take back relevance as an array aligned to that order. This both forces a
        # description-only verdict and keeps the prompt (input tokens) small.
        catalog = "\n".join(
            f"{i + 1}. {c.description}" for i, c in enumerate(candidates)
        )
        digest = _digest_conversation(req.messages, self._s.classifier_digest_chars)
        user_msg = (
            "Candidate models (numbered):\n"
            + catalog
            + "\n\nConversation (most recent last):\n"
            + digest
        )
        return {
            "model": self._s.classifier_model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens": self._s.classifier_max_tokens,
            "stream": False,
        }

    def _parse(self, text: str, candidates: list[ModelSpec]) -> ClassifierResult:
        obj = json.loads(_extract_json(text))
        # ``r`` is a per-candidate relevance array, positionally aligned to the
        # numbered catalog sent in _build_payload.
        rels = obj.get("r")
        if not isinstance(rels, list) or not rels:
            raise ValueError("classifier produced no usable scores")
        scores: list[CandidateScore] = []
        for i, c in enumerate(candidates):
            raw = rels[i] if i < len(rels) else None
            try:
                # A missing/garbled entry falls back to neutral, never hard-fails.
                score = _clamp(float(raw))
            except (TypeError, ValueError):
                score = _NEUTRAL_SUITABILITY
            scores.append(CandidateScore(model=c.model, score=score))
        diff_raw = obj.get("d")
        difficulty = _clamp(float(diff_raw)) if diff_raw is not None else None
        return ClassifierResult(scores=scores, difficulty=difficulty)

    def _heuristic(self, candidates: list[ModelSpec]) -> ClassifierResult:
        # No signal available: treat every candidate as equally relevant and give no
        # difficulty estimate, so the router skips capability-fit and lets its
        # deterministic cost/tier re-ranking decide.
        return ClassifierResult(
            scores=[
                CandidateScore(model=c.model, score=_NEUTRAL_SUITABILITY)
                for c in candidates
            ],
            difficulty=None,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_NEUTRAL_SUITABILITY = 0.5


def _aliases(candidates: list[ModelSpec]) -> list[str]:
    """Opaque per-position labels shown to the classifier in place of real model
    names, so it cannot let brand priors override the description/tier verdict.
    Position-based, so build- and parse-time mappings agree given the same list."""
    return [f"candidate-{i + 1}" for i in range(len(candidates))]


def _fmt_scores(result: ClassifierResult) -> str:
    return ", ".join(f"{s.model}={s.score:.3f}" for s in result.scores)


def _message_text(data: dict[str, Any]) -> str:
    """Extract the assistant text from a completion response.

    Prefers ``content``; for "thinking" models that may leave ``content`` empty,
    falls back to ``reasoning_content`` so an embedded JSON verdict can still be
    recovered.
    """
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if content:
        return content
    return msg.get("reasoning_content") or ""


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _digest_conversation(messages: list[dict[str, Any]], limit: int = 800) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "[non-text]")
                for p in content
                if isinstance(p, dict)
            )
        parts.append(f"{role}: {content}")
    text = "\n".join(parts)
    return text[-limit:]


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in classifier output")
    return text[start : end + 1]
