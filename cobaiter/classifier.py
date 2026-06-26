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
    "You assess a user's task against candidate models. Each candidate has only a "
    "`description` (what it is for) and an opaque `model` id — judge ONLY by the "
    "description; the id is meaningless. Ignore cost, speed, and capability — those "
    "are handled elsewhere.\n"
    "Return TWO things as JSON:\n"
    "1. `difficulty`: ONE number 0.0-1.0 for the whole task — how much skill and "
    "depth of reasoning it demands. 0.0 = trivial (greeting, simple lookup), 0.5 = "
    "moderate, 1.0 = very hard (deep/expert reasoning). Judge from what is asked, "
    "not from message length.\n"
    "2. `relevance` per candidate: 0.0-1.0 for how well the candidate's described "
    "use-case DOMAIN matches the task's TOPIC — not how capable, advanced, or fast "
    "it is. Judge ONLY the subject domain and IGNORE any wording about how hard, "
    "advanced, or complex a model is (a hard task does NOT make an 'advanced'-"
    "sounding model relevant). If a description says the model is unsuitable for / "
    "only for a certain domain, score it ~0.0 on tasks OUTSIDE that domain, however "
    "impressive the wording. 1.0 = the description squarely covers this topic; 0.0 = "
    "clearly a different domain (e.g. a coding model on a math-proof or translation "
    "task); use graded values for partial fit. Spread the values — do NOT make "
    "everything 0 or 1.\n"
    "Output ONLY this compact JSON: "
    "{\"difficulty\":<float>,\"scores\":[{\"model\":\"<id>\",\"relevance\":<float>}, ...]}"
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
            # Diagnostic: the alias->real mapping actually sent, and the classifier's
            # verbatim reply. Confirms whether anonymisation is live (the reply should
            # reference candidate-N, never real model names) and exposes degenerate
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
        # Anonymise model names: the classifier is itself an LLM and would
        # otherwise let brand priors (e.g. a famous model name) override the
        # neutral description/tier. Opaque aliases force a description-only verdict.
        aliases = _aliases(candidates)
        catalog = [
            {"model": aliases[i], "description": c.description}
            for i, c in enumerate(candidates)
        ]
        digest = _digest_conversation(req.messages)
        user_msg = (
            "Candidate models:\n"
            + json.dumps(catalog)
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
        # Map the opaque aliases emitted by the classifier back to real names.
        aliases = _aliases(candidates)
        alias_to_model = {aliases[i]: c.model for i, c in enumerate(candidates)}
        scores: list[CandidateScore] = []
        seen: set[str] = set()
        for item in obj.get("scores", []):
            model = alias_to_model.get(item.get("model"))
            if model is not None and model not in seen:
                # ``relevance`` is the new key; tolerate a stray ``score`` too.
                raw = item.get("relevance", item.get("score"))
                scores.append(CandidateScore(model=model, score=_clamp(float(raw))))
                seen.add(model)
        # Ensure every candidate has a relevance; fill gaps with neutral suitability.
        for c in candidates:
            if c.model not in seen:
                scores.append(CandidateScore(model=c.model, score=_NEUTRAL_SUITABILITY))
        if not scores:
            raise ValueError("classifier produced no usable scores")
        diff_raw = obj.get("difficulty")
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


def _digest_conversation(messages: list[dict[str, Any]], limit: int = 4000) -> str:
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
