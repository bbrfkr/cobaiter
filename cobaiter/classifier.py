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

_SYSTEM = (
    "You are a cost-aware routing classifier. Given a conversation and a list of "
    "candidate LLMs (each with a capability tier and a free-text description of what "
    "it is best suited for), score how appropriate each candidate is for handling "
    "this conversation, from 0.0 (poor) to 1.0 (ideal). Goal: pick the CHEAPEST model "
    "that is clearly SUFFICIENT for the task — do NOT reward raw capability. "
    "First match the conversation's actual use-case to each candidate's description; "
    "a model whose described purpose does not fit must score low regardless of tier. "
    "Among models that fit the use-case, prefer lighter (cheaper, faster) tiers and "
    "give them the HIGHEST score whenever they are sufficient. Reserve higher scores "
    "for a stronger (rich) tier only when the task CLEARLY demands it — genuinely hard "
    "reasoning, long or complex multi-file coding, or nuanced long-form writing. "
    "For simple, short, factual, or casual exchanges (e.g. greetings, one-line "
    "questions), a light model must outscore the rich model. When two tiers are "
    "otherwise equally suitable, break the tie in favor of the lighter (cheaper) one. "
    "Respond ONLY with compact JSON: "
    "{\"scores\":[{\"model\":\"<name>\",\"score\":<float>}, ...]} "
    "covering exactly the candidate models."
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

        On any downstream/parse failure, falls back to tier-based heuristic scores
        so routing never hard-fails on the classifier.
        """
        if not candidates:
            return ClassifierResult(scores=[])
        names = [c.model for c in candidates]
        try:
            payload = self._build_payload(req, candidates)
            data = await self._client.chat(payload)
            text = _message_text(data)
            result = self._parse(text, candidates)
            log.info(
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
        catalog = [
            {"model": c.model, "tier": c.tier, "description": c.description}
            for c in candidates
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
        valid = {c.model for c in candidates}
        scores: list[CandidateScore] = []
        seen: set[str] = set()
        for item in obj.get("scores", []):
            model = item.get("model")
            if model in valid and model not in seen:
                scores.append(
                    CandidateScore(model=model, score=_clamp(float(item["score"])))
                )
                seen.add(model)
        # Ensure every candidate has a score; fill gaps heuristically.
        for c in candidates:
            if c.model not in seen:
                scores.append(CandidateScore(model=c.model, score=_tier_score(c.tier)))
        if not scores:
            raise ValueError("classifier produced no usable scores")
        return ClassifierResult(scores=scores)

    def _heuristic(self, candidates: list[ModelSpec]) -> ClassifierResult:
        return ClassifierResult(
            scores=[
                CandidateScore(model=c.model, score=_tier_score(c.tier))
                for c in candidates
            ]
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_TIER_BASELINE = {"rich": 0.8, "light": 0.5, "openweight": 0.4}


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


def _tier_score(tier: str) -> float:
    return _TIER_BASELINE.get(tier, 0.5)


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
