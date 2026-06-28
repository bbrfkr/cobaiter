"""Downstream-dependency resilience: never surface an opaque 500.

* LiteLLM (model gateway) unreachable -> clean 502, classifier degrades to heuristic.
* Valkey (state store) unreachable -> clean 503.
"""

from __future__ import annotations

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from cobaiter.classifier import Classifier
from cobaiter.config import Settings
from cobaiter.litellm_client import LiteLLMClient
from cobaiter.store import Store
from starlette.testclient import TestClient

from cobaiter.app import create_app
from conftest import FakeClassifier, FakeClient


def _settings() -> Settings:
    return Settings(_env_file=None, min_dwell_turns=3, soft_recheck_every=4)


# --- LiteLLM transport failure ------------------------------------------- #
async def test_litellm_transport_error_becomes_502_downstream():
    """A connection error to the gateway is wrapped as a 502 DownstreamError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no address associated with hostname")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://litellm:4000")
    client = LiteLLMClient(http, _settings())
    from cobaiter.litellm_client import DownstreamError

    with pytest.raises(DownstreamError) as ei:
        await client.chat({"model": "x", "messages": []})
    assert ei.value.status == 502
    assert ei.value.kind == "none"
    await http.aclose()


async def test_classifier_degrades_on_transport_error():
    """Classifier must fall back to heuristic scores, not raise, when the gateway dies."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://litellm:4000")
    client = LiteLLMClient(http, _settings())
    classifier = Classifier(client, _settings())

    from cobaiter.store import default_seed_specs

    candidates = default_seed_specs()[:2]
    from cobaiter.schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="cobaiter-auto", messages=[{"role": "user", "content": "hi"}]
    )
    result = await classifier.score(req, candidates)
    assert {s.model for s in result.scores} == {c.model for c in candidates}
    await http.aclose()


async def test_classifier_payload_includes_description_but_not_tier():
    """Each candidate's use-case description must reach the classifier prompt, but
    ``tier`` must NOT: the classifier judges use-case relevance only, and the router
    folds tier (capability-fit) in deterministically downstream."""
    import json as _json

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        seen["system"] = body["messages"][0]["content"]
        seen["user"] = body["messages"][-1]["content"]
        payload = {
            "choices": [
                {"message": {"content": _json.dumps({"d": 0.5, "r": [0.1, 0.2]})}}
            ]
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://litellm:4000")
    client = LiteLLMClient(http, _settings())
    classifier = Classifier(client, _settings())

    from cobaiter.schemas import ChatCompletionRequest, ModelSpec

    candidates = [
        ModelSpec(model="m-coding", tier=3, description="code generation and debugging"),
        ModelSpec(model="m-general", tier=1, description="general chat"),
    ]
    req = ChatCompletionRequest(
        model="cobaiter-auto", messages=[{"role": "user", "content": "hi"}]
    )
    result = await classifier.score(req, candidates)
    # Descriptions reach the classifier as a numbered list (no JSON, no model ids):
    # this anonymises brand (avoids bias) and keeps the prompt small.
    assert "1. code generation and debugging" in seen["user"]
    assert "2. general chat" in seen["user"]
    # Real model names and tier values stay OUT of the classifier prompt.
    assert "m-coding" not in seen["user"] and "m-general" not in seen["user"]
    assert "tier" not in seen["user"].lower()
    # The prompt asks for a difficulty estimate (consumed by the router's
    # capability-fit), not a final score.
    assert "difficulty" in seen["system"]
    # ``r`` is parsed positionally back onto the candidates, ``d`` is the difficulty.
    by = {s.model: s.score for s in result.scores}
    assert by == {"m-coding": 0.1, "m-general": 0.2}
    assert result.difficulty == 0.5
    await http.aclose()


def test_digest_keeps_instruction_head_and_recent_tail():
    """When the conversation exceeds the budget the digest must keep BOTH ends.

    A title-generation request states its (trivial) action up-front and embeds the
    real conversation as the body; a tail-only digest would hide that action and
    show only the embedded (possibly hard) content, inflating the difficulty
    estimate. Keeping the head preserves the actual instruction."""
    from cobaiter.classifier import _digest_conversation

    head_instruction = "Generate a concise 3-5 word title for the following chat"
    tail_marker = "MOST_RECENT_LINE"
    messages = [
        {"role": "user", "content": head_instruction + " " + ("filler " * 500) + tail_marker}
    ]
    digest = _digest_conversation(messages, limit=300)
    assert "title" in digest  # the up-front action survives
    assert tail_marker in digest  # the recent tail survives
    assert len(digest) <= 300 + len("\n…\n")


# --- Valkey state-store failure ------------------------------------------ #
class _BrokenRedis:
    """Every operation raises a redis ConnectionError."""

    async def get(self, *a, **k):
        raise RedisConnectionError("valkey down")

    async def set(self, *a, **k):
        raise RedisConnectionError("valkey down")

    async def hgetall(self, *a, **k):
        raise RedisConnectionError("valkey down")

    async def hget(self, *a, **k):
        raise RedisConnectionError("valkey down")

    async def ping(self, *a, **k):
        raise RedisConnectionError("valkey down")

    async def aclose(self):
        pass


def test_valkey_unavailable_returns_503():
    settings = _settings()
    store = Store(_BrokenRedis(), settings)
    app = create_app(
        settings=settings,
        store=store,
        client=FakeClient(),
        classifier=FakeClassifier({"claude-opus-4-8": 0.9}),
        seed=False,
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/chat/completions",
            headers={"x-cobaiter-conversation-id": "c1"},
            json={"model": "cobaiter-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
        assert "state store unavailable" in r.json()["detail"]
        # healthz stays up but reports degraded.
        assert tc.get("/healthz").json()["status"] == "degraded"
