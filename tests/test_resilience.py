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
                {"message": {"content": _json.dumps({"difficulty": 0.5, "scores": []})}}
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
    await classifier.score(req, candidates)
    assert "code generation and debugging" in seen["user"]
    assert "general chat" in seen["user"]
    catalog = _json.loads(seen["user"].split("Candidate models:\n", 1)[1].split("\n\n", 1)[0])
    # Real model names are anonymised to opaque aliases (avoids brand bias) and the
    # catalog carries description ONLY — tier/cost stay out of the classifier.
    assert all(set(c.keys()) == {"model", "description"} for c in catalog)
    assert "m-coding" not in seen["user"] and "m-general" not in seen["user"]
    assert all(c["model"].startswith("candidate-") for c in catalog)
    # The prompt asks for a difficulty estimate (consumed by the router's
    # capability-fit), not a final score.
    assert "difficulty" in seen["system"]
    await http.aclose()


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
