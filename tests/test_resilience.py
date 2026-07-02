"""Downstream-dependency resilience: never surface an opaque 500.

* LiteLLM (model gateway) unreachable -> clean 502, classifier degrades to heuristic.
* Valkey (state store) unreachable -> clean 503.
"""

from __future__ import annotations

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from cobaiter.classifier import EmbeddingClassifier
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
    """Classifier must fall back to neutral relevance, not raise, when the gateway
    dies; the heuristic difficulty is still produced (it needs no network)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://litellm:4000")
    client = LiteLLMClient(http, _settings())
    classifier = EmbeddingClassifier(client, _settings())

    from cobaiter.schemas import ChatCompletionRequest, ModelSpec

    candidates = [
        ModelSpec(model="m-coding", tier=3, description="code generation"),
        ModelSpec(model="m-general", tier=1, description="general chat"),
    ]
    req = ChatCompletionRequest(
        model="cobaiter-auto", messages=[{"role": "user", "content": "hi"}]
    )
    result = await classifier.score(req, candidates)
    assert {s.model for s in result.scores} == {c.model for c in candidates}
    # Neutral relevance for everyone; difficulty still estimated heuristically.
    assert {s.score for s in result.scores} == {0.5}
    assert result.difficulty is not None
    await http.aclose()


def test_task_text_keeps_instruction_head_and_recent_tail():
    """When the latest user message exceeds the budget it must keep BOTH ends.

    A title-generation request states its (trivial) action up-front and embeds the
    real conversation as the body; a tail-only truncation would hide that action
    and show only the embedded (possibly hard) content, inflating the difficulty
    estimate. Keeping the head preserves the actual instruction."""
    from cobaiter.classifier import _task_text

    head_instruction = "Generate a concise 3-5 word title for the following chat"
    tail_marker = "MOST_RECENT_LINE"
    latest_user = head_instruction + " " + ("filler " * 500) + tail_marker
    text = _task_text(latest_user, limit=300)
    assert "title" in text  # the up-front action survives
    assert tail_marker in text  # the recent tail survives
    assert len(text) <= 300 + len("\n…\n")


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
