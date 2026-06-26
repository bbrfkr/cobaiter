"""End-to-end app tests via Starlette TestClient with injected fakes."""

from __future__ import annotations

import fakeredis.aioredis as fakeaioredis
import pytest
from starlette.testclient import TestClient

from cobaiter.app import create_app
from cobaiter.config import Settings
from cobaiter.litellm_client import DownstreamError
from cobaiter.store import Store

from conftest import FakeClassifier, FakeClient


@pytest.fixture
def app_ctx():
    settings = Settings(
        _env_file=None, min_dwell_turns=3, soft_recheck_every=4, switch_margin=0.15
    )
    store = Store(fakeaioredis.FakeRedis(decode_responses=True), settings)
    client = FakeClient()
    classifier = FakeClassifier({"claude-opus-4-8": 0.9})
    app = create_app(
        settings=settings, store=store, client=client, classifier=classifier, seed=True
    )
    with TestClient(app) as tc:
        yield tc, client, classifier


def _chat(tc, conv, content="hello", stream=False):
    return tc.post(
        "/v1/chat/completions",
        headers={"x-cobaiter-conversation-id": conv},
        json={"model": "cobaiter-auto", "messages": [{"role": "user", "content": content}], "stream": stream},
    )


def test_health_and_models(app_ctx):
    tc, _, _ = app_ctx
    assert tc.get("/healthz").json()["status"] == "ok"
    ids = [m["id"] for m in tc.get("/v1/models").json()["data"]]
    assert "cobaiter-auto" in ids
    assert "claude-opus-4-8" in ids


def test_auto_select_and_sticky(app_ctx):
    tc, _, classifier = app_ctx
    r1 = _chat(tc, "c1")
    assert r1.status_code == 200
    assert r1.headers["x-cobaiter-route"] == "classifier-select"
    assert r1.headers["x-cobaiter-model"] == "claude-opus-4-8"
    assert r1.json()["choices"][0]["message"]["content"] == "reply::claude-opus-4-8"

    r2 = _chat(tc, "c1")
    assert r2.headers["x-cobaiter-route"] == "pinned"
    assert r2.headers["x-cobaiter-model"] == "claude-opus-4-8"


def test_app_level_failover_on_quota(app_ctx):
    tc, client, _ = app_ctx
    # Pin to opus, then make opus raise a quota error downstream.
    _chat(tc, "c1")
    client.errors["claude-opus-4-8"] = DownstreamError(
        429, "quota exceeded", "quota"
    )
    r = _chat(tc, "c1")
    assert r.status_code == 200
    assert r.headers["x-cobaiter-route"] == "failover"
    assert r.headers["x-cobaiter-model"] == "claude-sonnet-4-6"
    assert r.json()["model"] == "claude-sonnet-4-6"


def test_bad_request_is_not_failed_over(app_ctx):
    tc, client, _ = app_ctx
    _chat(tc, "c1")
    client.errors["claude-opus-4-8"] = DownstreamError(400, "invalid", "none")
    r = _chat(tc, "c1")
    assert r.status_code == 502  # propagated, not failed over


def test_streaming_passthrough(app_ctx):
    tc, _, _ = app_ctx
    r = _chat(tc, "c1", stream=True)
    assert r.status_code == 200
    assert r.headers["x-cobaiter-model"] == "claude-opus-4-8"
    assert "data:" in r.text


def test_admin_model_and_conversation(app_ctx):
    tc, _, _ = app_ctx
    put = tc.put(
        "/admin/models",
        json={"model": "glm-4", "context_window": 128000, "cost": 0, "tier": 1, "supports_tools": True},
    )
    assert put.json()["status"] == "ok"

    _chat(tc, "c1")
    conv = tc.get("/admin/conversations/id:c1")
    assert conv.status_code == 200
    assert conv.json()["model"] == "claude-opus-4-8"

    assert tc.delete("/admin/conversations/id:c1").json()["deleted"] is True
    assert tc.get("/admin/conversations/id:c1").status_code == 404
