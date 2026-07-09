"""Golden-set regression test against the REAL embedding classifier + REAL
``models.yaml`` registry (unlike tests/test_classifier.py, which uses a
deterministic FAKE embedding space so relevance/difficulty maths can be
unit-tested without a network call).

Requires a live LiteLLM gateway with ``/v1/embeddings`` configured (see
``COBAITER_EMBEDDING_MODEL``). This is deliberately NOT run as part of the
default ``uv run pytest`` — it is collected (so it's visible, not silently
missing) but SKIPPED unless explicitly opted into:

    docker compose up -d valkey litellm   # or point COBAITER_LITELLM_BASE_URL elsewhere
    COBAITER_RUN_GOLDEN=1 uv run pytest -m golden -v

Run this whenever ``models.yaml`` (descriptions/task_examples), the embedding
model, or ``COBAITER_EMBEDDING_REL_BAND``/anchors change, to catch a routing
regression against real embeddings instead of the fake space. See
``tests/fixtures/routing_cases.yaml`` for the case set (it must be kept in
sync with ``models.yaml``'s model set by hand) and the ``cobaiter-multi-
domain-routing`` project memory for why domain-boundary cases are included.
"""

from __future__ import annotations

import os
from pathlib import Path

import fakeredis.aioredis as fakeaioredis
import pytest
import yaml

from cobaiter.classifier import EmbeddingClassifier
from cobaiter.config import get_settings
from cobaiter.litellm_client import LiteLLMClient
from cobaiter.registry import load_model_registry
from cobaiter.router import RouteEngine
from cobaiter.store import Store

from conftest import user_req

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        os.environ.get("COBAITER_RUN_GOLDEN") != "1",
        reason="requires a live LiteLLM/embedding backend; set COBAITER_RUN_GOLDEN=1",
    ),
]

_FIXTURES = Path(__file__).parent / "fixtures" / "routing_cases.yaml"
# The production registry file, read directly rather than via
# settings.models_config (whose default is empty / a container path) so this
# test always exercises the actual file that ships with the repo, regardless
# of what a developer's local .env happens to point at.
_MODELS_YAML = Path(__file__).resolve().parents[1] / "models.yaml"


def _load_cases() -> list[dict]:
    doc = yaml.safe_load(_FIXTURES.read_text(encoding="utf-8"))
    return doc["cases"]


@pytest.fixture
async def golden_engine():
    settings = get_settings()
    client = LiteLLMClient.create(settings)
    classifier = EmbeddingClassifier(client, settings)
    # Conversation state doesn't need to be live — only the embedding calls
    # need a real gateway — so fakeredis is enough for the Store.
    store = Store(fakeaioredis.FakeRedis(decode_responses=True), settings)
    await store.replace_models(load_model_registry(_MODELS_YAML))
    engine = RouteEngine(store, client, classifier, settings)
    yield engine
    await store.close()
    await client.close()


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
async def test_golden_routing_case(golden_engine, case):
    # Full production pipeline: constraint filter -> classifier -> capability-
    # fit -> cost/tier re-ranking, exercised end to end (not a partial slice).
    decision = await golden_engine.decide(
        user_req(case["message"]), header_id=case["name"]
    )
    acceptable = case.get("acceptable_models") or []
    avoid = case.get("avoid_models") or []
    if acceptable:
        assert decision.model in acceptable, (
            f"{case['name']}: routed to {decision.model!r} (route={decision.route.value}), "
            f"expected one of {acceptable}"
        )
    for bad in avoid:
        assert decision.model != bad, (
            f"{case['name']}: routed to avoided model {bad!r} (route={decision.route.value})"
        )
