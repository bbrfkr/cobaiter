"""Externally-managed model registry: load from config + reconcile store."""

from __future__ import annotations

import fakeredis.aioredis as fakeaioredis
import pytest

from cobaiter.config import Settings
from cobaiter.registry import RegistryConfigError, load_model_registry
from cobaiter.store import Store, default_seed_specs

_YAML = """
models:
  - model: bbrfkr-llm-general
    tier: rich
    description: "general-purpose chat and reasoning"
    context_window: 32768
    multimodal: true
    supports_tools: true
    is_local: true
    fallback_chain: [bbrfkr-llm-general-no-think]
  - model: bbrfkr-llm-general-no-think
    tier: light
    context_window: 32768
    multimodal: false
    supports_tools: true
    is_local: true
"""


def test_load_registry_from_yaml(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(_YAML)
    specs = load_model_registry(f)
    by = {s.model: s for s in specs}
    assert set(by) == {"bbrfkr-llm-general", "bbrfkr-llm-general-no-think"}
    assert by["bbrfkr-llm-general"].tier == "rich"
    assert by["bbrfkr-llm-general"].description == "general-purpose chat and reasoning"
    assert by["bbrfkr-llm-general"].fallback_chain == ["bbrfkr-llm-general-no-think"]
    assert by["bbrfkr-llm-general-no-think"].multimodal is False
    # description is optional: entries that omit it default to empty string.
    assert by["bbrfkr-llm-general-no-think"].description == ""


def test_missing_config_raises(tmp_path):
    with pytest.raises(RegistryConfigError):
        load_model_registry(tmp_path / "nope.yaml")


def test_empty_models_raises(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("models: []\n")
    with pytest.raises(RegistryConfigError):
        load_model_registry(f)


async def test_replace_models_clears_stale_entries(tmp_path):
    settings = Settings(_env_file=None)
    store = Store(fakeaioredis.FakeRedis(decode_responses=True), settings)
    # Pre-populate with the old default seed (claude-*), simulating a prior config.
    await store.seed_models(default_seed_specs())
    assert any(s.model == "claude-opus-4-8" for s in await store.list_models())

    # Reconcile to a new registry; stale claude-* entries must be gone.
    f = tmp_path / "models.yaml"
    f.write_text(_YAML)
    written = await store.replace_models(load_model_registry(f))
    assert written == 2
    models = {s.model for s in await store.list_models()}
    assert models == {"bbrfkr-llm-general", "bbrfkr-llm-general-no-think"}
    assert await store.models_for_tier("rich") == ["bbrfkr-llm-general"]
