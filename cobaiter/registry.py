"""Load the model registry from an externally-managed config file.

The registry maps each downstream (LiteLLM) model name to its routing-relevant
attributes. These split into two kinds:

* Capabilities that *could* be discovered from LiteLLM (``multimodal``,
  ``context_window``, ``supports_tools``) but are often unset there, and
* cobaiter-only routing **policy** (``tier``, ``fallback_chain``) that has no
  source in LiteLLM at all.

Rather than hardcode this, it is hand-managed in a YAML/JSON file and injected.
The file is the source of truth: the registry is reconciled to match it exactly
on startup (see ``Store.replace_models``).

Expected shape (YAML)::

    models:
      - model: bbrfkr-llm-general
        tier: rich
        context_window: 32768
        multimodal: false
        supports_tools: true
        is_local: true
        fallback_chain: [bbrfkr-llm-general-no-think]
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import ModelSpec


class RegistryConfigError(Exception):
    """The registry config file is missing required structure or is unreadable."""


def load_model_registry(path: str | Path) -> list[ModelSpec]:
    """Parse ``path`` into a list of ``ModelSpec``.

    Accepts either a top-level ``models:`` list or a bare list of model entries.
    YAML is a superset of JSON, so JSON files parse too.
    """
    p = Path(path)
    if not p.is_file():
        raise RegistryConfigError(f"registry config not found: {p}")

    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - thin wrapper
        raise RegistryConfigError(f"invalid registry config {p}: {exc}") from exc

    if isinstance(doc, dict):
        entries = doc.get("models", [])
    elif isinstance(doc, list):
        entries = doc
    else:
        raise RegistryConfigError(
            f"registry config {p} must be a mapping with 'models:' or a list"
        )

    if not entries:
        raise RegistryConfigError(f"registry config {p} defines no models")

    specs: list[ModelSpec] = []
    for entry in entries:
        try:
            specs.append(ModelSpec.model_validate(entry))
        except Exception as exc:  # noqa: BLE001 - surface which entry is bad
            raise RegistryConfigError(
                f"invalid model entry in {p}: {entry!r}: {exc}"
            ) from exc
    return specs
