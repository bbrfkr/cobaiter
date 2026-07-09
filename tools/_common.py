"""Shared helpers for the routing-tuning tools.

Keeps the eval prompt set, the target server URL, and cloud-model detection in
one place so the individual tools stay small and never hardcode a host/IP.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from cobaiter.registry import load_model_registry
from cobaiter.schemas import ChatCompletionRequest

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_PROMPTS_FILE = _TOOLS_DIR / "eval_prompts.yaml"

# Prompt = (domain, level, text).
Prompt = tuple[str, str, str]


def load_eval_prompts(path: str | Path | None = None) -> list[Prompt]:
    """Load the domain/level/text eval prompts from ``eval_prompts.yaml``."""
    p = Path(path) if path else _PROMPTS_FILE
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [(e["domain"], e["level"], e["text"]) for e in doc["prompts"]]


def default_base_url() -> str:
    """Cobaiter server URL for the LIVE tools (seed_logs / validate_routing).

    Override with COBAITER_TOOLS_BASE_URL or the tool's --base-url flag.
    """
    return os.environ.get("COBAITER_TOOLS_BASE_URL", "http://localhost:8080")


def cloud_models(models_config: str = "models.yaml") -> set[str]:
    """Names of the non-local (cloud) models, from the registry file — so the
    tools don't hardcode which models are 'rich/cloud'."""
    path = Path(models_config)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    specs = load_model_registry(path)
    return {s.model for s in specs if not s.is_local}


def registry_specs(models_config: str = "models.yaml"):
    path = Path(models_config)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return load_model_registry(path)


def as_request(text: str) -> ChatCompletionRequest:
    """A single-user-message request, the unit both the offline classifier and
    the live server score."""
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": text}]}
    )
