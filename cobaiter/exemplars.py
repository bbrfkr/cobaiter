"""Load the classifier's difficulty exemplars from an externally-managed file.

The exemplars are the two small task-phrase sets ("easy" and "hard") that the
classifier anchors task *difficulty* against (see cobaiter.classifier for the
embedding-similarity method). Like the model registry (registry.py) and the
difficulty anchors (Settings.difficulty_*_anchor), they are deployment- and
embedding-model-specific tuning data, so they are injected from a file rather
than hardcoded — letting them be tuned without a code change / image rebuild.

Empty config path = the caller uses the built-in defaults in cobaiter.classifier.

Expected shape (YAML; JSON parses too, being a YAML subset)::

    easy:
      - "こんにちは"
      - "今日の天気は？"
    hard:
      - "この命題を厳密に証明してください"
      - "この契約の法的リスクを多角的に分析してください"
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ExemplarConfigError(Exception):
    """The difficulty-exemplars config file is missing required structure or is unreadable."""


def load_difficulty_exemplars(path: str | Path) -> tuple[list[str], list[str]]:
    """Parse ``path`` into ``(easy_exemplars, hard_exemplars)``.

    Both keys are required and must be non-empty lists of non-empty strings —
    the difficulty ratio needs at least one exemplar on each side to be defined.
    """
    p = Path(path)
    if not p.is_file():
        raise ExemplarConfigError(f"difficulty-exemplars config not found: {p}")

    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - thin wrapper
        raise ExemplarConfigError(f"invalid difficulty-exemplars config {p}: {exc}") from exc

    if not isinstance(doc, dict):
        raise ExemplarConfigError(
            f"difficulty-exemplars config {p} must be a mapping with 'easy:' and 'hard:'"
        )

    easy = _clean_list(doc.get("easy"), p, "easy")
    hard = _clean_list(doc.get("hard"), p, "hard")
    return easy, hard


def _clean_list(value: object, p: Path, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ExemplarConfigError(f"difficulty-exemplars config {p}: '{key}' must be a list")
    items = [str(v).strip() for v in value if str(v).strip()]
    if not items:
        raise ExemplarConfigError(
            f"difficulty-exemplars config {p}: '{key}' must have at least one non-empty entry"
        )
    return items
