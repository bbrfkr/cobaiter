"""Externally-managed difficulty exemplars: config loader + classifier injection."""

from __future__ import annotations

import pytest

from cobaiter.classifier import (
    _EASY_EXEMPLARS,
    _HARD_EXEMPLARS,
    EmbeddingClassifier,
)
from cobaiter.config import Settings
from cobaiter.exemplars import ExemplarConfigError, load_difficulty_exemplars

_YAML = """
easy:
  - "こんにちは"
  - "  今何時ですか？  "
hard:
  - "この命題を厳密に証明してください"
  - "この契約の法的リスクを多角的に分析してください"
"""


def _write(tmp_path, text):
    p = tmp_path / "exemplars.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid(tmp_path):
    easy, hard = load_difficulty_exemplars(_write(tmp_path, _YAML))
    assert easy == ["こんにちは", "今何時ですか？"]  # stripped
    assert hard == [
        "この命題を厳密に証明してください",
        "この契約の法的リスクを多角的に分析してください",
    ]


def test_missing_file(tmp_path):
    with pytest.raises(ExemplarConfigError):
        load_difficulty_exemplars(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "text",
    [
        "easy: [a]\n",  # hard missing
        "hard: [a]\n",  # easy missing
        "easy: []\nhard: [a]\n",  # easy empty
        "easy: [a]\nhard: []\n",  # hard empty
        "- just\n- a\n- list\n",  # not a mapping
        "easy: a\nhard: b\n",  # values not lists
    ],
)
def test_invalid_shapes(tmp_path, text):
    with pytest.raises(ExemplarConfigError):
        load_difficulty_exemplars(_write(tmp_path, text))


def test_classifier_defaults_to_builtins():
    clf = EmbeddingClassifier(client=None, settings=Settings(_env_file=None))
    assert clf._easy_exemplars == list(_EASY_EXEMPLARS)
    assert clf._hard_exemplars == list(_HARD_EXEMPLARS)
    assert clf._difficulty_exemplars == list(_EASY_EXEMPLARS) + list(_HARD_EXEMPLARS)


def test_classifier_honors_injected_exemplars():
    clf = EmbeddingClassifier(
        client=None,
        settings=Settings(_env_file=None),
        easy_exemplars=["hi"],
        hard_exemplars=["prove this theorem"],
    )
    assert clf._easy_exemplars == ["hi"]
    assert clf._hard_exemplars == ["prove this theorem"]
    assert clf._difficulty_exemplars == ["hi", "prove this theorem"]
