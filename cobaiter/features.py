"""Cheap, non-LLM feature extraction from a chat request.

Provides three things used by the router:

* ``conversation_key`` — hybrid identity (explicit id, else fingerprint of the head).
* ``extract_constraints`` — hard routing constraints (multimodal / tools / privacy / tokens).
* helpers for the soft re-evaluation gate (code-block / size signals).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import tiktoken

from .schemas import ChatCompletionRequest, Constraints

# Header / field names recognised as an explicit conversation id.
CONV_ID_HEADER = "x-cobaiter-conversation-id"
PRIVACY_HEADER = "x-cobaiter-privacy"

_CODE_FENCE = re.compile(r"```")
# Single shared encoder; cl100k_base is a good cross-model proxy for token counts.
_ENCODER = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------- #
# Conversation identity
# --------------------------------------------------------------------------- #
def conversation_key(req: ChatCompletionRequest, header_id: str | None) -> str:
    """Return a stable conversation key.

    Priority: explicit header id -> metadata.conversation_id -> user field ->
    fingerprint of the conversation head (system prompt + first user message).
    """
    explicit = header_id
    if not explicit and req.metadata:
        explicit = req.metadata.get("conversation_id")
    if not explicit and req.user:
        explicit = req.user
    if explicit:
        return f"id:{explicit}"
    return f"fp:{_fingerprint(req)}"


def _fingerprint(req: ChatCompletionRequest) -> str:
    system = " ".join(_text(m) for m in req.messages if m.get("role") == "system")
    first_user = next(
        (_text(m) for m in req.messages if m.get("role") == "user"), ""
    )
    norm = (system.strip() + "\n␟\n" + first_user.strip()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #
def extract_constraints(
    req: ChatCompletionRequest, *, privacy_header: str | None = None
) -> Constraints:
    return Constraints(
        needs_multimodal=_has_image(req.messages),
        needs_tools=bool(req.tools or req.functions),
        needs_local=_needs_privacy(req, privacy_header),
        estimated_tokens=estimate_tokens(req.messages),
    )


def _needs_privacy(req: ChatCompletionRequest, privacy_header: str | None) -> bool:
    if privacy_header and privacy_header.lower() in ("1", "true", "local", "yes"):
        return True
    if req.metadata:
        val = req.metadata.get("privacy") or req.metadata.get("local")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("1", "true", "local", "yes")
    return False


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "image",
                    "input_image",
                ):
                    return True
    return False


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough prompt-token estimate. ~4 tokens of overhead per message."""
    total = 0
    for m in messages:
        total += 4
        total += len(_ENCODER.encode(_text(m)))
    return total


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return " ".join(parts)
    return ""


# --------------------------------------------------------------------------- #
# Cheap soft-change signals (stage-1 gate)
# --------------------------------------------------------------------------- #
def count_code_blocks(messages: list[dict[str, Any]]) -> int:
    """Number of code fences across the whole conversation (an even count = N blocks)."""
    fences = sum(len(_CODE_FENCE.findall(_text(m))) for m in messages)
    return fences // 2


def count_user_messages(messages: list[dict[str, Any]]) -> int:
    """Number of ``user``-role messages — a proxy for how many user turns occurred.

    A single user *instruction* given to an agent expands into many downstream
    chat/completions calls (the agentic loop: assistant tool-calls + ``tool``
    results, re-sent each round). Those round-trips only append ``assistant`` /
    ``tool`` messages, so this count stays flat *within* one instruction and
    increases only when a genuinely new user message arrives. The router uses the
    delta to tell "new user turn" (a routing opportunity) from "mid-instruction
    round-trip" (must stay pinned)."""
    return sum(1 for m in messages if m.get("role") == "user")
