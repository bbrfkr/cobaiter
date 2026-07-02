"""Valkey-backed persistence: conversation state + model registry + decision log.

Valkey is wire-compatible with Redis, so the ``redis.asyncio`` client is used.
All values are stored as JSON strings.

Keys
----
``cobaiter:conv:<key>``   string(JSON ConversationState), per-conversation, TTL'd
``cobaiter:models``       hash model -> JSON ModelSpec
``cobaiter:decisions``    stream, one entry(JSON DecisionLogEntry) per field "data"
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import redis.asyncio as redis

from .config import Settings
from .schemas import ConversationState, DecisionLogEntry, ModelSpec

log = logging.getLogger("cobaiter")

DECISIONS_KEY = "cobaiter:decisions"

CONV_PREFIX = "cobaiter:conv:"
MODELS_KEY = "cobaiter:models"


class Store:
    """Async wrapper around Valkey for cobaiter's persistent state."""

    def __init__(self, client: redis.Redis, settings: Settings) -> None:
        self._r = client
        self._settings = settings

    @classmethod
    def from_url(cls, settings: Settings) -> "Store":
        client = redis.from_url(settings.valkey_url, decode_responses=True)
        return cls(client, settings)

    async def close(self) -> None:
        await self._r.aclose()

    async def ping(self) -> bool:
        return bool(await self._r.ping())

    # ------------------------------------------------------------------ #
    # Conversation state
    # ------------------------------------------------------------------ #
    async def get_conversation(self, key: str) -> ConversationState | None:
        raw = await self._r.get(CONV_PREFIX + key)
        if raw is None:
            return None
        return ConversationState.model_validate_json(raw)

    async def set_conversation(self, key: str, state: ConversationState) -> None:
        await self._r.set(
            CONV_PREFIX + key,
            state.model_dump_json(),
            ex=self._settings.conv_ttl_seconds,
        )

    async def delete_conversation(self, key: str) -> bool:
        return bool(await self._r.delete(CONV_PREFIX + key))

    # ------------------------------------------------------------------ #
    # Model registry
    # ------------------------------------------------------------------ #
    async def put_model(self, spec: ModelSpec) -> None:
        await self._r.hset(MODELS_KEY, spec.model, spec.model_dump_json())

    async def get_model(self, model: str) -> ModelSpec | None:
        raw = await self._r.hget(MODELS_KEY, model)
        if raw is None:
            return None
        return ModelSpec.model_validate_json(raw)

    async def list_models(self) -> list[ModelSpec]:
        raw = await self._r.hgetall(MODELS_KEY)
        return [ModelSpec.model_validate_json(v) for v in raw.values()]

    async def delete_model(self, model: str) -> bool:
        return bool(await self._r.hdel(MODELS_KEY, model))

    # ------------------------------------------------------------------ #
    # Decision log (offline recalibration input)
    # ------------------------------------------------------------------ #
    async def log_decision(self, entry: DecisionLogEntry) -> None:
        """Append one decision to the ``cobaiter:decisions`` stream.

        Best-effort: any Valkey failure is logged and swallowed here so a decision
        logging hiccup never fails or delays the routing decision itself (this is
        called synchronously from the request path, same as other Store calls).
        The stream is trimmed to ``decision_log_maxlen`` (approximate trim, cheap)
        so it never grows unbounded.
        """
        try:
            await self._r.xadd(
                DECISIONS_KEY,
                {"data": entry.model_dump_json()},
                maxlen=self._settings.decision_log_maxlen,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001 - logging must never break routing
            log.warning("decision log write failed: %s", exc)

    async def read_decisions(self, count: int = 1000) -> list[DecisionLogEntry]:
        """Return up to ``count`` most recent logged decisions, oldest first."""
        raw = await self._r.xrevrange(DECISIONS_KEY, count=count)
        entries = [DecisionLogEntry.model_validate_json(fields["data"]) for _, fields in raw]
        entries.reverse()
        return entries

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #
    async def seed_models(
        self, specs: Iterable[ModelSpec], *, overwrite: bool = False
    ) -> int:
        """Insert registry entries that are missing (or overwrite all). Returns count written."""
        written = 0
        for spec in specs:
            if not overwrite and await self.get_model(spec.model) is not None:
                continue
            await self.put_model(spec)
            written += 1
        return written

    async def replace_models(self, specs: Iterable[ModelSpec]) -> int:
        """Make the registry match ``specs`` exactly.

        Drops any pre-existing entries (e.g. stale models from a previous config)
        before writing, so the externally-managed config file is the single source
        of truth. Returns the number of models written.
        """
        await self._r.delete(MODELS_KEY)
        written = 0
        for spec in specs:
            await self.put_model(spec)
            written += 1
        return written


def default_seed_specs() -> list[ModelSpec]:
    """A reasonable starter registry; tune via the /admin/models API."""
    return [
        ModelSpec(
            model="claude-opus-4-8",
            context_window=200_000,
            multimodal=True,
            supports_tools=True,
            is_local=False,
            cost=5.0,
            tier=3,
            fallback_chain=["claude-sonnet-4-6", "claude-haiku-4-5"],
        ),
        ModelSpec(
            model="claude-sonnet-4-6",
            context_window=200_000,
            multimodal=True,
            supports_tools=True,
            is_local=False,
            cost=3.0,
            tier=3,
            fallback_chain=["claude-haiku-4-5"],
        ),
        ModelSpec(
            model="claude-haiku-4-5",
            context_window=200_000,
            multimodal=True,
            supports_tools=True,
            is_local=False,
            cost=1.0,
            tier=1,
            fallback_chain=[],
        ),
        ModelSpec(
            model="qwen2.5",
            context_window=32_768,
            multimodal=False,
            supports_tools=True,
            is_local=True,
            cost=0.0,
            tier=1,
            fallback_chain=[],
        ),
    ]
