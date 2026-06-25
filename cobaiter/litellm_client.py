"""Thin async client for the downstream LiteLLM gateway.

Responsibilities:
* Forward chat completions (streaming and non-streaming).
* Expose ``/v1/models``.
* Read per-model credit headroom from LiteLLM's spend/budget endpoints.
* Classify downstream errors into availability categories for failover.

cobaiter deliberately keeps the *model choice* itself; LiteLLM's own silent
fallbacks are not relied upon (they would make the pinned binding stale).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings
from .schemas import AvailabilityError


class DownstreamError(Exception):
    """A downstream HTTP error, tagged with an availability category."""

    def __init__(self, status: int, body: str, kind: AvailabilityError) -> None:
        super().__init__(f"downstream {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.kind = kind


def classify_error(status: int, body: str) -> AvailabilityError:
    """Map an HTTP status + body to an availability category."""
    blob = body.lower()
    if status == 400 and "context" in blob and (
        "length" in blob or "window" in blob or "maximum" in blob
    ):
        return "context_length"
    if "context_length_exceeded" in blob or "context window" in blob:
        return "context_length"
    if status == 429:
        if "quota" in blob or "insufficient" in blob or "billing" in blob or "credit" in blob:
            return "quota"
        return "rate_limit"
    if status in (402,):
        return "quota"
    if status in (503, 529) or "overloaded" in blob:
        return "overloaded"
    return "none"


class LiteLLMClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._c = client
        self._s = settings
        self._credit_cache: dict[str, tuple[float, float | None]] = {}

    @classmethod
    def create(cls, settings: Settings) -> "LiteLLMClient":
        headers = {}
        if settings.litellm_api_key:
            headers["Authorization"] = f"Bearer {settings.litellm_api_key}"
        client = httpx.AsyncClient(
            base_url=settings.litellm_base_url,
            headers=headers,
            timeout=settings.request_timeout,
        )
        return cls(client, settings)

    async def close(self) -> None:
        await self._c.aclose()

    # ------------------------------------------------------------------ #
    # Chat completions
    # ------------------------------------------------------------------ #
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming completion. Raises DownstreamError on availability errors."""
        try:
            resp = await self._c.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if resp.status_code >= 400:
            self._raise(resp.status_code, resp.text)
        return resp.json()

    async def chat_stream(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Streaming completion.

        The initial response status is checked *before* yielding, so availability
        errors at stream start can still trigger failover. Once bytes flow, errors
        are propagated (no auto-switch) to avoid duplicated output.
        """
        try:
            async with self._c.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise(resp.status_code, body)
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    def _raise(self, status: int, body: str) -> None:
        raise DownstreamError(status, body, classify_error(status, body))

    @staticmethod
    def _transport_error(exc: httpx.HTTPError) -> DownstreamError:
        """Wrap a transport-level failure (gateway unreachable / timeout).

        Tagged as a 502 with kind ``none``: every model is reached through the same
        gateway, so failing over to another model would not help. This surfaces a
        clean 502 instead of an unhandled 500.
        """
        return DownstreamError(502, f"gateway unreachable: {exc}", "none")

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    async def list_models(self) -> list[str]:
        try:
            resp = await self._c.get("/v1/models")
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        data = resp.json().get("data", [])
        return [m["id"] for m in data if "id" in m]

    # ------------------------------------------------------------------ #
    # Credit headroom (delegated to LiteLLM budget/spend)
    # ------------------------------------------------------------------ #
    async def credit_remaining(self, model: str) -> float | None:
        """Remaining USD budget for ``model`` per LiteLLM, or None if unknown.

        Looks up the LiteLLM model-info / spend endpoints. Unknown == unconstrained
        (None), so absence of budget data never blocks routing.
        """
        cached = self._credit_cache.get(model)
        now = time.monotonic()
        if cached and now - cached[0] < self._s.credit_cache_ttl:
            return cached[1]

        remaining = await self._fetch_credit(model)
        self._credit_cache[model] = (now, remaining)
        return remaining

    async def _fetch_credit(self, model: str) -> float | None:
        try:
            resp = await self._c.get("/model/info", params={"litellm_model_id": model})
            if resp.status_code != 200:
                resp = await self._c.get("/model/info")
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", [])
        except httpx.HTTPError:
            return None

        for entry in data:
            info = entry.get("model_info", {}) if isinstance(entry, dict) else {}
            name = entry.get("model_name") or info.get("id")
            if name != model:
                continue
            budget = info.get("max_budget")
            spend = info.get("spend", 0.0) or 0.0
            if budget is None:
                return None
            try:
                return float(budget) - float(spend)
            except (TypeError, ValueError):
                return None
        return None

    def invalidate_credit(self, model: str) -> None:
        self._credit_cache.pop(model, None)
