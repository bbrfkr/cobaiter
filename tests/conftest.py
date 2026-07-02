"""Shared test fixtures and fakes."""

from __future__ import annotations

import fakeredis.aioredis as fakeaioredis
import pytest

from cobaiter.config import Settings
from cobaiter.litellm_client import DownstreamError
from cobaiter.router import RouteEngine
from cobaiter.schemas import CandidateScore, ClassifierDiagnostics, ClassifierResult
from cobaiter.store import Store, default_seed_specs


class FakeClient:
    """Stand-in for LiteLLMClient.

    ``credit`` maps model -> remaining headroom (None == unconstrained).
    ``errors`` maps model -> DownstreamError to raise from chat()/chat_stream().
    """

    def __init__(self) -> None:
        self.credit: dict[str, float | None] = {}
        self.errors: dict[str, DownstreamError] = {}
        self.calls: list[str] = []

    async def chat(self, payload):
        model = payload["model"]
        self.calls.append(model)
        if model in self.errors:
            raise self.errors[model]
        return {
            "id": "cmpl-test",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": f"reply::{model}"}}],
        }

    async def chat_stream(self, payload):
        model = payload["model"]
        self.calls.append(model)
        if model in self.errors:
            raise self.errors[model]
        for chunk in (b"data: a\n\n", b"data: [DONE]\n\n"):
            yield chunk

    async def credit_remaining(self, model: str):
        return self.credit.get(model)

    def invalidate_credit(self, model: str) -> None:
        self.credit.pop(model, None)

    async def close(self) -> None:
        pass


class FakeClassifier:
    """Returns scores from a per-model table; records call count."""

    def __init__(self, table: dict[str, float] | None = None) -> None:
        self.table = table or {}
        self.calls = 0
        # None (default) mirrors the "no usable estimate" case: the router skips
        # capability-fit entirely. Set to a float to exercise it in a test.
        self.difficulty: float | None = None
        # None (default) mirrors "nothing to log" (see EmbeddingClassifier.score);
        # set to a ClassifierDiagnostics to exercise decision logging in a test.
        self.raw: ClassifierDiagnostics | None = None

    async def score(self, req, candidates):
        self.calls += 1
        # Unlisted models default to 0.0 (unsuitable) so a test only needs to set
        # scores for the models it cares about; others won't win the re-ranking.
        return ClassifierResult(
            scores=[
                CandidateScore(model=c.model, score=self.table.get(c.model, 0.0))
                for c in candidates
            ],
            difficulty=self.difficulty,
            raw=self.raw,
        )


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps tests hermetic: never read the developer's local .env.
    return Settings(
        _env_file=None,
        min_dwell_turns=3,
        soft_recheck_every=4,
        switch_margin=0.15,
        credit_floor=0.0,
    )


@pytest.fixture
async def store(settings):
    s = Store(fakeaioredis.FakeRedis(decode_responses=True), settings)
    await s.seed_models(default_seed_specs())
    yield s
    await s.close()


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def engine(store, client, classifier, settings) -> RouteEngine:
    return RouteEngine(store, client, classifier, settings)


def user_req(content: str = "hello", **kw):
    from cobaiter.schemas import ChatCompletionRequest

    return ChatCompletionRequest(
        model="cobaiter-auto", messages=[{"role": "user", "content": content}], **kw
    )


def convo_req(user_turns: int, *, last: str = "hello", **kw):
    """Build a request for a conversation that has had ``user_turns`` user messages.

    Each earlier user turn is paired with an assistant reply; the final user
    message carries ``last``. Re-sending with an incremented ``user_turns`` is how
    a genuine *new user turn* is simulated (the router keys routing decisions off
    the user-message count, not off repeated identical single-message requests)."""
    from cobaiter.schemas import ChatCompletionRequest

    messages: list[dict] = []
    for i in range(user_turns - 1):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    messages.append({"role": "user", "content": last})
    return ChatCompletionRequest(model="cobaiter-auto", messages=messages, **kw)


def agentic_followup_req(prior_user_turns: int, *, tool_payload: str = "result", **kw):
    """Build a *mid-instruction* request: a tool round-trip with NO new user turn.

    Same user-message count as ``convo_req(prior_user_turns)`` but with an extra
    assistant(tool_calls) + tool(result) pair appended — exactly what an agent
    re-sends while looping on a single instruction."""
    from cobaiter.schemas import ChatCompletionRequest

    messages: list[dict] = []
    for i in range(prior_user_turns - 1):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    messages.append({"role": "user", "content": "do the task"})
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]})
    messages.append({"role": "tool", "tool_call_id": "t1", "content": tool_payload})
    return ChatCompletionRequest(model="cobaiter-auto", messages=messages, **kw)
