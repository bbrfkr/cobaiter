"""FastAPI application: OpenAI-compatible proxy + admin API.

Wiring is done via ``create_app`` so tests can inject fakes (Store / LiteLLMClient
/ Classifier) without a real Valkey or gateway.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from redis.exceptions import RedisError

from .classifier import EmbeddingClassifier
from .config import Settings, get_settings
from .features import CONV_ID_HEADER, PRIVACY_HEADER, extract_constraints
from .litellm_client import DownstreamError, LiteLLMClient
from .registry import RegistryConfigError, load_model_registry
from .router import RouteEngine
from .schemas import ChatCompletionRequest, ModelSpec, Route
from .store import Store, default_seed_specs

log = logging.getLogger("cobaiter")


def _configure_logging(level: int | str = logging.INFO) -> None:
    """Ensure cobaiter's logger emits to stderr (uvicorn's default config does not
    attach a handler for it, so INFO records would otherwise be swallowed).

    ``level`` may be a logging constant or a level name (e.g. "DEBUG"); an
    unrecognised name falls back to INFO.
    """
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [cobaiter] %(message)s")
        )
        log.addHandler(handler)
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    log.setLevel(level)
    log.propagate = False


def create_app(
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    client: LiteLLMClient | None = None,
    classifier: EmbeddingClassifier | None = None,
    seed: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.store = store or Store.from_url(settings)
        app.state.client = client or LiteLLMClient.create(settings)
        app.state.classifier = classifier or EmbeddingClassifier(
            app.state.client, settings
        )
        app.state.engine = RouteEngine(
            app.state.store, app.state.client, app.state.classifier, settings
        )
        if seed:
            try:
                specs = _load_registry_specs(settings)
                # Config file is the source of truth; reconcile the registry to it
                # (this also clears any stale models from a previous config).
                count = await app.state.store.replace_models(specs)
                log.info("model registry loaded: %d models", count)
            except RegistryConfigError as exc:
                log.error("model registry config error: %s", exc)
            except Exception as exc:  # noqa: BLE001 - seeding is best-effort at startup
                log.warning("model registry seeding skipped: %s", exc)
        yield
        await app.state.store.close()
        await app.state.client.close()

    def _load_registry_specs(settings: Settings) -> list[ModelSpec]:
        if settings.models_config:
            return load_model_registry(settings.models_config)
        return default_seed_specs()

    app = FastAPI(title="cobaiter", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RedisError)
    async def _state_store_unavailable(request: Request, exc: RedisError):
        # The Valkey state store is required for sticky routing; if it is
        # unreachable, fail explicitly with 503 rather than an opaque 500.
        return JSONResponse(
            status_code=503,
            content={"detail": f"state store unavailable: {exc}"},
        )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    async def healthz(request: Request):
        try:
            ok = await request.app.state.store.ping()
        except Exception:  # noqa: BLE001
            ok = False
        return {"status": "ok" if ok else "degraded", "valkey": ok}

    @app.get("/v1/models")
    async def list_models(request: Request):
        settings: Settings = request.app.state.settings
        specs = await request.app.state.store.list_models()
        data = [{"id": settings.virtual_model, "object": "model", "owned_by": "cobaiter"}]
        data += [{"id": s.model, "object": "model", "owned_by": "cobaiter"} for s in specs]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        settings: Settings = request.app.state.settings
        engine: RouteEngine = request.app.state.engine
        client: LiteLLMClient = request.app.state.client

        body = await request.json()
        req = ChatCompletionRequest.model_validate(body)
        header_id = request.headers.get(CONV_ID_HEADER)
        privacy_header = request.headers.get(PRIVACY_HEADER)

        decision = await engine.decide(
            req, header_id=header_id, privacy_header=privacy_header
        )
        constraints = extract_constraints(req, privacy_header=privacy_header)

        # The routing decision (with score) is logged once by RouteEngine.decide as
        # the canonical "decision:" line; keep only a debug breadcrumb here.
        log.debug(
            "route: conv=%s route=%s model=%s stream=%s",
            decision.conversation_key, decision.route.value, decision.model, req.stream,
        )

        headers = {
            "x-cobaiter-model": decision.model,
            "x-cobaiter-route": decision.route.value,
            "x-cobaiter-conversation": decision.conversation_key,
        }

        if req.stream:
            return await _stream_response(
                client, engine, req, decision, constraints, headers
            )
        return await _json_response(
            client, engine, req, decision, constraints, headers, settings
        )

    # --- Admin: model registry --------------------------------------- #
    @app.get("/admin/models")
    async def admin_list_models(request: Request):
        specs = await request.app.state.store.list_models()
        return {"data": [s.model_dump() for s in specs]}

    @app.put("/admin/models")
    async def admin_put_model(request: Request):
        spec = ModelSpec.model_validate(await request.json())
        await request.app.state.store.put_model(spec)
        return {"status": "ok", "model": spec.model}

    @app.delete("/admin/models/{model}")
    async def admin_delete_model(request: Request, model: str):
        removed = await request.app.state.store.delete_model(model)
        return {"status": "ok", "deleted": removed}

    # --- Admin: conversation bindings -------------------------------- #
    @app.get("/admin/conversations/{key:path}")
    async def admin_get_conversation(request: Request, key: str):
        state = await request.app.state.store.get_conversation(key)
        if state is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return state.model_dump()

    @app.delete("/admin/conversations/{key:path}")
    async def admin_delete_conversation(request: Request, key: str):
        removed = await request.app.state.store.delete_conversation(key)
        return {"status": "ok", "deleted": removed}


# --------------------------------------------------------------------------- #
# Response helpers (with one round of pre-flight failover)
# --------------------------------------------------------------------------- #
async def _json_response(
    client, engine, req, decision, constraints, headers, settings
):
    attempts = 0
    model = decision.model
    while True:
        attempts += 1
        try:
            data = await client.chat(req.to_downstream(model))
            headers["x-cobaiter-model"] = model
            log.info(
                "served: conv=%s model=%s route=%s",
                decision.conversation_key, model, headers["x-cobaiter-route"],
            )
            return JSONResponse(content=data, headers=headers)
        except DownstreamError as exc:
            if exc.kind == "none" or attempts > 5:
                log.warning(
                    "downstream failed (no failover): conv=%s model=%s kind=%s -> 502",
                    decision.conversation_key, model, exc.kind,
                )
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            new_decision = await engine.failover_to(
                decision.conversation_key, constraints
            )
            if new_decision is None or new_decision.model == model:
                raise HTTPException(status_code=503, detail="no available model") from exc
            log.info(
                "failover: conv=%s %s -> %s (kind=%s)",
                decision.conversation_key, model, new_decision.model, exc.kind,
            )
            model = new_decision.model
            headers["x-cobaiter-route"] = Route.FAILOVER.value


async def _stream_response(client, engine, req, decision, constraints, headers):
    """Stream with pre-flight failover.

    We open the upstream stream eagerly; if it fails *before* the first byte we
    can still fail over. Once bytes flow, errors propagate (no mid-stream switch).
    """
    model = decision.model
    attempts = 0
    while True:
        attempts += 1
        gen = client.chat_stream(req.to_downstream(model))
        try:
            first = await gen.__anext__()
        except StopAsyncIteration:
            first = None
        except DownstreamError as exc:
            if exc.kind == "none" or attempts > 5:
                log.warning(
                    "stream downstream failed (no failover): conv=%s model=%s kind=%s -> 502",
                    decision.conversation_key, model, exc.kind,
                )
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            new_decision = await engine.failover_to(
                decision.conversation_key, constraints
            )
            if new_decision is None or new_decision.model == model:
                raise HTTPException(status_code=503, detail="no available model") from exc
            log.info(
                "failover(stream): conv=%s %s -> %s (kind=%s)",
                decision.conversation_key, model, new_decision.model, exc.kind,
            )
            model = new_decision.model
            headers["x-cobaiter-route"] = Route.FAILOVER.value
            continue

        headers["x-cobaiter-model"] = model
        log.info(
            "served(stream): conv=%s model=%s route=%s",
            decision.conversation_key, model, headers["x-cobaiter-route"],
        )

        async def body():
            if first is not None:
                yield first
            async for chunk in gen:
                yield chunk

        return StreamingResponse(
            body(), media_type="text/event-stream", headers=headers
        )


# Default ASGI app for ``uvicorn cobaiter.app:app``.
app = create_app(seed=True)
