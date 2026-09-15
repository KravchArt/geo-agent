"""FastAPI application entrypoint.

Resource ownership: the engine and the Redis client are created **once, in the
lifespan**, stored on ``app.state``, and handed to handlers via dependencies.
They are never module-level singletons — both bind to the event loop that
created them, so a global would break in any other loop (tests, workers,
background tasks) and, for Redis, do so silently.

Endpoints:
* ``GET  /``             — service identity + active LLM mode.
* ``GET  /health``       — real liveness check that pings Postgres AND Redis.
* ``POST /api/v1/chat``  — the Phase 0 pipeline for one user message.
* ``POST /api/v1/chat/stream`` — that pipeline as an NDJSON progress stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.db.session import (
    create_engine,
    create_sessionmaker,
    get_session,
    ping_database,
)
from backend.app.llm.base import get_censorship_llm_client, get_scope_llm_client
from backend.app.redis.client import RedisClient, get_redis
from backend.app.services.conversations import ConversationNotFoundError, ConversationStore
from backend.app.services.gates import (
    ClassifierScopeGate,
    GateEvaluator,
    GuardianCensorshipGate,
    LLMCensorshipGate,
    LLMScopeGate,
    NoScoperScopeGate,
)
from backend.app.services.pipeline import run_pipeline
from backend.app.services.scope_classifier import ScopeClassifierClient
from backend.app.services.scope_ood import (
    OODIndex,
    ScopeEmbeddingClient,
    ScopeOODDetector,
    ScopeOODError,
)
from backend.app.services.user_context import build_user_location_reverse_geocoder
from backend.app.stores import RedisPlaceStore, RedisSourceStore
from backend.app.tools import build_runtime_tools
from common.models import (
    AgentRequest,
    AgentResponse,
    ConversationDetail,
    ConversationListResponse,
)
from tools import ToolExecutor

VERSION = "0.0.0"
REQUEST_IN_PROGRESS_DETAIL = "An agent request is already in progress for this session."
STREAM_HEARTBEAT_SECONDS = 15.0
logger = logging.getLogger(__name__)

_APPLICATION_LOG_NAMESPACES = ("backend", "tools")
_APPLICATION_LOG_HANDLER_MARKER = "_geoagent_stdout_handler"


def configure_application_logging() -> None:
    """Emit GeoAgent INFO lifecycle logs without enabling noisy dependency logs.

    Uvicorn configures only its own named loggers. Consequently module loggers
    such as ``backend.app.services.orchestrator`` otherwise have no INFO-capable
    handler in the container. Attach one handler to each application namespace
    and disable propagation so messages are printed exactly once.
    """

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for namespace in _APPLICATION_LOG_NAMESPACES:
        application_logger = logging.getLogger(namespace)
        application_logger.setLevel(logging.INFO)
        application_logger.propagate = False
        if any(
            getattr(handler, _APPLICATION_LOG_HANDLER_MARKER, False)
            for handler in application_logger.handlers
        ):
            continue
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        setattr(handler, _APPLICATION_LOG_HANDLER_MARKER, True)
        application_logger.addHandler(handler)


OptionalClientId = Annotated[
    str | None,
    Header(alias="X-Client-ID", min_length=1, max_length=128),
]
RequiredClientId = Annotated[
    str,
    Header(alias="X-Client-ID", min_length=1, max_length=128),
]
ConversationId = Annotated[str, Path(min_length=1, max_length=128)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create application-owned resources once and close them on shutdown."""

    configure_application_logging()
    settings = get_settings()

    engine = create_engine(settings)
    redis = RedisClient.from_settings(settings)

    async with AsyncExitStack() as stack:
        stack.push_async_callback(engine.dispose)
        stack.push_async_callback(redis.close)

        http_client = await stack.enter_async_context(
            httpx.AsyncClient(
                timeout=httpx.Timeout(float(settings.tools_http_timeout)),
                # Route tool traffic through TOOLS_HTTP_PROXY when configured, and
                # ignore HTTP_PROXY/ALL_PROXY from the environment: routing is a
                # deployment decision, not something a stray shell variable should
                # change. It also avoids httpx refusing to start on an ALL_PROXY
                # scheme it cannot parse (e.g. socks://).
                proxy=settings.tools_http_proxy,
                trust_env=False,
            )
        )

        # Refs live in Redis, not in the process: an in-memory ref stops
        # resolving after a restart or on a second worker, which silently breaks
        # every answer that cites it.
        place_store = RedisPlaceStore(redis)
        source_store = RedisSourceStore(redis)
        user_location_reverse_geocoder = build_user_location_reverse_geocoder(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

        # Opt-in model gate. Cheap to build (no weights — it calls an OpenAI-
        # compatible endpoint); None means the pipeline keeps the regex gate.
        censorship_gate: GateEvaluator | None = None
        if settings.censorship_provider == "model":
            censorship_gate = LLMCensorshipGate(
                get_censorship_llm_client(settings),
                model=settings.censorship_model,
            )
        elif settings.censorship_provider == "guardian":
            censorship_gate = GuardianCensorshipGate(
                get_censorship_llm_client(settings),
                model=settings.censorship_model,
            )

        # Scope classifier. Cheap to build (no weights — every option here calls a
        # remote endpoint); None means the pipeline keeps the regex gate.
        scope_gate: GateEvaluator | None = None
        if settings.scope_provider == "llm":
            scope_gate = LLMScopeGate(get_scope_llm_client(settings), model=settings.scope_model)
        elif settings.scope_provider == "classifier":
            # The encoder answers alone outside its grey zone; inside it, and on
            # any failure, the LLM scoper decides — it is the one that sees history.
            assert settings.scope_classifier_base_url is not None  # enforced in config
            assert settings.scope_classifier_model is not None
            # Its own client, never the tools one. The tools client carries the
            # routing and timeouts of outbound provider traffic; the classifier is
            # a local endpoint on the critical path, and borrowing that client makes
            # its call fail in ways that do not surface — it hangs until the timeout
            # and the gate silently degrades to the LLM scoper, then to regex.
            classifier_http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=httpx.Timeout(float(settings.scope_classifier_timeout)),
                    trust_env=False,
                )
            )
            # Distance-to-corpus check. Optional and fail-open by construction: a
            # missing index or an unreachable embedding endpoint leaves the gate
            # exactly as it was, because this only ever *adds* deferrals to the
            # LLM scoper — it can never let something through on its own.
            ood_detector: ScopeOODDetector | None = None
            if settings.scope_ood_mode != "off":
                try:
                    ood_index = OODIndex.load(settings.scope_ood_index_path or "")
                    ood_detector = ScopeOODDetector(
                        ScopeEmbeddingClient(
                            base_url=settings.scope_ood_base_url
                            or settings.scope_classifier_base_url,
                            model=settings.scope_ood_model or settings.scope_classifier_model,
                            api_key=settings.scope_classifier_api_key,
                            prefix=settings.scope_classifier_prefix,
                            timeout=settings.scope_classifier_timeout,
                            http_client=classifier_http_client,
                        ),
                        ood_index,
                        quantile=settings.scope_ood_quantile,
                    )
                    logger.info(
                        "scope_ood_ready mode=%s rows=%d quantile=%.3f threshold=%.4f",
                        settings.scope_ood_mode,
                        ood_index.rows,
                        settings.scope_ood_quantile,
                        ood_detector.threshold,
                    )
                except ScopeOODError:
                    logger.warning(
                        "scope_ood_unavailable continuing_without_it path=%r",
                        settings.scope_ood_index_path,
                        exc_info=True,
                    )

            scope_gate = ClassifierScopeGate(
                ScopeClassifierClient(
                    base_url=settings.scope_classifier_base_url,
                    model=settings.scope_classifier_model,
                    api_key=settings.scope_classifier_api_key,
                    prefix=settings.scope_classifier_prefix,
                    temperature=settings.scope_classifier_temperature,
                    timeout=settings.scope_classifier_timeout,
                    http_client=classifier_http_client,
                ),
                LLMScopeGate(get_scope_llm_client(settings), model=settings.scope_model),
                grey_low=settings.scope_classifier_grey_low,
                grey_high=settings.scope_classifier_grey_high,
                ood=ood_detector,
                ood_enforce=settings.scope_ood_mode == "enforce",
                confirm_reject=settings.scope_classifier_confirm_reject,
            )
        elif settings.scope_provider == "no_scoper":
            scope_gate = NoScoperScopeGate()

        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
            source_store=source_store,
        )
        tool_executor = ToolExecutor(runtime_tools)

        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        app.state.redis = redis

        app.state.tools_http_client = http_client
        app.state.place_store = place_store
        app.state.source_store = source_store
        app.state.user_location_reverse_geocoder = user_location_reverse_geocoder
        app.state.tools = runtime_tools
        app.state.tool_executor = tool_executor
        app.state.censorship_gate = censorship_gate
        app.state.scope_gate = scope_gate

        yield


app = FastAPI(title="Geo-Agent", version=VERSION, lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    settings = get_settings()
    return {"name": "geoagent", "version": VERSION, "llm_mode": settings.llm_mode}


@app.get("/health")
async def health(request: Request, redis: RedisClient = Depends(get_redis)) -> JSONResponse:
    """Return 200 only if both Postgres and Redis are reachable, else 503."""
    settings = get_settings()

    try:
        postgres_ok = await ping_database(request.app.state.engine)
    except Exception:
        postgres_ok = False

    try:
        redis_ok = await redis.ping()
    except Exception:
        redis_ok = False

    healthy = postgres_ok and redis_ok
    payload = {
        "status": "ok" if healthy else "unhealthy",
        "checks": {"postgres": postgres_ok, "redis": redis_ok},
        "llm_mode": settings.llm_mode,
    }
    return JSONResponse(status_code=200 if healthy else 503, content=payload)


def _conversation_not_found() -> HTTPException:
    # Ownership mismatches deliberately look identical to absent ids to prevent
    # conversation enumeration.
    return HTTPException(
        status_code=404,
        detail="Conversation not found",
        headers={"Cache-Control": "private, no-store", "Vary": "X-Client-ID"},
    )


def _disable_conversation_caching(response: Response) -> None:
    """Conversation payloads are private to the browser-scoping header."""

    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "X-Client-ID"


async def _check_chat_access(
    *,
    db: AsyncSession,
    session_id: str,
    client_id: str | None,
) -> ConversationStore:
    store = ConversationStore(db)
    try:
        await store.ensure_chat_access(session_id, client_id)
    except ConversationNotFoundError:
        raise _conversation_not_found() from None
    return store


@app.get("/api/v1/conversations", response_model=ConversationListResponse)
async def list_conversations(
    response: Response,
    client_id: RequiredClientId,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_session),
) -> ConversationListResponse:
    _disable_conversation_caching(response)
    return await ConversationStore(db).list_conversations(
        client_id,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(
    session_id: ConversationId,
    client_id: RequiredClientId,
    response: Response,
    db: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    _disable_conversation_caching(response)
    try:
        return await ConversationStore(db).get_conversation(session_id, client_id)
    except ConversationNotFoundError:
        raise _conversation_not_found() from None


@app.delete(
    "/api/v1/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    session_id: ConversationId,
    client_id: RequiredClientId,
    db: AsyncSession = Depends(get_session),
    redis: RedisClient = Depends(get_redis),
) -> Response:
    try:
        await ConversationStore(db).delete_conversation(session_id, client_id)
    except ConversationNotFoundError:
        raise _conversation_not_found() from None
    await db.commit()
    try:
        await redis.delete_session(session_id)
    except Exception:
        logger.warning(
            "conversation_cache_delete_failed session_id=%s",
            session_id,
            exc_info=True,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/v1/chat", response_model=AgentResponse)
async def chat(
    request: Request,
    payload: AgentRequest,
    client_id: OptionalClientId = None,
    db: AsyncSession = Depends(get_session),
    redis: RedisClient = Depends(get_redis),
) -> AgentResponse:
    """Run the complete Phase 0 pipeline for one user message."""
    lock_owner = str(uuid.uuid4())
    if not await redis.acquire_request_lock(payload.session_id, lock_owner):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=REQUEST_IN_PROGRESS_DETAIL,
        )

    tools = request.app.state.tools
    try:
        conversation_store = await _check_chat_access(
            db=db,
            session_id=payload.session_id,
            client_id=client_id,
        )
        return await run_pipeline(
            payload,
            db=db,
            redis=redis,
            censorship_gate=request.app.state.censorship_gate,
            scope_gate=request.app.state.scope_gate,
            tool_executor=request.app.state.tool_executor,
            tool_specs=[tool.spec for tool in tools.values()],
            place_store=request.app.state.place_store,
            source_store=request.app.state.source_store,
            user_location_reverse_geocoder=request.app.state.user_location_reverse_geocoder,
            client_id=client_id,
            conversation_store=conversation_store,
        )
    finally:
        await redis.release_request_lock(payload.session_id, lock_owner)


@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: Request,
    payload: AgentRequest,
    client_id: OptionalClientId = None,
    db: AsyncSession = Depends(get_session),
    redis: RedisClient = Depends(get_redis),
) -> StreamingResponse:
    """Stream pipeline progress and answer deltas, followed by the full response.

    Each line is one JSON object. The terminal event is either ``result`` with
    the unchanged :class:`AgentResponse`, or ``error`` if the pipeline failed.
    ``answer_reset`` tells clients to discard streamed text that is not safe or
    did not become the final answer.
    """
    lock_owner = str(uuid.uuid4())
    if not await redis.acquire_request_lock(payload.session_id, lock_owner):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=REQUEST_IN_PROGRESS_DETAIL,
        )

    try:
        conversation_store = await _check_chat_access(
            db=db,
            session_id=payload.session_id,
            client_id=client_id,
        )
    except Exception:
        await redis.release_request_lock(payload.session_id, lock_owner)
        raise

    tools = request.app.state.tools

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        release_task: asyncio.Task[bool] | None = None

        async def release_lock() -> None:
            """Release once, and let cleanup survive cancellation of the stream.

            Client disconnects cancel the response iterator. Keeping the Redis
            operation in its own shielded task prevents that cancellation from
            leaving the session locked until its TTL expires.
            """
            nonlocal release_task
            if release_task is None:
                release_task = asyncio.create_task(
                    redis.release_request_lock(payload.session_id, lock_owner)
                )
            await asyncio.shield(release_task)

        def report(event: dict[str, object]) -> None:
            queue.put_nowait(event)

        async def run() -> None:
            try:
                response = await run_pipeline(
                    payload,
                    db=db,
                    redis=redis,
                    censorship_gate=request.app.state.censorship_gate,
                    scope_gate=request.app.state.scope_gate,
                    tool_executor=request.app.state.tool_executor,
                    tool_specs=[tool.spec for tool in tools.values()],
                    place_store=request.app.state.place_store,
                    source_store=request.app.state.source_store,
                    user_location_reverse_geocoder=request.app.state.user_location_reverse_geocoder,
                    progress_callback=report,
                    client_id=client_id,
                    conversation_store=conversation_store,
                )
                await queue.put({"type": "result", "data": response.model_dump(mode="json")})
            except Exception:
                logger.exception(
                    "chat_stream_pipeline_failed session_id=%s",
                    payload.session_id,
                )
                await queue.put(
                    {
                        "type": "error",
                        "message": "Pipeline request failed. Please retry.",
                    }
                )
            finally:
                try:
                    await release_lock()
                finally:
                    await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=STREAM_HEARTBEAT_SECONDS,
                    )
                except TimeoutError:
                    # Keep proxies and the UI's read timeout from mistaking a
                    # slow provider turn for a dead connection. This carries no
                    # model content and is intentionally not persisted.
                    yield json.dumps({"type": "heartbeat"}) + "\n"
                    continue
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Also await cleanup here: if `run` was cancelled before its first
            # instruction, its own finally block never had a chance to execute.
            await release_lock()

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
