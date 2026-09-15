from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

import backend.app.main as main_module
from backend.app.db.session import get_session
from backend.app.main import app
from backend.app.redis.client import RedisClient, get_redis
from common.models import AgentRequest


class _Response:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "request_id": "req-1",
            "session_id": "session-1",
            "status": "completed",
            "rejection_reason": None,
            "answer": "done",
            "gates": {},
            "llm": None,
        }


class _RedisStub:
    def __init__(self, *, acquired: bool = True) -> None:
        self.acquired = acquired
        self.acquire_calls: list[tuple[str, str]] = []
        self.release_calls: list[tuple[str, str]] = []

    async def acquire_request_lock(self, session_id: str, owner: str) -> bool:
        self.acquire_calls.append((session_id, owner))
        return self.acquired

    async def release_request_lock(self, session_id: str, owner: str) -> bool:
        self.release_calls.append((session_id, owner))
        return True


async def test_stream_endpoint_emits_ndjson_progress_and_terminal_result(monkeypatch):
    redis = _RedisStub()

    async def session_override():
        yield object()

    async def redis_override():
        return redis

    async def fake_run_pipeline(*_args: Any, **kwargs: Any) -> _Response:
        report = kwargs["progress_callback"]
        report({"type": "answer_delta", "delta": "do"})
        report({"type": "answer_delta", "delta": "ne"})
        report({"type": "answer_reset"})
        report(
            {
                "type": "progress",
                "stage": "react",
                "status": "running",
                "message": "ReAct step 1",
            }
        )
        report(
            {
                "type": "progress",
                "stage": "tool",
                "status": "completed",
                "message": "Tool echo completed.",
                "tool_name": "echo",
                "ok": True,
            }
        )
        return _Response()

    async def fake_check_chat_access(**_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main_module, "_check_chat_access", fake_check_chat_access)
    for name, value in {
        "tools": {},
        "censorship_gate": None,
        "scope_gate": None,
        "tool_executor": object(),
        "place_store": None,
        "source_store": None,
        "user_location_reverse_geocoder": None,
    }.items():
        monkeypatch.setattr(app.state, name, value, raising=False)

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_redis] = redis_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/stream",
                json={"session_id": "session-1", "message": "Plan a trip"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["x-accel-buffering"] == "no"
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == [
        "answer_delta",
        "answer_delta",
        "answer_reset",
        "progress",
        "progress",
        "result",
    ]
    assert "".join(event["delta"] for event in events[:2]) == "done"
    assert events[3]["stage"] == "react"
    assert events[4]["tool_name"] == "echo"
    assert events[-1]["data"]["answer"] == "done"
    assert len(redis.acquire_calls) == 1
    assert redis.release_calls == redis.acquire_calls


async def test_closing_stream_releases_session_lock(monkeypatch):
    redis = _RedisStub()
    pipeline_started = asyncio.Event()

    async def fake_run_pipeline(*_args: Any, **kwargs: Any) -> _Response:
        kwargs["progress_callback"](
            {
                "type": "progress",
                "stage": "react",
                "status": "running",
                "message": "Waiting for the model",
            }
        )
        pipeline_started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled pipeline must not finish normally")

    async def fake_check_chat_access(**_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main_module, "_check_chat_access", fake_check_chat_access)
    state = SimpleNamespace(
        tools={},
        censorship_gate=None,
        scope_gate=None,
        tool_executor=object(),
        place_store=None,
        source_store=None,
        user_location_reverse_geocoder=None,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    payload = AgentRequest(session_id="session-1", message="Plan a trip")

    response = await main_module.chat_stream(
        request=cast(Request, request),
        payload=payload,
        client_id=None,
        db=cast(AsyncSession, object()),
        redis=cast(RedisClient, redis),
    )
    iterator = cast(AsyncGenerator[str, None], response.body_iterator)
    first = await anext(iterator)
    assert json.loads(first)["type"] == "progress"
    await pipeline_started.wait()

    waiting_for_next_event = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0)
    waiting_for_next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_for_next_event

    assert len(redis.acquire_calls) == 1
    assert redis.release_calls == redis.acquire_calls


async def test_stream_emits_heartbeat_while_pipeline_is_silent(monkeypatch):
    redis = _RedisStub()
    pipeline_started = asyncio.Event()

    async def fake_run_pipeline(*_args: Any, **_kwargs: Any) -> _Response:
        pipeline_started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled pipeline must not finish normally")

    async def fake_check_chat_access(**_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(main_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main_module, "_check_chat_access", fake_check_chat_access)
    monkeypatch.setattr(main_module, "STREAM_HEARTBEAT_SECONDS", 0.01)
    state = SimpleNamespace(
        tools={},
        censorship_gate=None,
        scope_gate=None,
        tool_executor=object(),
        place_store=None,
        source_store=None,
        user_location_reverse_geocoder=None,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    response = await main_module.chat_stream(
        request=cast(Request, request),
        payload=AgentRequest(session_id="session-1", message="Plan a trip"),
        client_id=None,
        db=cast(AsyncSession, object()),
        redis=cast(RedisClient, redis),
    )
    iterator = cast(AsyncGenerator[str, None], response.body_iterator)

    next_event = asyncio.create_task(anext(iterator))
    await pipeline_started.wait()
    assert json.loads(await next_event) == {"type": "heartbeat"}
    await iterator.aclose()

    assert redis.release_calls == redis.acquire_calls


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/chat/stream"])
async def test_chat_endpoints_reject_a_second_in_flight_request(path: str, monkeypatch):
    redis = _RedisStub(acquired=False)

    async def session_override():
        yield object()

    async def redis_override():
        return redis

    async def pipeline_must_not_run(*_args: Any, **_kwargs: Any) -> _Response:
        raise AssertionError("pipeline must not run when the session is busy")

    monkeypatch.setattr(main_module, "run_pipeline", pipeline_must_not_run)
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_redis] = redis_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                path,
                json={"session_id": "session-1", "message": "Another request"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json() == {"detail": main_module.REQUEST_IN_PROGRESS_DETAIL}
    assert len(redis.acquire_calls) == 1
    assert redis.release_calls == []
