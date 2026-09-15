"""Redis wrapper tests."""

from __future__ import annotations

import pytest

from backend.app.redis.client import RedisClient


def test_key_namespacing():
    # Pure, no live Redis needed.
    assert RedisClient.session_key("s1", "step") == "session:s1:step"
    assert RedisClient.echo_key("abc123") == "echo:abc123"


@pytest.mark.integration
async def test_session_roundtrip(redis_client: RedisClient):
    await redis_client.set_session_value("s1", "step", "hello", ttl=60)
    assert await redis_client.get_session_value("s1", "step") == "hello"
    assert await redis_client.get_session_value("s1", "missing") is None


@pytest.mark.integration
async def test_echo_roundtrip(redis_client: RedisClient):
    await redis_client.set_echo("hash123", "tool-payload", ttl=60)
    assert await redis_client.get_echo("hash123") == "tool-payload"


@pytest.mark.integration
async def test_request_lock_allows_only_one_owner(redis_client: RedisClient):
    session_id = "request-lock-test"

    assert await redis_client.acquire_request_lock(session_id, "owner-1") is True
    assert await redis_client.acquire_request_lock(session_id, "owner-2") is False
    assert await redis_client.release_request_lock(session_id, "owner-2") is False
    assert await redis_client.release_request_lock(session_id, "owner-1") is True
    assert await redis_client.acquire_request_lock(session_id, "owner-2") is True
    assert await redis_client.release_request_lock(session_id, "owner-2") is True


@pytest.mark.integration
async def test_delete_session_removes_known_state_and_history(redis_client: RedisClient):
    session_id = "delete-session"
    for field in (
        "last_user_message",
        "last_gate_results",
        "last_pipeline_status",
        "last_answer",
    ):
        await redis_client.set_session_value(session_id, field, "value", ttl=60)
    await redis_client.append_history(session_id, "user", "hello")

    await redis_client.delete_session(session_id)

    assert await redis_client.get_history(session_id) == []
    for field in (
        "last_user_message",
        "last_gate_results",
        "last_pipeline_status",
        "last_answer",
    ):
        assert await redis_client.get_session_value(session_id, field) is None
