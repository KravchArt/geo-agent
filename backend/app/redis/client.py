"""Thin async Redis wrapper with namespaced keys and a TTL policy.

Key namespaces
--------------
* ``session:<session_id>:<field>`` — per-session scratch state.
* ``echo:<tool_hash>``             — echo-grounding map: hash -> tool payload.

The client is **owned by the application lifespan** and lives on ``app.state``
(see :mod:`backend.app.main`). No process-wide singleton: a Redis connection
binds to the event loop that created it, and unlike SQLAlchemy (which hides this
behind ``pool_pre_ping``) redis-py does not recover — a leaked client is simply
dead in the next loop.

Only the client + key/TTL conventions live here. Serialization of values and the
hashing scheme for echo-grounding are the caller's responsibility (TODO: phase 1).
"""

from __future__ import annotations

import json
from typing import cast

from fastapi import Request
from redis.asyncio import Redis

from backend.app.config import Settings, get_settings

SESSION_NS = "session"
ECHO_NS = "echo"
_SESSION_FIELDS = (
    "history",
    "last_user_message",
    "last_gate_results",
    "last_pipeline_status",
    "last_answer",
)


class RedisClient:
    """Namespaced async Redis client. No business logic."""

    def __init__(
        self,
        url: str,
        *,
        session_ttl: int,
        echo_ttl: int,
        ref_ttl: int = 86_400,
        request_lock_ttl: int = 600,
    ) -> None:
        self._redis: Redis = Redis.from_url(url, decode_responses=True)
        self._session_ttl = session_ttl
        self._echo_ttl = echo_ttl
        self._ref_ttl = ref_ttl
        self._request_lock_ttl = request_lock_ttl

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RedisClient:
        settings = settings or get_settings()
        return cls(
            settings.redis_url,
            session_ttl=settings.redis_session_ttl,
            echo_ttl=settings.redis_echo_ttl,
            ref_ttl=settings.redis_ref_ttl,
            request_lock_ttl=settings.agent_request_lock_ttl,
        )

    @property
    def ref_ttl(self) -> int:
        return self._ref_ttl

    # --- health ---
    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    # --- key builders ---
    @staticmethod
    def session_key(session_id: str, field: str) -> str:
        return f"{SESSION_NS}:{session_id}:{field}"

    @staticmethod
    def echo_key(tool_hash: str) -> str:
        return f"{ECHO_NS}:{tool_hash}"

    # --- session state ---
    async def set_session_value(
        self, session_id: str, field: str, value: str, ttl: int | None = None
    ) -> None:
        await self._redis.set(
            self.session_key(session_id, field), value, ex=ttl or self._session_ttl
        )

    async def get_session_value(self, session_id: str, field: str) -> str | None:
        value = await self._redis.get(self.session_key(session_id, field))
        return None if value is None else str(value)

    async def acquire_request_lock(self, session_id: str, owner: str) -> bool:
        """Atomically reserve a session for one in-flight agent request."""
        acquired = await self._redis.set(
            self.session_key(session_id, "request_lock"),
            owner,
            ex=self._request_lock_ttl,
            nx=True,
        )
        return bool(acquired)

    async def release_request_lock(self, session_id: str, owner: str) -> bool:
        """Release only the lock owned by this request.

        Comparing and deleting in one Lua operation prevents an expired request
        from deleting a newer request's lock.
        """
        key = self.session_key(session_id, "request_lock")
        deleted = await self._redis.eval(
            """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            end
            return 0
            """,
            1,
            key,
            owner,
        )
        return bool(deleted)

    async def delete_session(self, session_id: str) -> None:
        """Delete only the known per-session cache keys.

        Place/source refs and echo records are global content-addressed data and
        intentionally remain untouched.
        """

        await self._redis.delete(
            *(self.session_key(session_id, field) for field in _SESSION_FIELDS)
        )

    # --- echo-grounding map (hash -> tool payload) ---
    async def set_echo(self, tool_hash: str, value: str, ttl: int | None = None) -> None:
        await self._redis.set(self.echo_key(tool_hash), value, ex=ttl or self._echo_ttl)

    async def get_echo(self, tool_hash: str) -> str | None:
        value = await self._redis.get(self.echo_key(tool_hash))
        return None if value is None else str(value)

    # --- conversation history (newest first, capped) ---
    async def append_history(
        self, session_id: str, role: str, content: str, *, limit: int = 20
    ) -> None:
        """Record one turn, keeping only the most recent ``limit`` entries.

        A capped list, not a growing log: this exists to give the gates and the
        model enough context to resolve a follow-up like "и что рядом?", not to
        be a transcript store.
        """
        key = self.session_key(session_id, "history")
        await self._redis.lpush(key, json.dumps({"role": role, "content": content}))
        await self._redis.ltrim(key, 0, limit - 1)
        await self._redis.expire(key, self._session_ttl)

    async def get_history(self, session_id: str, *, limit: int = 6) -> list[dict[str, str]]:
        """Return the last turns in chronological order (oldest first)."""
        raw = await self._redis.lrange(self.session_key(session_id, "history"), 0, limit - 1)
        turns: list[dict[str, str]] = []
        for item in reversed(raw):
            try:
                turns.append(json.loads(item))
            except (TypeError, ValueError):
                continue
        return turns

    # --- generic namespaced values (used by the ref stores: place:/source:) ---
    async def set_namespaced(
        self, namespace: str, key: str, value: str, ttl: int | None = None
    ) -> None:
        await self._redis.set(f"{namespace}:{key}", value, ex=ttl or self._ref_ttl)

    async def get_namespaced(self, namespace: str, key: str) -> str | None:
        value = await self._redis.get(f"{namespace}:{key}")
        return None if value is None else str(value)

    async def close(self) -> None:
        await self._redis.aclose()


def get_redis(request: Request) -> RedisClient:
    """FastAPI dependency: the app-owned Redis client."""
    return cast(RedisClient, request.app.state.redis)
