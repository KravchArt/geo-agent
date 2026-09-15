"""Redis access layer: a thin, namespaced async client wrapper.

Interface + connection management only. Echo-grounding logic (how a tool call is
hashed, what gets cached) is intentionally NOT implemented here — that belongs to
a later phase.
"""

from backend.app.redis.client import RedisClient, get_redis

__all__ = ["RedisClient", "get_redis"]
