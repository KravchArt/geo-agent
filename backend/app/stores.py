"""Redis-backed stores for the refs the model passes around.

The in-memory stores are fine for a single process, but a ref that stops
resolving is a broken conversation: the model cites ``plc_a1b2c3d4e5`` and the
adapter can no longer turn it into a point. That happens on every restart and on
every second worker. These implementations move the records into Redis under the
namespaces declared in :mod:`tools.refs` (``place:<ref>`` / ``source:<ref>``).

They implement the ``PlaceStore``/``SourceStore`` protocols from ``tools`` — the
dependency points backend -> tools, never the other way round.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.app.redis.client import RedisClient
from tools.geo.place_store import PlaceStore
from tools.refs import PLACE_NS, SOURCE_NS, PlaceRecord, PlaceRef, SourceRecord, SourceRef
from tools.web.source_store import SourceStore


class RedisPlaceStore(PlaceStore):
    """Place records at ``place:<ref>``, JSON-encoded, TTL from settings."""

    def __init__(self, redis: RedisClient, *, ttl: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl

    async def save(self, record: PlaceRecord) -> None:
        await self._redis.set_namespaced(
            PLACE_NS, record.ref, record.model_dump_json(), ttl=self._ttl
        )

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        for record in records:
            await self.save(record)

    async def get(self, ref: PlaceRef) -> PlaceRecord | None:
        raw = await self._redis.get_namespaced(PLACE_NS, ref)
        return None if raw is None else PlaceRecord.model_validate_json(raw)


class RedisSourceStore(SourceStore):
    """Source records at ``source:<ref>``, JSON-encoded, TTL from settings."""

    def __init__(self, redis: RedisClient, *, ttl: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl

    async def save_many(self, sources: Sequence[SourceRecord]) -> None:
        for source in sources:
            await self._redis.set_namespaced(
                SOURCE_NS, source.ref, source.model_dump_json(), ttl=self._ttl
            )

    async def get(self, ref: SourceRef) -> SourceRecord | None:
        raw = await self._redis.get_namespaced(SOURCE_NS, ref)
        return None if raw is None else SourceRecord.model_validate_json(raw)
