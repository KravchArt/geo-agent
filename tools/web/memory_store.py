"""In-memory source store for local development and tests."""

from __future__ import annotations

from collections.abc import Sequence

from tools.refs import SourceRecord, SourceRef
from tools.web.source_store import SourceStore


class InMemorySourceStore(SourceStore):
    def __init__(self) -> None:
        self._sources: dict[SourceRef, SourceRecord] = {}

    async def save_many(self, sources: Sequence[SourceRecord]) -> None:
        for source in sources:
            self._sources[source.ref] = source

    async def get(self, ref: SourceRef) -> SourceRecord | None:
        return self._sources.get(ref)
