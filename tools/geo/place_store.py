"""Shared storage contract and in-memory store for opaque place records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tools.refs import PlaceRecord, PlaceRef


@runtime_checkable
class PlaceStore(Protocol):
    """Store and retrieve full place records by their opaque references."""

    async def save(self, record: PlaceRecord) -> None:
        """Persist a place record under its ref."""
        ...

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        """Persist a batch of place records."""
        ...

    async def get(self, ref: PlaceRef) -> PlaceRecord | None:
        """Return a record by ref, or None when it does not exist."""
        ...


class InMemoryPlaceStore(PlaceStore):
    """Local development store for opaque place refs."""

    def __init__(self) -> None:
        self._records: dict[PlaceRef, PlaceRecord] = {}

    async def save(self, record: PlaceRecord) -> None:
        self._records[record.ref] = record

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        self._records.update({record.ref: record for record in records})

    async def get(self, ref: PlaceRef) -> PlaceRecord | None:
        return self._records.get(ref)
