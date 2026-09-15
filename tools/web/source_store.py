"""Storage contract for source records hidden from the model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tools.refs import SourceRecord, SourceRef


@runtime_checkable
class SourceStore(Protocol):
    async def save_many(self, sources: Sequence[SourceRecord]) -> None:
        """Persist a batch of source records."""
        ...

    async def get(self, ref: SourceRef) -> SourceRecord | None:
        """Return a source record by ref, or None when it does not exist."""
        ...
