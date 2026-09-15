"""Service contract for internal place geocoding."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tools.geo.geocoding.schemas import GeocodePlaceInput, GeocodePlaceOutput
from tools.observability import ToolExecutionContext


@runtime_checkable
class GeocoderService(Protocol):
    """Internal service that resolves free-text places into place refs."""

    provider: str

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        """Resolve a normalized geocoding request.

        Implementations may call Yandex Geocoder, check Redis cache, or return
        deterministic fake data in tests. The model should not call this service
        directly.
        """
        ...
