from __future__ import annotations

from tools.geo import GeocodePlaceInput, GeocodePlaceOutput, GeocoderService
from tools.observability import ToolExecutionContext


class FakeGeocoderService:
    provider = "fake_geocoder"

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        return GeocodePlaceOutput()


async def test_fake_geocoder_service_matches_protocol():
    """Verify that fake geocoder service matches protocol."""

    service = FakeGeocoderService()

    assert isinstance(service, GeocoderService)

    result = await service.geocode(
        GeocodePlaceInput(query="Красная площадь"),
        ToolExecutionContext(),
    )

    assert result == GeocodePlaceOutput()
