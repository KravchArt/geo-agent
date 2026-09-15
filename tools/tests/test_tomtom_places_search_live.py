"""Opt-in live check for TomTom ``open_now`` place filtering.

Run from the repository root with a real ``TOMTOM_API_KEY`` in ``.env``:

    RUN_LIVE_PLACES=1 uv run pytest -q \
        tools/tests/test_tomtom_places_search_live.py -s

The preloaded Moscow ``area_ref`` avoids geocoding, so this test performs one
real TomTom Search request and exercises only the places-search tool layer.
It is an upstream smoke test; deterministic UTC/local-time correctness belongs
to ``test_tomtom_opening_hours.py`` and ``test_tomtom_places_search.py``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import httpx
import pytest

from backend.app.config import Settings
from tools.geo.geocoding import GeocodedPlaceResolver
from tools.geo.geocoding.schemas import GeocodePlaceInput, GeocodePlaceOutput
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search import (
    PlacesSearchCoordinator,
    PlacesSearchScopeResolver,
)
from tools.geo.places_search.schemas import PlacesSearchInput
from tools.geo.places_search.tomtom import TomTomPlacesSearchProvider, TomTomSearchClient
from tools.geo.text_place_resolution import TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref

pytestmark = pytest.mark.live_places


class _NeverGeocoder:
    provider = "never_geocoder"

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        raise AssertionError("preloaded area_ref must not invoke geocoding")


class _RecordingClock:
    """Expose the exact instant the provider used without fixing a stale date."""

    def __init__(self) -> None:
        self.calls = 0
        self.evaluated_at: datetime | None = None

    def __call__(self) -> datetime:
        self.calls += 1
        self.evaluated_at = datetime.now(UTC)
        return self.evaluated_at


async def test_tomtom_returns_only_places_confirmed_open_now() -> None:
    if os.environ.get("RUN_LIVE_PLACES") != "1":
        pytest.skip("set RUN_LIVE_PLACES=1 to call the real TomTom API")

    # Force provider lists to empty so a developer's .env does not trigger
    # unrelated Settings guardrails. The TomTom values are still loaded from it.
    settings = Settings(app_env="dev", places_search_providers=[])
    if settings.tomtom_api_key is None:
        pytest.skip("TOMTOM_API_KEY is not configured")

    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("live-test:moscow")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6173,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(
                west=36.803101,
                south=55.142174,
                east=37.967427,
                north=56.021251,
            ),
            origin=RecordOrigin.GEOCODE,
        )
    )
    geocoder = _NeverGeocoder()
    geocoded_place_resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)
    clock = _RecordingClock()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
    ) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(
                api_key=settings.tomtom_api_key,
                http_client=http_client,
                base_url=settings.tomtom_search_base_url,
            ),
            place_store=store,
            clock=clock,
        )
        coordinator = PlacesSearchCoordinator(
            providers=[provider],
            scope_resolver=PlacesSearchScopeResolver(
                geocoded_place_resolver=geocoded_place_resolver,
                text_place_resolver=TextPlaceResolver(
                    geocoded_place_resolver=geocoded_place_resolver,
                ),
            ),
        )
        context = ToolExecutionContext()
        result = await coordinator.search(
            PlacesSearchInput(
                mode="area",
                query="аптеки",
                category="pharmacy",
                area_ref=area_ref,
                open_now=True,
                limit=10,
            ),
            context,
        )

    assert clock.calls == 1
    assert clock.evaluated_at is not None
    assert clock.evaluated_at.tzinfo is not None
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "tomtom_search"
    assert context.upstream_calls[0].operation == "search"

    print(
        json.dumps(
            {
                "opening_status_evaluated_at": clock.evaluated_at.isoformat(),
                "result": result.model_dump(mode="json"),
                "warnings": list(context.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not result.places:
        pytest.skip(
            "TomTom returned no currently open pharmacies with verifiable schedules; "
            "the live request and provider clock completed successfully"
        )

    assert all(place.is_open_now is True for place in result.places)
    assert all(place.hours_text is not None for place in result.places)
