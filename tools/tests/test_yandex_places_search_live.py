"""Opt-in live check for Yandex ``open_now`` place filtering.

Run from the repository root with a real organisation-search key in ``.env``:

    RUN_LIVE_PLACES=1 uv run pytest -q \
        tools/tests/test_yandex_places_search_live.py -s

The preloaded Moscow ``area_ref`` avoids geocoding, so this test performs one
real Yandex Organization Search request and exercises only the tool layer.
"""

from __future__ import annotations

import json
import os

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
from tools.geo.places_search.yandex.client import YandexOrganisationSearchClient
from tools.geo.places_search.yandex.provider import YandexPlacesSearchProvider
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


async def test_yandex_returns_only_places_confirmed_open_now() -> None:
    if os.environ.get("RUN_LIVE_PLACES") != "1":
        pytest.skip("set RUN_LIVE_PLACES=1 to call the real Yandex API")

    settings = Settings(app_env="dev", places_search_providers=[])
    if settings.yandex_organisation_search_api_key is None:
        pytest.skip("YANDEX_ORGANISATION_SEARCH_API_KEY is not configured")

    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("live-test:yandex:moscow")
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

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
    ) as http_client:
        provider = YandexPlacesSearchProvider(
            client=YandexOrganisationSearchClient(
                api_key=settings.yandex_organisation_search_api_key,
                http_client=http_client,
            ),
            place_store=store,
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
                query="рестораны",
                category="restaurant",
                area_ref=area_ref,
                open_now=True,
                limit=10,
            ),
            context,
        )

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))

    assert result.places, "Yandex returned no currently open restaurants in Moscow"
    assert all(place.is_open_now is True for place in result.places)
    assert all(place.hours_text is not None for place in result.places)
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "yandex_organisation_search"
    assert context.upstream_calls[0].operation == "search"
