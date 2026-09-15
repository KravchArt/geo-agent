"""OpenStreetMap-backed places-search provider tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import parse_qs

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError
from tools.geo.geocoding.schemas import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ToponymKind,
)
from tools.geo.geocoding.service import GeocoderService
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search.osm.client import OsmOverpassClient
from tools.geo.places_search.osm.provider import OsmPlacesSearchProvider, _address
from tools.geo.places_search.schemas import PlacesSearchInput, SearchMode
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref


class NeverGeocoder:
    provider = "never"

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        raise AssertionError("preloaded refs must not call the geocoder")


class RecordingReverseGeocoder:
    provider = "reverse"

    def __init__(
        self,
        addresses: dict[str, str],
        *,
        kinds: dict[str, ToponymKind] | None = None,
    ) -> None:
        self.addresses = addresses
        self.kinds = kinds or {}
        self.queries: list[GeocodePlaceInput] = []

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.queries.append(args)
        address = self.addresses.get(args.query)
        if address is None:
            return GeocodePlaceOutput()

        match = PlaceMatch(
            ref=mint_place_ref(f"reverse:{args.query}"),
            name=address,
            address=address,
            kind=self.kinds.get(args.query, ToponymKind.HOUSE),
        )
        return GeocodePlaceOutput(best=match, matches=[match])


def _provider(
    *,
    http_client: httpx.AsyncClient,
    store: InMemoryPlaceStore,
    geocoder: GeocoderService | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OsmPlacesSearchProvider:
    selected_geocoder = geocoder or NeverGeocoder()
    return OsmPlacesSearchProvider(
        client=OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test",
        ),
        place_store=store,
        geocoder=selected_geocoder,
        clock=clock or (lambda: datetime.now(UTC)),
    )


def _payload(*elements: dict[str, object], total: int) -> dict[str, object]:
    return {
        "version": 0.6,
        "generator": "Overpass API 0.7.62.4",
        "osm3s": {
            "timestamp_osm_base": "2026-07-23T10:20:52Z",
            "copyright": "OpenStreetMap contributors",
        },
        "elements": [
            *elements,
            {
                "type": "count",
                "id": 0,
                "tags": {
                    "nodes": str(total),
                    "ways": "0",
                    "relations": "0",
                    "total": str(total),
                },
            },
        ],
    }


def _area(*, area_id: int = 3_600_000_900) -> dict[str, object]:
    return {
        "type": "area",
        "id": area_id,
    }


def test_address_removes_postcode_and_rejects_postcode_only():
    assert (
        _address(
            {
                "addr:full": "Москва, Лесная улица 8, 123060",
                "addr:postcode": "123060",
                "addr:housenumber": "8",
            }
        )
        == "Москва, Лесная улица 8"
    )
    assert (
        _address(
            {
                "addr:full": "Москва, Лесная улица 8, 123060",
                "addr:housenumber": "8",
            }
        )
        == "Москва, Лесная улица 8"
    )
    assert _address({"addr:postcode": "123060"}) == ""
    assert _address({"addr:full": "123060"}) == ""
    assert _address({"addr:street": "Лесная улица"}) == ""
    assert _address({"addr:housenumber": "8"}) == ""


async def test_provider_open_now_keeps_only_confirmed_open_places() -> None:
    area_ref = mint_place_ref("area:osm-open-now")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.75,
            lon=37.62,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=37.3, south=55.5, east=37.9, north=55.9),
            origin=RecordOrigin.GEOCODE,
        )
    )
    payload = _payload(
        _area(),
        {
            "type": "node",
            "id": 501,
            "lat": 55.75,
            "lon": 37.62,
            "tags": {
                "name": "Открытый ресторан",
                "amenity": "restaurant",
                "addr:street": "Первая улица",
                "addr:housenumber": "1",
                "opening_hours": "Sa 14:00-16:00",
            },
        },
        {
            "type": "node",
            "id": 502,
            "lat": 55.751,
            "lon": 37.621,
            "tags": {
                "name": "Закрытый ресторан",
                "amenity": "restaurant",
                "addr:street": "Вторая улица",
                "addr:housenumber": "2",
                "opening_hours": "Sa 16:00-18:00",
            },
        },
        {
            "type": "node",
            "id": 503,
            "lat": 55.752,
            "lon": 37.622,
            "tags": {
                "name": "Ресторан с ошибкой в расписании",
                "amenity": "restaurant",
                "addr:street": "Третья улица",
                "addr:housenumber": "3",
                "opening_hours": "not a valid schedule",
            },
        },
        total=3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.content.decode())["data"][0]
        assert '["amenity"="restaurant"]["opening_hours"]' in query
        assert '["opening_hours"="24/7"]' not in query
        return httpx.Response(200, json=payload)

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = _provider(
            http_client=http_client,
            store=store,
            clock=lambda: datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="рестораны",
                category="restaurant",
                area_ref=area_ref,
                open_now=True,
            ),
            context,
        )

    assert provider.supports_open_now is True
    assert [place.id for place in result.places] == ["node/501"]
    assert result.places[0].is_open_now is True
    assert await store.get(mint_place_ref("osm:node/502")) is None
    assert await store.get(mint_place_ref("osm:node/503")) is None
    assert context.warnings == (
        "OpenStreetMap schedules could not verify the current opening status of 1 result(s); "
        "those results were omitted.",
    )


async def test_area_search_maps_tags_counts_results_and_persists_ref():
    area_ref = mint_place_ref("yandex:geo:moscow")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.75,
            lon=37.62,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    payload = _payload(
        _area(),
        {
            "type": "node",
            "id": 101,
            "lat": 55.75,
            "lon": 37.62,
            "tags": {
                "name": "Простая кофейня",
                "amenity": "cafe",
                "cuisine": "coffee_shop",
                "addr:street": "Улица без графика",
            },
        },
        {
            "type": "node",
            "id": 102,
            "lat": 55.751,
            "lon": 37.621,
            "tags": {
                "name:ru": "Кофейня с данными",
                "name": "Cafe with data",
                "amenity": "cafe",
                "cuisine": "coffee_shop",
                "addr:street": "Никольская улица",
                "addr:housenumber": "10",
                "addr:postcode": "101000",
                "contact:phone": "+7 495 111-22-33",
                "opening_hours": "24/7",
                "wheelchair": "yes",
                "website": "https://example.test",
            },
        },
        {
            "type": "node",
            "id": 103,
            "lat": 55.752,
            "lon": 37.622,
            "tags": {
                "name": "Кофейня только с индексом",
                "amenity": "cafe",
                "cuisine": "coffee_shop",
                "addr:postcode": "123060",
            },
        },
        total=3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.content.decode())["data"][0]
        assert 'nwr["amenity"="cafe"]["cuisine"~"(^|;)coffee_shop(;|$)"]' in query
        assert (
            'rel["boundary"="administrative"]["name"="Москва"]'
            "(55.100000,36.800000,56.000000,38.000000);" in query
        )
        assert ".searchBoundaries map_to_area ->.searchArea;" in query
        assert "(area.searchArea)" in query
        assert "out body geom" not in query
        assert '["addr:city"' not in query
        assert '["addr:place"' not in query
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = _provider(http_client=http_client, store=store)
        result = await provider.search(
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="кофейни",
                category="coffee_shop",
                area_ref=area_ref,
                limit=2,
            ),
            context,
        )

    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].operation == "search"
    assert result.returned_count == 1
    # All three raw elements were received. One lacked a usable address and was
    # filtered locally, so the normalized response itself was not truncated.
    assert result.truncated is False
    assert result.area is not None
    assert result.area.ref == area_ref
    assert [place.id for place in result.places] == ["node/102"]

    place = result.places[0]
    assert place.id == "node/102"
    assert place.name == "Кофейня с данными"
    assert place.address == "Никольская улица 10"
    assert place.categories == ["coffee_shop", "cafe"]
    assert place.phones == ["+7 495 111-22-33"]
    assert place.hours_text == "24/7"
    assert place.open_24h is True
    assert place.accessibility == ["wheelchair_access"]

    record = await store.get(place.ref)
    assert record is not None
    assert record.provider == "osm"
    assert record.provider_id == "node/102"
    assert record.provider_uri == "https://www.openstreetmap.org/node/102"


async def test_area_search_preserves_provider_order_with_different_completeness():
    area_ref = mint_place_ref("yandex:geo:moscow-order")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.75,
            lon=37.62,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    payload = _payload(
        _area(area_id=3_600_000_910),
        {
            "type": "node",
            "id": 301,
            "lat": 55.75,
            "lon": 37.62,
            "tags": {
                "name": "Яблоко",
                "amenity": "restaurant",
                "addr:street": "Первая улица",
                "addr:housenumber": "1",
                "opening_hours": "Mo-Fr 09:00-18:00",
            },
        },
        {
            "type": "node",
            "id": 302,
            "lat": 55.751,
            "lon": 37.621,
            "tags": {
                "name": "Альфа",
                "amenity": "restaurant",
                "addr:street": "Вторая улица",
                "addr:housenumber": "2",
                "phone": "+7 495 000-00-00",
                "opening_hours": "24/7",
                "website": "https://example.test",
                "wikidata": "Q1",
                "brand": "Полный набор данных",
            },
        },
        total=2,
    )

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = _provider(http_client=http_client, store=store)
        result = await provider.search(
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="рестораны",
                category="restaurant",
                area_ref=area_ref,
                limit=2,
            ),
            context,
        )

    assert [place.name for place in result.places] == ["Яблоко", "Альфа"]


async def test_area_search_reports_missing_overpass_area():
    area_ref = mint_place_ref("yandex:geo:missing-osm-area")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Несуществующий город",
            address="Россия, Несуществующий город",
            lat=55.75,
            lon=37.62,
            kind="locality",
            locality="Несуществующий город",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_payload(total=0)))
    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = _provider(http_client=http_client, store=store)
        with pytest.raises(ToolExecutionError) as exc_info:
            await provider.search(
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="рестораны",
                    category="restaurant",
                    area_ref=area_ref,
                ),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND


async def test_near_search_sorts_by_distance_and_echoes_anchor():
    anchor_ref = mint_place_ref("yandex:geo:red-square")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Красная площадь",
            address="Москва, Красная площадь",
            lat=55.75,
            lon=37.62,
            origin=RecordOrigin.GEOCODE,
        )
    )

    payload = _payload(
        {
            "type": "node",
            "id": 201,
            "lat": 55.758,
            "lon": 37.62,
            "tags": {
                "name": "Дальняя кофейня",
                "amenity": "cafe",
                "cuisine": "coffee_shop",
                "addr:street": "Дальняя улица",
                "addr:housenumber": "1",
                "opening_hours": "Mo-Fr 09:00-18:00",
            },
        },
        {
            "type": "way",
            "id": 202,
            "center": {"lat": 55.751, "lon": 37.62},
            "tags": {
                "name": "Ближняя кофейня",
                "amenity": "cafe",
                "cuisine": "coffee_shop",
                "addr:street": "Ближняя улица",
                "addr:housenumber": "2",
                "phone": "+7 495 000-00-00",
                "opening_hours": "24/7",
                "website": "https://example.test",
            },
        },
        total=2,
    )

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = _provider(http_client=http_client, store=store)
        result = await provider.search(
            PlacesSearchInput(
                mode=SearchMode.NEAR,
                query="кофейни",
                category="coffee_shop",
                near=anchor_ref,
                radius_m=2_000,
                limit=2,
            ),
            ToolExecutionContext(),
        )

    assert result.anchor == anchor_ref
    assert [place.id for place in result.places] == ["way/202", "node/201"]
    assert result.places[0].distance_m is not None
    assert result.places[1].distance_m is not None
    assert result.places[0].distance_m < result.places[1].distance_m


async def test_near_search_excludes_explicitly_different_locality():
    anchor_ref = mint_place_ref("yandex:geo:nizhny-novgorod")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Площадь Революции, 2",
            address="Нижний Новгород, площадь Революции, 2",
            lat=56.32236,
            lon=43.947147,
            locality="Нижний Новгород",
            origin=RecordOrigin.GEOCODE,
        )
    )
    payload = _payload(
        {
            "type": "relation",
            "id": 7779491,
            "center": {"lat": 56.3190899, "lon": 43.931384},
            "tags": {
                "name": "ТПУ «Канавинский»",
                "amenity": "bus_station",
            },
        },
        {
            "type": "way",
            "id": 166745310,
            "center": {"lat": 56.3557941, "lon": 44.0737089},
            "tags": {
                "name": "Автостанция Бор",
                "amenity": "bus_station",
                "addr:city": "Бор",
                "addr:street": "улица Крупской",
                "addr:housenumber": "21",
            },
        },
        total=2,
    )

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as http_client:
        result = await _provider(
            http_client=http_client,
            store=store,
            geocoder=RecordingReverseGeocoder(
                {
                    "43.931384,56.319090": (
                        "Россия, городской округ Нижний Новгород, Канавинский район"
                    )
                },
                kinds={"43.931384,56.319090": ToponymKind.DISTRICT},
            ),
        ).search(
            PlacesSearchInput(
                mode=SearchMode.NEAR,
                query="автовокзалы",
                category="bus_station",
                near=anchor_ref,
                radius_m=10_000,
                limit=7,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["relation/7779491"]


async def test_near_search_reverse_geocodes_only_addressless_top_three():
    anchor_ref = mint_place_ref("yandex:geo:reverse-anchor")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Точка поиска",
            address="Казань, Точка поиска",
            lat=55.75,
            lon=37.62,
            origin=RecordOrigin.GEOCODE,
        )
    )

    payload = _payload(
        {
            "type": "node",
            "id": 401,
            "lat": 55.751,
            "lon": 37.62,
            "tags": {"name": "Первая аптека", "amenity": "pharmacy"},
        },
        {
            "type": "node",
            "id": 402,
            "lat": 55.752,
            "lon": 37.62,
            "tags": {
                "name": "Вторая аптека",
                "amenity": "pharmacy",
                "addr:street": "Вторая улица",
                "addr:housenumber": "2",
            },
        },
        {
            "type": "node",
            "id": 403,
            "lat": 55.753,
            "lon": 37.62,
            "tags": {"name": "Третья аптека", "amenity": "pharmacy"},
        },
        {
            "type": "node",
            "id": 404,
            "lat": 55.754,
            "lon": 37.62,
            "tags": {"name": "Четвёртая аптека", "amenity": "pharmacy"},
        },
        {
            "type": "node",
            "id": 405,
            "lat": 55.755,
            "lon": 37.62,
            "tags": {
                "name": "Пятая аптека",
                "amenity": "pharmacy",
                "addr:street": "Пятая улица",
                "addr:housenumber": "5",
            },
        },
        total=5,
    )
    geocoder = RecordingReverseGeocoder(
        {
            "37.620000,55.751000": "Россия, Казань, Первая улица, 1, 420000",
            "37.620000,55.753000": "Россия, Казань, Третья улица, 3, 420003",
        },
        kinds={"37.620000,55.753000": ToponymKind.STREET},
    )

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = _provider(
            http_client=http_client,
            store=store,
            geocoder=geocoder,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode=SearchMode.NEAR,
                query="аптеки",
                category="pharmacy",
                near=anchor_ref,
                radius_m=2_000,
                limit=5,
            ),
            context,
        )

    assert [query.query for query in geocoder.queries] == [
        "37.620000,55.751000",
        "37.620000,55.753000",
    ]
    assert [place.id for place in result.places] == [
        "node/401",
        "node/402",
        "node/403",
        "node/404",
        "node/405",
    ]
    assert result.places[0].address == "Россия, Казань, Первая улица, 1"
    assert result.places[2].address == "Россия, Казань, Третья улица, 3"
    assert result.places[3].address == ""
    assert context.warnings == (
        "The geocoder could only provide an approximate address for 1 "
        "OpenStreetMap result(s); routing still uses the exact provider "
        "coordinates stored under each ref.",
        "OpenStreetMap supplied no address for 1 result(s); routing still uses "
        "the exact provider coordinates stored under each ref.",
    )
