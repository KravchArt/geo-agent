"""Tests for coverage-aware 2GIS place search and named-anchor selection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search.schemas import PlaceCategory, PlacesSearchInput
from tools.geo.places_search.twogis import (
    TwoGisNamedPoiResolver,
    TwoGisPlacesSearchProvider,
    TwoGisSearchClient,
)
from tools.geo.places_search.twogis.categories import TWOGIS_CATEGORY_SPECS
from tools.geo.places_search.twogis.opening_hours import summarize_schedule
from tools.geo.places_search.twogis.schemas import TwoGisSchedule
from tools.geo.text_place_resolution import PlaceResolutionArea
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref


def _region_payload(
    *,
    covered: bool = True,
    country_code: str = "ru",
    name: str = "Москва",
    settlements: list[str] | None = None,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    if covered:
        items.append(
            {
                "id": "32",
                "name": name,
                "type": "region",
                "country_code": country_code,
                "settlements": settlements if settlements is not None else ["Барвиха"],
                "satellites": [{"name": "Химки"}],
            }
        )
    return {
        "meta": {"api_version": "2.0.test", "code": 200},
        "result": {"items": items, "total": len(items)},
    }


def _item(
    item_id: str,
    *,
    name: str,
    item_type: str,
    lat: float,
    lon: float,
    city: str = "Москва",
    subtype: str | None = None,
    route_type: str | None = None,
    rubric_alias: str | None = None,
    rubric_name: str | None = None,
    rubric_id: str = "rubric",
    address: str | None = None,
    schedule: dict[str, object] | None = None,
    reviews: dict[str, object] | None = None,
    name_ex: dict[str, str] | None = None,
    building_id: str | None = None,
    brand_id: str | None = None,
    brand_name: str | None = None,
    org_id: str | None = None,
    org_name: str | None = None,
    region_id: str | None = None,
    building_name: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": item_id,
        "name": name,
        "type": item_type,
        "point": {"lat": lat, "lon": lon},
        "full_name": f"{city}, {name}",
        "adm_div": [
            {"id": "1", "name": "Россия", "type": "country"},
            {"id": "city", "name": city, "type": "city", "is_default": True},
        ],
    }
    if subtype is not None:
        result["subtype"] = subtype
    if route_type is not None:
        result["route_type"] = route_type
    if rubric_alias is not None and rubric_name is not None:
        result["rubrics"] = [
            {
                "id": rubric_id,
                "alias": rubric_alias,
                "name": rubric_name,
                "kind": "primary",
            }
        ]
    if address is not None:
        result["full_address_name"] = address
    if schedule is not None:
        result["schedule"] = schedule
    if reviews is not None:
        result["reviews"] = reviews
    if name_ex is not None:
        result["name_ex"] = name_ex
    if building_id is not None:
        result["address"] = {"building_id": building_id}
    if building_name is not None:
        result["building_name"] = building_name
    if brand_id is not None:
        result["brand"] = {"id": brand_id, "name": brand_name}
    if org_id is not None:
        result["org"] = {"id": org_id, "name": org_name}
    if region_id is not None:
        result["region_id"] = region_id
    return result


def _items_payload(*items: dict[str, object]) -> dict[str, object]:
    return {
        "meta": {"api_version": "3.0.test", "code": 200},
        "result": {"items": list(items), "total": len(items)},
    }


def _rubric_payload(
    *,
    rubric_id: str = "rubric",
    name: str = "Restaurants",
    alias: str = "restorany",
) -> dict[str, object]:
    return {
        "meta": {"api_version": "2.0.test", "code": 200},
        "result": {
            "items": [{"id": rubric_id, "name": name, "alias": alias}],
            "total": 1,
        },
    }


def _rubric_payload_for_query(query: str) -> dict[str, object]:
    aliases = {
        "кофейни": "kofejjni",
        "музеи": "muzei",
        "рестораны": "restorany",
    }
    return _rubric_payload(name=query, alias=aliases.get(query.casefold(), query.casefold()))


def test_twogis_category_mapping_covers_the_public_taxonomy() -> None:
    assert frozenset(TWOGIS_CATEGORY_SPECS) == frozenset(PlaceCategory)


async def test_rubric_search_selects_all_allowed_aliases_in_mapping_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "meta": {"api_version": "2.0.test", "code": 200},
                "result": {
                    "items": [
                        {"id": "women", "name": "Женская одежда", "alias": "women"},
                        {"id": "noise", "name": "Ателье", "alias": "tailors"},
                        {"id": "men", "name": "Мужская одежда", "alias": "men"},
                    ],
                    "total": 3,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        rubrics = await TwoGisSearchClient(api_key="key", http_client=http_client).find_rubrics(
            query="магазины одежды",
            region_id="32",
            locale="ru_RU",
            allowed_aliases=("men", "women"),
            context=ToolExecutionContext(),
        )

    assert [rubric.id for rubric in rubrics] == ["men", "women"]


async def test_rubric_search_selects_exact_match_after_fuzzy_first_result() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "meta": {"api_version": "2.0.test", "code": 200},
                "result": {
                    "items": [
                        {"id": "attractions", "name": "Аттракционы", "alias": "attrakciony"},
                        {"id": "parks", "name": "Парки", "alias": "parki"},
                    ],
                    "total": 2,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        rubrics = await TwoGisSearchClient(api_key="key", http_client=http_client).find_rubrics(
            query="парки",
            region_id="209",
            locale="ru_AZ",
            context=ToolExecutionContext(),
        )

    assert [rubric.id for rubric in rubrics] == ["parks"]
    assert requests[0].url.params["page_size"] == "50"


async def test_rubric_search_rejects_fuzzy_results_without_exact_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "meta": {"api_version": "2.0.test", "code": 200},
                "result": {
                    "items": [
                        {"id": "cafe", "name": "Cafe / Restaurants", "alias": "cafe_restaurants"}
                    ],
                    "total": 1,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        rubrics = await TwoGisSearchClient(api_key="key", http_client=http_client).find_rubrics(
            query="restaurants",
            region_id="173",
            locale="en_CY",
            context=ToolExecutionContext(),
        )

    assert rubrics == ()


async def test_rubric_backend_exception_is_retryable_provider_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "meta": {
                    "api_version": "2.0.test",
                    "code": 403,
                    "error": {
                        "type": "backendException",
                        "message": "Temporary backend failure",
                    },
                }
            },
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(ToolExecutionError) as exc_info:
            await TwoGisSearchClient(
                api_key="key",
                http_client=http_client,
            ).find_rubrics(
                query="аптеки",
                region_id="19",
                locale="ru_RU",
                allowed_aliases=("apteki",),
                context=context,
            )

    error = exc_info.value
    assert error.status_code == 403
    assert error.provider_code == "backendException"
    assert error.failure_kind is ToolFailureKind.HTTP_STATUS
    assert error.retryable is True
    assert context.upstream_calls[0].operation == "rubric_search"
    assert context.upstream_calls[0].retryable is True


def _full_week(start: str = "00:00", end: str = "24:00") -> dict[str, object]:
    schedule: dict[str, object] = {
        day: {"working_hours": [{"from": start, "to": end}]}
        for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    }
    if start == "00:00" and end == "24:00":
        schedule["is_24x7"] = True
    return schedule


def test_24x7_schedule_is_open_without_explicit_daily_periods() -> None:
    summary = summarize_schedule(
        TwoGisSchedule(is_24x7=True),
        lat=55.75,
        lon=37.62,
        now=datetime(2026, 8, 3, 12, tzinfo=UTC),
    )

    assert summary.text == "круглосуточно"
    assert summary.open_24h is True
    assert summary.is_open_now is True


def test_weekly_schedule_collapses_days_with_identical_hours() -> None:
    schedule = TwoGisSchedule.model_validate(
        {
            **{
                day: {"working_hours": [{"from": "12:00", "to": "24:00"}]}
                for day in ("Mon", "Tue", "Wed", "Thu")
            },
            **{
                day: {"working_hours": [{"from": "12:00", "to": "02:00"}]} for day in ("Fri", "Sat")
            },
            "Sun": {"working_hours": [{"from": "12:00", "to": "24:00"}]},
        }
    )

    summary = summarize_schedule(
        schedule,
        lat=55.75,
        lon=37.62,
        now=datetime(2026, 8, 3, 12, tzinfo=UTC),
    )

    assert summary.text == "пн–чт 12:00–24:00; пт–сб 12:00–02:00; вс 12:00–24:00"
    assert summary.open_24h is False


def test_uniform_weekly_schedule_is_rendered_as_daily() -> None:
    summary = summarize_schedule(
        TwoGisSchedule.model_validate(_full_week("12:00", "23:00")),
        lat=55.75,
        lon=37.62,
        now=datetime(2026, 8, 3, 12, tzinfo=UTC),
    )

    assert summary.text == "ежедневно 12:00–23:00"
    assert summary.open_24h is False


def _transport(
    items_payload: dict[str, object],
    *,
    covered: bool = True,
    city: str = "Москва",
    city_id: str = "city-id",
    country_code: str = "ru",
    region_name: str | None = None,
    region_settlements: list[str] | None = None,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/2.0/region/search":
            return httpx.Response(
                200,
                json=_region_payload(
                    covered=covered,
                    country_code=country_code,
                    # Provider tests are concerned with places, not Region API
                    # fuzziness.  Model a region whose primary city is the
                    # query unless the test deliberately supplies another one.
                    name=region_name or request.url.params.get("q", city),
                    settlements=region_settlements,
                ),
            )
        if request.url.path == "/2.0/catalog/rubric/search":
            return httpx.Response(
                200,
                json=_rubric_payload_for_query(request.url.params["q"]),
            )
        if request.url.path == "/3.0/items":
            if request.url.params.get("type") == "adm_div.city":
                return httpx.Response(
                    200,
                    json=_items_payload(
                        _item(
                            city_id,
                            name=city,
                            item_type="adm_div",
                            subtype="city",
                            lat=55.7558,
                            lon=37.6176,
                            city=city,
                            region_id="32",
                        )
                    ),
                )
            return httpx.Response(200, json=items_payload)
        raise AssertionError(f"unexpected 2GIS path: {request.url.path}")

    return httpx.MockTransport(handler), requests


async def test_region_coverage_is_cached_and_accepts_a_satellite_locality() -> None:
    transport, requests = _transport(_items_payload(), region_name="Москва")
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        first, second = await asyncio.gather(
            client.find_region("Химки", context),
            client.find_region("Химки", context),
        )

    assert first is second
    assert first is not None
    assert first.id == "32"
    assert len(requests) == 1
    assert requests[0].url.params["q"] == "Химки"
    assert requests[0].url.params["locale"] == "ru_RU"
    assert [call.operation for call in context.upstream_calls] == ["region_search"]


async def test_client_uses_a_short_per_request_timeout() -> None:
    transport, requests = _transport(_items_payload(), region_name="Москва")
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client, timeout_s=3)
        await client.find_region("Москва", ToolExecutionContext())

    assert requests[0].extensions["timeout"] == {
        "connect": 3,
        "read": 3,
        "write": 3,
        "pool": 3,
    }


async def test_region_coverage_accepts_a_named_settlement() -> None:
    transport, _ = _transport(_items_payload(), region_name="Москва")
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)

        region = await client.find_region("Барвиха", ToolExecutionContext())

    assert region is not None
    assert region.id == "32"


async def test_region_coverage_rejects_a_single_unrelated_ranked_region() -> None:
    """A lone Region API response is not proof that it covers the input city."""

    transport, requests = _transport(_items_payload(), region_name="Moscow")
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)

        region = await client.find_region("Krakow", ToolExecutionContext())

    assert region is None
    assert len(requests) == 1
    assert requests[0].url.path == "/2.0/region/search"
    assert requests[0].url.params["q"] == "Krakow"


async def test_region_coverage_by_resolved_point_is_cached() -> None:
    transport, requests = _transport(_items_payload())
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        first = await client.find_region_at_point(
            lon=37.6176,
            lat=55.7558,
            context=context,
        )
        second = await client.find_region_at_point(
            lon=37.6176,
            lat=55.7558,
            context=context,
        )

    assert first is second
    assert first is not None
    assert first.id == "32"
    assert len(requests) == 1
    assert requests[0].url.path == "/2.0/region/search"
    assert requests[0].url.params["q"] == "37.617600,55.755800"
    assert requests[0].url.params["type"] == "region"
    assert requests[0].url.params["fields"] == "items.country_code"
    assert [call.operation for call in context.upstream_calls] == ["region_point_search"]


async def test_city_lookup_by_resolved_point_stays_inside_covered_project() -> None:
    transport, requests = _transport(_items_payload())
    async with httpx.AsyncClient(transport=transport) as http_client:
        city = await TwoGisSearchClient(api_key="key", http_client=http_client).find_city_at_point(
            lon=37.6176,
            lat=55.7558,
            expected_region_id="32",
            locale="ru_RU",
            context=ToolExecutionContext(),
        )

    assert city is not None
    assert city.id == "city-id"
    assert len(requests) == 1
    assert requests[0].url.path == "/3.0/items"
    assert requests[0].url.params["lon"] == "37.617600"
    assert requests[0].url.params["lat"] == "55.755800"
    assert requests[0].url.params["type"] == "adm_div.city"
    assert requests[0].url.params["locale"] == "ru_RU"


async def test_region_coverage_clarifies_multiple_exact_settlements() -> None:
    payload = {
        "meta": {"api_version": "2.0.test", "code": 200},
        "result": {
            "items": [
                {
                    "id": "la",
                    "name": "Central Louisiana",
                    "type": "region",
                    "country_code": "us",
                    "settlements": ["Alexandria"],
                },
                {
                    "id": "va",
                    "name": "Northern Virginia",
                    "type": "region",
                    "country_code": "us",
                    "settlements": ["Alexandria"],
                },
            ],
            "total": 2,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.find_region("Alexandria", ToolExecutionContext())
        selected = await client.find_region(
            "Alexandria, Northern Virginia, US",
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.INVALID_INPUT
    clarification = exc_info.value.clarification
    assert clarification is not None
    assert clarification.kind == "select_area_query"
    assert [option.value for option in clarification.options] == [
        "Alexandria, Central Louisiana, US",
        "Alexandria, Northern Virginia, US",
    ]

    assert selected is not None
    assert selected.id == "va"


async def test_city_lookup_resolves_exact_city_id_and_is_cached() -> None:
    transport, requests = _transport(_items_payload())
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        first = await client.find_city(
            "Москва",
            context=context,
            expected_region_id="32",
            country_code="ru",
        )
        second = await client.find_city(
            "Москва, Москва, RU",
            context=context,
            expected_region_id="32",
            country_code="RU",
        )

    assert first is second
    assert first is not None
    assert first.id == "city-id"
    assert len(requests) == 1
    assert requests[0].url.params["q"] == "Москва"
    assert requests[0].url.params["type"] == "adm_div.city"
    assert "region_id" not in requests[0].url.params
    assert requests[0].url.params["locale"] == "ru_RU"
    assert set(requests[0].url.params["fields"].split(",")) == {
        "items.region_id",
        "items.adm_div",
        "items.point",
    }
    assert [call.operation for call in context.upstream_calls] == ["city_search"]


async def test_city_lookup_accepts_unique_canonical_name_in_the_expected_region() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/3.0/items"
        return httpx.Response(
            200,
            json=_items_payload(
                _item(
                    "4504222397630173",
                    name="Москва",
                    item_type="adm_div",
                    subtype="city",
                    lat=55.7588,
                    lon=37.6178,
                    region_id="32",
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        city = await TwoGisSearchClient(api_key="key", http_client=http_client).find_city(
            "Moscow",
            context=ToolExecutionContext(),
            expected_region_id="32",
            country_code="ru",
        )

    assert city is not None
    assert city.id == "4504222397630173"
    assert city.name == "Москва"
    assert requests[0].url.params["locale"] == "en_RU"


async def test_city_lookup_rejects_city_from_another_twogis_region() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_items_payload(
                _item(
                    "other-city-id",
                    name="Москва",
                    item_type="adm_div",
                    subtype="city",
                    lat=55.7588,
                    lon=37.6178,
                    region_id="other-project",
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        city = await TwoGisSearchClient(api_key="key", http_client=http_client).find_city(
            "Moscow",
            context=ToolExecutionContext(),
            expected_region_id="32",
            country_code="ru",
        )

    assert city is None


async def test_places_search_uses_english_locale_for_latin_query() -> None:
    transport, requests = _transport(_items_payload())
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)

        await client.search_places(
            query="restaurants",
            city_id="city-id",
            context=ToolExecutionContext(),
        )

    assert len(requests) == 1
    assert requests[0].url.params["locale"] == "en_RU"


async def test_rubric_search_requires_region_and_query_for_large_radius() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)

        with pytest.raises(ValueError, match="requires region_id"):
            await client.search_places(
                query=None,
                rubric_id="rubric",
                context=ToolExecutionContext(),
            )

        with pytest.raises(ValueError, match="cannot exceed 2000"):
            await client.search_places(
                query=None,
                rubric_id="rubric",
                region_id="32",
                center=(37.62, 55.75),
                radius_m=3_000,
                context=ToolExecutionContext(),
            )


async def test_area_search_uses_query_language_when_city_is_cyrillic() -> None:
    requests: list[httpx.Request] = []
    restaurant = _item(
        "restaurant-id",
        name="The Бык",
        item_type="branch",
        lat=55.7558,
        lon=37.6176,
        rubric_alias="restorany",
        rubric_name="Рестораны",
        address="Москва, Ветошный переулок, 13",
        city="Москва",
    )
    consulting = _item(
        "consulting-id",
        name="Restaurant consulting",
        item_type="branch",
        lat=55.6508,
        lon=37.5410,
        rubric_alias="restorannyj_konsalting",
        rubric_name="Restaurant consulting",
        rubric_id="wrong-rubric",
        city="Москва",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/2.0/region/search":
            return httpx.Response(200, json=_region_payload(name="Moscow"))
        if request.url.path == "/2.0/catalog/rubric/search":
            return httpx.Response(
                200,
                json=_rubric_payload_for_query(request.url.params["q"]),
            )
        if request.url.path != "/3.0/items":
            raise AssertionError(f"unexpected 2GIS path: {request.url.path}")
        if request.url.params.get("city_id") is None:
            assert request.url.params["q"] == "Moscow"
            return httpx.Response(
                200,
                json=_items_payload(
                    _item(
                        "4504222397630173",
                        name="Москва",
                        item_type="adm_div",
                        subtype="city",
                        lat=55.7588,
                        lon=37.6178,
                        region_id="32",
                    )
                ),
            )
        return httpx.Response(200, json=_items_payload(consulting, restaurant))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        store = InMemoryPlaceStore()
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="restaurants",
                category="restaurant",
                city="Moscow",
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["restaurant-id"]
    assert result.places[0].name == "The Byk"
    assert result.places[0].address == "Moscow, Vetoshnyy pereulok, 13"
    assert result.places[0].categories == ["restaurant"]
    stored = await store.get(result.places[0].ref)
    assert stored is not None
    assert stored.name == "The Бык"
    assert stored.address == "Москва, Ветошный переулок, 13"
    assert requests[2].url.params["q"] == "рестораны"
    assert requests[2].url.params["region_id"] == "32"
    assert requests[2].url.params["locale"] == "ru_RU"
    assert requests[3].url.params["city_id"] == "4504222397630173"
    assert requests[3].url.params["rubric_id"] == "rubric"
    assert "q" not in requests[3].url.params
    assert requests[3].url.params["locale"] == "en_RU"


async def test_named_resolver_uses_query_language_when_city_is_cyrillic() -> None:
    transport, requests = _transport(
        _items_payload(
            _item(
                "restaurant-id",
                name="Restaurant",
                item_type="branch",
                lat=55.7558,
                lon=37.6176,
                rubric_alias="restorany",
                rubric_name="Restaurants",
            )
        )
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolved = await TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        ).resolve_named_poi(
            query="Restaurant",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert requests[1].url.params["locale"] == "en_RU"


async def test_region_search_classifies_404_as_not_found() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/2.0/region/search"
        requests.append(request)
        return httpx.Response(
            404,
            json={
                "meta": {
                    "api_version": "2.0.test",
                    "code": 404,
                    "error": {
                        "message": "Results not found",
                        "type": "itemNotFound",
                    },
                }
            },
        )

    first_context = ToolExecutionContext()
    cached_context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisSearchClient(api_key="key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as first_exc_info:
            await client.find_region("Париж", first_context)
        with pytest.raises(ToolExecutionError) as cached_exc_info:
            await client.find_region("Париж", cached_context)

    for exc_info in (first_exc_info, cached_exc_info):
        assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
        assert exc_info.value.failure_kind is ToolFailureKind.COVERAGE_MISS
        assert str(exc_info.value) == "2GIS does not cover the requested locality"
        assert exc_info.value.status_code == 404
        assert exc_info.value.provider_code == "itemNotFound"
    assert len(requests) == 1
    assert first_context.upstream_calls[0].error_code == ToolErrorCode.NOT_FOUND.value
    assert first_context.upstream_calls[0].failure_kind == ToolFailureKind.COVERAGE_MISS.value
    assert cached_context.upstream_calls == ()


async def test_named_resolver_skips_places_request_outside_coverage() -> None:
    transport, requests = _transport(_items_payload(), covered=False)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await resolver.resolve_named_poi(
            query="Louvre",
            city="Paris",
            context=ToolExecutionContext(),
        )

    assert result is None
    assert [request.url.path for request in requests] == ["/2.0/region/search"]
    assert requests[0].url.params["locale"] == "en_RU"


async def test_named_resolver_filters_an_explicit_park_type() -> None:
    payload = _items_payload(
        _item(
            "pier",
            name="Парк Горького-2",
            item_type="branch",
            lat=55.7310,
            lon=37.5977,
            rubric_alias="pristani",
            rubric_name="Пристани",
        ),
        _item(
            "stop",
            name="Парк Горького",
            item_type="station",
            subtype="stop",
            route_type="bus",
            lat=55.7322,
            lon=37.6045,
        ),
        _item(
            "park",
            name="Центральный парк культуры и отдыха им. М. Горького",
            item_type="branch",
            lat=55.7282,
            lon=37.6002,
            rubric_alias="parki",
            rubric_name="Парки",
            schedule=_full_week(),
            name_ex={"short_name": "Парк Горького"},
        ),
        _item(
            "gate",
            name="Арка Главного входа в ПКиО им. Горького",
            item_type="building",
            lat=55.7315,
            lon=37.6033,
        ),
    )
    transport, requests = _transport(payload)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        resolved = await resolver.resolve_named_poi(
            query="Парк Горького",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "park"
    assert resolved.record.name.startswith("Центральный парк")
    assert await store.get(resolved.ref) == resolved.record
    assert requests[1].url.params["q"] == "Парк Горького, Москва"
    assert requests[1].url.params["page_size"] == "10"
    fields = set(requests[1].url.params["fields"].split(","))
    assert {"items.address", "items.brand", "items.org"} <= fields


async def test_named_resolver_first_address_uses_native_city_scope() -> None:
    payload = _items_payload(
        _item(
            "park",
            name="Швейцария, парк культуры и отдыха",
            item_type="branch",
            lat=56.2730,
            lon=43.9850,
            rubric_alias="parki",
            rubric_name="Парки",
            address="Нижний Новгород, проспект Гагарина, 35",
        )
    )
    transport, requests = _transport(
        payload,
        city="Нижний Новгород",
        city_id="nizhny-novgorod-id",
    )
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        resolved = await resolver.resolve_first_address(
            query="парк Швейцария",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "park"
    assert [request.url.path for request in requests] == [
        "/2.0/region/search",
        "/3.0/items",
        "/3.0/items",
    ]
    places_request = requests[-1]
    assert places_request.url.params["q"] == "парк Швейцария"
    assert places_request.url.params["city_id"] == "nizhny-novgorod-id"


async def test_named_resolver_uses_area_point_for_coverage_lookup() -> None:
    payload = _items_payload(
        _item(
            "museum",
            name="Терем русского самовара, музей",
            item_type="branch",
            lat=56.6426,
            lon=43.4652,
            rubric_alias="muzei",
            rubric_name="Музеи",
            city="Городец",
        )
    )
    transport, requests = _transport(payload, city="Городец", region_name="Нижний Новгород")
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        resolved = await resolver.resolve_named_poi(
            query="Терем русского самовара",
            city="Городец",
            area=PlaceResolutionArea(
                name="Городец",
                ref="plc_f1e2d3c4b5",
                lat=56.6441,
                lon=43.4722,
                bounds=GeoBounds(west=43.3, south=56.5, east=43.6, north=56.8),
            ),
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert requests[0].url.path == "/2.0/region/search"
    assert requests[0].url.params["q"] == "43.472200,56.644100"


async def test_named_resolver_prefers_exact_office_name_over_museum_and_store() -> None:
    payload = _items_payload(
        _item(
            "museum",
            name="Яндекс Музей",
            item_type="branch",
            lat=55.7350,
            lon=37.5880,
            rubric_alias="muzei",
            rubric_name="Музеи",
            address="Москва, улица Тимура Фрунзе, 11 ст13",
        ),
        _item(
            "office",
            name="Яндекс",
            item_type="branch",
            lat=55.7339,
            lon=37.5870,
            rubric_alias="internet_kompanii",
            rubric_name="Интернет-компании",
            address="Москва, улица Льва Толстого, 16",
        ),
        _item(
            "store",
            name="Яндекс, фирменный магазин",
            item_type="branch",
            lat=55.7351,
            lon=37.5881,
            rubric_alias="firmennye_magaziny",
            rubric_name="Фирменные магазины",
            address="Москва, улица Тимура Фрунзе, 11 ст13",
            name_ex={"primary": "Яндекс"},
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="офис Яндекса на Красной Розе",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "office"
    assert resolved.record.address == "Москва, улица Льва Толстого, 16"


async def test_named_resolver_accepts_unique_city_and_type_match_when_name_differs() -> None:
    payload = _items_payload(
        _item(
            "samovar-museum",
            name="Терем русского самовара, музей",
            item_type="branch",
            lat=56.642596,
            lon=43.465153,
            city="Городец",
            rubric_alias="muzei",
            rubric_name="Музеи",
            name_ex={
                "extension": "музей",
                "primary": "Терем русского самовара",
                "short_name": "Терем русского самовара",
            },
            address="Городец, набережная Революции, 11",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Музей Самоваров",
            city="Городец",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "samovar-museum"


async def test_named_resolver_does_not_pick_first_of_multiple_type_only_matches() -> None:
    payload = _items_payload(
        _item(
            "first-museum",
            name="Первый городской музей",
            item_type="branch",
            lat=56.642596,
            lon=43.465153,
            city="Городец",
            rubric_alias="muzei",
            rubric_name="Музеи",
        ),
        _item(
            "second-museum",
            name="Второй городской музей",
            item_type="branch",
            lat=56.650000,
            lon=43.480000,
            city="Городец",
            rubric_alias="muzei",
            rubric_name="Музеи",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Музей Самоваров",
            city="Городец",
            context=ToolExecutionContext(),
        )

    assert resolved is None


async def test_explicit_type_filter_preserves_provider_order_among_parks() -> None:
    payload = _items_payload(
        _item(
            "provider-first",
            name="Парк Горького-2",
            item_type="branch",
            lat=55.7310,
            lon=37.5977,
            rubric_alias="parki",
            rubric_name="Парки",
        ),
        _item(
            "exact-second",
            name="Парк Горького",
            item_type="branch",
            lat=55.7311,
            lon=37.5978,
            rubric_alias="parki",
            rubric_name="Парки",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Парк Горького",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "provider-first"


async def test_named_resolver_accepts_first_building_address() -> None:
    payload = _items_payload(
        _item(
            "building",
            name="проспект Гагарина, 35",
            item_type="building",
            lat=56.2893,
            lon=43.9802,
            city="Нижний Новгород",
            address="Нижний Новгород, проспект Гагарина, 35",
        ),
        _item(
            "branch",
            name="Организация в соседнем здании",
            item_type="branch",
            lat=56.2895,
            lon=43.9805,
            city="Нижний Новгород",
        ),
    )
    transport, _ = _transport(payload)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        resolved = await resolver.resolve_named_poi(
            query="проспект Гагарина, 35",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "building"
    assert resolved.record.address == "Нижний Новгород, проспект Гагарина, 35"


async def test_named_resolver_accepts_first_toponym() -> None:
    payload = _items_payload(
        _item(
            "park",
            name="Парк Швейцария",
            item_type="adm_div",
            subtype="place",
            lat=56.2683,
            lon=43.9738,
            city="Нижний Новгород",
        )
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Парк Швейцария",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "park"


async def test_named_resolver_accepts_short_city_inside_qualified_locality() -> None:
    payload = _items_payload(
        _item(
            "klpk",
            name="Кировский лесопромышленный колледж",
            item_type="branch",
            lat=58.592666,
            lon=49.667756,
            city="Киров",
            rubric_alias="kolledzhi",
            rubric_name="Колледжи",
            address="Киров, Владимирская улица, 115",
            name_ex={"short_name": "КЛПК"},
        )
    )
    transport, _ = _transport(
        payload,
        city="Киров",
        region_name="Кировская область",
        region_settlements=["Киров"],
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="КЛПК",
            city="Киров, Кировская область",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "klpk"
    assert resolved.record.name == "Кировский лесопромышленный колледж"


async def test_named_resolver_keeps_only_real_spatial_name_groups() -> None:
    payload = _items_payload(
        _item(
            "residential",
            name="Алые паруса, жилой комплекс",
            item_type="building",
            lat=58.604362,
            lon=49.656704,
            city="Киров",
            address="Киров, Октябрьский проспект, 117",
        ),
        _item(
            "office",
            name="Алые паруса, представительство в городе",
            item_type="branch",
            lat=58.588125,
            lon=49.636351,
            city="Киров",
            address="Киров, улица Сурикова, 19",
            name_ex={
                "primary": "Алые паруса",
                "extension": "представительство в городе",
            },
        ),
        _item(
            "street-stop",
            name="Октябрьский проспект",
            item_type="station",
            subtype="stop",
            route_type="bus",
            lat=58.603380,
            lon=49.657156,
            city="Киров",
        ),
        _item(
            "square",
            name="Сквер Алые паруса",
            item_type="adm_div",
            subtype="place",
            lat=58.603663,
            lon=49.652765,
            city="Киров",
            rubric_alias="parki",
            rubric_name="Парки",
        ),
        _item(
            "kindergarten",
            name="Алые паруса",
            item_type="branch",
            lat=58.590186,
            lon=49.598093,
            city="Киров",
            address="Киров, улица Космонавта Владислава Волкова, 2/2",
            rubric_alias="detskie_sady",
            rubric_name="Детские сады",
            org_id="kindergarten-org",
            org_name="Алые паруса, МКДОУ Детский сад №51",
            name_ex={
                "primary": "Алые паруса",
                "legal_name": "МКДОУ Детский сад №51",
            },
        ),
    )
    transport, _ = _transport(
        payload,
        city="Киров",
        region_name="Кировская область",
        region_settlements=["Киров"],
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="Алые паруса",
                city="Киров, Кировская область",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:residential"),
        mint_place_ref("twogis:item:kindergarten"),
    ]
    assert [option.label for option in exc_info.value.clarification.options] == [
        "Алые паруса, жилой комплекс",
        "Алые паруса, МКДОУ Детский сад №51",
    ]


async def test_named_resolver_treats_same_building_as_one_anchor_group() -> None:
    payload = _items_payload(
        _item(
            "mall",
            name="Океанис, торгово-развлекательный центр",
            item_type="branch",
            lat=56.2861,
            lon=43.9799,
            city="Нижний Новгород",
            building_id="oceanis-building",
        ),
        _item(
            "waterpark",
            name="Океанис, аквапарк",
            item_type="branch",
            lat=56.2852,
            lon=43.9788,
            city="Нижний Новгород",
            building_id="oceanis-building",
        ),
        _item(
            "fitness",
            name="Океанис, фитнес-клуб",
            item_type="branch",
            lat=56.2848,
            lon=43.9782,
            city="Нижний Новгород",
            building_id="oceanis-building",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Океанис",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "mall"


async def test_named_resolver_clarifies_distant_branches_of_one_brand() -> None:
    payload = _items_payload(
        _item(
            "first",
            name="Пятёрочка",
            item_type="branch",
            lat=56.3200,
            lon=44.0000,
            city="Нижний Новгород",
            address="Нижний Новгород, улица Белинского, 63",
            building_id="building-1",
            brand_id="brand-5ka",
            brand_name="Пятёрочка",
        ),
        _item(
            "second",
            name="Пятёрочка",
            item_type="branch",
            lat=56.2900,
            lon=43.9800,
            city="Нижний Новгород",
            address="Нижний Новгород, проспект Гагарина, 105",
            building_id="building-2",
            brand_id="brand-5ka",
            brand_name="Пятёрочка",
        ),
        _item(
            "third",
            name="Пятёрочка",
            item_type="branch",
            lat=56.3400,
            lon=43.9200,
            city="Нижний Новгород",
            address="Нижний Новгород, Московское шоссе, 30",
            building_id="building-3",
            brand_id="brand-5ka",
            brand_name="Пятёрочка",
        ),
    )
    transport, _ = _transport(payload)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="Пятёрочка",
                city="Нижний Новгород",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:first"),
        mint_place_ref("twogis:item:second"),
        mint_place_ref("twogis:item:third"),
    ]


async def test_ambiguity_keeps_standalone_first_before_brand_branches() -> None:
    payload = _items_payload(
        _item(
            "standalone",
            name="Ромашка",
            item_type="branch",
            lat=56.3200,
            lon=44.0000,
            city="Нижний Новгород",
            address="Нижний Новгород, первая улица, 1",
        ),
        _item(
            "brand-1",
            name="Ромашка",
            item_type="branch",
            lat=56.2900,
            lon=43.9800,
            city="Нижний Новгород",
            address="Нижний Новгород, вторая улица, 2",
            brand_id="brand-romashka",
            brand_name="Ромашка",
        ),
        _item(
            "brand-2",
            name="Ромашка",
            item_type="branch",
            lat=56.3400,
            lon=43.9200,
            city="Нижний Новгород",
            address="Нижний Новгород, третья улица, 3",
            brand_id="brand-romashka",
            brand_name="Ромашка",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="Ромашка",
                city="Нижний Новгород",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert exc_info.value.clarification.options[0].value == mint_place_ref("twogis:item:standalone")


async def test_named_resolver_preserves_distant_cross_type_groups() -> None:
    payload = _items_payload(
        _item(
            "metro",
            name="ВДНХ",
            item_type="station",
            subtype="metro",
            route_type="metro",
            lat=55.8211,
            lon=37.6414,
            rubric_alias="stancii_metro",
            rubric_name="Станции метро",
        ),
        _item(
            "park",
            name="ВДНХ",
            item_type="branch",
            lat=55.8263,
            lon=37.6377,
            rubric_alias="parki",
            rubric_name="Парки",
            name_ex={"primary": "ВДНХ", "short_name": "ВДНХ"},
        ),
        _item(
            "hotel",
            name="Cosmos Москва ВДНХ Отель",
            item_type="branch",
            lat=55.8228,
            lon=37.6470,
            rubric_alias="gostinicy",
            rubric_name="Гостиницы",
        ),
    )
    transport, _ = _transport(payload)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="ВДНХ",
                city="Москва",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:metro"),
        mint_place_ref("twogis:item:park"),
        mint_place_ref("twogis:item:hotel"),
    ]


async def test_named_resolver_treats_prefixed_subobjects_as_one_landmark() -> None:
    payload = _items_payload(
        _item(
            "station",
            name="Лужники",
            item_type="station",
            lat=55.7200,
            lon=37.5630,
        ),
        _item(
            "aqua",
            name="Лужники, аквакомплекс",
            item_type="branch",
            lat=55.7165,
            lon=37.5450,
            org_id="aqua-org",
            org_name="Лужники, аквакомплекс",
        ),
        _item(
            "arena",
            name="Лужники, большая спортивная арена",
            item_type="branch",
            lat=55.7158,
            lon=37.5530,
            org_id="arena-org",
            org_name="Лужники, большая спортивная арена",
        ),
        _item(
            "cable-car",
            name="Московская канатная дорога",
            item_type="branch",
            lat=55.7107,
            lon=37.5458,
            org_id="cable-org",
            org_name="Московская канатная дорога",
        ),
        _item(
            "small-arena",
            name="Лужники, малая спортивная арена",
            item_type="branch",
            lat=55.7202,
            lon=37.5560,
            org_id="arena-org",
            org_name="Лужники, большая спортивная арена",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Лужники",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "station"


async def test_named_resolver_limits_ambiguous_groups_in_provider_order() -> None:
    payload = _items_payload(
        _item(
            "metro-red",
            name="Сокольники",
            item_type="station",
            subtype="metro",
            route_type="metro",
            lat=55.7892,
            lon=37.6797,
        ),
        _item(
            "metro-bkl",
            name="Сокольники",
            item_type="station",
            subtype="metro",
            route_type="metro",
            lat=55.8210,
            lon=37.6820,
        ),
        _item(
            "park",
            name="Сокольники",
            item_type="branch",
            lat=55.8500,
            lon=37.6768,
            rubric_alias="parki",
            rubric_name="Парки",
        ),
        _item(
            "district",
            name="Сокольники",
            item_type="adm_div",
            lat=55.8800,
            lon=37.6780,
        ),
        _item(
            "building",
            name="Сокольники",
            item_type="building",
            lat=55.9100,
            lon=37.6770,
        ),
    )
    transport, _ = _transport(payload)
    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="Сокольники",
                city="Москва",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:metro-red"),
        mint_place_ref("twogis:item:metro-bkl"),
        mint_place_ref("twogis:item:park"),
    ]


async def test_named_resolver_drops_technical_object_beside_addressable_park() -> None:
    addressable = _item(
        "switzerland-addressable",
        name="Швейцария, парк культуры и отдыха",
        item_type="branch",
        lat=56.284648,
        lon=43.978388,
        city="Нижний Новгород",
        rubric_alias="parki",
        rubric_name="Парки",
        address="Нижний Новгород, проспект Гагарина, 35",
        building_id="addressable-building",
    )
    technical = _item(
        "switzerland-object",
        name="Швейцария, парк",
        item_type="branch",
        lat=56.274582,
        lon=43.973089,
        city="Нижний Новгород",
        rubric_alias="parki",
        rubric_name="Парки",
        building_id="technical-building",
    )
    technical["full_name"] = "Нижний Новгород, Объект"
    transport, _ = _transport(_items_payload(addressable, technical))

    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="парк Швейцария",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "switzerland-addressable"


async def test_named_resolver_keeps_lone_technical_object() -> None:
    technical = _item(
        "lone-object",
        name="Безымянный парк",
        item_type="branch",
        lat=56.274582,
        lon=43.973089,
        city="Нижний Новгород",
        rubric_alias="parki",
        rubric_name="Парки",
        building_id="technical-building",
    )
    technical["full_name"] = "Нижний Новгород, Объект"
    transport, _ = _transport(_items_payload(technical))

    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Безымянный парк",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "lone-object"


async def test_named_resolver_preserves_distant_generic_type_groups() -> None:
    payload = _items_payload(
        _item(
            "switzerland",
            name="Парк Швейцария",
            item_type="branch",
            lat=56.2846,
            lon=43.9784,
            city="Нижний Новгород",
            rubric_alias="parki",
            rubric_name="Парки",
        ),
        _item(
            "victory",
            name="Парк Победы",
            item_type="branch",
            lat=56.3288,
            lon=44.0452,
            city="Нижний Новгород",
            rubric_alias="parki",
            rubric_name="Парки",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="парк",
                city="Нижний Новгород",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:switzerland"),
        mint_place_ref("twogis:item:victory"),
    ]


async def test_named_resolver_deduplicates_nearby_station_alias() -> None:
    payload = _items_payload(
        _item(
            "main",
            name="Динамо",
            item_type="station",
            subtype="metro",
            route_type="metro",
            lat=55.789818,
            lon=37.558106,
        ),
        _item(
            "alias",
            name="Метро Динамо",
            item_type="station",
            subtype="metro",
            route_type="metro",
            lat=55.789636,
            lon=37.557672,
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="Динамо",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "main"


async def test_named_resolver_uses_shopping_center_hint_with_rubric_alias() -> None:
    payload = _items_payload(
        _item(
            "mall",
            name="Фантастика",
            item_type="branch",
            lat=56.3075,
            lon=44.0728,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
        ),
        _item(
            "cinema",
            name="Фантастика",
            item_type="branch",
            lat=56.3077,
            lon=44.0730,
            city="Нижний Новгород",
            rubric_alias="kinoteatry",
            rubric_name="Кинотеатры",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="ТРК Фантастика",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "mall"


async def test_named_resolver_prefers_exact_mall_alias_over_prefixed_name() -> None:
    payload = _items_payload(
        _item(
            "nebo",
            name="Небо, торгово-развлекательный комплекс",
            item_type="branch",
            lat=56.308936,
            lon=43.986493,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
            name_ex={"primary": "Небо"},
            address="Нижний Новгород, Большая Покровская улица, 82",
        ),
        _item(
            "seventh-sky",
            name="Седьмое небо, торгово-развлекательный центр",
            item_type="branch",
            lat=56.339632,
            lon=43.956987,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
            name_ex={"primary": "Седьмое небо"},
            address="Нижний Новгород, улица Бетанкура, 1",
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        resolved = await resolver.resolve_named_poi(
            query="ТЦ Небо",
            city="Нижний Новгород",
            context=ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.provider_id == "nebo"


async def test_named_resolver_keeps_multiple_exact_type_hint_matches_ambiguous() -> None:
    payload = _items_payload(
        _item(
            "nebo-first",
            name="Небо, торговый центр",
            item_type="branch",
            lat=56.3089,
            lon=43.9865,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
            name_ex={"primary": "Небо"},
        ),
        _item(
            "nebo-second",
            name="Небо, торговый центр",
            item_type="branch",
            lat=56.2500,
            lon=43.8500,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
            name_ex={"primary": "Небо"},
        ),
        _item(
            "seventh-sky",
            name="Седьмое небо, торговый центр",
            item_type="branch",
            lat=56.3396,
            lon=43.9570,
            city="Нижний Новгород",
            rubric_alias="torgovye_centry",
            rubric_name="Торговые центры",
            name_ex={"primary": "Седьмое небо"},
        ),
    )
    transport, _ = _transport(payload)
    async with httpx.AsyncClient(transport=transport) as http_client:
        resolver = TwoGisNamedPoiResolver(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        with pytest.raises(AmbiguousPlaceError) as exc_info:
            await resolver.resolve_named_poi(
                query="ТЦ Небо",
                city="Нижний Новгород",
                context=ToolExecutionContext(),
            )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("twogis:item:nebo-first"),
        mint_place_ref("twogis:item:nebo-second"),
    ]


async def test_places_provider_skips_places_request_outside_coverage() -> None:
    transport, requests = _transport(_items_payload(), covered=False)
    store = InMemoryPlaceStore()
    anchor_ref = mint_place_ref("test:paris-anchor")
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Лувр",
            address="Париж, улица Риволи",
            lat=48.8606,
            lon=2.3376,
            locality="Париж",
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="near",
                query="кафе",
                category="cafe",
                near=anchor_ref,
                radius_m=1_000,
            ),
            ToolExecutionContext(),
        )

    assert result.anchor == anchor_ref
    assert result.places == []
    assert [request.url.path for request in requests] == ["/2.0/region/search"]


async def test_min_rating_filters_results_and_uses_twogis_rating_parameters() -> None:
    payload = _items_payload(
        _item(
            "high",
            name="Высокий рейтинг",
            item_type="branch",
            lat=55.7542,
            lon=37.6210,
            rubric_alias="restorany",
            rubric_name="Рестораны",
            reviews={"general_rating": "4.8", "general_review_count": 456},
        ),
        _item(
            "equal",
            name="На пороге",
            item_type="branch",
            lat=55.7543,
            lon=37.6211,
            rubric_alias="restorany",
            rubric_name="Рестораны",
            reviews={"general_rating": 4.6, "general_review_count": 123},
        ),
        _item(
            "low",
            name="Ниже порога",
            item_type="branch",
            lat=55.7544,
            lon=37.6212,
            rubric_alias="restorany",
            rubric_name="Рестораны",
            reviews={"general_rating": 4.5, "general_review_count": 99},
        ),
        _item(
            "none",
            name="Без рейтинга",
            item_type="branch",
            lat=55.7545,
            lon=37.6213,
            rubric_alias="restorany",
            rubric_name="Рестораны",
        ),
    )
    transport, requests = _transport(payload)

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="рестораны",
                category="restaurant",
                city="Москва",
                min_rating=4.6,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["high", "equal"]
    assert [place.rating for place in result.places] == [4.8, 4.6]
    assert requests[3].url.params["sort"] == "rating"
    assert requests[3].url.params["has_rating"] == "true"


async def test_min_rating_returns_web_search_signal_outside_twogis_coverage() -> None:
    transport, requests = _transport(_items_payload(), covered=False)

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await provider.search(
                PlacesSearchInput(
                    mode="area",
                    query="рестораны",
                    category="restaurant",
                    city="Париж",
                    min_rating=4.6,
                ),
                ToolExecutionContext(),
            )

    error = exc_info.value
    assert error.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert error.failure_kind is ToolFailureKind.COVERAGE_MISS
    assert error.provider == "twogis"
    assert "web_search" in str(error)
    assert [request.url.path for request in requests] == ["/2.0/region/search"]


async def test_area_search_uses_geocoded_point_for_twogis_coverage_and_city_id() -> None:
    cafe = _item(
        "cafe",
        name="Кофейня",
        item_type="branch",
        lat=55.7542,
        lon=37.6210,
        rubric_alias="kofeyni",
        rubric_name="Кофейни",
        reviews={
            "general_rating": 4.9,
            "general_review_count": 456,
            "rating": 4.7,
            "review_count": 123,
        },
    )
    transport, requests = _transport(_items_payload(cafe))
    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("test:moscow-area")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6176,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="кофейни",
                category="coffee_shop",
                area_ref=area_ref,
            ),
            ToolExecutionContext(),
        )

    assert [place.name for place in result.places] == ["Кофейня"]
    assert result.places[0].rating == 4.9
    assert result.places[0].review_count == 456
    assert requests[0].url.params["q"] == "37.617600,55.755800"
    assert requests[0].url.params["type"] == "region"
    assert requests[1].url.params["type"] == "adm_div.city"
    assert requests[1].url.params["lon"] == "37.617600"
    assert requests[1].url.params["lat"] == "55.755800"
    assert "q" not in requests[1].url.params
    assert "region_id" not in requests[1].url.params
    assert requests[2].url.path == "/2.0/catalog/rubric/search"
    assert requests[3].url.params["city_id"] == "city-id"
    assert requests[3].url.params["region_id"] == "32"
    assert requests[3].url.params["rubric_id"] == "rubric"
    assert "items.reviews" in requests[3].url.params["fields"].split(",")
    assert "point1" not in requests[3].url.params
    assert "point2" not in requests[3].url.params


async def test_area_search_uses_query_locale_and_trusts_city_id_over_locality_text() -> None:
    museum = _item(
        "museum",
        name="Музей",
        item_type="branch",
        lat=40.1772,
        lon=44.5035,
        city="Ереван",
        rubric_alias="muzei",
        rubric_name="Музеи",
    )
    transport, requests = _transport(
        _items_payload(museum),
        city="Yerevan",
        country_code="am",
    )
    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("test:yerevan-area")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Yerevan",
            address="Yerevan, Armenia",
            lat=40.1772,
            lon=44.5035,
            kind="locality",
            locality="Yerevan",
            bounds=GeoBounds(west=44.3, south=40.0, east=44.7, north=40.3),
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        result = await TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        ).search(
            PlacesSearchInput(
                mode="area",
                query="музеи",
                category="museum",
                area_ref=area_ref,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["museum"]
    assert requests[1].url.params["type"] == "adm_div.city"
    assert requests[1].url.params["locale"] == "ru_AM"
    assert requests[3].url.params["city_id"] == "city-id"
    assert requests[3].url.params["locale"] == "ru_AM"


async def test_area_search_resolves_raw_city_through_twogis_without_area_ref() -> None:
    cafe = _item(
        "cafe",
        name="Кофейня",
        item_type="branch",
        lat=55.7542,
        lon=37.6210,
        rubric_alias="kofeyni",
        rubric_name="Кофейни",
    )
    transport, requests = _transport(_items_payload(cafe))

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="кофейни",
                category="coffee_shop",
                city="Москва",
            ),
            ToolExecutionContext(),
        )

    assert [place.name for place in result.places] == ["Кофейня"]
    assert result.area is None
    assert [request.url.path for request in requests] == [
        "/2.0/region/search",
        "/3.0/items",
        "/2.0/catalog/rubric/search",
        "/3.0/items",
    ]
    assert requests[0].url.params["q"] == "Москва"
    assert requests[1].url.params["q"] == "Москва"
    assert requests[1].url.params["type"] == "adm_div.city"
    assert "region_id" not in requests[1].url.params
    assert requests[3].url.params["city_id"] == "city-id"
    assert requests[3].url.params["region_id"] == "32"
    assert requests[3].url.params["rubric_id"] == "rubric"


async def test_area_search_uses_the_supported_uzbekistan_locale() -> None:
    restaurant = _item(
        "restaurant",
        name="Ресторан",
        item_type="branch",
        lat=41.3111,
        lon=69.2797,
        city="Ташкент",
        rubric_alias="restorany",
        rubric_name="Рестораны",
    )
    transport, requests = _transport(
        _items_payload(restaurant),
        city="Ташкент",
        country_code="uz",
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="рестораны",
                category="restaurant",
                city="Ташкент",
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["restaurant"]
    assert requests[1].url.params["locale"] == "ru_UZ"
    assert requests[2].url.params["locale"] == "ru_UZ"
    assert requests[3].url.params["locale"] == "ru_UZ"


async def test_area_named_search_preserves_twogis_relevance_order() -> None:
    provider_first = _item(
        "provider-first",
        name="Луна парк",
        item_type="branch",
        lat=55.74,
        lon=37.60,
    )
    exact_name_second = _item(
        "exact-second",
        name="Луна",
        item_type="branch",
        lat=55.76,
        lon=37.64,
    )
    transport, _ = _transport(_items_payload(provider_first, exact_name_second))
    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("test:moscow-named-order")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6176,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Луна",
                area_ref=area_ref,
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == [
        "provider-first",
        "exact-second",
    ]


async def test_provider_resolve_path_selects_first_card_with_explicit_address() -> None:
    no_address = _item(
        "no-address",
        name="Первая карточка",
        item_type="branch",
        lat=55.73,
        lon=37.59,
    )
    provider_first_with_address = _item(
        "provider-first-with-address",
        name="Гостиница у вокзала",
        item_type="branch",
        lat=55.74,
        lon=37.60,
        address="Москва, Первая улица, 1",
    )
    semantic_match = _item(
        "semantic-match",
        name="Московский вокзал",
        item_type="branch",
        lat=55.75,
        lon=37.61,
        address="Москва, Вокзальная площадь, 1",
    )
    transport, _ = _transport(
        _items_payload(no_address, provider_first_with_address, semantic_match)
    )
    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("test:moscow-resolve-first-address")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6176,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search_first_address(
            PlacesSearchInput(
                mode="area",
                query="Московский вокзал",
                area_ref=area_ref,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["provider-first-with-address"]


async def test_area_named_search_ignores_same_named_transit_stop_for_exact_poi() -> None:
    landmark = _item(
        "red-square-poi",
        name="Красная площадь",
        item_type="branch",
        lat=55.753084,
        lon=37.622133,
        rubric_alias="tochki_interesa",
        rubric_name="Точки интереса",
        address="Москва, Объект",
    )
    bus_stop = _item(
        "red-square-stop",
        name="Красная площадь",
        item_type="station",
        subtype="stop",
        route_type="bus",
        lat=55.751882,
        lon=37.625104,
        address="Москва, Красная площадь",
    )
    transport, _ = _transport(_items_payload(landmark, bus_stop))

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Красная площадь",
                city="Москва",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["red-square-poi"]


async def test_area_named_search_keeps_stop_for_explicit_stop_query() -> None:
    landmark = _item(
        "red-square-poi",
        name="Красная площадь",
        item_type="branch",
        lat=55.753084,
        lon=37.622133,
    )
    bus_stop = _item(
        "red-square-stop",
        name="Красная площадь",
        item_type="station",
        subtype="stop",
        route_type="bus",
        lat=55.751882,
        lon=37.625104,
    )
    transport, _ = _transport(_items_payload(landmark, bus_stop))

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="остановка Красная площадь",
                city="Москва",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["red-square-stop"]


async def test_area_named_dedup_keeps_first_provider_record() -> None:
    provider_first = _item(
        "provider-first",
        name="Луна",
        item_type="branch",
        lat=55.75000,
        lon=37.62000,
    )
    richer_alias_second = _item(
        "richer-second",
        name="Луна парк",
        item_type="branch",
        lat=55.75010,
        lon=37.62010,
        address="Москва, Лунная улица, 1",
        rubric_alias="razvlekatelnye_centry",
        rubric_name="Развлекательные центры",
        schedule=_full_week(),
    )
    transport, _ = _transport(_items_payload(provider_first, richer_alias_second))
    store = InMemoryPlaceStore()
    area_ref = mint_place_ref("test:moscow-named-dedup-order")
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6176,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Луна",
                area_ref=area_ref,
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["provider-first"]


async def test_area_named_dedup_treats_different_cards_for_one_building_as_one_place() -> None:
    building = _item(
        "residential-building",
        name="Алые паруса, жилой комплекс",
        item_type="building",
        lat=58.604362,
        lon=49.656704,
        city="Киров",
        address="Киров, Октябрьский проспект, 117",
        building_id="same-building",
        building_name="Алые паруса",
    )
    mall = _item(
        "shopping-centre",
        name="Алые паруса, торговый центр",
        item_type="branch",
        lat=58.604365,
        lon=49.656841,
        city="Киров",
        address="Киров, Октябрьский проспект, 117",
        rubric_alias="torgovye_centry",
        rubric_name="Торговые центры",
        building_id="same-building",
        name_ex={"primary": "Алые паруса"},
    )
    transport, _ = _transport(
        _items_payload(building, mall),
        city="Киров",
        region_name="Кировская область",
        region_settlements=["Киров"],
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Алые паруса",
                city="Киров",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["residential-building"]


async def test_area_named_search_ignores_unrequested_qualified_branch() -> None:
    landmark = _item(
        "landmark",
        name="Алые паруса, жилой комплекс",
        item_type="building",
        lat=58.604362,
        lon=49.656704,
        city="Киров",
        building_name="Алые паруса",
    )
    office = _item(
        "office",
        name="Алые паруса, представительство в городе",
        item_type="branch",
        lat=58.588125,
        lon=49.636351,
        city="Киров",
        name_ex={
            "primary": "Алые паруса",
            "extension": "представительство в городе",
        },
    )
    transport, _ = _transport(
        _items_payload(landmark, office),
        city="Киров",
        region_name="Кировская область",
        region_settlements=["Киров"],
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        bare_result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Алые паруса",
                city="Киров",
                limit=5,
            ),
            ToolExecutionContext(),
        )
        explicit_result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="представительство Алые паруса",
                city="Киров",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in bare_result.places] == ["landmark"]
    assert [place.id for place in explicit_result.places] == ["office"]


async def test_area_named_search_keeps_qualified_card_when_no_plain_card_exists() -> None:
    desired_bar = _item(
        "estestvoznanie",
        name="Естествознание, стриптиз-бар",
        item_type="branch",
        lat=56.8329,
        lon=60.5997,
        city="Екатеринбург",
        address="Екатеринбург, улица 8 Марта, 13",
        name_ex={
            "primary": "Естествознание",
            "extension": "стриптиз-бар",
        },
    )
    same_venue_alias = _item(
        "zazhigalka",
        name="Zажигалка, стриптиз-бар",
        item_type="branch",
        lat=56.8329,
        lon=60.5997,
        city="Екатеринбург",
        address="Екатеринбург, улица 8 Марта, 13",
        name_ex={
            "primary": "Zажигалка",
            "short_name": "Zажигалка & Естествознание",
            "extension": "стриптиз-бар",
        },
    )
    transport, _ = _transport(
        _items_payload(desired_bar, same_venue_alias),
        city="Екатеринбург",
        region_name="Свердловская область",
        region_settlements=["Екатеринбург"],
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Естествознание",
                city="Екатеринбург",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["estestvoznanie"]


async def test_area_named_dedup_merges_nearby_different_object_types() -> None:
    landmark = _item(
        "landmark",
        name="Алые паруса, жилой комплекс",
        item_type="building",
        lat=58.604362,
        lon=49.656704,
        city="Киров",
        building_name="Алые паруса",
    )
    nearby_stop = _item(
        "nearby-stop",
        name="Алые паруса",
        item_type="station",
        subtype="stop",
        route_type="bus",
        lat=58.604408,
        lon=49.657360,
        city="Киров",
    )
    transport, _ = _transport(
        _items_payload(landmark, nearby_stop),
        city="Киров",
        region_name="Кировская область",
        region_settlements=["Киров"],
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=InMemoryPlaceStore(),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                query="Алые паруса",
                city="Киров",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["landmark"]


async def test_places_provider_returns_route_ready_nearby_results_with_hours() -> None:
    cafe = _item(
        "cafe",
        name="Кофейня рядом",
        item_type="branch",
        lat=55.7542,
        lon=37.6210,
        rubric_alias="kofeyni",
        rubric_name="Кофейни",
        address="Москва, Никольская улица, 1",
        schedule=_full_week(),
    )
    transport, requests = _transport(_items_payload(cafe))
    store = InMemoryPlaceStore()
    anchor_ref = mint_place_ref("test:anchor")
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Красная площадь",
            address="Москва, Красная площадь",
            lat=55.7539,
            lon=37.6208,
            locality="Москва",
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
            clock=lambda: datetime(2026, 8, 3, 12, tzinfo=UTC),
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="near",
                query="кофейни",
                category="coffee_shop",
                near=anchor_ref,
                radius_m=3_000,
                open_now=True,
            ),
            ToolExecutionContext(),
        )

    assert result.anchor == anchor_ref
    assert len(result.places) == 1
    place = result.places[0]
    assert place.name == "Кофейня рядом"
    assert place.categories == ["Кофейни"]
    assert place.phones == []
    assert place.open_24h is True
    assert place.is_open_now is True
    assert place.distance_m is not None and place.distance_m < 100
    saved = await store.get(place.ref)
    assert saved is not None
    assert saved.provider == "twogis"
    assert requests[1].url.params["type"] == "adm_div.city"
    assert requests[1].url.params["locale"] == "ru_RU"
    assert requests[2].url.path == "/2.0/catalog/rubric/search"
    assert requests[3].url.params["point"] == "37.620800,55.753900"
    assert requests[3].url.params["radius"] == "3000"
    assert requests[3].url.params["q"] == "кофейни"
    assert requests[3].url.params["work_time"] == "now"
    assert requests[3].url.params["page_size"] == "10"
    assert requests[3].url.params["region_id"] == "32"
    assert requests[3].url.params["rubric_id"] == "rubric"


async def test_near_search_uses_query_locale_for_defensive_locality_filter() -> None:
    museum = _item(
        "museum",
        name="Музей рядом",
        item_type="branch",
        lat=40.1773,
        lon=44.5036,
        city="Ереван",
        rubric_alias="muzei",
        rubric_name="Музеи",
    )
    transport, requests = _transport(
        _items_payload(museum),
        city="Ереван",
        country_code="am",
    )
    store = InMemoryPlaceStore()
    anchor_ref = mint_place_ref("test:yerevan-anchor")
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Republic Square",
            address="Yerevan, Republic Square",
            lat=40.1772,
            lon=44.5035,
            locality="Yerevan",
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        result = await TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        ).search(
            PlacesSearchInput(
                mode="near",
                query="музеи",
                category="museum",
                near=anchor_ref,
                radius_m=2_000,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["museum"]
    assert requests[1].url.params["type"] == "adm_div.city"
    assert requests[1].url.params["locale"] == "ru_AM"
    assert requests[3].url.params["point"] == "44.503500,40.177200"
    assert requests[3].url.params["locale"] == "ru_AM"


async def test_near_search_still_sorts_by_distance() -> None:
    farther_provider_first = _item(
        "farther-first",
        name="Дальняя кофейня",
        item_type="branch",
        lat=55.7600,
        lon=37.6300,
        rubric_alias="kofeyni",
        rubric_name="Кофейни",
    )
    closer_provider_second = _item(
        "closer-second",
        name="Ближняя кофейня",
        item_type="branch",
        lat=55.7540,
        lon=37.6210,
        rubric_alias="kofeyni",
        rubric_name="Кофейни",
    )
    transport, _ = _transport(_items_payload(farther_provider_first, closer_provider_second))
    store = InMemoryPlaceStore()
    anchor_ref = mint_place_ref("test:distance-order-anchor")
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Красная площадь",
            address="Москва, Красная площадь",
            lat=55.7539,
            lon=37.6208,
            locality="Москва",
            origin=RecordOrigin.GEOCODE,
        )
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        provider = TwoGisPlacesSearchProvider(
            client=TwoGisSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="near",
                query="кофейни",
                category="coffee_shop",
                near=anchor_ref,
                radius_m=2_000,
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == [
        "closer-second",
        "farther-first",
    ]
    first_distance = result.places[0].distance_m
    second_distance = result.places[1].distance_m
    assert first_distance is not None
    assert second_distance is not None
    assert first_distance < second_distance
