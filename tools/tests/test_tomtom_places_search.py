"""TomTom places-search client and provider tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search.schemas import PlaceCategory, PlacesSearchInput
from tools.geo.places_search.tomtom import (
    TomTomPlacesSearchProvider,
    TomTomSearchClient,
    TomTomSearchEndpoint,
)
from tools.geo.places_search.tomtom import provider as tomtom_provider
from tools.geo.places_search.tomtom.categories import (
    TOMTOM_CATEGORY_SPECS,
    tomtom_category_spec,
)
from tools.geo.places_search.tomtom.matching import name_match_score
from tools.geo.places_search.tomtom.opening_hours import TomTomOpeningHoursSummary
from tools.geo.places_search.tomtom.schemas import TomTomSearchResult
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref


def _search_payload() -> dict[str, object]:
    first_day = datetime.now(UTC).astimezone(ZoneInfo("Europe/Moscow")).date()
    last_day = first_day + timedelta(days=7)
    return {
        "summary": {
            "numResults": 2,
            "totalResults": 2,
        },
        "results": [
            {
                "type": "POI",
                "id": "parking",
                "score": 0.999,
                "poi": {
                    "name": "Московский Вокзал 1",
                    "classifications": [
                        {
                            "code": "OPEN_PARKING_AREA",
                            "names": [{"name": "open parking area"}],
                        }
                    ],
                },
                "address": {
                    "freeformAddress": "Площадь Революции, Нижний Новгород",
                    "municipality": "Нижний Новгород",
                },
                "position": {"lat": 56.321443, "lon": 43.945893},
            },
            {
                "type": "POI",
                "id": "station",
                "score": 0.997,
                "poi": {
                    "name": "Московский Вокзал",
                    "phone": "+7 831 248-28-00",
                    "classifications": [
                        {
                            "code": "RAILWAY_STATION",
                            "names": [
                                {"name": "railway station"},
                                {"name": "national"},
                            ],
                        }
                    ],
                    "openingHours": {
                        "mode": "nextSevenDays",
                        "timeRanges": [
                            {
                                "startTime": {
                                    "date": first_day.isoformat(),
                                    "hour": 0,
                                    "minute": 0,
                                },
                                "endTime": {
                                    "date": last_day.isoformat(),
                                    "hour": 0,
                                    "minute": 0,
                                },
                            }
                        ],
                    },
                    "timeZone": {"ianaId": "Europe/Moscow"},
                },
                "address": {
                    "freeformAddress": "Площадь Революции, 2, Нижний Новгород",
                    "municipality": "Нижний Новгород",
                },
                "position": {"lat": 56.32184, "lon": 43.94647},
                "entryPoints": [
                    {
                        "type": "main",
                        "position": {"lat": 56.32163, "lon": 43.94673},
                    }
                ],
            },
        ],
    }


async def test_client_builds_bounded_fuzzy_search_request_and_records_metrics() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(
            api_key="secret-key",
            http_client=http_client,
            base_url="https://tomtom.example.test/search/2",
        )
        context = ToolExecutionContext()
        payload = await client.search(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query="Московский вокзал, Нижний Новгород",
            limit=20,
            bbox=GeoBounds(west=43.7, south=56.1, east=44.1, north=56.5),
            context=context,
        )

    assert payload.summary.total_results == 2
    assert captured_request is not None
    assert unquote(captured_request.url.path).endswith(
        "/search/Московский вокзал, Нижний Новгород.json"
    )
    assert captured_request.url.params["idxSet"] == "POI"
    assert captured_request.url.params["topLeft"] == "56.500000,43.700000"
    assert captured_request.url.params["btmRight"] == "56.100000,44.100000"
    assert captured_request.url.params["language"] == "NGT"
    assert captured_request.url.params["openingHours"] == "nextSevenDays"
    assert captured_request.url.params["timeZone"] == "iana"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.upstream_calls[0].status_code == 200


async def test_client_builds_radius_constrained_category_search_request() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        await client.search(
            endpoint=TomTomSearchEndpoint.NEARBY,
            query=None,
            category_ids=(7397,),
            limit=20,
            center=(37.657, 55.774),
            radius_m=1000,
            context=ToolExecutionContext(),
        )

    assert captured_request is not None
    assert unquote(captured_request.url.path).endswith("/nearbySearch/.json")
    assert "idxSet" not in captured_request.url.params
    assert captured_request.url.params["categorySet"] == "7397"
    assert captured_request.url.params["lat"] == "55.774000"
    assert captured_request.url.params["lon"] == "37.657000"
    assert captured_request.url.params["radius"] == "1000"


async def test_client_uses_explicit_response_language() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        await client.search(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query="Бранденбургские ворота, Берлин",
            language="ru-RU",
            limit=20,
            context=ToolExecutionContext(),
        )

    assert captured_request is not None
    assert captured_request.url.params["language"] == "ru-RU"


async def test_client_keeps_named_query_in_fuzzy_search_with_category_ids() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        await client.search(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query="Пятёрочка",
            category_ids=(7332005,),
            limit=20,
            center=(44.005, 56.326),
            radius_m=3000,
            context=ToolExecutionContext(),
        )

    assert captured_request is not None
    assert unquote(captured_request.url.path).endswith("/search/Пятёрочка.json")
    assert captured_request.url.params["idxSet"] == "POI"
    assert captured_request.url.params["categorySet"] == "7332005"
    assert captured_request.url.params["radius"] == "3000"


async def test_client_classifies_invalid_success_payload_as_failed_upstream_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 1, "totalResults": 1},
                "results": [{"type": "POI", "id": "missing-required-fields"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                endpoint=TomTomSearchEndpoint.FUZZY,
                query="Парк Горького, Москва",
                limit=20,
                context=context,
            )

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.status_code == 200
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].failure_kind == ToolFailureKind.INVALID_SCHEMA.value


async def test_client_classifies_non_poi_result_as_invalid_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _search_payload()
        results = payload["results"]
        assert isinstance(results, list)
        first_result = results[0]
        assert isinstance(first_result, dict)
        first_result["type"] = "Geography"
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                endpoint=TomTomSearchEndpoint.FUZZY,
                query="Парк Горького, Москва",
                limit=20,
                context=context,
            )

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.status_code == 200
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].failure_kind == ToolFailureKind.INVALID_SCHEMA.value


async def test_client_omits_blank_name_cards_without_rejecting_valid_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _search_payload()
        results = payload["results"]
        assert isinstance(results, list)
        first_result = results[0]
        assert isinstance(first_result, dict)
        poi = first_result["poi"]
        assert isinstance(poi, dict)
        poi["name"] = ""
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        result = await client.search(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query="Московский вокзал, Нижний Новгород",
            limit=20,
            context=context,
        )

    assert [item.id for item in result.results] == ["station"]
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.warnings == ("TomTom omitted 1 malformed result(s) with blank names.",)


async def test_client_rejects_response_when_all_result_names_are_blank() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _search_payload()
        results = payload["results"]
        assert isinstance(results, list)
        for raw_result in results:
            assert isinstance(raw_result, dict)
            poi = raw_result["poi"]
            assert isinstance(poi, dict)
            poi["name"] = " "
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        context = ToolExecutionContext()

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                endpoint=TomTomSearchEndpoint.FUZZY,
                query="супермаркеты",
                limit=20,
                context=context,
            )

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.warnings == ()


def test_tomtom_category_mapping_covers_the_public_contract() -> None:
    assert set(TOMTOM_CATEGORY_SPECS) == set(PlaceCategory)
    assert all(spec.query and spec.classification_codes for spec in TOMTOM_CATEGORY_SPECS.values())
    text_only_categories = {
        category for category, spec in TOMTOM_CATEGORY_SPECS.items() if not spec.category_ids
    }
    # TomTom has no sufficiently narrow standard IDs for these concepts. A
    # broader ID would make queryless Nearby Search return unrelated places.
    assert text_only_categories == {
        PlaceCategory.BICYCLE_RENTAL,
        PlaceCategory.BICYCLE_STORE,
    }
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.ATM].category_ids == (7397,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.BUS_STATION].category_ids == (9942,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.BUS_STATION].supports_queryless_nearby is False
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.FUEL].category_ids == (7311,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.HOTEL].category_ids == (7314003,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.PARKING].category_ids == (7313, 7369)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.PHARMACY].category_ids == (7326,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.RESTAURANT].category_ids == (7315,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.SUPERMARKET].category_ids == (7332005,)
    assert TOMTOM_CATEGORY_SPECS[PlaceCategory.CONVENIENCE_STORE].category_ids == (9361009,)


def test_parent_category_accepts_children_but_not_other_taxonomies() -> None:
    assert tomtom_provider._is_same_or_child_category(7315, 7315) is True
    assert tomtom_provider._is_same_or_child_category(7315036, 7315) is True
    assert tomtom_provider._is_same_or_child_category(7326, 7315) is False
    # A leaf category is exact: siblings must not be accepted.
    assert tomtom_provider._is_same_or_child_category(7315036, 7315036) is True
    assert tomtom_provider._is_same_or_child_category(7315015, 7315036) is False


def test_provider_deduplicates_same_point_aliases_but_keeps_colocated_businesses() -> None:
    def match(
        result_id: str,
        name: str,
        *,
        phone: str | None = None,
        has_hours: bool = False,
    ) -> tomtom_provider._TomTomMatch:
        poi: dict[str, object] = {
            "name": name,
            "classifications": [
                {
                    "code": "PARKING_GARAGE",
                    "names": [{"name": "parking garage"}],
                }
            ],
        }
        if phone is not None:
            poi["phone"] = phone
        if has_hours:
            poi["openingHours"] = {
                "mode": "nextSevenDays",
                "timeRanges": [],
            }
        return tomtom_provider._TomTomMatch(
            result=TomTomSearchResult.model_validate(
                {
                    "type": "POI",
                    "id": result_id,
                    "score": 0.9,
                    "poi": poi,
                    "address": {
                        "freeformAddress": "Пулковское шоссе, 41",
                        "municipality": "Санкт-Петербург",
                    },
                    "position": {"lat": 59.799219, "lon": 30.275278},
                }
            ),
            address="Пулковское шоссе, 41, Санкт-Петербург",
            name_score=(0, 0),
            distance_m=140,
            opening_hours=TomTomOpeningHoursSummary(
                text=None,
                open_24h=None,
                is_open_now=None,
            ),
        )

    matches = tomtom_provider._deduplicate_matches(
        [
            match("p1-sparse", "Крытый паркинг P1"),
            match(
                "p1-rich",
                "P1",
                phone="+7 921 858-60-88",
                has_hours=True,
            ),
            match("p2-primary", "Краткосрочная парковка P2"),
            match("p2-alias", "P2 Краткосрочная"),
            match("vtb", "ВТБ Банк"),
            match("rosselkhoz", "Россельхозбанк"),
        ]
    )

    assert [match.result.id for match in matches] == [
        "p1-rich",
        "p2-primary",
        "vtb",
        "rosselkhoz",
    ]


def test_provider_selects_endpoint_from_typed_search_intent() -> None:
    anchor_ref = mint_place_ref("anchor:brand-search")
    category = tomtom_category_spec(PlaceCategory.SUPERMARKET)

    named_plan = tomtom_provider._build_search_plan(
        PlacesSearchInput(
            mode="near",
            near=anchor_ref,
            query="Пятёрочка",
            radius_m=3000,
            limit=5,
        ),
        None,
    )
    nearby_category_plan = tomtom_provider._build_search_plan(
        PlacesSearchInput(
            mode="near",
            near=anchor_ref,
            query="супермаркеты",
            category="supermarket",
            radius_m=3000,
            limit=5,
        ),
        category,
    )
    area_category_plan = tomtom_provider._build_search_plan(
        PlacesSearchInput(
            mode="area",
            city="Нижний Новгород",
            query="супермаркеты",
            category="supermarket",
            limit=5,
        ),
        category,
    )

    assert named_plan.endpoint is TomTomSearchEndpoint.FUZZY
    assert named_plan.query == "Пятёрочка"
    assert named_plan.category_ids == ()
    assert named_plan.is_named_search is True

    assert nearby_category_plan.endpoint is TomTomSearchEndpoint.NEARBY
    assert nearby_category_plan.query is None
    assert nearby_category_plan.category_ids == (7332005,)
    assert nearby_category_plan.is_named_search is False

    assert area_category_plan.endpoint is TomTomSearchEndpoint.CATEGORY
    assert area_category_plan.query == "supermarkets hypermarkets"
    assert area_category_plan.category_ids == (7332005,)
    assert area_category_plan.is_named_search is False


def test_nearby_is_used_only_for_safe_queryless_category_ids() -> None:
    anchor_ref = mint_place_ref("anchor:all-category-plans")

    for category, spec in TOMTOM_CATEGORY_SPECS.items():
        query = f"user phrase for {category.value}"
        plan = tomtom_provider._build_search_plan(
            PlacesSearchInput(
                mode="near",
                near=anchor_ref,
                query=query,
                category=category,
            ),
            spec,
        )

        if spec.category_ids and spec.supports_queryless_nearby:
            expected_endpoint = TomTomSearchEndpoint.NEARBY
        elif spec.category_ids:
            expected_endpoint = TomTomSearchEndpoint.CATEGORY
        else:
            expected_endpoint = TomTomSearchEndpoint.FUZZY
        assert plan.endpoint is expected_endpoint, category
        assert plan.category_ids == spec.category_ids
        if not spec.category_ids:
            assert plan.query == query
            assert plan.is_named_search is False
        elif not spec.supports_queryless_nearby:
            assert plan.query == spec.query
            assert plan.is_named_search is False


def test_bus_station_requires_specific_taxonomy_name_within_parent_category() -> None:
    category = tomtom_category_spec(PlaceCategory.BUS_STATION)
    base: dict[str, Any] = {
        "type": "POI",
        "id": "transport",
        "poi": {
            "name": "Transport",
            "categorySet": [{"id": 9942002}],
            "classifications": [
                {
                    "code": "PUBLIC_TRANSPORT_STOP",
                    "names": [{"name": "bus station"}],
                }
            ],
        },
        "address": {"freeformAddress": "Address"},
        "position": {"lat": 56.0, "lon": 44.0},
    }
    bus_station = TomTomSearchResult.model_validate(base)
    bus_stop = TomTomSearchResult.model_validate(
        {
            **base,
            "poi": {
                **base["poi"],
                "classifications": [
                    {
                        "code": "PUBLIC_TRANSPORT_STOP",
                        "names": [{"name": "bus stop"}],
                    }
                ],
            },
        }
    )

    assert tomtom_provider._matches_category(bus_station, category) is True
    assert tomtom_provider._matches_category(bus_stop, category) is False


def test_name_match_accepts_two_word_qualifier_only_for_an_intact_phrase() -> None:
    assert name_match_score(
        "Санкт-Петербургский Дом Книги",
        "Дом книги",
        locality="Санкт-Петербург",
    ) == (1, 2)
    assert (
        name_match_score(
            "Торговый центр Дом Книги",
            "Дом книги",
            locality="Санкт-Петербург",
        )
        is None
    )


def test_name_match_accepts_one_letter_variant_with_one_type_word() -> None:
    assert name_match_score("Медео каток", "Медеу", locality="Алматы") == (2, 2)
    assert name_match_score("Мир", "Мар", locality="Москва") is None
    assert name_match_score("Медео большой каток", "Медеу", locality="Алматы") is None


async def test_client_preserves_tomtom_authentication_error_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detailedError": {
                    "code": "FORBIDDEN",
                    "message": "Invalid key",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="bad-key", http_client=http_client)
        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                endpoint=TomTomSearchEndpoint.FUZZY,
                query="coffee",
                limit=5,
                center=(37.62, 55.75),
                radius_m=1000,
                context=context,
            )

    error = exc_info.value
    assert error.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert error.failure_kind is ToolFailureKind.AUTHENTICATION
    assert error.provider_code == "FORBIDDEN"
    assert error.retryable is False
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].provider_code == "FORBIDDEN"


async def test_client_classifies_explicit_tomtom_403_quota_as_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detailedError": {
                    "code": "Forbidden",
                    "message": "Account is over QPD limit",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="limited-key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                endpoint=TomTomSearchEndpoint.FUZZY,
                query="coffee",
                limit=5,
                center=(37.62, 55.75),
                radius_m=1000,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.RATE_LIMITED
    assert exc_info.value.retryable is True


async def test_provider_excludes_auxiliary_when_exact_poi_exists_and_persists_entrance() -> None:
    area_ref = mint_place_ref("area:nizhny-novgorod")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Нижний Новгород",
            address="Россия, Нижний Новгород",
            lat=56.3269,
            lon=43.9361,
            kind="locality",
            locality="Нижний Новгород",
            bounds=GeoBounds(west=43.6, south=56.1, east=44.2, north=56.5),
            origin=RecordOrigin.GEOCODE,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="Московский вокзал",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["station"]
    place = result.places[0]
    assert place.name == "Московский Вокзал"
    assert place.categories == ["railway station", "national"]
    assert place.phones == ["+7 831 248-28-00"]
    assert place.hours_text == "Open 24 hours for the next 7 days"
    assert place.open_24h is True
    assert place.is_open_now is True
    assert result.returned_count == 1
    assert result.truncated is False
    assert result.area is not None
    assert result.area.ref == area_ref
    record = await store.get(place.ref)
    assert record is not None
    assert record.provider == "tomtom"
    assert (record.lat, record.lon) == (56.32163, 43.94673)


async def test_provider_resolve_path_selects_first_card_with_address() -> None:
    area_ref = mint_place_ref("area:nizhny-novgorod-resolve-first")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Нижний Новгород",
            address="Россия, Нижний Новгород",
            lat=56.3269,
            lon=43.9361,
            kind="locality",
            locality="Нижний Новгород",
            bounds=GeoBounds(west=43.6, south=56.1, east=44.2, north=56.5),
            origin=RecordOrigin.GEOCODE,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search_first_address(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="Московский вокзал",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["parking"]


async def test_empty_filtered_result_is_not_marked_truncated() -> None:
    area_ref = mint_place_ref("area:nizhny-novgorod-empty-filtered")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Нижний Новгород",
            address="Россия, Нижний Новгород",
            lat=56.3269,
            lon=43.9361,
            kind="locality",
            locality="Нижний Новгород",
            bounds=GeoBounds(west=43.6, south=56.1, east=44.2, north=56.5),
            origin=RecordOrigin.GEOCODE,
        )
    )
    payload = _search_payload()
    payload["summary"] = {"numResults": 2, "totalResults": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="рестораны",
                category="restaurant",
            ),
            ToolExecutionContext(),
        )

    assert result.places == []
    assert result.truncated is False


async def test_area_named_search_preserves_tomtom_relevance_order() -> None:
    area_ref = mint_place_ref("area:nizhny-novgorod-provider-order")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Нижний Новгород",
            address="Россия, Нижний Новгород",
            lat=56.3269,
            lon=43.9361,
            kind="locality",
            locality="Нижний Новгород",
            bounds=GeoBounds(west=43.6, south=56.1, east=44.2, north=56.5),
            origin=RecordOrigin.GEOCODE,
        )
    )

    def result_item(
        result_id: str,
        name: str,
        *,
        lat: float,
        lon: float,
    ) -> dict[str, object]:
        return {
            "type": "POI",
            "id": result_id,
            "score": 0.9,
            "poi": {
                "name": name,
                "classifications": [
                    {
                        "code": "RAILWAY_STATION",
                        "names": [{"name": "railway station"}],
                    }
                ],
            },
            "address": {
                "freeformAddress": f"{name}, Нижний Новгород",
                "municipality": "Нижний Новгород",
            },
            "position": {"lat": lat, "lon": lon},
        }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 2, "totalResults": 2},
                "results": [
                    result_item(
                        "provider-first",
                        "Московский вокзал пассажирский",
                        lat=56.3218,
                        lon=43.9465,
                    ),
                    result_item(
                        "exact-second",
                        "Московский вокзал",
                        lat=56.31,
                        lon=43.99,
                    ),
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="Московский вокзал",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == [
        "provider-first",
        "exact-second",
    ]


async def test_provider_open_now_keeps_only_confirmed_open_places() -> None:
    area_ref = mint_place_ref("area:moscow-open-now")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6173,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=37.3, south=55.5, east=37.9, north=55.9),
            origin=RecordOrigin.GEOCODE,
        )
    )
    reference_time = datetime(2026, 7, 30, 9, 0, tzinfo=UTC)

    def result(
        result_id: str,
        name: str,
        *,
        hours: tuple[int, int] | None,
    ) -> dict[str, object]:
        poi: dict[str, object] = {
            "name": name,
            "categorySet": [{"id": 9376002}],
            "classifications": [
                {
                    "code": "CAFE_PUB",
                    "names": [{"name": "café"}],
                }
            ],
        }
        if hours is not None:
            start_hour, end_hour = hours
            poi["openingHours"] = {
                "mode": "nextSevenDays",
                "timeRanges": [
                    {
                        "startTime": {
                            "date": "2026-07-30",
                            "hour": start_hour,
                            "minute": 0,
                        },
                        "endTime": {
                            "date": "2026-07-30",
                            "hour": end_hour,
                            "minute": 0,
                        },
                    }
                ],
            }
            poi["timeZone"] = {"ianaId": "Europe/Moscow"}

        return {
            "type": "POI",
            "id": result_id,
            "score": 0.9,
            "poi": poi,
            "address": {
                "freeformAddress": f"Москва, улица {result_id}",
                "municipality": "Москва",
            },
            "position": {"lat": 55.75, "lon": 37.62},
        }

    payload = {
        "summary": {"numResults": 3, "totalResults": 3},
        "results": [
            result("open", "Открытая кофейня", hours=(8, 22)),
            result("closed", "Закрытая кофейня", hours=(14, 22)),
            result("unknown", "Кофейня без расписания", hours=None),
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
            clock=lambda: reference_time,
        )
        context = ToolExecutionContext()
        search_result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="кафе",
                category="cafe",
                open_now=True,
            ),
            context,
        )

    assert provider.supports_open_now is True
    assert [place.id for place in search_result.places] == ["open"]
    assert search_result.places[0].is_open_now is True
    assert await store.get(mint_place_ref("tomtom:poi:closed")) is None
    assert await store.get(mint_place_ref("tomtom:poi:unknown")) is None
    assert context.warnings == (
        "TomTom did not provide enough schedule or time-zone data to verify the current "
        "opening status of 1 result(s); those results were omitted.",
    )


async def test_provider_prefers_numeric_category_over_broad_classification() -> None:
    area_ref = mint_place_ref("area:moscow")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Москва",
            address="Россия, Москва",
            lat=55.7558,
            lon=37.6173,
            kind="locality",
            locality="Москва",
            bounds=GeoBounds(west=37.3, south=55.5, east=37.9, north=55.9),
            origin=RecordOrigin.GEOCODE,
        )
    )
    captured_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_path
        captured_path = unquote(request.url.path)
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 3, "totalResults": 3},
                "results": [
                    {
                        "type": "POI",
                        "id": "atm",
                        "score": 0.9,
                        "poi": {
                            "name": "Банкомат",
                            "categorySet": [{"id": 7397}],
                            "classifications": [
                                {
                                    "code": "CASH_DISPENSER",
                                    "names": [{"name": "automatic teller machine"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Комсомольская площадь, 2",
                            "municipality": "Москва",
                        },
                        "position": {"lat": 55.775, "lon": 37.657},
                    },
                    {
                        "type": "POI",
                        "id": "monument",
                        "score": 0.8,
                        "poi": {
                            "name": "Памятный банкомат",
                            "categorySet": [{"id": 9902}],
                            "classifications": [
                                {
                                    # The broad classification is deliberately
                                    # plausible; the specific numeric category
                                    # proves that this is not an ATM.
                                    "code": "CASH_DISPENSER",
                                    "names": [{"name": "cash dispenser"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Комсомольская площадь",
                            "municipality": "Москва",
                        },
                        "position": {"lat": 55.776, "lon": 37.658},
                    },
                    {
                        "type": "POI",
                        "id": "atm-outside-area",
                        "score": 0.7,
                        "poi": {
                            "name": "Банкомат в другом городе",
                            "categorySet": [{"id": 7397}],
                            "classifications": [
                                {
                                    "code": "CASH_DISPENSER",
                                    "names": [{"name": "cash dispenser"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Улица Чехова, 33",
                            "municipality": "Москва",
                        },
                        "position": {"lat": 55.7657, "lon": 38.2},
                    },
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="банкоматы",
                category="atm",
                limit=10,
            ),
            ToolExecutionContext(),
        )

    assert captured_path.endswith("/categorySearch/cash dispenser.json")
    assert [place.id for place in result.places] == ["atm"]
    assert result.places[0].categories == ["automatic teller machine"]


async def test_area_provider_accepts_localized_municipality_inside_city_bounds() -> None:
    area_ref = mint_place_ref("area:frankfurt-am-main")
    store = InMemoryPlaceStore()
    bounds = GeoBounds(west=8.45, south=49.95, east=8.85, north=50.25)
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Frankfurt am Main",
            address="Frankfurt am Main, Germany",
            lat=50.1109,
            lon=8.6821,
            kind="locality",
            locality="Frankfurt am Main",
            bounds=bounds,
            origin=RecordOrigin.GEOCODE,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["topLeft"] == "50.250000,8.450000"
        assert request.url.params["btmRight"] == "49.950000,8.850000"
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 2, "totalResults": 2},
                "results": [
                    {
                        "type": "POI",
                        "id": "frankfurt-coffee",
                        "score": 0.99,
                        "poi": {
                            "name": "Kaffeestube",
                            "categorySet": [{"id": 9376006}],
                            "classifications": [
                                {
                                    "code": "CAFE_PUB",
                                    "names": [{"name": "coffee shop"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Frankfurt, Germany",
                            "municipality": "Frankfurt",
                        },
                        "position": {"lat": 50.111, "lon": 8.682},
                    },
                    {
                        "type": "POI",
                        "id": "eschborn-coffee",
                        "score": 0.98,
                        "poi": {
                            "name": "Coffee in Eschborn",
                            "categorySet": [{"id": 9376006}],
                            "classifications": [
                                {
                                    "code": "CAFE_PUB",
                                    "names": [{"name": "coffee shop"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Eschborn, Germany",
                            "municipality": "Eschborn",
                        },
                        # Inside Frankfurt's rectangular bbox, but outside its
                        # municipality and therefore must not be returned.
                        "position": {"lat": 50.14, "lon": 8.57},
                    },
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="coffee shops",
                category="coffee_shop",
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["frankfurt-coffee"]


async def test_area_provider_uses_cached_russian_locality_and_rejects_neighbour() -> None:
    area_ref = mint_place_ref("area:vienna")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=area_ref,
            name="Vienna",
            address="Vienna, Austria",
            lat=48.209206,
            lon=16.372778,
            kind="locality",
            locality="Vienna",
            localized_localities={"ru-RU": "Вена"},
            bounds=GeoBounds(west=16.18, south=48.11, east=16.58, north=48.33),
            origin=RecordOrigin.GEOCODE,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["language"] == "ru-RU"
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 2, "totalResults": 2},
                "results": [
                    {
                        "type": "POI",
                        "id": "hofburg",
                        "score": 0.99,
                        "poi": {
                            "name": "Хофбург",
                            "categorySet": [{"id": 7376}],
                            "classifications": [
                                {
                                    "code": "IMPORTANT_TOURIST_ATTRACTION",
                                    "names": [{"name": "important tourist attraction"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Heldenplatz, 1010 Вена",
                            "municipality": "Вена",
                        },
                        "position": {"lat": 48.205468, "lon": 16.364909},
                    },
                    {
                        "type": "POI",
                        "id": "neighbour",
                        "score": 0.98,
                        "poi": {
                            "name": "Соседнее место",
                            "categorySet": [{"id": 7376}],
                            "classifications": [
                                {
                                    "code": "IMPORTANT_TOURIST_ATTRACTION",
                                    "names": [{"name": "important tourist attraction"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "Внутри прямоугольника",
                            "municipality": "Клостернойбург",
                        },
                        "position": {"lat": 48.30, "lon": 16.32},
                    },
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="area",
                area_ref=area_ref,
                query="достопримечательности",
                category="attraction",
                limit=15,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["hofburg"]


async def test_provider_omits_incompatible_house_numbers_at_identical_coordinates() -> None:
    anchor_ref = mint_place_ref("anchor:palace-square")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Дворцовая площадь, 2",
            address="Россия, Санкт-Петербург, Дворцовая площадь, 2",
            lat=59.940082,
            lon=30.312814,
            locality="Санкт-Петербург",
            origin=RecordOrigin.GEOCODE,
        )
    )

    def hotel(
        *,
        result_id: str,
        name: str,
        address: str,
        lat: float,
        lon: float,
        phone: str | None = None,
        municipality: str = "Санкт-Петербург",
    ) -> dict[str, object]:
        poi: dict[str, object] = {
            "name": name,
            "classifications": [
                {
                    "code": "HOTEL_MOTEL",
                    "names": [{"name": "hotel"}],
                }
            ],
        }
        if phone is not None:
            poi["phone"] = phone
        return {
            "type": "POI",
            "id": result_id,
            "score": 0.9,
            "poi": poi,
            "address": {
                "freeformAddress": address,
                "municipality": municipality,
            },
            "position": {"lat": lat, "lon": lon},
        }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/nearbySearch/.json")
        assert request.url.params["categorySet"] == "7314003"
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 6, "totalResults": 6},
                "results": [
                    hotel(
                        result_id="valid",
                        name="At The Hermitage",
                        address="Дворцовая набережная, 6",
                        lat=59.94136,
                        lon=30.3133,
                        phone="+7 812 000-00-00",
                    ),
                    hotel(
                        result_id="conflict-136",
                        name="Акварели",
                        address="Невский проспект, 136",
                        lat=59.93701,
                        lon=30.31289,
                    ),
                    hotel(
                        result_id="conflict-47",
                        name="Мини-Отель На Невском",
                        address="Невский проспект, 47",
                        lat=59.93701,
                        lon=30.31289,
                    ),
                    hotel(
                        result_id="same-building-9",
                        name="Отель в одном комплексе",
                        address="Улица Крымский Вал, 9",
                        lat=59.939,
                        lon=30.311,
                        phone="+7 812 000-00-01",
                    ),
                    hotel(
                        result_id="same-building-9-59",
                        name="Другой отель в том же комплексе",
                        address="Улица Крымский Вал, 9/59",
                        lat=59.939,
                        lon=30.311,
                        phone="+7 812 000-00-02",
                    ),
                    hotel(
                        result_id="outside-locality",
                        name="Отель в другом городе",
                        address="Улица Крупской, 21",
                        lat=59.9,
                        lon=30.314,
                        municipality="Бор",
                    ),
                ],
            },
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="near",
                near=anchor_ref,
                query="отели",
                category="hotel",
                radius_m=2000,
                limit=10,
            ),
            context,
        )

    assert [place.id for place in result.places] == [
        "valid",
        "same-building-9",
        "same-building-9-59",
    ]
    assert context.warnings == (
        "TomTom returned 2 places with incompatible house numbers at identical "
        "coordinates; those ambiguous places were omitted.",
    )
    assert await store.get(mint_place_ref("tomtom:poi:conflict-136")) is None
    assert await store.get(mint_place_ref("tomtom:poi:conflict-47")) is None


async def test_nearby_provider_uses_radius_instead_of_localized_municipality() -> None:
    anchor_ref = mint_place_ref("anchor:paris")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=anchor_ref,
            name="Центр Парижа",
            address="Париж",
            lat=48.856613,
            lon=2.352222,
            locality="Париж",
            origin=RecordOrigin.GEOCODE,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/nearbySearch/.json")
        assert request.url.params["categorySet"] == "9376006"
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 1, "totalResults": 1},
                "results": [
                    {
                        "type": "POI",
                        "id": "paris-coffee",
                        "score": 0.99,
                        "poi": {
                            "name": "Café de Paris",
                            "categorySet": [{"id": 9376006}],
                            "classifications": [
                                {
                                    "code": "CAFE_PUB",
                                    "names": [{"name": "coffee shop"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "10 Rue de Rivoli, Paris",
                            "municipality": "Paris",
                        },
                        "position": {"lat": 48.857, "lon": 2.353},
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = TomTomPlacesSearchProvider(
            client=TomTomSearchClient(api_key="key", http_client=http_client),
            place_store=store,
        )
        result = await provider.search(
            PlacesSearchInput(
                mode="near",
                near=anchor_ref,
                query="кофейни",
                category="coffee_shop",
                radius_m=3_000,
                limit=5,
            ),
            ToolExecutionContext(),
        )

    assert [place.id for place in result.places] == ["paris-coffee"]
    assert result.places[0].distance_m is not None
    assert result.places[0].distance_m < 100
