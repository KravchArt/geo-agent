"""Tests for the TomTom named-POI fallback used by text-place resolution."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding import (
    GeocodedPlaceResolver,
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
)
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search import PlacesSearchScopeResolver
from tools.geo.places_search.schemas import PlacesSearchInput
from tools.geo.places_search.tomtom.client import (
    TomTomSearchClient,
    TomTomSearchEndpoint,
)
from tools.geo.places_search.tomtom.named_poi_resolver import TomTomNamedPoiResolver
from tools.geo.places_search.tomtom.provider import TomTomPlacesSearchProvider
from tools.geo.places_search.tomtom.schemas import TomTomSearchResponse
from tools.geo.text_place_resolution import TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, RecordOrigin, mint_place_ref


def _result(
    result_id: str,
    *,
    name: str,
    address: str,
    municipality: str,
    lat: float,
    lon: float,
    score: float = 1.0,
    classification: str = "PARK",
    entry_lat: float | None = None,
    entry_lon: float | None = None,
) -> dict[str, Any]:
    entry_points: list[dict[str, object]] = []
    if entry_lat is not None and entry_lon is not None:
        entry_points.append(
            {
                "type": "main",
                "position": {"lat": entry_lat, "lon": entry_lon},
            }
        )
    return {
        "type": "POI",
        "id": result_id,
        "score": score,
        "poi": {
            "name": name,
            "classifications": [
                {
                    "code": classification,
                    "names": [{"name": "park"}],
                }
            ],
        },
        "address": {
            "freeformAddress": address,
            "municipality": municipality,
        },
        "position": {"lat": lat, "lon": lon},
        "entryPoints": entry_points,
    }


def _response(*results: dict[str, Any]) -> TomTomSearchResponse:
    return TomTomSearchResponse.model_validate(
        {
            "summary": {
                "numResults": len(results),
                "totalResults": len(results),
            },
            "results": list(results),
        }
    )


class StaticTomTomClient:
    def __init__(self, response: TomTomSearchResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def search(self, **kwargs: object) -> TomTomSearchResponse:
        self.calls.append(dict(kwargs))
        return self.response


class AmbiguousParkGeocoder:
    provider = "fake_geocoder"

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        first = PlaceMatch(
            ref="plc_a1b2c3d4e5",
            name="Парк Горького",
            address="Россия, Москва, Парк Горького",
        )
        second = PlaceMatch(
            ref="plc_b2c3d4e5f6",
            name="Парк Горького",
            address="Россия, Москва, улица Крымский Вал",
        )
        return GeocodePlaceOutput(
            best=first,
            matches=[first, second],
            ambiguous=True,
        )


async def test_named_poi_resolver_selects_first_provider_result() -> None:
    store = InMemoryPlaceStore()
    client = StaticTomTomClient(
        _response(
            _result(
                "generic-attraction",
                name="Парк Горького",
                address="Москва, 119121",
                municipality="Москва",
                lat=55.728935,
                lon=37.601208,
                score=2.0,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            ),
            _result(
                "parking",
                name="Парк Горького",
                address="улица Крымский Вал, 9, Москва",
                municipality="Москва",
                lat=55.7300,
                lon=37.6020,
                classification="OPEN_PARKING_AREA",
            ),
            _result(
                "park",
                name="Парк Горького",
                address="улица Крымский Вал, 9, Москва",
                municipality="Москва",
                lat=55.7296,
                lon=37.6017,
                score=0.9,
                entry_lat=55.7290,
                entry_lon=37.6013,
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(client=client, place_store=store)

    resolved = await resolver.resolve_named_poi(
        query="Парк Горького",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.ref == mint_place_ref("tomtom:poi:generic-attraction")
    assert (resolved.record.lat, resolved.record.lon) == (55.728935, 37.601208)
    assert resolved.record.provider == "tomtom"
    assert resolved.record.origin is RecordOrigin.PLACES_SEARCH
    assert await store.get(resolved.ref) == resolved.record
    assert client.calls[0]["endpoint"] is TomTomSearchEndpoint.FUZZY
    assert client.calls[0]["query"] == "Парк Горького, Москва"
    assert client.calls[0]["language"] == "ru-RU"


async def test_named_poi_resolver_first_address_path_skips_semantic_filtering() -> None:
    store = InMemoryPlaceStore()
    client = StaticTomTomClient(
        _response(
            _result(
                "blank-address",
                name="Unusable first card",
                address=" ",
                municipality="London",
                lat=51.50,
                lon=-0.13,
            ),
            _result(
                "provider-first-with-address",
                name="Station Hotel",
                address="1 Station Road, London",
                municipality="London",
                lat=51.51,
                lon=-0.12,
                classification="HOTEL_MOTEL",
            ),
            _result(
                "semantic-match",
                name="London Railway Station",
                address="2 Station Road, London",
                municipality="London",
                lat=51.52,
                lon=-0.11,
                classification="RAILWAY_STATION",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(client=client, place_store=store)

    resolved = await resolver.resolve_first_address(
        query="London Railway Station",
        city="London",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.provider_id == "provider-first-with-address"


@pytest.mark.parametrize(
    ("query", "city", "expected_language"),
    [
        ("Бранденбургские ворота", "Берлин", "ru-RU"),
        ("Brandenburger Tor", "Berlin", "en-US"),
    ],
)
async def test_named_poi_resolver_localizes_response_to_query_script(
    query: str,
    city: str,
    expected_language: str,
) -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "gate",
                name=query,
                address=f"Pariser Platz, {city}",
                municipality=city,
                lat=52.5163,
                lon=13.3777,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            )
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query=query,
        city=city,
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.name == query
    assert client.calls[0]["language"] == expected_language


@pytest.mark.parametrize(
    ("query", "city", "expected_language", "expected_question_prefix"),
    [
        ("ВДНХ", "Москва", "ru-RU", "Какой объект"),
        ("National Gallery", "London", "en-US", "Which place"),
    ],
)
async def test_named_poi_resolver_clarifies_distinct_anchors_in_query_language(
    query: str,
    city: str,
    expected_language: str,
    expected_question_prefix: str,
) -> None:
    store = InMemoryPlaceStore()
    client = StaticTomTomClient(
        _response(
            _result(
                "first-anchor",
                name=query,
                address=f"First address, {city}",
                municipality=city,
                lat=51.50,
                lon=-0.12,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            ),
            _result(
                "second-anchor",
                name=query,
                address=f"Second address, {city}",
                municipality=city,
                lat=51.55,
                lon=-0.18,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(client=client, place_store=store)

    with pytest.raises(AmbiguousPlaceError) as exc_info:
        await resolver.resolve_named_poi(
            query=query,
            city=city,
            context=ToolExecutionContext(),
        )

    clarification = exc_info.value.clarification
    assert clarification is not None
    assert clarification.kind == "select_anchor"
    assert clarification.question.startswith(expected_question_prefix)
    assert [option.value for option in clarification.options] == [
        mint_place_ref("tomtom:poi:first-anchor"),
        mint_place_ref("tomtom:poi:second-anchor"),
    ]
    records = [await store.get(option.value) for option in clarification.options]
    assert all(record is not None for record in records)
    assert client.calls[0]["language"] == expected_language


async def test_named_poi_resolver_keeps_first_provider_ranked_exact_match() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "first",
                name="ВДНХ",
                address="проспект Мира, 119, Москва",
                municipality="Москва",
                lat=55.8298,
                lon=37.6321,
                score=3.8455998898,
            ),
            _result(
                "second",
                name="ВДНХ",
                address="проспект Мира, 121, Москва",
                municipality="Москва",
                lat=55.8299,
                lon=37.6323,
                score=3.8455998898,
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="ВДНХ",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.ref == mint_place_ref("tomtom:poi:first")


async def test_named_poi_resolver_selects_top_score_when_gap_exceeds_three_percent() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "rink",
                name="Медео каток",
                address="Алматы",
                municipality="Алматы",
                lat=43.157502,
                lon=77.059022,
                score=4.0,
                classification="ICE_SKATING_RINK",
            ),
            _result(
                "utility",
                name="Медеу",
                address="Улица Дуйсенова, Алматы",
                municipality="Алматы",
                lat=43.253148,
                lon=76.879576,
                score=3.0,
                classification="PRIMARY_RESOURCE_UTILITY",
            ),
            _result(
                "shop",
                name="Медеу",
                address="Улица Есенжанова, Алматы",
                municipality="Алматы",
                lat=43.24014,
                lon=76.876476,
                score=2.0,
                classification="SHOP",
            ),
            _result(
                "sports-center",
                name="Медеу",
                address="Алматы",
                municipality="Алматы",
                lat=43.157139,
                lon=77.059469,
                score=1.0,
                classification="SPORTS_CENTER",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="Медеу",
        city="Алматы",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.provider_id == "rink"


async def test_named_poi_resolver_clarifies_heterogeneous_exact_names() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "utility",
                name="Медеу",
                address="Улица Дуйсенова, Алматы",
                municipality="Алматы",
                lat=43.253148,
                lon=76.879576,
                classification="PRIMARY_RESOURCE_UTILITY",
            ),
            _result(
                "shop",
                name="Медеу",
                address="Улица Есенжанова, Алматы",
                municipality="Алматы",
                lat=43.24014,
                lon=76.876476,
                classification="SHOP",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    with pytest.raises(AmbiguousPlaceError) as exc_info:
        await resolver.resolve_named_poi(
            query="Медеу",
            city="Алматы",
            context=ToolExecutionContext(),
        )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("tomtom:poi:utility"),
        mint_place_ref("tomtom:poi:shop"),
    ]


async def test_named_poi_resolver_clarifies_distant_partial_matches() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "large",
                name="Большой Центральный парк",
                address="Первая улица, Москва",
                municipality="Москва",
                lat=55.75,
                lon=37.61,
            ),
            _result(
                "small",
                name="Малый Центральный парк",
                address="Вторая улица, Москва",
                municipality="Москва",
                lat=55.76,
                lon=37.62,
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    with pytest.raises(AmbiguousPlaceError) as exc_info:
        await resolver.resolve_named_poi(
            query="Центральный парк",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert exc_info.value.clarification is not None
    assert [option.value for option in exc_info.value.clarification.options] == [
        mint_place_ref("tomtom:poi:large"),
        mint_place_ref("tomtom:poi:small"),
    ]


async def test_named_poi_resolver_rejects_a_result_from_another_city() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "wrong-city",
                name="Парк Горького",
                address="улица Горького, Казань",
                municipality="Казань",
                lat=55.79,
                lon=49.12,
            )
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="Парк Горького",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is None


async def test_named_poi_resolver_uses_city_bounds_across_localized_city_names() -> None:
    berlin_bounds = GeoBounds(west=13.08, south=52.33, east=13.76, north=52.68)
    client = StaticTomTomClient(
        _response(
            _result(
                "kaliningrad-gate",
                name="Бранденбургские Ворота",
                address="улица Багратиона, 137, Калининград",
                municipality="Калининград",
                lat=54.6978,
                lon=20.5033,
                score=4.7,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            ),
            _result(
                "berlin-gate",
                name="Бранденбургские ворота",
                address="Pariser Platz, Berlin",
                municipality="Berlin",
                lat=52.5163,
                lon=13.3777,
                score=4.0,
                classification="IMPORTANT_TOURIST_ATTRACTION",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="Бранденбургские ворота",
        city="Берлин",
        city_bounds=berlin_bounds,
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.provider_id == "berlin-gate"
    assert resolved.record.locality == "Berlin"
    assert client.calls[0]["bbox"] == berlin_bounds


async def test_named_poi_resolver_respects_explicit_type_hint() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "commercial-building",
                name="Авиапарк",
                address="Ленинградский проспект, 68с2, Москва",
                municipality="Москва",
                lat=55.8048,
                lon=37.5230,
                score=10.0,
                classification="COMMERCIAL_BUILDING",
            ),
            _result(
                "shopping-center",
                name="Авиапарк",
                address="Ходынский бульвар, 4, Москва",
                municipality="Москва",
                lat=55.7906,
                lon=37.5299,
                score=8.0,
                classification="SHOPPING_CENTER",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="торговый центр Авиапарк",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.ref == mint_place_ref("tomtom:poi:shopping-center")


async def test_named_poi_resolver_prefers_exact_name_over_marketing_prefix() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "provider-winner",
                name="DoubleTree Amsterdam Centraal Station",
                address="Oosterdoksstraat 4, Amsterdam",
                municipality="Berlin",
                lat=52.52,
                lon=13.40,
                score=10.0,
                classification="HOTEL_MOTEL",
            ),
            _result(
                "lexical-and-typed-match",
                name="Amsterdam Centraal Station",
                address="Stationsplein, Amsterdam",
                municipality="Berlin",
                lat=52.521,
                lon=13.41,
                score=8.0,
                classification="RAILWAY_STATION",
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="Amsterdam Centraal Station",
        city="Berlin",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.provider_id == "lexical-and-typed-match"


@pytest.mark.parametrize(
    ("second_score", "clarifies"),
    [
        (9.71, True),
        (9.70, False),
    ],
)
async def test_named_poi_resolver_uses_three_percent_relative_score_gap(
    second_score: float,
    clarifies: bool,
) -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "first",
                name="Any query",
                address="First address, Berlin",
                municipality="Berlin",
                lat=52.50,
                lon=13.40,
                score=10.0,
            ),
            _result(
                "second",
                name="Any query",
                address="Second address, Berlin",
                municipality="Berlin",
                lat=52.60,
                lon=13.50,
                score=second_score,
            ),
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    if clarifies:
        with pytest.raises(AmbiguousPlaceError):
            await resolver.resolve_named_poi(
                query="Any query",
                city="Berlin",
                context=ToolExecutionContext(),
            )
    else:
        resolved = await resolver.resolve_named_poi(
            query="Any query",
            city="Berlin",
            context=ToolExecutionContext(),
        )
        assert resolved is not None
        assert resolved.record.provider_id == "first"


async def test_named_poi_resolver_keeps_exact_name_despite_type_word() -> None:
    client = StaticTomTomClient(
        _response(
            _result(
                "station-hotel",
                name="Station Hotel",
                address="London",
                municipality="London",
                lat=51.50,
                lon=-0.12,
                classification="HOTEL_MOTEL",
            )
        )
    )
    resolver = TomTomNamedPoiResolver(
        client=client,
        place_store=InMemoryPlaceStore(),
    )

    resolved = await resolver.resolve_named_poi(
        query="Station Hotel",
        city="London",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record.provider_id == "station-hotel"


async def test_near_search_resolves_named_anchor_before_category_search() -> None:
    """Exercise geocoder fallback and category search within one tool call."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "/nearbySearch/" not in request.url.path:
            return httpx.Response(
                200,
                json={
                    "summary": {"numResults": 1, "totalResults": 1},
                    "results": [
                        _result(
                            "park",
                            name="Парк Горького",
                            address="улица Крымский Вал, 9, Москва",
                            municipality="Москва",
                            lat=55.7296,
                            lon=37.6017,
                            entry_lat=55.7301,
                            entry_lon=37.6032,
                        )
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "summary": {"numResults": 1, "totalResults": 1},
                "results": [
                    {
                        "type": "POI",
                        "id": "restaurant",
                        "score": 0.9,
                        "poi": {
                            "name": "Ресторан рядом",
                            "categorySet": [{"id": 7315002}],
                            "classifications": [
                                {
                                    "code": "RESTAURANT",
                                    "names": [{"name": "restaurant"}],
                                }
                            ],
                        },
                        "address": {
                            "freeformAddress": "улица Крымский Вал, 10, Москва",
                            "municipality": "Москва",
                        },
                        "position": {"lat": 55.7310, "lon": 37.6040},
                    }
                ],
            },
        )

    store = InMemoryPlaceStore()
    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomSearchClient(api_key="key", http_client=http_client)
        geocoded_place_resolver = GeocodedPlaceResolver(
            geocoder=AmbiguousParkGeocoder(),
            place_store=store,
        )
        text_place_resolver = TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=(
                TomTomNamedPoiResolver(
                    client=client,
                    place_store=store,
                ),
            ),
        )
        provider = TomTomPlacesSearchProvider(
            client=client,
            place_store=store,
        )
        args = await PlacesSearchScopeResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            text_place_resolver=text_place_resolver,
        ).resolve_scope(
            PlacesSearchInput(
                mode="near",
                query="рестораны",
                category="restaurant",
                near_query="Парк Горького",
                city="Москва",
                radius_m=1500,
                limit=7,
            ),
            context,
        )
        result = await provider.search(args, context)

    assert len(requests) == 2
    assert "/search/" in requests[0].url.path
    assert requests[0].url.params["idxSet"] == "POI"
    assert requests[0].url.params["language"] == "ru-RU"
    assert requests[1].url.path.endswith("/nearbySearch/.json")
    assert requests[1].url.params["categorySet"] == "7315"
    assert requests[1].url.params["language"] == "ru-RU"
    assert result.anchor == mint_place_ref("tomtom:poi:park")
    assert [place.name for place in result.places] == ["Ресторан рядом"]
    assert [call.provider for call in context.upstream_calls] == [
        "tomtom_search",
        "tomtom_search",
    ]
