"""Tests for the Yandex-backed organisation-search provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding import GeocodedPlaceResolver
from tools.geo.geocoding.schemas import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ToponymKind,
)
from tools.geo.geocoding.service import GeocoderService
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search.provider import PlacesSearchProvider
from tools.geo.places_search.resolution import PlacesSearchScopeResolver
from tools.geo.places_search.schemas import (
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
    SearchMode,
)
from tools.geo.places_search.yandex.client import (
    YandexOrganisationSearchClient,
)
from tools.geo.places_search.yandex.provider import (
    YandexPlacesSearchProvider,
)
from tools.geo.text_place_resolution import TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref


def _organisation(
    *,
    company_id: str,
    name: str,
    address: str,
    lon: float,
    lat: float,
    open_24h: bool | None = None,
    availabilities: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    company: dict[str, object] = {
        "id": company_id,
        "name": name,
        "Address": {"formatted": address},
        "Categories": [{"name": "Кофейня"}],
    }

    if open_24h is not None:
        company["Hours"] = {
            "text": "круглосуточно" if open_24h else "ежедневно, 09:00–22:00",
            "Availabilities": [
                {"Everyday": True, "TwentyFourHours": open_24h},
            ],
        }
    elif availabilities is not None:
        company["Hours"] = {
            "text": "структурированное расписание",
            "Availabilities": availabilities,
        }

    return {
        "type": "Feature",
        "properties": {
            "CompanyMetaData": company,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
    }


def _payload(
    *features: dict[str, object],
    found: int,
) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "properties": {
            "ResponseMetaData": {
                "SearchResponse": {
                    "found": found,
                },
            },
        },
        "features": list(features),
    }


class RecordingGeocoder:
    provider = "fake_geocoder"

    def __init__(self, store: InMemoryPlaceStore) -> None:
        self._store = store
        self.calls: list[GeocodePlaceInput] = []
        self.ref = mint_place_ref("yandex:geo:red-square")
        self.area_ref = mint_place_ref("yandex:geo:moscow")

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        context.record_upstream_call(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=120,
        )

        if args.query == "Москва" and args.city is None:
            await self._store.save(
                PlaceRecord(
                    ref=self.area_ref,
                    name="Москва",
                    address="Россия, Москва",
                    lat=55.755864,
                    lon=37.617698,
                    kind="locality",
                    locality="Москва",
                    bounds=GeoBounds(
                        west=36.803101,
                        south=55.142174,
                        east=37.967427,
                        north=56.021251,
                    ),
                    origin=RecordOrigin.GEOCODE,
                ),
            )
            match = PlaceMatch(
                ref=self.area_ref,
                name="Москва",
                address="Россия, Москва",
                kind=ToponymKind.LOCALITY,
            )
            return GeocodePlaceOutput(best=match, matches=[match])

        await self._store.save(
            PlaceRecord(
                ref=self.ref,
                name="Красная площадь",
                address="Россия, Москва, Красная площадь",
                lat=55.753544,
                lon=37.621202,
                origin=RecordOrigin.GEOCODE,
            ),
        )

        match = PlaceMatch(
            ref=self.ref,
            name="Красная площадь",
            address="Россия, Москва, Красная площадь",
        )

        return GeocodePlaceOutput(
            best=match,
            matches=[match],
        )


class NonPersistingGeocoder(RecordingGeocoder):
    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        match = PlaceMatch(
            ref=self.ref,
            name="Красная площадь",
            address="Россия, Москва, Красная площадь",
        )
        return GeocodePlaceOutput(best=match, matches=[match])


class UnboundedAreaGeocoder(RecordingGeocoder):
    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        context.record_upstream_call(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=120,
        )
        await self._store.save(
            PlaceRecord(
                ref=self.area_ref,
                name="Москва",
                address="Россия, Москва",
                lat=55.755864,
                lon=37.617698,
                kind="locality",
                locality="Москва",
                origin=RecordOrigin.GEOCODE,
            ),
        )
        match = PlaceMatch(
            ref=self.area_ref,
            name="Москва",
            address="Россия, Москва",
            kind=ToponymKind.LOCALITY,
        )
        return GeocodePlaceOutput(best=match, matches=[match])


class AmbiguousAreaGeocoder(RecordingGeocoder):
    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        context.record_upstream_call(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=120,
        )

        matches: list[PlaceMatch] = []
        records: list[PlaceRecord] = []

        for region, west in (("Регион А", 36.0), ("Регион Б", 38.0)):
            ref = mint_place_ref(f"yandex:geo:moscow:{region}")
            records.append(
                PlaceRecord(
                    ref=ref,
                    name="Москва",
                    address=f"Россия, {region}, Москва",
                    lat=55.75,
                    lon=west + 0.5,
                    kind="locality",
                    locality="Москва",
                    bounds=GeoBounds(
                        west=west,
                        south=55.0,
                        east=west + 1.0,
                        north=56.0,
                    ),
                    origin=RecordOrigin.GEOCODE,
                ),
            )
            matches.append(
                PlaceMatch(
                    ref=ref,
                    name="Москва",
                    address=f"Россия, {region}, Москва",
                    kind=ToponymKind.LOCALITY,
                ),
            )

        await self._store.save_many(records)
        return GeocodePlaceOutput(
            best=matches[0],
            matches=matches,
            ambiguous=True,
        )


class RecordingPlaceStore(InMemoryPlaceStore):
    def __init__(self) -> None:
        super().__init__()
        self.save_calls = 0
        self.save_many_calls: list[list[PlaceRecord]] = []

    async def save(self, record: PlaceRecord) -> None:
        self.save_calls += 1
        await super().save(record)

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        batch = list(records)
        self.save_many_calls.append(batch)
        await super().save_many(batch)


def _provider(
    *,
    client: YandexOrganisationSearchClient,
    place_store: InMemoryPlaceStore,
    geocoder: GeocoderService,
    clock: Callable[[], datetime] | None = None,
) -> YandexPlacesSearchProvider:
    return YandexPlacesSearchProvider(
        client=client,
        place_store=place_store,
        clock=clock or (lambda: datetime.now(UTC)),
    )


async def _search_with_resolved_scope(
    provider: YandexPlacesSearchProvider,
    args: PlacesSearchInput,
    context: ToolExecutionContext,
    *,
    place_store: InMemoryPlaceStore,
    geocoder: GeocoderService,
) -> PlacesSearchOutput:
    geocoded_place_resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=place_store,
    )
    resolver = PlacesSearchScopeResolver(
        geocoded_place_resolver=geocoded_place_resolver,
        text_place_resolver=TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
        ),
    )
    resolved_args = await resolver.resolve_scope(args, context)
    return await provider.search(resolved_args, context)


async def test_provider_open_now_keeps_only_confirmed_open_places() -> None:
    payload = _payload(
        _organisation(
            company_id="open",
            name="Открытая кофейня",
            address="Москва, Первая улица, 1",
            lon=37.62,
            lat=55.75,
            availabilities=[
                {
                    "Everyday": True,
                    "Intervals": [{"from": "14:00:00", "to": "16:00:00"}],
                }
            ],
        ),
        _organisation(
            company_id="closed",
            name="Закрытая кофейня",
            address="Москва, Вторая улица, 2",
            lon=37.621,
            lat=55.751,
            availabilities=[
                {
                    "Everyday": True,
                    "Intervals": [{"from": "16:00:00", "to": "18:00:00"}],
                }
            ],
        ),
        _organisation(
            company_id="unknown",
            name="Кофейня без расписания",
            address="Москва, Третья улица, 3",
            lon=37.622,
            lat=55.752,
        ),
        found=3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["results"] == "20"
        return httpx.Response(200, json=payload)

    store = RecordingPlaceStore()
    geocoder = RecordingGeocoder(store)
    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
            clock=lambda: datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        )
        result = await _search_with_resolved_scope(
            provider,
            PlacesSearchInput(
                mode="area",
                query="кофейни",
                city="Москва",
                open_now=True,
            ),
            context,
            place_store=store,
            geocoder=geocoder,
        )

    assert provider.supports_open_now is True
    assert [place.id for place in result.places] == ["open"]
    assert result.places[0].is_open_now is True
    assert await store.get(mint_place_ref("yandex:organisation:closed")) is None
    assert await store.get(mint_place_ref("yandex:organisation:unknown")) is None
    assert context.warnings == (
        "Yandex schedules could not verify the current opening status of 1 result(s); "
        "those results were omitted.",
    )


async def test_area_search_filters_24h_places_and_saves_returned_records():
    """Verify that area search filters 24h places and saves returned records."""

    payload = _payload(
        _organisation(
            company_id="always-open",
            name="Круглосуточная кофейня",
            address="Москва, Никольская улица, 17",
            lon=37.6231,
            lat=55.7581,
            open_24h=True,
        ),
        _organisation(
            company_id="daytime",
            name="Дневная кофейня",
            address="Москва, Мясницкая улица, 24",
            lon=37.6380,
            lat=55.7630,
            open_24h=False,
        ),
        found=2,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["text"] == "кофейни"
        assert request.url.params["results"] == "30"
        assert request.url.params["bbox"] == ("36.803101,55.142174~37.967427,56.021251")
        assert request.url.params["rspn"] == "1"
        assert "ll" not in request.url.params
        assert "spn" not in request.url.params

        return httpx.Response(200, json=payload)

    store = RecordingPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        context = ToolExecutionContext()
        result = await _search_with_resolved_scope(
            service,
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="кофейни",
                category="cafe",
                city="Москва",
                open_24h=True,
            ),
            context,
            place_store=store,
            geocoder=geocoder,
        )

    assert isinstance(service, PlacesSearchProvider)
    assert geocoder.calls == [GeocodePlaceInput(query="Москва", limit=5, locality_only=True)]
    assert result.returned_count == 1
    assert result.truncated is False
    assert result.area == ResolvedSearchArea(
        ref=geocoder.area_ref,
        name="Москва",
        address="Россия, Москва",
    )
    assert result.anchor is None
    assert [place.name for place in result.places] == ["Круглосуточная кофейня"]

    stored = await store.get(result.places[0].ref)

    assert stored is not None
    assert stored.provider == "yandex"
    assert stored.lat == 55.7581
    assert stored.lon == 37.6231
    assert stored.origin is RecordOrigin.PLACES_SEARCH
    assert store.save_calls == 1
    assert len(store.save_many_calls) == 1
    assert [record.provider_id for record in store.save_many_calls[0]] == ["always-open"]
    assert [call.provider for call in context.upstream_calls] == [
        "yandex_geocoder",
        "yandex_organisation_search",
    ]


async def test_area_ref_reuses_resolved_city_without_geocoding_again():
    """Verify that area ref reuses resolved city without geocoding again."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_payload(found=0))

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        first_context = ToolExecutionContext()
        first = await _search_with_resolved_scope(
            service,
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="рестораны",
                city="Москва",
            ),
            first_context,
            place_store=store,
            geocoder=geocoder,
        )
        assert first.area is not None

        second_context = ToolExecutionContext()
        second = await service.search(
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="музеи",
                area_ref=first.area.ref,
            ),
            second_context,
        )

    assert geocoder.calls == [GeocodePlaceInput(query="Москва", limit=5, locality_only=True)]
    assert [request.url.params["text"] for request in requests] == ["рестораны", "музеи"]
    assert all(
        request.url.params["bbox"] == "36.803101,55.142174~37.967427,56.021251"
        for request in requests
    )
    assert second.area == first.area
    assert [call.provider for call in second_context.upstream_calls] == [
        "yandex_organisation_search"
    ]


async def test_area_ref_rejects_unknown_ref_without_geocoding():
    """Verify that area ref rejects unknown ref without geocoding."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("organisation search must not run for an unknown area ref")

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="музеи",
                    area_ref="plc_0000000000",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.UNKNOWN_REF
    assert str(exc_info.value) == (
        "Search area ref plc_0000000000 was not found. Pass city to resolve it again."
    )
    assert geocoder.calls == []


@pytest.mark.parametrize(
    ("kind", "bounds"),
    [
        (
            None,
            GeoBounds(
                west=36.803101,
                south=55.142174,
                east=37.967427,
                north=56.021251,
            ),
        ),
        ("locality", None),
    ],
)
async def test_area_ref_must_identify_a_bounded_locality(
    kind: str | None,
    bounds: GeoBounds | None,
):
    """Verify that area ref must identify a bounded locality."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("organisation search must not run for an invalid area ref")

    ref = mint_place_ref(f"invalid-area:{kind}:{bounds}")
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=ref,
            name="Не область поиска",
            address="Не область поиска",
            lat=55.75,
            lon=37.61,
            kind=kind,
            bounds=bounds,
            origin=RecordOrigin.PLACES_SEARCH,
        )
    )
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="музеи",
                    area_ref=ref,
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.INVALID_INPUT
    assert str(exc_info.value) == (f"Place ref {ref} does not identify a bounded city search area.")
    assert geocoder.calls == []


async def test_area_search_rejects_locality_without_bounds():
    """Verify that area search rejects locality without bounds."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("organisation search must not run without city bounds")

    store = InMemoryPlaceStore()
    geocoder = UnboundedAreaGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="кофейни",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert str(exc_info.value) == "Could not resolve a bounded city search area: 'Москва'"


async def test_area_search_uses_first_ranked_city():
    """Verify that area search uses the geocoder's top-ranked bounded city."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["bbox"] == "36.000000,55.000000~37.000000,56.000000"
        return httpx.Response(200, json=_payload(found=0))

    store = InMemoryPlaceStore()
    geocoder = AmbiguousAreaGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        result = await _search_with_resolved_scope(
            service,
            PlacesSearchInput(
                mode=SearchMode.AREA,
                query="кофейни",
                city="Москва",
            ),
            ToolExecutionContext(),
            place_store=store,
            geocoder=geocoder,
        )

    assert result.places == []


async def test_near_query_geocodes_anchor_then_filters_to_true_radius():
    """Verify that near query geocodes anchor then filters to true radius."""

    payload = _payload(
        _organisation(
            company_id="nearby",
            name="Кофейня рядом",
            address="Москва, Никольская улица, 17",
            lon=37.6231,
            lat=55.7581,
        ),
        _organisation(
            company_id="far-away",
            name="Далёкая кофейня",
            address="Москва, Мясницкая улица, 24",
            lon=37.6380,
            lat=55.7630,
        ),
        found=2,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["text"] == "кофейни"
        assert request.url.params["ll"] == "37.621202,55.753544"
        assert request.url.params["rspn"] == "1"

        return httpx.Response(200, json=payload)

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        context = ToolExecutionContext()
        result = await _search_with_resolved_scope(
            service,
            PlacesSearchInput(
                mode=SearchMode.NEAR,
                query="кофейни",
                near_query="Красная площадь",
                city="Москва",
                radius_m=700,
            ),
            context,
            place_store=store,
            geocoder=geocoder,
        )

    assert geocoder.calls == [
        GeocodePlaceInput(
            query="Красная площадь",
            city="Москва",
            limit=5,
        ),
        GeocodePlaceInput(query="Москва", limit=5, locality_only=True),
    ]
    assert result.anchor == geocoder.ref
    assert result.area is not None
    assert result.area.ref == geocoder.area_ref
    assert [place.name for place in result.places] == ["Кофейня рядом"]
    assert result.places[0].distance_m is not None
    assert result.places[0].distance_m < 700
    assert [call.provider for call in context.upstream_calls] == [
        "yandex_geocoder",
        "yandex_geocoder",
        "yandex_organisation_search",
    ]


async def test_near_query_rejects_unmatched_anchor_before_places_request():
    """Verify near mode does not use unrelated locality matches as its anchor."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("organisation search must not run for an ambiguous anchor")

    store = InMemoryPlaceStore()
    geocoder = AmbiguousAreaGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.NEAR,
                    query="кофейни",
                    near_query="Красная площадь",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert str(exc_info.value) == ("Could not resolve the nearby-search anchor: 'Красная площадь'")


async def test_near_query_rejects_geocoder_that_does_not_persist_anchor():
    """Verify that near query rejects geocoder that does not persist anchor."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("organisation search must not be called without a stored anchor")

    store = InMemoryPlaceStore()
    geocoder = NonPersistingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.NEAR,
                    query="кофейни",
                    near_query="Красная площадь",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == (
        "Internal geocoder returned a place ref without persisting its record"
    )
    assert exc_info.value.provider == "fake_geocoder"
    assert exc_info.value.status_code is None
    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert exc_info.value.retryable is False


async def test_area_search_maps_invalid_yandex_schema_to_safe_upstream_error():
    """Verify that area search maps invalid Yandex schema to safe upstream error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="кофейни",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == (
        "Yandex organisation search returned data in an unexpected format"
    )
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.status_code is None
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ValidationError)


async def test_client_value_error_is_an_internal_contract_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that client value error is an internal contract error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request must not be sent")

    client_error = ValueError("defensive client validation failed")
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        async def fail_search(**_: object) -> dict[str, object]:
            raise client_error

        monkeypatch.setattr(client, "search", fail_search)
        store = InMemoryPlaceStore()
        geocoder = RecordingGeocoder(store)
        service = _provider(
            client=client,
            place_store=store,
            geocoder=geocoder,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="кофейни",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    error = exc_info.value
    assert error.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert error.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert error.provider == "yandex_organisation_search"
    assert error.retryable is False
    assert error.__cause__ is client_error
    assert "defensive client validation failed" not in str(error)


async def test_invalid_materialized_output_is_an_internal_contract_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that invalid materialized output is an internal contract error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request must not be sent")

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store)
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = _provider(
            client=YandexOrganisationSearchClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
            geocoder=geocoder,
        )

        async def fail_materialization(
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchOutput:
            del args, context
            return PlacesSearchOutput.model_validate({"anchor": "not-a-place-ref"})

        monkeypatch.setattr(service, "_search", fail_materialization)

        with pytest.raises(ToolExecutionError) as exc_info:
            await _search_with_resolved_scope(
                service,
                PlacesSearchInput(
                    mode=SearchMode.AREA,
                    query="кофейни",
                    city="Москва",
                ),
                ToolExecutionContext(),
                place_store=store,
                geocoder=geocoder,
            )

    error = exc_info.value
    assert error.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert error.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert error.provider == "yandex_organisation_search"
    assert error.retryable is False
    assert isinstance(error.__cause__, ValidationError)
