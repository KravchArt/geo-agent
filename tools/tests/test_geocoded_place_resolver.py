"""Tests for geocoded places, search scopes, and routing-input resolution."""

from __future__ import annotations

import pytest

from tools.base import (
    ToolClarification,
    ToolClarificationOption,
    ToolErrorCode,
    ToolExecutionError,
    ToolFailureKind,
)
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding import (
    GeocodedPlaceResolver,
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    PlaceResolutionContractError,
    ToponymKind,
    select_unique_place_match,
)
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.places_search.provider import (
    AmbiguousSearchAnchorError,
    AnchorNotFoundError,
)
from tools.geo.places_search.resolution import PlacesSearchScopeLoader, PlacesSearchScopeResolver
from tools.geo.places_search.schemas import PlacesSearchInput
from tools.geo.routing import RoutingInput, RoutingPlaceLoader, RoutingPlaceResolver
from tools.geo.text_place_resolution import PlaceResolutionArea, TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, ResolvedPlace


class StaticGeocoder:
    provider = "fake_geocoder"

    def __init__(
        self,
        *,
        output: GeocodePlaceOutput,
        store: InMemoryPlaceStore,
        record: PlaceRecord | None = None,
    ) -> None:
        self._output = output
        self._store = store
        self._record = record
        self.calls: list[GeocodePlaceInput] = []

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        if self._record is not None:
            await self._store.save(self._record)
        return self._output


class LocalizingStaticGeocoder(StaticGeocoder):
    def __init__(
        self,
        *,
        localized_city: str,
        output: GeocodePlaceOutput,
        store: InMemoryPlaceStore,
        record: PlaceRecord | None = None,
    ) -> None:
        super().__init__(output=output, store=store, record=record)
        self._localized_city = localized_city
        self.reverse_calls: list[tuple[float, float, str]] = []

    async def reverse_geocode_city(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
    ) -> str | None:
        self.reverse_calls.append((lat, lon, language))
        return self._localized_city


class StaticNamedPoiResolver:
    requires_city_bounds = False

    def __init__(
        self,
        result: ResolvedPlace | None,
        *,
        provider: str = "static",
        error: ToolExecutionError | None = None,
        requires_city_bounds: bool = False,
    ) -> None:
        self.provider = provider
        self.requires_city_bounds = requires_city_bounds
        self._result = result
        self._error = error
        self.calls: list[tuple[str, str]] = []
        self.first_address_calls: list[tuple[str, str]] = []
        self.received_city_bounds: list[GeoBounds | None] = []
        self.received_areas: list[PlaceResolutionArea | None] = []

    async def resolve_named_poi(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        self.calls.append((query, city))
        self.received_city_bounds.append(city_bounds)
        self.received_areas.append(area)
        if self._error is not None:
            raise self._error
        return self._result

    async def resolve_first_address(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        self.first_address_calls.append((query, city))
        return await self.resolve_named_poi(
            query=query,
            city=city,
            context=context,
            city_bounds=city_bounds,
            area=area,
        )


def _scope_resolver(
    geocoded_place_resolver: GeocodedPlaceResolver,
    *,
    named_poi_resolvers: tuple[StaticNamedPoiResolver, ...] = (),
) -> PlacesSearchScopeResolver:
    return PlacesSearchScopeResolver(
        geocoded_place_resolver=geocoded_place_resolver,
        text_place_resolver=TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=named_poi_resolvers,
            named_first=True,
        ),
    )


def _match(
    ref: str,
    *,
    name: str,
    address: str,
    kind: ToponymKind | None = None,
    precision: str | None = None,
    relevance_score: float | None = None,
    match_confidence_score: float | None = None,
    municipality: str | None = None,
    country_secondary_subdivision: str | None = None,
    country_subdivision: str | None = None,
    country_code: str | None = None,
    provider_entity_type: str | None = None,
) -> PlaceMatch:
    return PlaceMatch(
        ref=ref,
        name=name,
        address=address,
        kind=kind,
        precision=precision,
        municipality=municipality,
        country_secondary_subdivision=country_secondary_subdivision,
        country_subdivision=country_subdivision,
        country_code=country_code,
        provider_entity_type=provider_entity_type,
        relevance_score=relevance_score,
        match_confidence_score=match_confidence_score,
    )


def test_selector_returns_none_when_geocoder_found_nothing() -> None:
    """Verify an empty geocoder response remains an ordinary not-found result."""

    assert (
        select_unique_place_match(
            GeocodePlaceOutput(),
            query="Неизвестное место",
            city=None,
        )
        is None
    )


def test_selector_accepts_a_single_candidate() -> None:
    """Verify one candidate can be selected without additional heuristics."""

    match = _match(
        "plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=match, matches=[match]),
        query="Красная площадь",
        city=None,
    )

    assert selected == match


def test_selector_uses_city_address_components() -> None:
    """Verify an explicit city removes otherwise identical candidates elsewhere."""

    moscow = _match(
        "plc_a1b2c3d4e5",
        name="Центральный парк",
        address="Россия, Москва, Центральный парк",
    )
    kazan = _match(
        "plc_b2c3d4e5f6",
        name="Центральный парк",
        address="Россия, Татарстан, Казань, Центральный парк",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=moscow, matches=[moscow, kazan], ambiguous=True),
        query="Центральный парк",
        city="Москва",
    )

    assert selected == moscow


def test_selector_accepts_russian_rendering_of_ukrainian_i() -> None:
    """Keep a correct Ukrainian anchor found for a Russian-language request."""

    maidan = _match(
        "plc_a1b2c3d4e5",
        name="Майдан Незалежності",
        address="Майдан Незалежності, Киев, Украина",
        kind=ToponymKind.STREET,
        precision="other",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=maidan, matches=[maidan]),
        query="Майдан Незалежности",
        city="Киев",
    )

    assert selected == maidan


def test_selector_accepts_city_inside_a_longer_address_component() -> None:
    """Provider-specific city prefixes must not reject the correct address."""

    house = _match(
        "plc_a1b2c3d4e5",
        name="Большая Покровская, 2",
        address=(
            "Россия, Нижегородская область, городской округ Нижний Новгород, Большая Покровская, 2"
        ),
        kind=ToponymKind.HOUSE,
        precision="number",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=house, matches=[house]),
        query="Большая Покровская улица, 2",
        city="Нижний Новгород",
    )

    assert selected == house


def test_selector_does_not_ignore_a_different_house_number() -> None:
    house = _match(
        "plc_a1b2c3d4e5",
        name="Большая Покровская, 4",
        address="Россия, Нижний Новгород, Большая Покровская, 4",
        kind=ToponymKind.HOUSE,
        precision="number",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=house, matches=[house]),
        query="Большая Покровская улица, 2",
        city="Нижний Новгород",
    )

    assert selected is None


def test_selector_accepts_provider_added_house_letter() -> None:
    house = _match(
        "plc_a1b2c3d4e5",
        name="Большая Покровская улица, 2А",
        address="Россия, Нижний Новгород, Большая Покровская улица, 2А",
        kind=ToponymKind.HOUSE,
        precision="exact",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=house, matches=[house]),
        query="Большая Покровская улица, 2",
        city="Нижний Новгород",
    )

    assert selected == house


def test_selector_rejects_a_different_explicit_house_letter() -> None:
    house = _match(
        "plc_a1b2c3d4e5",
        name="Большая Покровская улица, 2А",
        address="Россия, Нижний Новгород, Большая Покровская улица, 2А",
        kind=ToponymKind.HOUSE,
        precision="exact",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=house, matches=[house]),
        query="Большая Покровская улица, 2Б",
        city="Нижний Новгород",
    )

    assert selected is None


def test_selector_rejects_city_fallback_for_a_specific_address() -> None:
    """A matching city is not a valid substitute for the requested house."""

    city = _match(
        "plc_a1b2c3d4e5",
        name="Нижний Новгород",
        address="Россия, Нижегородская область, Нижний Новгород",
        kind=ToponymKind.LOCALITY,
        precision="other",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=city, matches=[city]),
        query="площадь Революции, 2А",
        city="Нижний Новгород",
    )

    assert selected is None


def test_selector_keeps_exact_house_and_drops_city_fallback() -> None:
    """An exact house wins even when Yandex also returns the containing city."""

    city = _match(
        "plc_a1b2c3d4e5",
        name="Нижний Новгород",
        address="Россия, Нижегородская область, Нижний Новгород",
        kind=ToponymKind.LOCALITY,
        precision="other",
    )
    house = _match(
        "plc_b2c3d4e5f6",
        name="площадь Революции, 2А",
        address="Россия, Нижний Новгород, площадь Революции, 2А",
        kind=ToponymKind.HOUSE,
        precision="exact",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=house, matches=[house, city], ambiguous=True),
        query="площадь Революции, 2А",
        city="Нижний Новгород",
    )

    assert selected == house


def test_selector_prefers_one_exact_name() -> None:
    """Verify an exact name wins over a looser geocoder candidate."""

    exact = _match(
        "plc_a1b2c3d4e5",
        name="ВДНХ",
        address="Россия, Москва, ВДНХ",
    )
    loose = _match(
        "plc_b2c3d4e5f6",
        name="станция метро ВДНХ",
        address="Россия, Москва, станция метро ВДНХ",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=exact, matches=[exact, loose], ambiguous=True),
        query="ВДНХ",
        city=None,
    )

    assert selected == exact


def test_selector_prefers_one_exact_precision_match() -> None:
    """Verify exact house precision disambiguates candidates with non-identical names."""

    exact = _match(
        "plc_a1b2c3d4e5",
        name="Тверская улица, 1",
        address="Россия, Москва, Тверская улица, 1",
        precision="exact",
    )
    approximate = _match(
        "plc_b2c3d4e5f6",
        name="Тверская улица, 3",
        address="Россия, Москва, Тверская улица, 3",
        precision="near",
    )

    selected = select_unique_place_match(
        GeocodePlaceOutput(best=exact, matches=[exact, approximate], ambiguous=True),
        query="Тверская 1",
        city="Москва",
    )

    assert selected == exact


def test_selector_rejects_multiple_equally_plausible_candidates() -> None:
    """Verify unresolved ambiguity becomes a safe input error instead of a guess."""

    first = _match(
        "plc_a1b2c3d4e5",
        name="Центральный парк",
        address="Россия, Москва, Центральный парк",
    )
    second = _match(
        "plc_b2c3d4e5f6",
        name="Центральный парк",
        address="Россия, Казань, Центральный парк",
    )

    with pytest.raises(AmbiguousPlaceError) as exc_info:
        select_unique_place_match(
            GeocodePlaceOutput(best=first, matches=[first, second], ambiguous=True),
            query="Центральный парк",
            city=None,
        )

    assert exc_info.value.error_code is ToolErrorCode.INVALID_INPUT
    assert str(exc_info.value) == (
        "Place is ambiguous: 'Центральный парк'. Specify a more precise name or full address."
    )


async def test_places_scope_loader_loads_existing_ref_from_store() -> None:
    """A prepared ref is loaded without introducing a geocoder dependency."""

    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        lat=55.7539,
        lon=37.6208,
        origin=RecordOrigin.GEOCODE,
    )
    await store.save(record)
    loader = PlacesSearchScopeLoader(place_store=store)

    resolved = await loader.load_anchor(record.ref)

    assert resolved == record


async def test_geocoded_place_resolver_geocodes_and_returns_persisted_record() -> None:
    """Verify text resolution selects a candidate and returns its hidden record."""

    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        lat=55.7539,
        lon=37.6208,
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_place(
        query="Красная площадь",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == record
    assert geocoder.calls == [GeocodePlaceInput(query="Красная площадь", city="Москва", limit=5)]


async def test_geocoded_place_resolver_resolves_bounded_locality() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Москва",
        address="Россия, Москва",
        lat=55.7558,
        lon=37.6176,
        kind=ToponymKind.LOCALITY.value,
        locality="Москва",
        bounds=GeoBounds(
            west=36.8,
            south=55.1,
            east=38.0,
            north=56.0,
        ),
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_bounded_locality(
        "Москва",
        ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == record
    assert geocoder.calls == [GeocodePlaceInput(query="Москва", limit=5, locality_only=True)]


async def test_russian_area_search_caches_localized_foreign_city_name() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Vienna",
        address="Vienna, Austria",
        lat=48.209206,
        lon=16.372778,
        kind=ToponymKind.LOCALITY.value,
        locality="Vienna",
        bounds=GeoBounds(west=16.18, south=48.11, east=16.58, north=48.33),
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = LocalizingStaticGeocoder(
        localized_city="Вена",
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="area",
            query="достопримечательности",
            category="attraction",
            city="Vienna",
            limit=15,
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.area_ref == record.ref
    assert geocoder.reverse_calls == [(record.lat, record.lon, "ru-RU")]
    stored = await store.get(record.ref)
    assert stored is not None
    assert stored.localized_localities == {"ru-RU": "Вена"}


async def test_russian_follow_up_localizes_existing_english_area_ref() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Vienna",
        address="Vienna, Austria",
        lat=48.209206,
        lon=16.372778,
        kind=ToponymKind.LOCALITY.value,
        locality="Vienna",
        bounds=GeoBounds(west=16.18, south=48.11, east=16.58, north=48.33),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save(record)
    geocoder = LocalizingStaticGeocoder(
        localized_city="Вена",
        output=GeocodePlaceOutput(),
        store=store,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="area",
            query="музеи",
            category="museum",
            area_ref=record.ref,
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.area_ref == record.ref
    assert geocoder.calls == []
    assert geocoder.reverse_calls == [(record.lat, record.lon, "ru-RU")]
    stored = await store.get(record.ref)
    assert stored is not None
    assert stored.localized_localities == {"ru-RU": "Вена"}


async def test_bounded_locality_prefers_primary_city_label_over_namesakes() -> None:
    store = InMemoryPlaceStore()
    city = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Казань",
        address="Россия, Республика Татарстан (Татарстан), Казань",
        lat=55.7961,
        lon=49.1064,
        kind=ToponymKind.LOCALITY.value,
        locality="Казань",
        bounds=GeoBounds(west=48.8, south=55.6, east=49.4, north=56.0),
        origin=RecordOrigin.GEOCODE,
    )
    village = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="деревня Казань",
        address="Россия, Кировская область, деревня Казань",
        lat=58.0,
        lon=48.0,
        kind=ToponymKind.LOCALITY.value,
        locality="деревня Казань",
        bounds=GeoBounds(west=47.9, south=57.9, east=48.1, north=58.1),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([city, village])
    city_match = _match(
        city.ref,
        name=city.name,
        address=city.address,
        kind=ToponymKind.LOCALITY,
    )
    village_match = _match(
        village.ref,
        name=village.name,
        address=village.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(
            best=city_match,
            matches=[city_match, village_match],
            ambiguous=True,
        ),
        store=store,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_bounded_locality(
        "Казань, Россия",
        ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == city


async def test_bounded_locality_keeps_top_ranked_local_name_over_literal_namesake() -> None:
    store = InMemoryPlaceStore()
    copenhagen = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="København",
        address="København, Region Hovedstaden, Danmark",
        lat=55.6761,
        lon=12.5683,
        kind=ToponymKind.LOCALITY.value,
        locality="København",
        bounds=GeoBounds(west=12.45, south=55.55, east=12.7, north=55.78),
        origin=RecordOrigin.GEOCODE,
    )
    namesake = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Copenhagen",
        address="Copenhagen, Lewis, NY, United States",
        lat=43.8934,
        lon=-75.6735,
        kind=ToponymKind.LOCALITY.value,
        locality="Copenhagen",
        bounds=GeoBounds(west=-75.8, south=43.8, east=-75.5, north=44.0),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([copenhagen, namesake])
    matches = [
        _match(
            copenhagen.ref,
            name=copenhagen.name,
            address=copenhagen.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.5,
        ),
        _match(
            namesake.ref,
            name=namesake.name,
            address=namesake.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.3,
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_bounded_locality(
        "Copenhagen",
        ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == copenhagen


async def test_bounded_locality_ignores_disallowed_country_before_ambiguity() -> None:
    """A US namesake must not make a supported European city ambiguous."""

    store = InMemoryPlaceStore()
    edinburgh = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Эдинбург",
        address="Эдинбург, Midlothian, SCT, Соединённое Королевство",
        lat=55.9533,
        lon=-3.1883,
        kind=ToponymKind.LOCALITY.value,
        locality="Эдинбург",
        bounds=GeoBounds(west=-3.5, south=55.8, east=-3.0, north=56.1),
        origin=RecordOrigin.GEOCODE,
    )
    texas = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Эдинбург",
        address="Эдинбург, Hidalgo, TX, США",
        lat=26.3017,
        lon=-98.1633,
        kind=ToponymKind.LOCALITY.value,
        locality="Эдинбург",
        bounds=GeoBounds(west=-98.3, south=26.2, east=-98.0, north=26.4),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([edinburgh, texas])
    matches = [
        _match(
            edinburgh.ref,
            name=edinburgh.name,
            address=edinburgh.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.55,
            municipality="Эдинбург",
            country_subdivision="SCT",
            country_code="GB",
        ),
        _match(
            texas.ref,
            name=texas.name,
            address=texas.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.46,
            municipality="Эдинбург",
            country_subdivision="TX",
            country_code="US",
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=store,
        allowed_country_codes={"GB", "RU"},
    )

    resolved = await resolver.geocode_bounded_locality("Эдинбург", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record == edinburgh


async def test_bounded_locality_prefers_supported_exact_city_over_prefix_matches() -> None:
    """Filtering must expose Porto rather than the higher-ranked Porto Alegre."""

    store = InMemoryPlaceStore()
    porto_alegre = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Порту-Алегри",
        address="Порту-Алегри, Риу-Гранди-ду-Сул, Бразилия",
        lat=-30.0346,
        lon=-51.2177,
        kind=ToponymKind.LOCALITY.value,
        locality="Порту-Алегри",
        bounds=GeoBounds(west=-51.4, south=-30.2, east=-51.0, north=-29.8),
        origin=RecordOrigin.GEOCODE,
    )
    porto = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Порту",
        address="Порту, Porto, Portugal Continental, Португалия",
        lat=41.1579,
        lon=-8.6291,
        kind=ToponymKind.LOCALITY.value,
        locality="Порту",
        bounds=GeoBounds(west=-8.8, south=41.0, east=-8.4, north=41.3),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([porto_alegre, porto])
    matches = [
        _match(
            porto_alegre.ref,
            name=porto_alegre.name,
            address=porto_alegre.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.65,
            municipality="Порту-Алегри",
            country_subdivision="Риу-Гранди-ду-Сул",
            country_code="BR",
        ),
        _match(
            porto.ref,
            name=porto.name,
            address=porto.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.45,
            municipality="Порту",
            country_subdivision="Portugal Continental",
            country_code="PT",
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=store,
        allowed_country_codes={"PT"},
    )

    resolved = await resolver.geocode_bounded_locality("Порту", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record == porto


async def test_bounded_locality_selects_top_ranked_local_name() -> None:
    store = InMemoryPlaceStore()
    rome = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Roma",
        address="Roma, Lazio, Italia",
        lat=41.8933,
        lon=12.4829,
        kind=ToponymKind.LOCALITY.value,
        locality="Roma",
        bounds=GeoBounds(west=12.2, south=41.7, east=12.8, north=42.1),
        origin=RecordOrigin.GEOCODE,
    )
    namesake = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Rome",
        address="Rome, Floyd, GA, United States",
        lat=34.257,
        lon=-85.1647,
        kind=ToponymKind.LOCALITY.value,
        locality="Rome",
        bounds=GeoBounds(west=-85.3, south=34.1, east=-85.0, north=34.4),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([rome, namesake])
    matches = [
        _match(
            rome.ref,
            name=rome.name,
            address=rome.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.5,
        ),
        _match(
            namesake.ref,
            name=namesake.name,
            address=namesake.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.4,
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_bounded_locality("Rome", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record == rome


async def test_bounded_locality_deduplicates_administrative_variants() -> None:
    store = InMemoryPlaceStore()
    city_centre = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Venezia Centro",
        address="Venezia, Veneto, Italia",
        lat=45.4372,
        lon=12.3346,
        kind=ToponymKind.LOCALITY.value,
        locality="Venezia",
        bounds=GeoBounds(west=12.30, south=45.41, east=12.38, north=45.46),
        origin=RecordOrigin.GEOCODE,
    )
    municipality = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Venezia",
        address="Venezia, Veneto, Italia",
        lat=45.4408,
        lon=12.3155,
        kind=ToponymKind.LOCALITY.value,
        locality="Venezia",
        bounds=GeoBounds(west=12.17, south=45.23, east=12.60, north=45.58),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([city_centre, municipality])
    matches = [
        _match(
            city_centre.ref,
            name=city_centre.name,
            address=city_centre.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.5,
            municipality="Venezia Centro",
            country_secondary_subdivision="Venezia",
            country_subdivision="Veneto",
            country_code="IT",
        ),
        _match(
            municipality.ref,
            name=municipality.name,
            address=municipality.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.49,
            municipality="Venezia",
            country_subdivision="Veneto",
            country_code="IT",
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    resolved = await resolver.geocode_bounded_locality(
        "Venice",
        ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == municipality


async def test_geocoded_place_resolver_resolves_unique_address_or_toponym() -> None:
    """The shared resolver contains only the geocoder-backed one-place path."""

    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        lat=55.7539,
        lon=37.6208,
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(record.ref, name=record.name, address=record.address)
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=store,
    )

    resolved = await resolver.geocode_place(
        query="Красная площадь",
        city="Москва",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == record


async def test_places_search_scope_replaces_text_anchor_with_resolved_ref() -> None:
    """Provider fallback receives one reusable anchor instead of the original text."""

    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Парк Горького",
        address="Россия, Москва, улица Крымский Вал, 9",
        lat=55.7296,
        lon=37.6017,
        locality="Москва",
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(record.ref, name=record.name, address=record.address)
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))
    args = PlacesSearchInput(
        mode="near",
        query="рестораны",
        category="restaurant",
        near_query="Парк Горького",
        city="Москва",
    )

    resolved_args = await resolver.resolve_scope(args, ToolExecutionContext())

    assert resolved_args.near == record.ref
    assert resolved_args.near_query is None
    assert resolved_args.city == "Москва"
    assert geocoder.calls == [
        GeocodePlaceInput(query="Парк Горького", city="Москва", limit=5),
        GeocodePlaceInput(query="Москва", limit=5, locality_only=True),
    ]


async def test_places_search_scope_resolves_duplicated_city_as_locality_anchor() -> None:
    """A city repeated as near_query and city must never enter named-POI fallback."""

    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_f1a2b3c4d5",
        name="Франкфурт-на-Майне",
        address="Германия, Франкфурт-на-Майне",
        lat=50.1109,
        lon=8.6821,
        kind=ToponymKind.LOCALITY.value,
        locality="Франкфурт-на-Майне",
        bounds=GeoBounds(west=8.4727, south=50.0155, east=8.8005, north=50.2272),
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    named_poi_resolver = StaticNamedPoiResolver(None)
    resolver = _scope_resolver(
        GeocodedPlaceResolver(geocoder=geocoder, place_store=store),
        named_poi_resolvers=(named_poi_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="restaurants",
            category="restaurant",
            near_query="Frankfurt am Main",
            city="Frankfurt am Main",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == record.ref
    assert resolved_args.near_query is None
    assert resolved_args.city is None
    assert resolved_args.area_ref == record.ref
    assert geocoder.calls == [
        GeocodePlaceInput(query="Frankfurt am Main", limit=5, locality_only=True)
    ]
    assert named_poi_resolver.calls == []


async def test_places_search_scope_reuses_area_ref_for_a_new_text_anchor() -> None:
    """A follow-up near_query loads its city from the prior reusable area ref."""

    store = InMemoryPlaceStore()
    area_record = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Москва",
        address="Москва, Россия",
        lat=55.7565,
        lon=37.6149,
        kind=ToponymKind.LOCALITY.value,
        locality="Москва",
        bounds=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.1),
        origin=RecordOrigin.GEOCODE,
    )
    anchor_record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="метро Римская",
        address="Москва, площадь Рогожская Застава",
        lat=55.7471,
        lon=37.6816,
        locality="Москва",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    await store.save(area_record)
    geocoder = StaticGeocoder(output=GeocodePlaceOutput(), store=store)
    named_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=anchor_record.ref, record=anchor_record)
    )
    resolver = _scope_resolver(
        GeocodedPlaceResolver(geocoder=geocoder, place_store=store),
        named_poi_resolvers=(named_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="кафе",
            category="cafe",
            near_query="метро Римская",
            area_ref=area_record.ref,
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == anchor_record.ref
    assert resolved_args.near_query is None
    assert resolved_args.city is None
    assert resolved_args.area_ref == area_record.ref
    assert named_resolver.calls == [("метро Римская", "Москва")]
    assert geocoder.calls == []


async def test_places_search_scope_accepts_country_suffix_on_locality_anchor() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_f1a2b3c4d5",
        name="Frankfurt am Main",
        address="Frankfurt am Main, Germany",
        lat=50.1109,
        lon=8.6821,
        kind=ToponymKind.LOCALITY.value,
        locality="Frankfurt am Main",
        bounds=GeoBounds(west=8.4727, south=50.0155, east=8.8005, north=50.2272),
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="restaurants",
            category="restaurant",
            near_query="Frankfurt am Main, Germany",
            city="Frankfurt am Main",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == record.ref
    assert geocoder.calls == [
        GeocodePlaceInput(query="Frankfurt am Main, Germany", limit=5, locality_only=True)
    ]


async def test_places_search_scope_uses_first_ranked_bounded_locality() -> None:
    store = InMemoryPlaceStore()
    bounds = GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0)
    first_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Москва",
        address="Россия, Москва",
        lat=55.7558,
        lon=37.6176,
        kind=ToponymKind.LOCALITY.value,
        locality="Москва",
        bounds=bounds,
        origin=RecordOrigin.GEOCODE,
    )
    second_record = first_record.model_copy(
        update={
            "ref": "plc_b2c3d4e5f6",
            "address": "США, штат Айдахо, Москва",
            "lat": 46.7324,
            "lon": -117.0002,
        }
    )
    await store.save_many([first_record, second_record])
    first_match = _match(
        first_record.ref,
        name=first_record.name,
        address=first_record.address,
        kind=ToponymKind.LOCALITY,
        relevance_score=2.58184,
    )
    second_match = _match(
        second_record.ref,
        name=second_record.name,
        address=second_record.address,
        kind=ToponymKind.LOCALITY,
        relevance_score=2.39858,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(
            best=first_match,
            matches=[first_match, second_match],
            ambiguous=True,
        ),
        store=store,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))

    resolved = await resolver.resolve_scope(
        PlacesSearchInput(mode="area", query="кафе", city="Москва"),
        ToolExecutionContext(),
    )

    assert resolved.area_ref == first_record.ref
    assert resolved.city is None


async def test_bounded_locality_selects_unique_best_textual_match_despite_near_scores() -> None:
    """English Minsk should not clarify against a lower-confidence Polish namesake."""

    store = InMemoryPlaceStore()
    belarus = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Минск",
        address="Минск, Беларусь",
        lat=53.9,
        lon=27.5667,
        kind=ToponymKind.LOCALITY.value,
        locality="Минск",
        bounds=GeoBounds(west=27.3, south=53.7, east=27.8, north=54.1),
        origin=RecordOrigin.GEOCODE,
    )
    poland = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Gmina Mińsk Mazowiecki",
        address="Gmina Mińsk Mazowiecki, Miński, Mazowieckie, Polska",
        lat=52.2,
        lon=21.6,
        kind=ToponymKind.LOCALITY.value,
        locality="Gmina Mińsk Mazowiecki",
        bounds=GeoBounds(west=21.4, south=52.0, east=21.8, north=52.4),
        origin=RecordOrigin.GEOCODE,
    )
    await store.save_many([belarus, poland])
    matches = [
        _match(
            belarus.ref,
            name=belarus.name,
            address=belarus.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.4498,
            match_confidence_score=1.0,
            municipality="Минск",
            country_subdivision="Минск",
            country_code="BY",
        ),
        _match(
            poland.ref,
            name=poland.name,
            address=poland.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=2.3515,
            match_confidence_score=0.8619,
            municipality="Gmina Mińsk Mazowiecki",
            country_subdivision="Mazowieckie",
            country_code="PL",
        ),
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=store,
        allowed_country_codes={"BY", "PL"},
    )

    resolved = await resolver.geocode_bounded_locality("Minsk", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record == belarus


@pytest.mark.parametrize(
    "args",
    [
        PlacesSearchInput(mode="area", query="restaurants", city="Alexandria"),
        PlacesSearchInput(
            mode="near",
            query="restaurants",
            near_query="Alexandria",
            city="Alexandria",
        ),
    ],
)
async def test_places_search_scope_selects_first_processed_locality(
    args: PlacesSearchInput,
) -> None:
    store = InMemoryPlaceStore()
    records = [
        PlaceRecord(
            ref="plc_a1b2c3d4e5",
            name="Alexandria",
            address="Alexandria, LA, United States",
            lat=31.3113,
            lon=-92.4451,
            kind=ToponymKind.LOCALITY.value,
            locality="Alexandria",
            bounds=GeoBounds(west=-92.6, south=31.2, east=-92.3, north=31.4),
            origin=RecordOrigin.GEOCODE,
        ),
        PlaceRecord(
            ref="plc_b2c3d4e5f6",
            name="Alexandria",
            address="Alexandria, VA, United States",
            lat=38.8048,
            lon=-77.0469,
            kind=ToponymKind.LOCALITY.value,
            locality="Alexandria",
            bounds=GeoBounds(west=-77.2, south=38.7, east=-76.9, north=38.9),
            origin=RecordOrigin.GEOCODE,
        ),
        PlaceRecord(
            ref="plc_c3d4e5f6a7",
            name="Alexandria",
            address="Alexandria, Romania",
            lat=43.9697,
            lon=25.3333,
            kind=ToponymKind.LOCALITY.value,
            locality="Alexandria",
            bounds=GeoBounds(west=25.2, south=43.8, east=25.5, north=44.1),
            origin=RecordOrigin.GEOCODE,
        ),
    ]
    await store.save_many(records)
    scores = [2.4241199, 2.327, 2.25]
    administrative_contexts = [
        ("Alexandria", "LA", "US"),
        ("Alexandria", "VA", "US"),
        ("Alexandria", "Teleorman", "RO"),
    ]
    matches = [
        _match(
            record.ref,
            name=record.name,
            address=record.address,
            kind=ToponymKind.LOCALITY,
            relevance_score=score,
            municipality=municipality,
            country_subdivision=country_subdivision,
            country_code=country_code,
        )
        for record, score, (municipality, country_subdivision, country_code) in zip(
            records,
            scores,
            administrative_contexts,
            strict=True,
        )
    ]
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=matches[0], matches=matches, ambiguous=True),
        store=store,
    )
    resolver = _scope_resolver(GeocodedPlaceResolver(geocoder=geocoder, place_store=store))

    resolved_args = await resolver.resolve_scope(args, ToolExecutionContext())

    assert resolved_args.city is None
    assert resolved_args.area_ref == records[0].ref
    if args.mode.value == "near":
        assert resolved_args.near == records[0].ref
        assert resolved_args.near_query is None


async def test_routing_place_resolver_replaces_text_waypoints_with_refs_once() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Парк Горького",
        address="Россия, Москва, улица Крымский Вал, 9",
        lat=55.7296,
        lon=37.6017,
        locality="Москва",
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(record.ref, name=record.name, address=record.address)
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=GeocodedPlaceResolver(
                geocoder=geocoder,
                place_store=store,
            )
        ),
        store,
    )
    text_point = {"query": "Парк Горького", "area": "Москва"}
    args = RoutingInput(
        mode="route",
        waypoints=[text_point, text_point, "plc_b2c3d4e5f6"],
    )

    resolved_args = await resolver.resolve_to_refs(args, ToolExecutionContext())

    assert resolved_args.waypoints == [record.ref, record.ref, "plc_b2c3d4e5f6"]
    assert geocoder.calls == [GeocodePlaceInput(query="Парк Горького", city="Москва", limit=5)]


async def test_routing_area_ref_reuses_locality_scope_and_bounds() -> None:
    class CountingPlaceStore(InMemoryPlaceStore):
        def __init__(self) -> None:
            super().__init__()
            self.get_calls: list[str] = []

        async def get(self, ref: str) -> PlaceRecord | None:
            self.get_calls.append(ref)
            return await super().get(ref)

    store = CountingPlaceStore()
    area_record = PlaceRecord(
        ref="plc_f1e2d3c4b5",
        name="Городец",
        address="Россия, Нижегородская область, Городец",
        lat=56.6441,
        lon=43.4722,
        kind=ToponymKind.LOCALITY,
        bounds=GeoBounds(west=43.3, south=56.5, east=43.6, north=56.8),
        origin=RecordOrigin.GEOCODE,
    )
    target_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Музей",
        address="Городец, набережная Революции, 11",
        lat=56.6426,
        lon=43.4652,
        locality="Городец",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    await store.save(area_record)
    geocoder = StaticGeocoder(output=GeocodePlaceOutput(), store=store)
    named = StaticNamedPoiResolver(
        ResolvedPlace(ref=target_record.ref, record=target_record),
        requires_city_bounds=True,
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=GeocodedPlaceResolver(
                geocoder=geocoder,
                place_store=store,
            ),
            named_poi_resolvers=(named,),
            named_first=True,
        ),
        store,
    )

    resolved = await resolver.resolve_to_refs(
        RoutingInput(
            mode="route",
            waypoints=[
                {"query": "Музей Самоваров", "area": area_record.ref},
                {"query": "Кремль", "area": area_record.ref},
            ],
        ),
        ToolExecutionContext(),
    )

    assert resolved.waypoints == [target_record.ref, target_record.ref]
    assert store.get_calls == [area_record.ref]
    assert geocoder.calls == []
    assert named.received_city_bounds == [area_record.bounds, area_record.bounds]
    assert all(area is not None and area.ref == area_record.ref for area in named.received_areas)


@pytest.mark.parametrize(
    ("record", "error_code"),
    [
        (None, ToolErrorCode.UNKNOWN_REF),
        (
            PlaceRecord(
                ref="plc_f1e2d3c4b5",
                name="Парк",
                address="Городец, парк",
                lat=56.64,
                lon=43.47,
                origin=RecordOrigin.PLACES_SEARCH,
            ),
            ToolErrorCode.INVALID_INPUT,
        ),
    ],
)
async def test_routing_area_ref_must_be_a_known_bounded_locality(
    record: PlaceRecord | None,
    error_code: ToolErrorCode,
) -> None:
    store = InMemoryPlaceStore()
    if record is not None:
        await store.save(record)
    geocoder = StaticGeocoder(output=GeocodePlaceOutput(), store=store)
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=GeocodedPlaceResolver(
                geocoder=geocoder,
                place_store=store,
            )
        ),
        store,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await resolver.resolve_to_refs(
            RoutingInput(
                mode="route",
                waypoints=[
                    {"query": "Музей", "area": "plc_f1e2d3c4b5"},
                    "plc_a1b2c3d4e5",
                ],
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is error_code
    assert geocoder.calls == []


async def test_routing_place_resolver_preserves_rank_groups_when_replacing_text() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Кремль",
        address="Россия, Москва, Кремль",
        lat=55.7520,
        lon=37.6175,
        locality="Москва",
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(record.ref, name=record.name, address=record.address)
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=GeocodedPlaceResolver(
                geocoder=geocoder,
                place_store=store,
            )
        ),
        store,
    )
    text_point = {"query": "Кремль", "area": "Москва"}
    args = RoutingInput(
        mode="rank",
        origins=[text_point],
        candidates=[text_point, "plc_b2c3d4e5f6"],
    )

    resolved_args = await resolver.resolve_to_refs(args, ToolExecutionContext())

    assert resolved_args.origins == [record.ref]
    assert resolved_args.candidates == [record.ref, "plc_b2c3d4e5f6"]
    assert geocoder.calls == [GeocodePlaceInput(query="Кремль", city="Москва", limit=5)]


async def test_routing_place_resolver_uses_named_fallback_after_geocoder_ambiguity() -> None:
    store = InMemoryPlaceStore()
    first = _match(
        "plc_a1b2c3d4e5",
        name="ВДНХ",
        address="Россия, Москва, ВДНХ",
        kind=ToponymKind.METRO,
    )
    second = _match(
        "plc_b2c3d4e5f6",
        name="ВДНХ",
        address="Россия, Москва, проспект Мира, 119",
    )
    fallback_record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="ВДНХ",
        address="Москва, проспект Мира, 119",
        lat=55.8263,
        lon=37.6377,
        locality="Москва",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    geocoded_place_resolver = GeocodedPlaceResolver(
        geocoder=StaticGeocoder(
            output=GeocodePlaceOutput(
                best=first,
                matches=[first, second],
                ambiguous=True,
            ),
            store=store,
        ),
        place_store=store,
    )
    named_poi_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=fallback_record.ref, record=fallback_record)
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=(named_poi_resolver,),
        ),
        store,
    )
    args = RoutingInput(
        mode="route",
        waypoints=[
            {"query": "ВДНХ", "area": "Москва"},
            "plc_d4e5f6a7b8",
        ],
    )

    resolved_args = await resolver.resolve_to_refs(args, ToolExecutionContext())

    assert resolved_args.waypoints == [fallback_record.ref, "plc_d4e5f6a7b8"]
    assert named_poi_resolver.calls == [("ВДНХ", "Москва")]


async def test_routing_place_resolver_uses_named_fallback_after_empty_geocoder() -> None:
    store = InMemoryPlaceStore()
    fallback_record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Ritmo X Giardino",
        address="Москва, улица Большая Дмитровка, 9",
        lat=55.7590,
        lon=37.6144,
        locality="Москва",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    geocoded_place_resolver = GeocodedPlaceResolver(
        geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
        place_store=store,
    )
    named_poi_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=fallback_record.ref, record=fallback_record)
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=(named_poi_resolver,),
        ),
        store,
    )

    resolved_args = await resolver.resolve_to_refs(
        RoutingInput(
            mode="route",
            waypoints=[
                {"query": "Ritmo X Giardino", "area": "Москва"},
                "plc_d4e5f6a7b8",
            ],
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.waypoints == [fallback_record.ref, "plc_d4e5f6a7b8"]
    assert named_poi_resolver.calls == [("Ritmo X Giardino", "Москва")]
    assert named_poi_resolver.first_address_calls == [("Ritmo X Giardino", "Москва")]


async def test_routing_place_resolver_uses_first_named_poi_when_ambiguous() -> None:
    store = InMemoryPlaceStore()
    first_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="ВДНХ метро",
        address="Москва, станция метро ВДНХ",
        lat=55.8212,
        lon=37.6417,
        origin=RecordOrigin.PLACES_SEARCH,
    )
    second_record = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="ВДНХ комплекс",
        address="Москва, проспект Мира, 119",
        lat=55.8298,
        lon=37.6328,
        origin=RecordOrigin.PLACES_SEARCH,
    )
    await store.save_many([first_record, second_record])
    clarification = ToolClarification(
        kind="select_anchor",
        question="Which VDNH do you mean?",
        options=[
            ToolClarificationOption(value="plc_a1b2c3d4e5", label="ВДНХ метро"),
            ToolClarificationOption(value="plc_b2c3d4e5f6", label="ВДНХ комплекс"),
        ],
    )
    geocoded_place_resolver = GeocodedPlaceResolver(
        geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
        place_store=store,
    )
    named_poi_resolver = StaticNamedPoiResolver(
        None,
        error=AmbiguousPlaceError("ВДНХ", clarification=clarification),
    )
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=(named_poi_resolver,),
        ),
        store,
    )

    context = ToolExecutionContext()
    resolved_args = await resolver.resolve_to_refs(
        RoutingInput(
            mode="route",
            waypoints=[
                {"query": "ВДНХ", "area": "Москва"},
                "plc_d4e5f6a7b8",
            ],
        ),
        context,
    )

    assert resolved_args.waypoints == [first_record.ref, "plc_d4e5f6a7b8"]
    assert context.warnings == (
        "Routing point 'ВДНХ' matched multiple places; using the provider-ranked first result "
        "'ВДНХ метро'.",
    )


async def test_routing_place_loader_rejects_unresolved_text() -> None:
    store = InMemoryPlaceStore()
    loader = RoutingPlaceLoader(store)
    args = RoutingInput(
        mode="route",
        waypoints=[
            {"query": "Кремль", "area": "Москва"},
            "plc_b2c3d4e5f6",
        ],
    )

    with pytest.raises(ValueError, match="only prepared place refs"):
        await loader.load_places(args.waypoints)


async def test_places_scope_uses_named_fallback_after_geocoder_ambiguity() -> None:
    store = InMemoryPlaceStore()
    first = _match(
        "plc_a1b2c3d4e5",
        name="Парк Горького",
        address="Россия, Москва, Парк Горького",
    )
    second = _match(
        "plc_b2c3d4e5f6",
        name="Парк Горького",
        address="Россия, Москва, улица Крымский Вал",
    )
    fallback_record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Парк Горького",
        address="улица Крымский Вал, 9, Москва",
        lat=55.7296,
        lon=37.6017,
        locality="Москва",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    named_poi_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=fallback_record.ref, record=fallback_record)
    )
    resolver = _scope_resolver(
        GeocodedPlaceResolver(
            geocoder=StaticGeocoder(
                output=GeocodePlaceOutput(
                    best=first,
                    matches=[first, second],
                    ambiguous=True,
                ),
                store=store,
            ),
            place_store=store,
        ),
        named_poi_resolvers=(named_poi_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="рестораны",
            near_query="Парк Горького",
            city="Москва",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == fallback_record.ref
    assert resolved_args.near_query is None
    assert named_poi_resolver.calls == [("Парк Горького", "Москва")]


async def test_places_scope_prefers_named_anchor_over_successful_geocoder() -> None:
    store = InMemoryPlaceStore()
    geocoder_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="ЖК Швейцария Парк",
        address="Нижний Новгород, ЖК Швейцария Парк",
        lat=56.262882,
        lon=43.975327,
        locality="Нижний Новгород",
        origin=RecordOrigin.GEOCODE,
    )
    geocoder_match = _match(
        geocoder_record.ref,
        name=geocoder_record.name,
        address=geocoder_record.address,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=geocoder_match, matches=[geocoder_match]),
        store=store,
        record=geocoder_record,
    )
    park_record = PlaceRecord(
        ref="plc_b2c3d4e5f6",
        name="Парк Швейцария",
        address="Нижний Новгород, проспект Гагарина, 35",
        lat=56.2683,
        lon=43.9738,
        locality="Нижний Новгород",
        provider="twogis",
        provider_id="park",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    named_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=park_record.ref, record=park_record),
        provider="twogis",
    )
    resolver = _scope_resolver(
        GeocodedPlaceResolver(geocoder=geocoder, place_store=store),
        named_poi_resolvers=(named_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="рестораны",
            near_query="Парк Швейцария",
            city="Нижний Новгород",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == park_record.ref
    assert named_resolver.calls == [("Парк Швейцария", "Нижний Новгород")]
    assert geocoder.calls == [
        GeocodePlaceInput(query="Нижний Новгород", limit=5, locality_only=True)
    ]


async def test_places_scope_falls_back_to_geocoder_after_empty_named_chain() -> None:
    store = InMemoryPlaceStore()
    address_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="проспект Гагарина, 35",
        address="Нижний Новгород, проспект Гагарина, 35",
        lat=56.2893,
        lon=43.9802,
        locality="Нижний Новгород",
        origin=RecordOrigin.GEOCODE,
    )
    address_match = _match(
        address_record.ref,
        name=address_record.name,
        address=address_record.address,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=address_match, matches=[address_match]),
        store=store,
        record=address_record,
    )
    named_resolver = StaticNamedPoiResolver(None, provider="twogis")
    resolver = _scope_resolver(
        GeocodedPlaceResolver(geocoder=geocoder, place_store=store),
        named_poi_resolvers=(named_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="рестораны",
            near_query="проспект Гагарина, 35",
            city="Нижний Новгород",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == address_record.ref
    assert named_resolver.calls == [("проспект Гагарина, 35", "Нижний Новгород")]
    assert geocoder.calls == [
        GeocodePlaceInput(
            query="проспект Гагарина, 35",
            city="Нижний Новгород",
            limit=5,
        ),
        GeocodePlaceInput(query="Нижний Новгород", limit=5, locality_only=True),
    ]


async def test_places_scope_uses_named_fallback_when_geocoder_found_nothing() -> None:
    store = InMemoryPlaceStore()
    fallback_record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Московский вокзал",
        address="площадь Революции, 2, Нижний Новгород",
        lat=56.3216,
        lon=43.9467,
        locality="Нижний Новгород",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    named_poi_resolver = StaticNamedPoiResolver(
        ResolvedPlace(ref=fallback_record.ref, record=fallback_record)
    )
    resolver = _scope_resolver(
        GeocodedPlaceResolver(
            geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
            place_store=store,
        ),
        named_poi_resolvers=(named_poi_resolver,),
    )

    resolved_args = await resolver.resolve_scope(
        PlacesSearchInput(
            mode="near",
            query="рестораны",
            near_query="Московский вокзал",
            city="Нижний Новгород",
        ),
        ToolExecutionContext(),
    )

    assert resolved_args.near == fallback_record.ref
    assert named_poi_resolver.calls == [("Московский вокзал", "Нижний Новгород")]


async def test_text_place_resolver_tries_next_named_provider_after_empty_result() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Лувр",
        address="Париж, Rue de Rivoli",
        lat=48.8606,
        lon=2.3376,
        locality="Париж",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    first = StaticNamedPoiResolver(None, provider="twogis")
    second = StaticNamedPoiResolver(
        ResolvedPlace(ref=record.ref, record=record),
        provider="tomtom",
    )
    resolver = TextPlaceResolver(
        geocoded_place_resolver=GeocodedPlaceResolver(
            geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
            place_store=store,
        ),
        named_poi_resolvers=(first, second),
    )

    resolved = await resolver.resolve(
        query="Лувр",
        city="Париж",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == record
    assert first.calls == [("Лувр", "Париж")]
    assert second.calls == [("Лувр", "Париж")]


async def test_text_place_resolver_resolves_bounds_only_for_provider_that_needs_them() -> None:
    store = InMemoryPlaceStore()
    city_bounds = GeoBounds(west=2.22, south=48.81, east=2.47, north=48.91)
    city_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Париж",
        address="Франция, Париж",
        lat=48.8566,
        lon=2.3522,
        kind=ToponymKind.LOCALITY.value,
        locality="Париж",
        bounds=city_bounds,
        origin=RecordOrigin.GEOCODE,
    )
    city_match = _match(
        city_record.ref,
        name=city_record.name,
        address=city_record.address,
        kind=ToponymKind.LOCALITY,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=city_match, matches=[city_match]),
        store=store,
        record=city_record,
    )
    record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Лувр",
        address="Rue de Rivoli, Paris",
        lat=48.8606,
        lon=2.3376,
        locality="Paris",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    first = StaticNamedPoiResolver(None, provider="twogis")
    second = StaticNamedPoiResolver(
        ResolvedPlace(ref=record.ref, record=record),
        provider="tomtom",
        requires_city_bounds=True,
    )
    resolver = TextPlaceResolver(
        geocoded_place_resolver=GeocodedPlaceResolver(
            geocoder=geocoder,
            place_store=store,
        ),
        named_poi_resolvers=(first, second),
    )

    resolved = await resolver.resolve(
        query="Лувр",
        city="Париж",
        context=ToolExecutionContext(),
    )

    assert resolved is not None
    assert resolved.record == record
    assert [call.query for call in geocoder.calls] == ["Лувр", "Париж"]
    assert first.received_city_bounds == [None]
    assert second.received_city_bounds == [city_bounds]


async def test_text_place_resolver_falls_back_after_retryable_named_provider_error() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_c3d4e5f6a7",
        name="Лувр",
        address="Париж, Rue de Rivoli",
        lat=48.8606,
        lon=2.3376,
        locality="Париж",
        origin=RecordOrigin.PLACES_SEARCH,
    )
    first = StaticNamedPoiResolver(
        None,
        provider="twogis",
        error=ToolExecutionError(
            ToolErrorCode.TIMEOUT,
            "2GIS timed out",
            provider="twogis_search",
            failure_kind=ToolFailureKind.TIMEOUT,
            retryable=True,
        ),
    )
    second = StaticNamedPoiResolver(
        ResolvedPlace(ref=record.ref, record=record),
        provider="tomtom",
    )
    resolver = TextPlaceResolver(
        geocoded_place_resolver=GeocodedPlaceResolver(
            geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
            place_store=store,
        ),
        named_poi_resolvers=(first, second),
    )
    context = ToolExecutionContext()

    resolved = await resolver.resolve(query="Лувр", city="Париж", context=context)

    assert resolved is not None
    assert resolved.record == record
    assert context.warnings == ("twogis named-place lookup failed; retrying with tomtom.",)


async def test_named_first_resolver_falls_back_to_geocoder_after_provider_error() -> None:
    store = InMemoryPlaceStore()
    geocoder_record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="проспект Гагарина, 35",
        address="Нижний Новгород, проспект Гагарина, 35",
        lat=56.2893,
        lon=43.9802,
        locality="Нижний Новгород",
        origin=RecordOrigin.GEOCODE,
    )
    geocoder_match = _match(
        geocoder_record.ref,
        name=geocoder_record.name,
        address=geocoder_record.address,
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=geocoder_match, matches=[geocoder_match]),
        store=store,
        record=geocoder_record,
    )
    named_resolver = StaticNamedPoiResolver(
        None,
        provider="twogis",
        error=ToolExecutionError(
            ToolErrorCode.TIMEOUT,
            "2GIS timed out",
            provider="twogis_search",
            failure_kind=ToolFailureKind.TIMEOUT,
            retryable=True,
        ),
    )
    resolver = TextPlaceResolver(
        geocoded_place_resolver=GeocodedPlaceResolver(
            geocoder=geocoder,
            place_store=store,
        ),
        named_poi_resolvers=(named_resolver,),
        named_first=True,
    )
    context = ToolExecutionContext()

    resolved = await resolver.resolve(
        query="проспект Гагарина, 35",
        city="Нижний Новгород",
        context=context,
    )

    assert resolved is not None
    assert resolved.record == geocoder_record
    assert context.warnings == ("twogis named-place lookup failed; retrying with geocoder.",)


async def test_places_scope_preserves_ambiguity_when_named_fallback_cannot_choose() -> None:
    store = InMemoryPlaceStore()
    first = _match(
        "plc_a1b2c3d4e5",
        name="Центральный парк",
        address="Россия, Москва, Центральный парк",
    )
    second = _match(
        "plc_b2c3d4e5f6",
        name="Центральный парк",
        address="Россия, Москва, другой Центральный парк",
    )
    named_poi_resolver = StaticNamedPoiResolver(None)
    resolver = _scope_resolver(
        GeocodedPlaceResolver(
            geocoder=StaticGeocoder(
                output=GeocodePlaceOutput(
                    best=first,
                    matches=[first, second],
                    ambiguous=True,
                ),
                store=store,
            ),
            place_store=store,
        ),
        named_poi_resolvers=(named_poi_resolver,),
    )

    with pytest.raises(AmbiguousSearchAnchorError, match="Specify a more precise name"):
        await resolver.resolve_scope(
            PlacesSearchInput(
                mode="near",
                query="рестораны",
                near_query="Центральный парк",
                city="Москва",
            ),
            ToolExecutionContext(),
        )

    assert named_poi_resolver.calls == [("Центральный парк", "Москва")]


async def test_places_scope_preserves_named_anchor_clarification() -> None:
    store = InMemoryPlaceStore()
    clarification = ToolClarification(
        kind="select_anchor",
        question="Which VDNH do you mean?",
        options=[
            ToolClarificationOption(value="plc_a1b2c3d4e5", label="ВДНХ метро"),
            ToolClarificationOption(value="plc_b2c3d4e5f6", label="ВДНХ комплекс"),
        ],
    )
    named_poi_resolver = StaticNamedPoiResolver(
        None,
        error=AmbiguousPlaceError("ВДНХ", clarification=clarification),
    )
    resolver = _scope_resolver(
        GeocodedPlaceResolver(
            geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
            place_store=store,
        ),
        named_poi_resolvers=(named_poi_resolver,),
    )

    with pytest.raises(AmbiguousSearchAnchorError) as exc_info:
        await resolver.resolve_scope(
            PlacesSearchInput(
                mode="near",
                query="рестораны",
                near_query="ВДНХ",
                city="Москва",
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.clarification == clarification


async def test_places_scope_reports_missing_anchor_without_named_result() -> None:
    store = InMemoryPlaceStore()
    named_poi_resolver = StaticNamedPoiResolver(None)
    resolver = _scope_resolver(
        GeocodedPlaceResolver(
            geocoder=StaticGeocoder(output=GeocodePlaceOutput(), store=store),
            place_store=store,
        ),
        named_poi_resolvers=(named_poi_resolver,),
    )

    with pytest.raises(AnchorNotFoundError, match="Could not resolve"):
        await resolver.resolve_scope(
            PlacesSearchInput(
                mode="near",
                query="рестораны",
                near_query="Неизвестное место",
                city="Москва",
            ),
            ToolExecutionContext(),
        )


async def test_geocoded_place_resolver_warns_about_provider_added_house_suffix() -> None:
    store = InMemoryPlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Большая Покровская улица, 2А",
        address="Россия, Нижний Новгород, Большая Покровская улица, 2А",
        lat=56.326,
        lon=44.005,
        kind=ToponymKind.HOUSE.value,
        precision="number",
        origin=RecordOrigin.GEOCODE,
    )
    match = _match(
        record.ref,
        name=record.name,
        address=record.address,
        kind=ToponymKind.HOUSE,
        precision="number",
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
        record=record,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)
    context = ToolExecutionContext()

    resolved = await resolver.geocode_place(
        query="Большая Покровская улица, 2",
        city="Нижний Новгород",
        context=context,
    )

    assert resolved is not None
    assert context.warnings == (
        "Geocoder normalized the requested address 'Большая Покровская улица, 2' "
        "to 'Большая Покровская улица, 2А'; verify the building suffix if exact "
        "entrance accuracy matters.",
    )


async def test_geocoded_place_resolver_rejects_unpersisted_geocoder_ref() -> None:
    """Verify a missing hidden record is reported as a geocoder contract failure."""

    store = InMemoryPlaceStore()
    match = _match(
        "plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
    )
    geocoder = StaticGeocoder(
        output=GeocodePlaceOutput(best=match, matches=[match]),
        store=store,
    )
    resolver = GeocodedPlaceResolver(geocoder=geocoder, place_store=store)

    with pytest.raises(PlaceResolutionContractError) as exc_info:
        await resolver.geocode_place(
            query="Красная площадь",
            city="Москва",
            context=ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert exc_info.value.provider == "fake_geocoder"
    assert exc_info.value.retryable is False
