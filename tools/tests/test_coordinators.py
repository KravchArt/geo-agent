from __future__ import annotations

import pytest

from tools.base import (
    ToolClarification,
    ToolClarificationOption,
    ToolErrorCode,
    ToolExecutionError,
    ToolFailureKind,
)
from tools.coordination import ProviderExecutionStrategy
from tools.geo.places_search.coordinator import PlacesSearchCoordinator
from tools.geo.places_search.schemas import (
    MAX_NAMED_PLACE_RESULTS,
    OrganisationCandidate,
    OrganisationResolutionStatus,
    Place,
    PlacesSearchInput,
    PlacesSearchOutput,
    SearchMode,
)
from tools.geo.routing import RouteInfo, RoutingCoordinator, RoutingInput, RoutingOutput
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.web.coordinator import WebSearchCoordinator
from tools.web.search import WebSearchInput, WebSearchOutput


class RecordingWebSearchProvider:
    def __init__(
        self,
        provider: str,
        *,
        error: ToolExecutionError | None = None,
    ) -> None:
        self.provider = provider
        self.error = error
        self.calls: list[WebSearchInput] = []

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> WebSearchOutput:
        self.calls.append(args)

        if self.error is not None:
            raise self.error

        return WebSearchOutput(query=args.query)


class RecordingPlacesSearchProvider:
    def __init__(
        self,
        provider: str,
        *,
        result: PlacesSearchOutput | None = None,
        error: ToolExecutionError | None = None,
        supports_open_now: bool = False,
        supports_min_rating: bool = False,
        resolves_area_natively: bool = False,
    ) -> None:
        self.provider = provider
        self.supports_open_now = supports_open_now
        self.supports_min_rating = supports_min_rating
        self.resolves_area_natively = resolves_area_natively
        self.result = result or PlacesSearchOutput()
        self.error = error
        self.calls: list[PlacesSearchInput] = []
        self.first_address_calls: list[PlacesSearchInput] = []

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result

    async def search_first_address(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        self.first_address_calls.append(args)
        return await self.search(args, context)


class RecordingPlacesSearchScopeResolver:
    def __init__(self, resolved_args: PlacesSearchInput | None = None) -> None:
        self._resolved_args = resolved_args
        self.calls: list[PlacesSearchInput] = []

    async def resolve_scope(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchInput:
        self.calls.append(args)
        return self._resolved_args or args


class RecordingRoutingInputResolver:
    def __init__(self, resolved_args: RoutingInput | None = None) -> None:
        self._resolved_args = resolved_args
        self.calls: list[RoutingInput] = []

    async def resolve_to_refs(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingInput:
        self.calls.append(args)
        return self._resolved_args or args


class RecordingRoutingProvider:
    def __init__(
        self,
        provider: str,
        *,
        error: ToolExecutionError | None = None,
        warning: str | None = None,
    ) -> None:
        self.provider = provider
        self.error = error
        self.warning = warning
        self.calls: list[RoutingInput] = []

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        self.calls.append(args)
        if self.warning is not None:
            context.add_warning(self.warning)
        if self.error is not None:
            if self.error.provider is not None and self.error.provider.endswith("_routing"):
                context.record_upstream_call(
                    provider=self.error.provider,
                    operation="build_route",
                    latency_ms=1,
                    outcome=UpstreamCallOutcome.FAILURE,
                    status_code=self.error.status_code,
                    error_code=self.error.error_code.value,
                    failure_kind=(
                        self.error.failure_kind.value
                        if self.error.failure_kind is not None
                        else None
                    ),
                    provider_code=self.error.provider_code,
                    retryable=self.error.retryable,
                )
            raise self.error

        context.record_upstream_call(
            provider=f"{self.provider}_routing",
            operation="build_route",
            latency_ms=1,
            outcome=UpstreamCallOutcome.SUCCESS,
            status_code=200,
        )
        return RoutingOutput(
            mode=args.mode,
            transport=args.transport,
            route=RouteInfo(length_m=100, duration_s=60),
        )


async def test_web_search_coordinator_uses_only_first_provider_by_default() -> None:
    """Verify that web search coordinator uses only first provider by default."""

    first = RecordingWebSearchProvider("first")
    second = RecordingWebSearchProvider("second")
    coordinator = WebSearchCoordinator(providers=[first, second])
    args = WebSearchInput(query="выставки в Москве")

    context = ToolExecutionContext()
    result = await coordinator.search(args, context)

    assert result == WebSearchOutput(query=args.query)
    assert first.calls == [args]
    assert second.calls == []


async def test_places_search_coordinator_uses_only_first_provider_by_default() -> None:
    """Verify that places search coordinator uses only first provider by default."""

    first = RecordingPlacesSearchProvider("first")
    second = RecordingPlacesSearchProvider("second")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[first, second],
        scope_resolver=scope_resolver,
        strategy=ProviderExecutionStrategy.FIRST,
    )
    args = PlacesSearchInput(
        mode=SearchMode.AREA,
        query="кофейни",
        city="Москва",
    )

    context = ToolExecutionContext()
    result = await coordinator.search(args, context)

    assert result == PlacesSearchOutput()
    assert scope_resolver.calls == [args]
    assert first.calls == [args]
    assert second.calls == []


async def test_places_search_resolve_returns_independent_batch_outcomes() -> None:
    """Resolve, disambiguate, miss, and fail candidates without losing batch order."""

    alpha = Place(
        ref="plc_a1b2c3d4e5",
        id="alpha",
        name="Alpha",
        address="Main Street 1, Berlin",
    )
    beta_first = Place(
        ref="plc_b2c3d4e5f6",
        id="beta-1",
        name="Beta",
        address="First Street 1, Berlin",
    )
    beta_second = Place(
        ref="plc_c3d4e5f6a7",
        id="beta-2",
        name="Beta",
        address="Second Street 2, Berlin",
    )

    class BatchProvider(RecordingPlacesSearchProvider):
        async def search(
            self,
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchOutput:
            self.calls.append(args)
            if args.query == "Broken":
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    "batch candidate timed out",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.TIMEOUT,
                    retryable=True,
                )
            return PlacesSearchOutput(
                places={
                    "Alpha": [alpha],
                    "Beta": [beta_first, beta_second],
                }.get(args.query or "", [])
            )

    provider = BatchProvider("batch", supports_open_now=True)

    class BatchScopeResolver(RecordingPlacesSearchScopeResolver):
        async def resolve_scope(
            self,
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchInput:
            self.calls.append(args)
            if args.area_ref is not None:
                return args
            return PlacesSearchInput(
                mode="area",
                query=args.query,
                area_ref="plc_d4e5f6a7b8",
            )

    scope_resolver = BatchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[provider],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="resolve",
        city="Berlin",
        open_now=True,
        organisations=[
            {"client_id": "a", "name": "Alpha"},
            {"client_id": "b", "name": "Beta", "address_hint": "Second Street"},
            {"client_id": "c", "name": "Beta"},
            {"client_id": "d", "name": "Missing"},
            {"client_id": "e", "name": "Broken"},
        ],
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert [item.client_id for item in result.resolved] == ["a", "b", "c", "d", "e"]
    assert [item.status for item in result.resolved] == [
        OrganisationResolutionStatus.RESOLVED,
        OrganisationResolutionStatus.RESOLVED,
        OrganisationResolutionStatus.RESOLVED,
        OrganisationResolutionStatus.NOT_FOUND,
        OrganisationResolutionStatus.ERROR,
    ]
    assert result.resolved[0].place == alpha
    assert result.resolved[1].place == beta_second
    assert result.resolved[2].place == beta_first
    assert result.resolved[4].error == "batch candidate timed out"
    assert result.returned_count == 3
    assert len(scope_resolver.calls) == 6
    assert scope_resolver.calls[0].city == "Berlin"
    assert all(call.area_ref == "plc_d4e5f6a7b8" for call in provider.calls)
    assert all(call.open_now is True for call in provider.calls)
    assert provider.first_address_calls == provider.calls


def test_places_search_resolve_matches_transliterated_address_hint() -> None:
    dostyk = Place(
        ref="plc_a1b2c3d4e5",
        id="navat-dostyk",
        name="Chaihana Navat, ресторан",
        address="Алматы, проспект Достык, 48",
    )
    other_branch = Place(
        ref="plc_b2c3d4e5f6",
        id="navat-abai",
        name="Chaihana Navat, ресторан",
        address="Алматы, проспект Абая, 23",
    )

    result = PlacesSearchCoordinator._select_resolution(
        OrganisationCandidate(
            client_id="navat_dostyk",
            name="NAVAT",
            address_hint="Dostyk Avenue",
        ),
        [dostyk, other_branch],
    )

    assert result.status is OrganisationResolutionStatus.RESOLVED
    assert result.place == dostyk


def test_places_search_resolve_keeps_results_when_address_hint_has_no_match() -> None:
    frank = Place(
        ref="plc_a1b2c3d4e5",
        id="frank-pokrovskaya",
        name="Frank by Баста, реберная",
        address="Нижний Новгород, Большая Покровская улица, 50",
    )

    result = PlacesSearchCoordinator._select_resolution(
        OrganisationCandidate(
            client_id="frank",
            name="Frank by Баста",
            address_hint="Покровка",
        ),
        [frank],
    )

    assert result.status is OrganisationResolutionStatus.RESOLVED
    assert result.place == frank


def test_places_search_resolve_selects_first_normalized_provider_result() -> None:
    residential_complex = Place(
        ref="plc_a1b2c3d4e5",
        id="residential-complex",
        name="Алые паруса, жилой комплекс",
        address="Киров, Октябрьский проспект, 117",
    )
    kindergarten = Place(
        ref="plc_b2c3d4e5f6",
        id="kindergarten",
        name="Алые паруса",
        address="Киров, улица Космонавта Владислава Волкова, 2/2",
    )

    result = PlacesSearchCoordinator._select_resolution(
        OrganisationCandidate(client_id="alye_parusa", name="Алые Паруса"),
        [residential_complex, kindergarten],
    )

    assert result.status is OrganisationResolutionStatus.RESOLVED
    assert result.place == residential_complex


async def test_places_search_resolve_reuses_selected_area_ref_without_mode_drift() -> None:
    place = Place(
        ref="plc_a1b2c3d4e5",
        id="alye-parusa",
        name="Алые паруса, жилой комплекс",
        address="Киров, Октябрьский проспект, 117",
    )
    provider = RecordingPlacesSearchProvider(
        "tomtom",
        result=PlacesSearchOutput(places=[place]),
    )
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[provider],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="resolve",
        area_ref="plc_d4e5f6a7b8",
        organisations=[{"client_id": "alye_parusa", "name": "Алые паруса"}],
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result.resolved[0].status is OrganisationResolutionStatus.RESOLVED
    assert result.resolved[0].place == place
    assert len(scope_resolver.calls) == 1
    assert scope_resolver.calls[0].area_ref == "plc_d4e5f6a7b8"
    assert len(provider.calls) == 1
    assert provider.calls[0].mode is SearchMode.AREA
    assert provider.calls[0].city is None
    assert provider.calls[0].area_ref == "plc_d4e5f6a7b8"


async def test_places_search_resolve_defers_geocoding_for_native_twogis_area() -> None:
    place = Place(
        ref="plc_a1b2c3d4e5",
        id="twogis-molot",
        name="Молот",
        address="Городец, улица Свердлова, 9/9",
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        result=PlacesSearchOutput(places=[place]),
        resolves_area_natively=True,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="resolve",
        city="Городец",
        organisations=[{"client_id": "molot", "name": "Молот"}],
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result.resolved[0].status is OrganisationResolutionStatus.RESOLVED
    assert result.resolved[0].place == place
    assert scope_resolver.calls == []
    assert len(twogis.calls) == 1
    assert twogis.calls[0].city == "Городец"
    assert twogis.calls[0].area_ref is None
    assert tomtom.calls == []


async def test_places_search_resolve_geocodes_only_after_twogis_fallback() -> None:
    area_ref = "plc_d4e5f6a7b8"
    place = Place(
        ref="plc_a1b2c3d4e5",
        id="tomtom-molot",
        name="Молот",
        address="Городец, улица Свердлова, 9/9",
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        error=ToolExecutionError(
            ToolErrorCode.NOT_FOUND,
            "2GIS search returned no results",
            provider="twogis_search",
            failure_kind=ToolFailureKind.HTTP_STATUS,
            retryable=False,
        ),
        resolves_area_natively=True,
    )
    tomtom = RecordingPlacesSearchProvider(
        "tomtom",
        result=PlacesSearchOutput(
            places=[place],
            area={"ref": area_ref, "name": "Городец", "address": "Городец, Россия"},
        ),
    )
    raw_item_args = PlacesSearchInput(
        mode="area",
        query="Молот",
        city="Городец",
        limit=MAX_NAMED_PLACE_RESULTS,
    )
    resolved_item_args = raw_item_args.model_copy(update={"city": None, "area_ref": area_ref})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_item_args)
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="resolve",
        city="Городец",
        organisations=[{"client_id": "molot", "name": "Молот"}],
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result.resolved[0].status is OrganisationResolutionStatus.RESOLVED
    assert twogis.calls == [raw_item_args]
    assert scope_resolver.calls == [raw_item_args]
    assert tomtom.calls == [resolved_item_args]


async def test_places_search_prioritizes_tomtom_and_falls_back_to_yandex_on_empty() -> None:
    area_ref = "plc_a1b2c3d4e5"
    tomtom = RecordingPlacesSearchProvider(
        "tomtom",
        result=PlacesSearchOutput(
            area={
                "ref": area_ref,
                "name": "Нижний Новгород",
                "address": "Россия, Нижний Новгород",
            }
        ),
    )
    yandex_result = PlacesSearchOutput(
        places=[
            {
                "ref": "plc_b2c3d4e5f6",
                "id": "yandex-1",
                "name": "Московский вокзал",
                "address": "Площадь Революции, 2А",
            }
        ]
    )
    yandex = RecordingPlacesSearchProvider("yandex", result=yandex_result)
    args = PlacesSearchInput(
        mode="area",
        query="Московский вокзал",
        city="Нижний Новгород",
    )
    resolved_args = args.model_copy(update={"city": None, "area_ref": area_ref})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, tomtom],
        scope_resolver=scope_resolver,
    )
    context = ToolExecutionContext()

    result = await coordinator.search(args, context)

    assert result == yandex_result
    assert scope_resolver.calls == [args]
    assert tomtom.calls == [resolved_args]
    assert yandex.calls == [resolved_args]
    assert context.warnings == ("TomTom place search returned no matches; retrying with Yandex.",)


async def test_places_search_prioritizes_twogis_before_tomtom_and_yandex() -> None:
    area_ref = "plc_a1b2c3d4e5"
    tomtom = RecordingPlacesSearchProvider("tomtom")
    twogis_result = PlacesSearchOutput(
        places=[
            {
                "ref": "plc_b2c3d4e5f6",
                "id": "twogis-cafe",
                "name": "Кофейня",
                "address": "Москва, Тверская улица, 1",
            }
        ]
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        result=twogis_result,
        resolves_area_natively=True,
    )
    yandex = RecordingPlacesSearchProvider("yandex")
    args = PlacesSearchInput(mode="area", query="кафе", city="Москва")
    resolved_args = args.model_copy(update={"city": None, "area_ref": area_ref})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, twogis, tomtom],
        scope_resolver=scope_resolver,
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result == twogis_result
    assert scope_resolver.calls == []
    assert twogis.calls == [args]
    assert tomtom.calls == []
    assert yandex.calls == []


async def test_places_search_falls_back_on_transient_tomtom_error() -> None:
    error = ToolExecutionError(
        ToolErrorCode.TIMEOUT,
        "TomTom search timed out",
        provider="tomtom_search",
        failure_kind=ToolFailureKind.TIMEOUT,
        retryable=True,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom", error=error)
    yandex_result = PlacesSearchOutput()
    yandex = RecordingPlacesSearchProvider("yandex", result=yandex_result)
    args = PlacesSearchInput(
        mode="near",
        query="кафе",
        category="cafe",
        near_query="Парк Горького",
        city="Москва",
    )
    resolved_args = args.model_copy(
        update={
            "near": "plc_a1b2c3d4e5",
            "near_query": None,
        }
    )
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, tomtom],
        scope_resolver=scope_resolver,
    )
    context = ToolExecutionContext()

    result = await coordinator.search(args, context)

    assert result == yandex_result
    assert scope_resolver.calls == [args]
    assert tomtom.calls == [resolved_args]
    assert yandex.calls == [resolved_args]
    assert context.warnings == ("TomTom place search timed out; retrying with Yandex.",)


async def test_places_search_falls_back_from_twogis_backend_exception_with_resolved_near() -> None:
    error = ToolExecutionError(
        ToolErrorCode.UPSTREAM_ERROR,
        "2GIS search backend temporarily failed",
        status_code=403,
        provider="twogis_search",
        provider_code="backendException",
        failure_kind=ToolFailureKind.HTTP_STATUS,
        retryable=True,
    )
    twogis = RecordingPlacesSearchProvider("twogis", error=error)
    tomtom_result = PlacesSearchOutput()
    tomtom = RecordingPlacesSearchProvider("tomtom", result=tomtom_result)
    args = PlacesSearchInput(
        mode="near",
        query="аптеки",
        category="pharmacy",
        near="кафе Молот",
        area="Городец",
        radius_m=3_000,
    )
    resolved_args = args.model_copy(
        update={
            "near": "plc_a1b2c3d4e5",
            "area": None,
        }
    )
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )
    context = ToolExecutionContext()

    result = await coordinator.search(args, context)

    assert result == tomtom_result
    assert scope_resolver.calls == [args]
    assert twogis.calls == [resolved_args]
    assert tomtom.calls == [resolved_args]
    assert context.warnings == ("2GIS place search failed; retrying with TomTom.",)


async def test_places_search_falls_back_on_not_found() -> None:
    error = ToolExecutionError(
        ToolErrorCode.NOT_FOUND,
        "2GIS search returned no results",
        status_code=404,
        provider="twogis_search",
        failure_kind=ToolFailureKind.HTTP_STATUS,
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        error=error,
        resolves_area_natively=True,
    )
    tomtom_result = PlacesSearchOutput()
    tomtom = RecordingPlacesSearchProvider("tomtom", result=tomtom_result)
    args = PlacesSearchInput(mode="area", query="кафе", city="Париж")
    resolved_args = args.model_copy(update={"city": None, "area_ref": "plc_a1b2c3d4e5"})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )
    context = ToolExecutionContext()

    result = await coordinator.search(args, context)

    assert result == tomtom_result
    assert scope_resolver.calls == [args]
    assert twogis.calls == [args]
    assert tomtom.calls == [resolved_args]
    assert context.warnings == ("2GIS place search returned no matches; retrying with TomTom.",)


async def test_places_search_reports_coverage_fallback_without_claiming_place_search() -> None:
    coverage_miss = ToolExecutionError(
        ToolErrorCode.NOT_FOUND,
        "2GIS does not cover the requested locality",
        status_code=404,
        provider="twogis_search",
        failure_kind=ToolFailureKind.COVERAGE_MISS,
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        error=coverage_miss,
        resolves_area_natively=True,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom")
    args = PlacesSearchInput(mode="area", query="museums", city="Berlin")
    resolved_args = args.model_copy(update={"city": None, "area_ref": "plc_a1b2c3d4e5"})
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=RecordingPlacesSearchScopeResolver(resolved_args),
    )
    context = ToolExecutionContext()

    await coordinator.search(args, context)

    assert context.warnings == ("2GIS does not cover the requested locality; using TomTom.",)


async def test_places_search_stops_for_twogis_region_clarification() -> None:
    clarification = ToolClarification(
        kind="select_area_query",
        question="Which Alexandria do you mean?",
        options=[
            ToolClarificationOption(
                value="Alexandria, Central Louisiana, US",
                label="Alexandria, Central Louisiana, US",
            ),
            ToolClarificationOption(
                value="Alexandria, Northern Virginia, US",
                label="Alexandria, Northern Virginia, US",
            ),
        ],
    )
    error = ToolExecutionError(
        ToolErrorCode.INVALID_INPUT,
        "City is ambiguous in 2GIS coverage: 'Alexandria'.",
        provider="twogis_search",
        retryable=False,
        clarification=clarification,
    )
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        error=error,
        resolves_area_natively=True,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom")
    args = PlacesSearchInput(mode="area", query="restaurants", city="Alexandria")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.search(args, ToolExecutionContext())

    assert exc_info.value.clarification == clarification
    assert twogis.calls == [args]
    assert tomtom.calls == []
    assert scope_resolver.calls == []


async def test_places_search_does_not_hide_tomtom_authentication_error() -> None:
    error = ToolExecutionError(
        ToolErrorCode.UPSTREAM_ERROR,
        "TomTom authentication failed",
        status_code=403,
        provider="tomtom_search",
        failure_kind=ToolFailureKind.AUTHENTICATION,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom", error=error)
    yandex = RecordingPlacesSearchProvider("yandex")
    args = PlacesSearchInput(mode="area", query="кафе", city="Москва")
    resolved_args = args.model_copy(
        update={
            "city": None,
            "area_ref": "plc_a1b2c3d4e5",
        }
    )
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, tomtom],
        scope_resolver=scope_resolver,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.search(
            args,
            ToolExecutionContext(),
        )

    assert exc_info.value is error
    assert scope_resolver.calls == [args]
    assert tomtom.calls == [resolved_args]
    assert yandex.calls == []


async def test_places_search_open_now_uses_only_capable_providers_in_priority_order() -> None:
    tomtom = RecordingPlacesSearchProvider("tomtom", supports_open_now=True)
    yandex = RecordingPlacesSearchProvider("yandex", supports_open_now=True)
    args = PlacesSearchInput(
        mode="area",
        query="кафе",
        city="Москва",
        open_now=True,
    )
    resolved_args = args.model_copy(update={"city": None, "area_ref": "plc_a1b2c3d4e5"})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, tomtom],
        scope_resolver=scope_resolver,
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result == PlacesSearchOutput()
    assert scope_resolver.calls == [args]
    assert tomtom.calls == [resolved_args]
    assert yandex.calls == [resolved_args]


async def test_places_search_min_rating_uses_only_twogis() -> None:
    twogis = RecordingPlacesSearchProvider(
        "twogis",
        supports_min_rating=True,
        resolves_area_natively=True,
    )
    tomtom = RecordingPlacesSearchProvider("tomtom")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom, twogis],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="area",
        query="рестораны",
        category="restaurant",
        city="Москва",
        min_rating=4.6,
    )

    result = await coordinator.search(args, ToolExecutionContext())

    assert result == PlacesSearchOutput()
    assert scope_resolver.calls == []
    assert twogis.calls == [args]
    assert tomtom.calls == []


async def test_places_search_min_rating_fails_without_twogis_before_scope_resolution() -> None:
    tomtom = RecordingPlacesSearchProvider("tomtom")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[tomtom],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="area",
        query="рестораны",
        category="restaurant",
        city="Москва",
        min_rating=4.6,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.search(args, ToolExecutionContext())

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert "web_search" in str(exc_info.value)
    assert scope_resolver.calls == []
    assert tomtom.calls == []


async def test_places_search_open_now_fails_before_scope_resolution_without_capable_provider() -> (
    None
):
    legacy = RecordingPlacesSearchProvider("legacy")
    scope_resolver = RecordingPlacesSearchScopeResolver()
    coordinator = PlacesSearchCoordinator(
        providers=[legacy],
        scope_resolver=scope_resolver,
    )
    args = PlacesSearchInput(
        mode="area",
        query="кафе",
        city="Москва",
        open_now=True,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.search(args, ToolExecutionContext())

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert scope_resolver.calls == []
    assert legacy.calls == []


async def test_places_search_open_now_falls_back_to_yandex_after_tomtom_failure() -> None:
    error = ToolExecutionError(
        ToolErrorCode.TIMEOUT,
        "TomTom search timed out",
        provider="tomtom_search",
        failure_kind=ToolFailureKind.TIMEOUT,
        retryable=True,
    )
    tomtom = RecordingPlacesSearchProvider(
        "tomtom",
        error=error,
        supports_open_now=True,
    )
    yandex = RecordingPlacesSearchProvider("yandex", supports_open_now=True)
    args = PlacesSearchInput(
        mode="area",
        query="кафе",
        city="Москва",
        open_now=True,
    )
    resolved_args = args.model_copy(update={"city": None, "area_ref": "plc_a1b2c3d4e5"})
    scope_resolver = RecordingPlacesSearchScopeResolver(resolved_args)
    coordinator = PlacesSearchCoordinator(
        providers=[yandex, tomtom],
        scope_resolver=scope_resolver,
    )

    context = ToolExecutionContext()
    result = await coordinator.search(args, context)

    assert result == PlacesSearchOutput()
    assert tomtom.calls == [resolved_args]
    assert yandex.calls == [resolved_args]
    assert context.warnings == ("TomTom place search timed out; retrying with Yandex.",)


async def test_routing_coordinator_uses_only_first_provider_by_default() -> None:
    """Verify that routing coordinator uses only first provider by default."""

    first = RecordingRoutingProvider("first")
    second = RecordingRoutingProvider("second")
    input_resolver = RecordingRoutingInputResolver()
    coordinator = RoutingCoordinator(
        providers=[first, second],
        input_resolver=input_resolver,
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )

    result = await coordinator.route(args, ToolExecutionContext())

    assert result.route is not None
    assert result.route.length_m == 100
    assert input_resolver.calls == [args]
    assert first.calls == [args]
    assert second.calls == []


async def test_routing_coordinator_prioritizes_graphhopper_over_osrm() -> None:
    graphhopper = RecordingRoutingProvider("graphhopper")
    osrm = RecordingRoutingProvider("osrm")
    coordinator = RoutingCoordinator(
        providers=[osrm, graphhopper],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )

    await coordinator.route(args, ToolExecutionContext())

    assert graphhopper.calls == [args]
    assert osrm.calls == []


async def test_routing_coordinator_prioritizes_twogis_and_falls_back_to_graphhopper() -> None:
    twogis = RecordingRoutingProvider(
        "twogis",
        error=ToolExecutionError(
            ToolErrorCode.NOT_FOUND,
            "2GIS route is outside coverage",
            provider="twogis_routing",
            failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
            retryable=False,
        ),
    )
    graphhopper = RecordingRoutingProvider("graphhopper")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, twogis],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )
    context = ToolExecutionContext()

    result = await coordinator.route(args, context)

    assert result.route is not None
    assert twogis.calls == [args]
    assert graphhopper.calls == [args]
    assert context.warnings == ("2GIS routing failed; retrying with GraphHopper.",)


async def test_routing_coordinator_falls_back_on_provider_capability_gap() -> None:
    twogis = RecordingRoutingProvider(
        "twogis",
        error=ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            "2GIS cannot disable traffic",
            provider="twogis_routing",
            retryable=False,
        ),
    )
    graphhopper = RecordingRoutingProvider("graphhopper")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, twogis],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
        use_traffic=False,
    )

    await coordinator.route(args, ToolExecutionContext())

    assert twogis.calls == [args]
    assert graphhopper.calls == [args]


async def test_routing_coordinator_falls_back_from_graphhopper_to_osrm() -> None:
    graphhopper = RecordingRoutingProvider(
        "graphhopper",
        error=ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "GraphHopper unavailable",
            provider="graphhopper",
            failure_kind=ToolFailureKind.NETWORK,
            retryable=True,
        ),
        warning="warning from failed provider",
    )
    osrm = RecordingRoutingProvider("osrm", warning="OSRM result warning")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )
    context = ToolExecutionContext()

    result = await coordinator.route(args, context)

    assert result.route is not None
    assert graphhopper.calls == [args]
    assert osrm.calls == [args]
    assert context.warnings == (
        "GraphHopper routing failed; retrying with OSRM.",
        "OSRM result warning",
    )


async def test_routing_coordinator_resolves_text_once_before_provider_fallback() -> None:
    graphhopper = RecordingRoutingProvider(
        "graphhopper",
        error=ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "GraphHopper unavailable",
            provider="graphhopper",
            failure_kind=ToolFailureKind.NETWORK,
            retryable=True,
        ),
    )
    osrm = RecordingRoutingProvider("osrm")
    args = RoutingInput(
        mode="route",
        waypoints=[
            {"query": "Кремль", "city": "Москва"},
            {"query": "Парк Горького", "city": "Москва"},
        ],
    )
    resolved_args = args.model_copy(update={"waypoints": ["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"]})
    input_resolver = RecordingRoutingInputResolver(resolved_args)
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=input_resolver,
    )

    result = await coordinator.route(args, ToolExecutionContext())

    assert result.route is not None
    assert input_resolver.calls == [args]
    assert graphhopper.calls == [resolved_args]
    assert osrm.calls == [resolved_args]


async def test_routing_coordinator_falls_back_after_graphhopper_http_timeout() -> None:
    graphhopper = RecordingRoutingProvider(
        "graphhopper",
        error=ToolExecutionError(
            ToolErrorCode.TIMEOUT,
            "GraphHopper request timed out",
            provider="graphhopper_routing",
            retryable=True,
        ),
    )
    osrm = RecordingRoutingProvider("osrm")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )
    context = ToolExecutionContext()

    result = await coordinator.route(args, context)

    assert result.route is not None
    assert graphhopper.calls == [args]
    assert osrm.calls == [args]
    assert context.warnings == ("GraphHopper routing timed out; retrying with OSRM.",)


@pytest.mark.parametrize(
    "error",
    [
        ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "GraphHopper authentication failed",
            status_code=401,
            provider="graphhopper_routing",
            failure_kind=ToolFailureKind.AUTHENTICATION,
            retryable=False,
        ),
        ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "GraphHopper provider contract failed",
            provider="graphhopper_routing",
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        ),
    ],
)
async def test_routing_coordinator_does_not_hide_operational_errors(
    error: ToolExecutionError,
) -> None:
    graphhopper = RecordingRoutingProvider("graphhopper", error=error)
    osrm = RecordingRoutingProvider("osrm")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )
    context = ToolExecutionContext()

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.route(args, context)

    assert exc_info.value is error
    assert osrm.calls == []
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE


async def test_invalid_schema_falls_back_but_remains_observable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graphhopper = RecordingRoutingProvider(
        "graphhopper",
        error=ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "GraphHopper returned malformed data",
            status_code=200,
            provider="graphhopper_routing",
            failure_kind=ToolFailureKind.INVALID_SCHEMA,
            retryable=False,
        ),
    )
    osrm = RecordingRoutingProvider("osrm")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )

    with caplog.at_level("ERROR", logger="tools.geo.routing.coordinator"):
        context = ToolExecutionContext()
        result = await coordinator.route(args, context)

    assert result.route is not None
    assert [call.outcome for call in context.upstream_calls] == [
        UpstreamCallOutcome.FAILURE,
        UpstreamCallOutcome.SUCCESS,
    ]
    assert context.upstream_calls[0].failure_kind == ToolFailureKind.INVALID_SCHEMA.value
    assert "routing_provider_fallback" in caplog.text
    assert "failure_kind=invalid_schema" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ToolExecutionError(
            ToolErrorCode.UNKNOWN_REF,
            "unknown place ref",
        ),
        ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "geocoder contract failure",
            provider="yandex_geocoder",
        ),
    ],
)
async def test_routing_coordinator_does_not_fallback_for_shared_errors(
    error: ToolExecutionError,
) -> None:
    graphhopper = RecordingRoutingProvider("graphhopper", error=error)
    osrm = RecordingRoutingProvider("osrm")
    coordinator = RoutingCoordinator(
        providers=[graphhopper, osrm],
        input_resolver=RecordingRoutingInputResolver(),
    )
    args = RoutingInput(
        mode="route",
        waypoints=["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.route(args, ToolExecutionContext())

    assert exc_info.value is error
    assert graphhopper.calls == [args]
    assert osrm.calls == []


@pytest.mark.parametrize(
    "coordinator",
    [
        lambda: WebSearchCoordinator(providers=[]),
        lambda: PlacesSearchCoordinator(providers=[]),
        lambda: RoutingCoordinator(providers=[]),
    ],
)
def test_coordinator_rejects_empty_provider_list(coordinator: object) -> None:
    """Verify that coordinator rejects empty provider list."""

    with pytest.raises(ValueError, match="requires at least one provider"):
        coordinator()  # type: ignore[operator]


def test_web_search_coordinator_rejects_duplicate_providers() -> None:
    """Verify that web search coordinator rejects duplicate providers."""

    with pytest.raises(ValueError, match="duplicate web search providers"):
        WebSearchCoordinator(
            providers=[
                RecordingWebSearchProvider("same"),
                RecordingWebSearchProvider("same"),
            ]
        )


def test_places_search_coordinator_rejects_duplicate_providers() -> None:
    """Verify that places search coordinator rejects duplicate providers."""

    with pytest.raises(ValueError, match="duplicate places search providers"):
        PlacesSearchCoordinator(
            providers=[
                RecordingPlacesSearchProvider("same"),
                RecordingPlacesSearchProvider("same"),
            ]
        )


def test_routing_coordinator_rejects_duplicate_providers() -> None:
    """Verify that routing coordinator rejects duplicate providers."""

    with pytest.raises(ValueError, match="duplicate routing providers"):
        RoutingCoordinator(
            providers=[
                RecordingRoutingProvider("same"),
                RecordingRoutingProvider("same"),
            ]
        )


@pytest.mark.parametrize(
    "coordinator",
    [
        lambda: WebSearchCoordinator(
            providers=[RecordingWebSearchProvider("web")],
            strategy=ProviderExecutionStrategy.PARALLEL,
        ),
        lambda: PlacesSearchCoordinator(
            providers=[RecordingPlacesSearchProvider("places")],
            strategy=ProviderExecutionStrategy.PARALLEL,
        ),
        lambda: RoutingCoordinator(
            providers=[RecordingRoutingProvider("routing")],
            strategy=ProviderExecutionStrategy.PARALLEL,
        ),
    ],
)
def test_coordinator_rejects_unimplemented_parallel_strategy(coordinator: object) -> None:
    """Verify that coordinator rejects unimplemented parallel strategy."""

    with pytest.raises(NotImplementedError, match=r"parallel .* strategy is not implemented"):
        coordinator()  # type: ignore[operator]


async def test_web_search_coordinator_preserves_provider_error() -> None:
    """Verify that web search coordinator preserves provider error."""

    error = ToolExecutionError(
        ToolErrorCode.TIMEOUT,
        "provider timed out",
        provider="web",
        retryable=True,
    )
    coordinator = WebSearchCoordinator(providers=[RecordingWebSearchProvider("web", error=error)])

    with pytest.raises(ToolExecutionError) as exc_info:
        await coordinator.search(
            WebSearchInput(query="новости"),
            ToolExecutionContext(),
        )

    assert exc_info.value is error
