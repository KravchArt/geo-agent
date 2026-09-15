from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

import httpx
import pytest
from pydantic import ValidationError

from backend.app import tools as runtime_tools_module
from backend.app.config import Settings
from backend.app.tools import build_runtime_tools
from tools.geo import (
    PlacesSearchInput,
    PlacesSearchOutput,
    RouteInfo,
    RoutingInput,
    RoutingOutput,
)
from tools.geo.geocoding import GeocodedPlaceResolver
from tools.geo.place_store import InMemoryPlaceStore, PlaceStore
from tools.geo.places_search import PlacesSearchScopeResolver
from tools.geo.places_search.tomtom import TomTomNamedPoiResolver
from tools.geo.places_search.twogis import TwoGisNamedPoiResolver
from tools.geo.routing import RoutingPlaceResolver
from tools.geo.text_place_resolution import NamedPoiResolver, TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.web import (
    InMemorySourceStore,
    WebSearchInput,
    WebSearchOutput,
)


def make_settings(
    *,
    places: list[Literal["yandex", "tomtom", "twogis"]],
    web: list[Literal["tavily", "exa", "firecrawl"]],
    routing: list[Literal["yandex", "graphhopper", "osrm", "twogis"]] | None = None,
    text_place_resolution: list[Literal["twogis", "tomtom"]] | None = None,
) -> Settings:
    configured_routing = routing or []
    configured_text_place_resolution = (
        text_place_resolution
        if text_place_resolution is not None
        else [name for name in ("twogis", "tomtom") if name in places]
        or (["tomtom"] if configured_routing else [])
    )
    return Settings(
        places_search_providers=places,
        routing_providers=configured_routing,
        text_place_resolution_providers=configured_text_place_resolution,
        web_search_providers=web,
        yandex_geocoder_api_key=None,
        yandex_organisation_search_api_key="test-search-key" if "yandex" in places else None,
        tomtom_api_key=("test-tomtom-key" if places or configured_routing else None),
        dgis_api_key=(
            "test-dgis-key"
            if "twogis" in places
            or "twogis" in configured_routing
            or "twogis" in configured_text_place_resolution
            else None
        ),
        yandex_routing_api_key=("test-routing-key" if "yandex" in configured_routing else None),
        graphhopper_api_key=(
            "test-graphhopper-key" if "graphhopper" in configured_routing else None
        ),
        tavily_api_key="test-tavily-key" if "tavily" in web else None,
        exa_api_key="test-exa-key" if "exa" in web else None,
        firecrawl_api_key="test-firecrawl-key" if "firecrawl" in web else None,
    )


@pytest.mark.parametrize(
    ("settings", "expected_names"),
    [
        (make_settings(places=[], web=[]), set()),
        (make_settings(places=["yandex"], web=[]), {"places_search"}),
        (make_settings(places=["tomtom"], web=[]), {"places_search"}),
        (make_settings(places=["twogis"], web=[]), {"places_search"}),
        (make_settings(places=[], web=["tavily"]), {"web_search"}),
        (make_settings(places=[], web=["exa"]), {"web_search"}),
        (make_settings(places=[], web=["firecrawl"]), {"web_search"}),
        (
            make_settings(places=[], web=[], routing=["yandex"]),
            {"routing_tool"},
        ),
        (
            make_settings(places=[], web=[], routing=["graphhopper"]),
            {"routing_tool"},
        ),
        (
            make_settings(places=[], web=[], routing=["osrm"]),
            {"routing_tool"},
        ),
        (
            make_settings(places=[], web=[], routing=["twogis"]),
            {"routing_tool"},
        ),
        (
            make_settings(places=["yandex"], web=["tavily"], routing=["yandex"]),
            {"places_search", "routing_tool", "web_search"},
        ),
    ],
)
async def test_build_runtime_tools_exposes_only_configured_providers(
    settings: Settings,
    expected_names: set[str],
) -> None:
    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

    assert set(runtime_tools) == expected_names


def test_settings_require_yandex_keys_when_provider_is_enabled() -> None:
    with pytest.raises(ValidationError, match="missing required provider keys"):
        Settings(
            places_search_providers=["yandex"],
            web_search_providers=[],
            yandex_organisation_search_api_key=None,
            tomtom_api_key="tomtom-key",
        )


def test_settings_reject_removed_osm_places_provider() -> None:
    with pytest.raises(ValidationError, match="places_search_providers"):
        Settings.model_validate(
            {
                "places_search_providers": ["osm"],
                "web_search_providers": [],
                "tomtom_api_key": "tomtom-key",
            }
        )


def test_settings_require_tomtom_key_when_enabled() -> None:
    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=["tomtom"],
            web_search_providers=[],
            tomtom_api_key=None,
        )


def test_settings_require_twogis_and_geocoder_keys_when_enabled() -> None:
    with pytest.raises(ValidationError, match="DGIS_API_KEY"):
        Settings(
            places_search_providers=["twogis"],
            web_search_providers=[],
            tomtom_api_key="tomtom-key",
            dgis_api_key=None,
        )

    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=["twogis"],
            web_search_providers=[],
            tomtom_api_key=None,
            dgis_api_key="dgis-key",
        )


def test_settings_require_tavily_key_when_provider_is_enabled() -> None:
    with pytest.raises(ValidationError, match="TAVILY_API_KEY is required"):
        Settings(
            places_search_providers=[],
            web_search_providers=["tavily"],
            tavily_api_key=None,
        )


def test_settings_require_exa_key_when_provider_is_enabled() -> None:
    with pytest.raises(ValidationError, match="EXA_API_KEY is required"):
        Settings(
            places_search_providers=[],
            web_search_providers=["exa"],
            exa_api_key=None,
        )


def test_settings_require_firecrawl_key_when_provider_is_enabled() -> None:
    with pytest.raises(ValidationError, match="FIRECRAWL_API_KEY is required"):
        Settings(
            places_search_providers=[],
            web_search_providers=["firecrawl"],
            firecrawl_api_key=None,
        )


def test_settings_require_yandex_routing_key_when_provider_is_enabled() -> None:
    with pytest.raises(ValidationError, match="YANDEX_ROUTING_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["yandex"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            yandex_routing_api_key=None,
        )


def test_settings_require_tomtom_geocoder_key_for_internal_routing_resolution() -> None:
    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["yandex"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            yandex_routing_api_key="routing-key",
            tomtom_api_key=None,
        )


def test_settings_require_graphhopper_and_geocoder_keys_when_enabled() -> None:
    with pytest.raises(ValidationError, match="GRAPHHOPPER_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["graphhopper"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            tomtom_api_key="tomtom-key",
            graphhopper_api_key=None,
        )

    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["graphhopper"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            tomtom_api_key=None,
            graphhopper_api_key="graphhopper-key",
        )


def test_settings_require_twogis_and_geocoder_keys_when_routing_is_enabled() -> None:
    with pytest.raises(ValidationError, match="DGIS_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["twogis"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            tomtom_api_key="tomtom-key",
            dgis_api_key=None,
        )

    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["twogis"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            tomtom_api_key=None,
            dgis_api_key="dgis-key",
        )


def test_settings_require_geocoder_key_when_osrm_is_enabled() -> None:
    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["osrm"],
            text_place_resolution_providers=[],
            web_search_providers=[],
            tomtom_api_key=None,
        )


def test_settings_require_tomtom_key_for_text_place_resolution() -> None:
    with pytest.raises(ValidationError, match="TOMTOM_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["osrm"],
            text_place_resolution_providers=["tomtom"],
            web_search_providers=[],
            tomtom_api_key=None,
        )


def test_settings_require_twogis_key_for_text_place_resolution() -> None:
    with pytest.raises(ValidationError, match="DGIS_API_KEY"):
        Settings(
            places_search_providers=[],
            routing_providers=["osrm"],
            text_place_resolution_providers=["twogis", "tomtom"],
            web_search_providers=[],
            tomtom_api_key="tomtom-key",
            dgis_api_key=None,
        )


def test_settings_reject_non_positive_tool_execution_timeout() -> None:
    """Verify that the complete tool-call deadline must be positive."""

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(tools_execution_timeout=0)


def test_settings_reject_non_positive_snap_warning_distance() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(routing_snap_warning_distance_m=0)


async def test_runtime_forwards_tool_execution_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that runtime settings reach the model-facing tool wrappers."""

    captured_timeout: object = None

    def fake_registry(**kwargs: object) -> MappingProxyType[str, object]:
        nonlocal captured_timeout
        captured_timeout = kwargs["execution_timeout_s"]
        return MappingProxyType({})

    monkeypatch.setattr(
        runtime_tools_module,
        "build_runtime_tool_registry",
        fake_registry,
    )

    async with httpx.AsyncClient() as http_client:
        build_runtime_tools(
            settings=Settings(tools_execution_timeout=7),
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

    assert captured_timeout == 7.0


async def test_yandex_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: PlacesSearchInput | None = None

    class FakeYandexPlacesSearchProvider:
        provider = "yandex"
        supports_open_now = True

        async def search(
            self,
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchOutput:
            nonlocal received_args
            received_args = args
            return PlacesSearchOutput()

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_yandex_places_search_provider",
        lambda **_: FakeYandexPlacesSearchProvider(),
    )

    settings = make_settings(places=["yandex"], web=[])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

        result = await runtime_tools["places_search"].run(
            {
                "mode": "area",
                "query": "кофейни",
                "area_ref": "plc_a1b2c3d4e5",
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.query == "кофейни"
    assert received_args.area_ref == "plc_a1b2c3d4e5"


async def test_tomtom_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: PlacesSearchInput | None = None

    class FakeTomTomPlacesSearchProvider:
        provider = "tomtom"
        supports_open_now = True

        async def search(
            self,
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchOutput:
            nonlocal received_args
            received_args = args
            return PlacesSearchOutput()

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_tomtom_places_search_provider",
        lambda **_: FakeTomTomPlacesSearchProvider(),
    )
    settings = make_settings(places=["tomtom"], web=[])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )
        result = await runtime_tools["places_search"].run(
            {
                "mode": "area",
                "query": "Московский вокзал",
                "area_ref": "plc_a1b2c3d4e5",
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.query == "Московский вокзал"


async def test_yandex_routing_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: RoutingInput | None = None

    class FakeYandexRoutingProvider:
        provider = "yandex"

        async def route(
            self,
            args: RoutingInput,
            context: ToolExecutionContext,
        ) -> RoutingOutput:
            nonlocal received_args
            received_args = args
            return RoutingOutput(
                mode=args.mode,
                transport=args.transport,
                route=RouteInfo(
                    length_m=1_500,
                    duration_s=420,
                    waypoint_order=[0, 1],
                ),
            )

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_yandex_routing_provider",
        lambda **_: FakeYandexRoutingProvider(),
    )

    settings = make_settings(places=[], web=[], routing=["yandex"])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

        result = await runtime_tools["routing_tool"].run(
            {
                "mode": "route",
                "transport": "walking",
                "waypoints": ["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.transport.value == "walking"
    assert received_args.waypoints == ["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"]


async def test_graphhopper_routing_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: RoutingInput | None = None

    class FakeGraphHopperRoutingProvider:
        provider = "graphhopper"

        async def route(
            self,
            args: RoutingInput,
            context: ToolExecutionContext,
        ) -> RoutingOutput:
            nonlocal received_args
            received_args = args
            return RoutingOutput(
                mode=args.mode,
                transport=args.transport,
                route=RouteInfo(
                    length_m=2_000,
                    duration_s=600,
                    waypoint_order=[0, 1],
                ),
            )

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_graphhopper_routing_provider",
        lambda **_: FakeGraphHopperRoutingProvider(),
    )
    settings = make_settings(places=[], web=[], routing=["graphhopper"])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )
        result = await runtime_tools["routing_tool"].run(
            {
                "mode": "route",
                "transport": "bicycle",
                "waypoints": ["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.transport.value == "bicycle"


async def test_places_and_routing_share_text_resolution_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing endpoints and places-search anchors use exactly one resolver."""

    resolvers: dict[str, object] = {}
    real_scope_resolver = PlacesSearchScopeResolver
    real_routing_place_resolver = RoutingPlaceResolver

    class FakePlacesProvider:
        provider = "tomtom"

        async def search(
            self,
            args: PlacesSearchInput,
            context: ToolExecutionContext,
        ) -> PlacesSearchOutput:
            return PlacesSearchOutput()

    class FakeRoutingProvider:
        provider = "yandex"

        async def route(
            self,
            args: RoutingInput,
            context: ToolExecutionContext,
        ) -> RoutingOutput:
            return RoutingOutput(
                mode=args.mode,
                transport=args.transport,
                route=RouteInfo(length_m=0, duration_s=0),
            )

    def build_places_provider(**kwargs: object) -> FakePlacesProvider:
        resolvers["places_store"] = kwargs["place_store"]
        return FakePlacesProvider()

    def build_routing_provider(**kwargs: object) -> FakeRoutingProvider:
        resolvers["routing_store"] = kwargs["place_store"]
        return FakeRoutingProvider()

    def build_scope_resolver(
        *,
        geocoded_place_resolver: GeocodedPlaceResolver,
        text_place_resolver: TextPlaceResolver,
    ) -> PlacesSearchScopeResolver:
        resolvers["scope_place"] = geocoded_place_resolver
        resolvers["scope_text"] = text_place_resolver
        return real_scope_resolver(
            geocoded_place_resolver=geocoded_place_resolver,
            text_place_resolver=text_place_resolver,
        )

    def build_routing_place_resolver(
        text_place_resolver: TextPlaceResolver,
        routing_place_store: PlaceStore,
    ) -> object:
        resolvers["routing_text"] = text_place_resolver
        resolvers["routing_resolution_store"] = routing_place_store
        return real_routing_place_resolver(text_place_resolver, routing_place_store)

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_tomtom_places_search_provider",
        build_places_provider,
    )
    monkeypatch.setattr(
        runtime_tools_module,
        "_build_yandex_routing_provider",
        build_routing_provider,
    )
    monkeypatch.setattr(
        runtime_tools_module,
        "PlacesSearchScopeResolver",
        build_scope_resolver,
    )
    monkeypatch.setattr(
        runtime_tools_module,
        "RoutingPlaceResolver",
        build_routing_place_resolver,
    )

    place_store = InMemoryPlaceStore()
    async with httpx.AsyncClient() as http_client:
        build_runtime_tools(
            settings=make_settings(
                places=["tomtom", "twogis"],
                web=[],
                routing=["yandex"],
            ),
            http_client=http_client,
            place_store=place_store,
            source_store=InMemorySourceStore(),
        )

    assert resolvers["places_store"] is place_store
    assert resolvers["routing_store"] is place_store
    assert resolvers["routing_resolution_store"] is place_store
    shared_resolver = resolvers["scope_place"]
    assert isinstance(shared_resolver, GeocodedPlaceResolver)
    text_place_resolver = resolvers["scope_text"]
    assert isinstance(text_place_resolver, TextPlaceResolver)
    assert text_place_resolver._geocoded_place_resolver is shared_resolver
    assert tuple(type(item) for item in text_place_resolver._named_poi_resolvers) == (
        TwoGisNamedPoiResolver,
        TomTomNamedPoiResolver,
    )
    routing_text_resolver = resolvers["routing_text"]
    assert isinstance(routing_text_resolver, TextPlaceResolver)
    assert routing_text_resolver is text_place_resolver
    assert tuple(type(item) for item in routing_text_resolver._named_poi_resolvers) == (
        TwoGisNamedPoiResolver,
        TomTomNamedPoiResolver,
    )
    assert routing_text_resolver._named_first is True


async def test_routing_only_uses_the_same_configured_named_first_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling places_search must not change routing text interpretation."""

    captured: dict[str, object] = {}
    real_text_place_resolver = TextPlaceResolver

    def build_text_place_resolver(
        *,
        geocoded_place_resolver: GeocodedPlaceResolver,
        named_poi_resolvers: tuple[NamedPoiResolver, ...],
        named_first: bool,
    ) -> TextPlaceResolver:
        captured.update(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=named_poi_resolvers,
            named_first=named_first,
        )
        return real_text_place_resolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=named_poi_resolvers,
            named_first=named_first,
        )

    monkeypatch.setattr(
        runtime_tools_module,
        "TextPlaceResolver",
        build_text_place_resolver,
    )

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=make_settings(
                places=[],
                web=[],
                routing=["osrm"],
                text_place_resolution=["twogis", "tomtom"],
            ),
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

    assert set(runtime_tools) == {"routing_tool"}
    assert captured["named_first"] is True
    named_poi_resolvers = captured["named_poi_resolvers"]
    assert isinstance(named_poi_resolvers, tuple)
    assert tuple(type(item) for item in named_poi_resolvers) == (
        TwoGisNamedPoiResolver,
        TomTomNamedPoiResolver,
    )


async def test_text_place_resolver_prefers_twogis_then_falls_back_to_tomtom() -> None:
    settings = make_settings(places=["tomtom", "twogis"], web=[])
    store = InMemoryPlaceStore()

    async with httpx.AsyncClient() as http_client:
        _, geocoded_place_resolver = runtime_tools_module._build_geocoded_place_dependencies(
            settings=settings,
            http_client=http_client,
            place_store=store,
        )
        assert geocoded_place_resolver is not None
        clients_by_name = runtime_tools_module._build_place_lookup_clients(
            settings=settings,
            http_client=http_client,
        )
        named_poi_resolvers_by_name = runtime_tools_module._build_named_poi_resolvers(
            clients_by_name=clients_by_name,
            place_store=store,
        )
        named_poi_resolvers = runtime_tools_module._select_named_poi_resolvers(
            resolvers_by_name=named_poi_resolvers_by_name,
            provider_names=settings.text_place_resolution_providers,
        )
        text_place_resolver = TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=named_poi_resolvers,
        )

    assert tuple(type(resolver) for resolver in text_place_resolver._named_poi_resolvers) == (
        TwoGisNamedPoiResolver,
        TomTomNamedPoiResolver,
    )


async def test_named_poi_and_search_providers_share_client_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(places=["tomtom", "twogis"], web=[])
    built_clients: dict[str, list[object]] = {"tomtom": [], "twogis": []}
    real_tomtom_builder = runtime_tools_module._build_tomtom_search_client
    real_twogis_builder = runtime_tools_module._build_twogis_search_client

    def build_tomtom_client(**kwargs: Any) -> object:
        client = real_tomtom_builder(**kwargs)
        built_clients["tomtom"].append(client)
        return client

    def build_twogis_client(**kwargs: Any) -> object:
        client = real_twogis_builder(**kwargs)
        built_clients["twogis"].append(client)
        return client

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_tomtom_search_client",
        build_tomtom_client,
    )
    monkeypatch.setattr(
        runtime_tools_module,
        "_build_twogis_search_client",
        build_twogis_client,
    )

    async with httpx.AsyncClient() as http_client:
        build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

    assert {name: len(clients) for name, clients in built_clients.items()} == {
        "tomtom": 1,
        "twogis": 1,
    }


async def test_tavily_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: WebSearchInput | None = None

    class FakeTavilyWebSearchProvider:
        provider = "tavily"

        async def search(
            self,
            args: WebSearchInput,
            context: ToolExecutionContext,
        ) -> WebSearchOutput:
            nonlocal received_args
            received_args = args
            return WebSearchOutput(
                query=args.query,
                results=[],
            )

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_tavily_web_search_provider",
        lambda **_: FakeTavilyWebSearchProvider(),
    )

    settings = make_settings(places=[], web=["tavily"])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

        result = await runtime_tools["web_search"].run(
            {
                "query": "выставки в Москве",
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.query == "выставки в Москве"


async def test_exa_provider_is_registered_and_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_args: WebSearchInput | None = None

    class FakeExaWebSearchProvider:
        provider = "exa"

        async def search(
            self,
            args: WebSearchInput,
            context: ToolExecutionContext,
        ) -> WebSearchOutput:
            nonlocal received_args
            received_args = args
            return WebSearchOutput(
                query=args.query,
                results=[],
            )

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_exa_web_search_provider",
        lambda **_: FakeExaWebSearchProvider(),
    )

    settings = make_settings(places=[], web=["exa"])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )

        result = await runtime_tools["web_search"].run(
            {
                "query": "выставки в Москве",
            }
        )

    assert result.ok is True
    assert received_args is not None
    assert received_args.query == "выставки в Москве"


async def test_exa_has_priority_when_both_web_providers_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_providers: list[str] = []

    class FakeWebSearchProvider:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        async def search(
            self,
            args: WebSearchInput,
            context: ToolExecutionContext,
        ) -> WebSearchOutput:
            called_providers.append(self.provider)
            return WebSearchOutput(query=args.query)

    monkeypatch.setattr(
        runtime_tools_module,
        "_build_exa_web_search_provider",
        lambda **_: FakeWebSearchProvider("exa"),
    )
    monkeypatch.setattr(
        runtime_tools_module,
        "_build_tavily_web_search_provider",
        lambda **_: FakeWebSearchProvider("tavily"),
    )

    settings = make_settings(places=[], web=["tavily", "exa"])

    async with httpx.AsyncClient() as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )
        result = await runtime_tools["web_search"].run({"query": "выставки в Москве"})

    assert result.ok is True
    assert called_providers == ["exa"]


@pytest.mark.parametrize(
    ("settings", "missing_key"),
    [
        (
            Settings.model_construct(
                places_search_providers=["yandex"],
                web_search_providers=[],
                yandex_organisation_search_api_key="search-key",
                tomtom_api_key=None,
            ),
            "TOMTOM_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=["yandex"],
                web_search_providers=[],
                yandex_organisation_search_api_key=None,
                tomtom_api_key="tomtom-key",
            ),
            "YANDEX_ORGANISATION_SEARCH_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=[],
                routing_providers=["yandex"],
                text_place_resolution_providers=[],
                web_search_providers=[],
                yandex_routing_api_key=None,
                tomtom_api_key="tomtom-key",
            ),
            "YANDEX_ROUTING_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=[],
                routing_providers=["yandex"],
                text_place_resolution_providers=[],
                web_search_providers=[],
                yandex_routing_api_key="routing-key",
                tomtom_api_key=None,
            ),
            "TOMTOM_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=[],
                web_search_providers=["tavily"],
                tavily_api_key=None,
            ),
            "TAVILY_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=[],
                web_search_providers=["exa"],
                exa_api_key=None,
            ),
            "EXA_API_KEY",
        ),
        (
            Settings.model_construct(
                places_search_providers=[],
                web_search_providers=["firecrawl"],
                firecrawl_api_key=None,
            ),
            "FIRECRAWL_API_KEY",
        ),
    ],
)
async def test_runtime_factory_rejects_missing_provider_key(
    settings: Settings,
    missing_key: str,
) -> None:
    async with httpx.AsyncClient() as http_client:
        with pytest.raises(
            ValueError,
            match=missing_key,
        ):
            build_runtime_tools(
                settings=settings,
                http_client=http_client,
                place_store=InMemoryPlaceStore(),
                source_store=InMemorySourceStore(),
            )
