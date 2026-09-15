"""Runtime assembly of model-facing tools from configured providers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import MappingProxyType
from typing import Any, TypeAlias

import httpx

from backend.app.config import Settings
from tools.base import Tool, ToolHandler
from tools.geo import PlacesSearchInput, PlacesSearchOutput, RoutingInput, RoutingOutput
from tools.geo.geocoding import GeocodedPlaceResolver, GeocoderService
from tools.geo.geocoding.tomtom import TomTomGeocoderClient, TomTomGeocoderService
from tools.geo.place_store import PlaceStore
from tools.geo.places_search import (
    PlacesSearchCoordinator,
    PlacesSearchProvider,
    PlacesSearchScopeResolver,
)
from tools.geo.places_search.tomtom import (
    TomTomNamedPoiResolver,
    TomTomPlacesSearchProvider,
    TomTomSearchClient,
)
from tools.geo.places_search.twogis import (
    TwoGisNamedPoiResolver,
    TwoGisPlacesSearchProvider,
    TwoGisSearchClient,
)
from tools.geo.places_search.yandex.client import YandexOrganisationSearchClient
from tools.geo.places_search.yandex.provider import YandexPlacesSearchProvider
from tools.geo.routing import RoutingCoordinator, RoutingPlaceResolver, RoutingProvider
from tools.geo.routing.graphhopper import (
    GraphHopperRoutingClient,
    GraphHopperRoutingProvider,
)
from tools.geo.routing.osrm import OsrmRoutingClient, OsrmRoutingProvider
from tools.geo.routing.twogis import TwoGisRoutingClient, TwoGisRoutingProvider
from tools.geo.routing.yandex import YandexRoutingClient, YandexRoutingProvider
from tools.geo.text_place_resolution import NamedPoiResolver, TextPlaceResolver
from tools.registry import build_runtime_tool_registry
from tools.web import (
    SourceStore,
    WebSearchCoordinator,
    WebSearchInput,
    WebSearchOutput,
    WebSearchProvider,
)
from tools.web.exa import ExaSearchClient, ExaWebSearchProvider
from tools.web.firecrawl import FirecrawlSearchClient, FirecrawlWebSearchProvider
from tools.web.tavily import TavilySearchClient, TavilyWebSearchProvider

WEB_SEARCH_PROVIDER_PRIORITY = ("exa", "tavily", "firecrawl")
PlaceLookupClient: TypeAlias = (
    YandexOrganisationSearchClient | TomTomSearchClient | TwoGisSearchClient
)


def build_runtime_tools(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
    source_store: SourceStore,
) -> MappingProxyType[str, Tool[Any, Any]]:
    """Build only tools backed by configured runtime providers."""

    _, geocoded_place_resolver = _build_geocoded_place_dependencies(
        settings=settings,
        http_client=http_client,
        place_store=place_store,
    )
    place_clients_by_name = _build_place_lookup_clients(
        settings=settings,
        http_client=http_client,
    )
    named_poi_resolvers_by_name = _build_named_poi_resolvers(
        clients_by_name=place_clients_by_name,
        place_store=place_store,
    )
    named_poi_resolvers = _select_named_poi_resolvers(
        resolvers_by_name=named_poi_resolvers_by_name,
        provider_names=settings.text_place_resolution_providers,
    )
    text_place_resolver = (
        TextPlaceResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            named_poi_resolvers=named_poi_resolvers,
            named_first=True,
        )
        if geocoded_place_resolver is not None
        else None
    )
    places_search_providers_by_name = _build_places_search_providers(
        settings=settings,
        clients_by_name=place_clients_by_name,
        place_store=place_store,
    )
    routing_providers_by_name = _build_routing_providers(
        settings=settings,
        http_client=http_client,
        place_store=place_store,
    )
    web_search_providers_by_name = _build_web_search_providers(
        settings=settings,
        http_client=http_client,
        source_store=source_store,
    )

    return build_runtime_tool_registry(
        places_search_handler=_build_places_search_handler(
            providers=tuple(
                places_search_providers_by_name[name]
                for name in settings.places_search_providers
                if name in places_search_providers_by_name
            ),
            geocoded_place_resolver=geocoded_place_resolver,
            text_place_resolver=text_place_resolver,
        ),
        routing_handler=_build_routing_handler(
            providers=tuple(
                routing_providers_by_name[name]
                for name in settings.routing_providers
                if name in routing_providers_by_name
            ),
            text_place_resolver=text_place_resolver,
            place_store=place_store,
        ),
        web_search_handler=_build_web_search_handler(
            providers=tuple(
                web_search_providers_by_name[name]
                for name in sorted(
                    settings.web_search_providers,
                    key=WEB_SEARCH_PROVIDER_PRIORITY.index,
                )
                if name in web_search_providers_by_name
            ),
        ),
        execution_timeout_s=float(settings.tools_execution_timeout),
    )


def _build_places_search_handler(
    *,
    providers: Sequence[PlacesSearchProvider],
    geocoded_place_resolver: GeocodedPlaceResolver | None,
    text_place_resolver: TextPlaceResolver | None,
) -> ToolHandler[PlacesSearchInput, PlacesSearchOutput] | None:
    if not providers:
        return None
    if geocoded_place_resolver is None:
        raise RuntimeError("Places search scope-resolution dependencies were not built")
    if text_place_resolver is None:
        raise RuntimeError("Places search text-resolution dependency was not built")

    coordinator = PlacesSearchCoordinator(
        providers=providers,
        scope_resolver=PlacesSearchScopeResolver(
            geocoded_place_resolver=geocoded_place_resolver,
            text_place_resolver=text_place_resolver,
        ),
    )
    return coordinator.search


def _build_place_lookup_clients(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> dict[str, PlaceLookupClient]:
    """Build clients needed by places search or routing-point resolution."""

    clients: dict[str, PlaceLookupClient] = {}
    if "yandex" in settings.places_search_providers:
        clients["yandex"] = YandexOrganisationSearchClient(
            api_key=_require_key(
                settings.yandex_organisation_search_api_key,
                "YANDEX_ORGANISATION_SEARCH_API_KEY",
            ),
            http_client=http_client,
        )
    needs_text_resolution = bool(settings.places_search_providers or settings.routing_providers)
    if "tomtom" in settings.places_search_providers or (
        needs_text_resolution and "tomtom" in settings.text_place_resolution_providers
    ):
        clients["tomtom"] = _build_tomtom_search_client(
            settings=settings,
            http_client=http_client,
        )
    if "twogis" in settings.places_search_providers or (
        needs_text_resolution and "twogis" in settings.text_place_resolution_providers
    ):
        clients["twogis"] = _build_twogis_search_client(
            settings=settings,
            http_client=http_client,
        )
    return clients


def _build_places_search_providers(
    *,
    settings: Settings,
    clients_by_name: dict[str, PlaceLookupClient],
    place_store: PlaceStore,
) -> dict[str, PlacesSearchProvider]:
    """Build all configured places-search providers."""

    providers: dict[str, PlacesSearchProvider] = {}

    if "yandex" in settings.places_search_providers:
        client = clients_by_name.get("yandex")
        if not isinstance(client, YandexOrganisationSearchClient):
            raise RuntimeError("Yandex places search client was not built")
        providers["yandex"] = _build_yandex_places_search_provider(
            client=client,
            place_store=place_store,
        )

    if "tomtom" in settings.places_search_providers:
        client = clients_by_name.get("tomtom")
        if not isinstance(client, TomTomSearchClient):
            raise RuntimeError("TomTom places search client was not built")
        providers["tomtom"] = _build_tomtom_places_search_provider(
            client=client,
            place_store=place_store,
        )

    if "twogis" in settings.places_search_providers:
        client = clients_by_name.get("twogis")
        if not isinstance(client, TwoGisSearchClient):
            raise RuntimeError("2GIS search client was not built")
        providers["twogis"] = _build_twogis_places_search_provider(
            client=client,
            place_store=place_store,
        )

    return providers


def _build_yandex_places_search_provider(
    *,
    client: YandexOrganisationSearchClient,
    place_store: PlaceStore,
) -> PlacesSearchProvider:
    return YandexPlacesSearchProvider(
        client=client,
        place_store=place_store,
    )


def _build_tomtom_places_search_provider(
    *,
    client: TomTomSearchClient,
    place_store: PlaceStore,
) -> PlacesSearchProvider:
    return TomTomPlacesSearchProvider(
        client=client,
        place_store=place_store,
    )


def _build_twogis_places_search_provider(
    *,
    client: TwoGisSearchClient,
    place_store: PlaceStore,
) -> PlacesSearchProvider:
    return TwoGisPlacesSearchProvider(
        client=client,
        place_store=place_store,
    )


def _build_named_poi_resolvers(
    *,
    clients_by_name: dict[str, PlaceLookupClient],
    place_store: PlaceStore,
) -> dict[str, NamedPoiResolver]:
    """Build each configured named-POI adapter exactly once."""

    named_poi_resolvers: dict[str, NamedPoiResolver] = {}
    twogis_client = clients_by_name.get("twogis")
    if twogis_client is not None:
        if not isinstance(twogis_client, TwoGisSearchClient):
            raise RuntimeError("Unexpected client registered for 2GIS")
        named_poi_resolvers["twogis"] = TwoGisNamedPoiResolver(
            client=twogis_client,
            place_store=place_store,
        )
    tomtom_client = clients_by_name.get("tomtom")
    if tomtom_client is not None:
        if not isinstance(tomtom_client, TomTomSearchClient):
            raise RuntimeError("Unexpected client registered for TomTom")
        named_poi_resolvers["tomtom"] = TomTomNamedPoiResolver(
            client=tomtom_client,
            place_store=place_store,
        )

    return named_poi_resolvers


def _select_named_poi_resolvers(
    *,
    resolvers_by_name: dict[str, NamedPoiResolver],
    provider_names: Iterable[str],
) -> tuple[NamedPoiResolver, ...]:
    """Select one ordered fallback chain from the shared adapter registry."""

    return tuple(resolvers_by_name[name] for name in provider_names if name in resolvers_by_name)


def _build_routing_handler(
    *,
    providers: Sequence[RoutingProvider],
    text_place_resolver: TextPlaceResolver | None,
    place_store: PlaceStore,
) -> ToolHandler[RoutingInput, RoutingOutput] | None:
    if not providers:
        return None
    if text_place_resolver is None:
        raise RuntimeError("Routing place-resolution dependencies were not built")

    coordinator = RoutingCoordinator(
        providers=providers,
        input_resolver=RoutingPlaceResolver(text_place_resolver, place_store),
    )
    return coordinator.route


def _build_routing_providers(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> dict[str, RoutingProvider]:
    providers: dict[str, RoutingProvider] = {}

    if "yandex" in settings.routing_providers:
        providers["yandex"] = _build_yandex_routing_provider(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

    if "graphhopper" in settings.routing_providers:
        providers["graphhopper"] = _build_graphhopper_routing_provider(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

    if "osrm" in settings.routing_providers:
        providers["osrm"] = _build_osrm_routing_provider(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

    if "twogis" in settings.routing_providers:
        providers["twogis"] = _build_twogis_routing_provider(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

    return providers


def _build_yandex_routing_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> RoutingProvider:
    api_key = _require_key(
        settings.yandex_routing_api_key,
        "YANDEX_ROUTING_API_KEY",
    )
    return YandexRoutingProvider(
        client=YandexRoutingClient(
            api_key=api_key,
            http_client=http_client,
        ),
        place_store=place_store,
    )


def _build_graphhopper_routing_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> RoutingProvider:
    api_key = _require_key(
        settings.graphhopper_api_key,
        "GRAPHHOPPER_API_KEY",
    )
    return GraphHopperRoutingProvider(
        client=GraphHopperRoutingClient(
            api_key=api_key,
            http_client=http_client,
            base_url=settings.graphhopper_base_url,
        ),
        place_store=place_store,
        snap_warning_distance_m=settings.routing_snap_warning_distance_m,
    )


def _build_osrm_routing_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> RoutingProvider:
    return OsrmRoutingProvider(
        client=OsrmRoutingClient(
            http_client=http_client,
            base_url=settings.osrm_base_url,
            walking_base_url=settings.osrm_walking_base_url,
            user_agent=settings.osrm_user_agent,
        ),
        place_store=place_store,
        snap_warning_distance_m=settings.routing_snap_warning_distance_m,
    )


def _build_twogis_routing_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> RoutingProvider:
    return TwoGisRoutingProvider(
        client=TwoGisRoutingClient(
            api_key=_require_key(settings.dgis_api_key, "DGIS_API_KEY"),
            http_client=http_client,
            base_url=settings.dgis_routing_base_url,
        ),
        place_store=place_store,
    )


def _build_geocoded_place_dependencies(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> tuple[
    GeocoderService | None,
    GeocodedPlaceResolver | None,
]:
    """Build one shared geocoder-backed resolver for model-facing geo tools."""

    needs_places_resolution = bool(settings.places_search_providers)
    needs_routing_resolution = bool(settings.routing_providers)
    if not needs_places_resolution and not needs_routing_resolution:
        return None, None

    geocoder = _build_tomtom_geocoder_service(
        settings=settings,
        http_client=http_client,
        place_store=place_store,
    )
    geocoded_place_resolver = GeocodedPlaceResolver(
        geocoder=geocoder,
        place_store=place_store,
        allowed_country_codes=settings.geocoding_allowed_country_codes,
    )
    return geocoder, geocoded_place_resolver


def _build_tomtom_geocoder_service(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> GeocoderService:
    return TomTomGeocoderService(
        client=TomTomGeocoderClient(
            api_key=_require_key(settings.tomtom_api_key, "TOMTOM_API_KEY"),
            http_client=http_client,
            base_url=settings.tomtom_search_base_url,
        ),
        place_store=place_store,
    )


def _build_tomtom_search_client(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> TomTomSearchClient:
    """Build the shared TomTom HTTP adapter configuration consistently."""

    return TomTomSearchClient(
        api_key=_require_key(settings.tomtom_api_key, "TOMTOM_API_KEY"),
        http_client=http_client,
        base_url=settings.tomtom_search_base_url,
    )


def _build_twogis_search_client(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> TwoGisSearchClient:
    return TwoGisSearchClient(
        api_key=_require_key(settings.dgis_api_key, "DGIS_API_KEY"),
        http_client=http_client,
        base_url=settings.dgis_catalog_base_url,
        timeout_s=float(settings.dgis_catalog_timeout),
    )


def _build_web_search_handler(
    *,
    providers: Sequence[WebSearchProvider],
) -> ToolHandler[WebSearchInput, WebSearchOutput] | None:
    if not providers:
        return None

    coordinator = WebSearchCoordinator(providers=providers)
    return coordinator.search


def _build_web_search_providers(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    source_store: SourceStore,
) -> dict[str, WebSearchProvider]:
    """Build all configured web-search providers."""

    providers: dict[str, WebSearchProvider] = {}

    if "exa" in settings.web_search_providers:
        providers["exa"] = _build_exa_web_search_provider(
            settings=settings,
            http_client=http_client,
            source_store=source_store,
        )

    if "tavily" in settings.web_search_providers:
        providers["tavily"] = _build_tavily_web_search_provider(
            settings=settings,
            http_client=http_client,
            source_store=source_store,
        )

    if "firecrawl" in settings.web_search_providers:
        providers["firecrawl"] = _build_firecrawl_web_search_provider(
            settings=settings,
            http_client=http_client,
            source_store=source_store,
        )

    return providers


def _build_tavily_web_search_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    source_store: SourceStore,
) -> WebSearchProvider:
    tavily_api_key = _require_key(
        settings.tavily_api_key,
        "TAVILY_API_KEY",
    )

    return TavilyWebSearchProvider(
        client=TavilySearchClient(
            api_key=tavily_api_key,
            http_client=http_client,
        ),
        source_store=source_store,
    )


def _build_exa_web_search_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    source_store: SourceStore,
) -> WebSearchProvider:
    exa_api_key = _require_key(
        settings.exa_api_key,
        "EXA_API_KEY",
    )

    return ExaWebSearchProvider(
        client=ExaSearchClient(
            api_key=exa_api_key,
            http_client=http_client,
        ),
        source_store=source_store,
    )


def _build_firecrawl_web_search_provider(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    source_store: SourceStore,
) -> WebSearchProvider:
    firecrawl_api_key = _require_key(
        settings.firecrawl_api_key,
        "FIRECRAWL_API_KEY",
    )

    return FirecrawlWebSearchProvider(
        client=FirecrawlSearchClient(
            api_key=firecrawl_api_key,
            http_client=http_client,
        ),
        source_store=source_store,
    )


def _require_key(value: str | None, name: str) -> str:
    """Defend factory use outside Settings validation."""

    if not value:
        raise ValueError(f"{name} is required for the selected provider")

    return value
