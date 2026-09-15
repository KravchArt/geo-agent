"""Prepare places-search scopes and load their hidden place records."""

from __future__ import annotations

from dataclasses import dataclass

from tools.base import ToolClarification, ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding import GeocodedPlaceResolver
from tools.geo.geocoding.schemas import ToponymKind
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.provider import (
    AmbiguousSearchAnchorError,
    AmbiguousSearchAreaError,
    AnchorNotFoundError,
    InvalidSearchAreaRefError,
    SearchAreaNotFoundError,
    UnknownPlaceRefError,
    UnknownSearchAreaRefError,
)
from tools.geo.places_search.schemas import PlacesSearchInput, SearchMode
from tools.geo.text_place_query import same_locality_anchor, uses_cyrillic
from tools.geo.text_place_resolution import TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, PlaceRef


@dataclass(frozen=True, slots=True)
class ResolvedArea:
    """A validated locality with the bounds required by search providers."""

    ref: PlaceRef
    name: str
    address: str
    lat: float
    lon: float
    bounds: GeoBounds
    localized_localities: dict[str, str]


class PlacesSearchScopeLoader:
    """Load coordinator-prepared refs for concrete search providers.

    Providers only consume resolved records. They do not need access to the
    geocoder or named-POI fallback policy that produced those refs.
    """

    def __init__(self, *, place_store: PlaceStore) -> None:
        self._place_store = place_store

    async def load_anchor(self, ref: PlaceRef) -> PlaceRecord:
        """Load a nearby-search anchor from the shared place store."""

        record = await self._place_store.get(ref)
        if record is None:
            raise UnknownPlaceRefError(ref)
        return record

    async def load_area(self, ref: PlaceRef) -> ResolvedArea:
        """Load and validate a bounded locality for an area search."""

        record = await self._place_store.get(ref)
        if record is None:
            raise UnknownSearchAreaRefError(ref)
        if record.kind != ToponymKind.LOCALITY.value or record.bounds is None:
            raise InvalidSearchAreaRefError(ref)

        return ResolvedArea(
            ref=ref,
            name=record.name,
            address=record.address,
            lat=record.lat,
            lon=record.lon,
            bounds=record.bounds,
            localized_localities=dict(record.localized_localities),
        )


class PlacesSearchScopeResolver:
    """Resolve model-facing area and nearby text into provider-ready refs."""

    def __init__(
        self,
        *,
        geocoded_place_resolver: GeocodedPlaceResolver,
        text_place_resolver: TextPlaceResolver,
    ) -> None:
        self._geocoded_place_resolver = geocoded_place_resolver
        self._text_place_resolver = text_place_resolver

    async def resolve_scope(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchInput:
        """Resolve textual scope once and replace it with a reusable place ref.

        The coordinator calls this before trying concrete search providers. Each
        provider can then load the same hidden coordinates from ``PlaceStore``
        without repeating geocoding or named-POI fallback after a provider
        error.
        """

        if args.mode is SearchMode.AREA:
            if args.area_ref is not None:
                if args.query is not None and uses_cyrillic(args.query):
                    await self._geocoded_place_resolver.localize_bounded_locality_ref(
                        args.area_ref,
                        language="ru-RU",
                        context=context,
                    )
                return args
            if args.city is None:
                raise self._internal_contract_error(
                    "Places search input lost both city and area_ref"
                )
            resolved_area_ref = await self._resolve_area(
                args.city,
                query=args.query,
                context=context,
            )
            return args.model_copy(
                update={
                    "area": resolved_area_ref,
                }
            )

        if args.near_ref is not None:
            return args
        if args.near_query is None:
            raise self._internal_contract_error("Places search input lost its required near anchor")

        area_ref = args.area_ref
        city = args.city
        if area_ref is not None:
            resolved_area = await self._geocoded_place_resolver.load_bounded_locality_ref(area_ref)
            if resolved_area is None:
                raise InvalidSearchAreaRefError(area_ref)
            city = resolved_area.record.name

        if city is not None and same_locality_anchor(args.near_query, city):
            anchor_ref = await self._resolve_locality_anchor(args.near_query, context)
            if area_ref is None:
                # The anchor itself is the bounded locality, so no second city
                # geocoder call is needed merely to mint the reusable scope.
                area_ref = anchor_ref
        else:
            anchor_ref = await self._resolve_anchor(
                query=args.near_query,
                city=city,
                context=context,
            )
            if area_ref is None and city is not None:
                # Resolve a portable bounded city ref after the anchor. Anchor
                # errors must retain their select_anchor semantics; failure to
                # materialize the reusable scope may degrade to the historical
                # textual-city path without invalidating a valid nearby search.
                try:
                    area_ref = await self._resolve_area(city, query=args.query, context=context)
                except ToolExecutionError:
                    area_ref = None
        return args.model_copy(
            update={
                "area": area_ref if area_ref is not None else city,
                "near": anchor_ref,
            }
        )

    async def _resolve_anchor(
        self,
        *,
        query: str,
        city: str | None,
        context: ToolExecutionContext,
    ) -> PlaceRef:
        """Resolve a textual anchor through the places-search-only policy."""

        try:
            resolved = await self._text_place_resolver.resolve(
                query=query,
                city=city,
                context=context,
            )
        except AmbiguousPlaceError as exc:
            raise AmbiguousSearchAnchorError(
                query,
                clarification=self._as_anchor_clarification(exc.clarification),
            ) from exc
        if resolved is None:
            raise AnchorNotFoundError(query)

        return resolved.ref

    async def _resolve_locality_anchor(
        self,
        query: str,
        context: ToolExecutionContext,
    ) -> PlaceRef:
        """Resolve a city used as the nearby anchor without named-POI fallback."""

        try:
            resolved = await self._geocoded_place_resolver.geocode_bounded_locality(
                query,
                context,
            )
        except AmbiguousPlaceError as exc:
            raise AmbiguousSearchAnchorError(
                query,
                clarification=self._as_anchor_clarification(exc.clarification),
            ) from exc
        if resolved is None:
            raise AnchorNotFoundError(query)
        return resolved.ref

    async def _resolve_area(
        self,
        city: str,
        *,
        query: str | None,
        context: ToolExecutionContext,
    ) -> PlaceRef:
        """Resolve a city into a bounded locality while preparing provider input."""

        try:
            resolved = await self._geocoded_place_resolver.geocode_bounded_locality(city, context)
        except AmbiguousPlaceError as exc:
            raise AmbiguousSearchAreaError(
                city,
                clarification=exc.clarification,
            ) from exc
        if resolved is None:
            raise SearchAreaNotFoundError(city)
        if resolved.record.bounds is None:
            raise self._internal_contract_error(
                "Place resolver returned a bounded locality without bounds"
            )

        # TomTom localizes POI municipality fields to the search language. If
        # a Russian request contains an English city exonym (Vienna), cache the
        # Russian municipality label (Вена) from the already resolved city's
        # center. The provider can then keep its strict bbox + municipality
        # filter instead of admitting neighbouring cities inside the bbox.
        if query is not None and uses_cyrillic(query) and not uses_cyrillic(city):
            resolved = await self._geocoded_place_resolver.localize_bounded_locality(
                resolved,
                language="ru-RU",
                context=context,
            )

        return resolved.ref

    @staticmethod
    def _as_anchor_clarification(
        clarification: ToolClarification | None,
    ) -> ToolClarification | None:
        """Retarget locality choices when the city is a nearby-search anchor."""

        if clarification is None or clarification.kind == "select_anchor":
            return clarification
        return clarification.model_copy(update={"kind": "select_anchor"})

    def _internal_contract_error(self, message: str) -> ToolExecutionError:
        return ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=self._geocoded_place_resolver.provider,
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        )
