"""Resolve arbitrary place text through provider search and geocoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding import GeocodedPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRef, ResolvedPlace


@dataclass(frozen=True, slots=True)
class PlaceResolutionArea:
    """Resolved locality scope shared by named-place providers."""

    name: str
    ref: PlaceRef | None = None
    lat: float | None = None
    lon: float | None = None
    bounds: GeoBounds | None = None


class NamedPoiResolver(Protocol):
    """Provider-backed lookup used only for named places and organisations."""

    provider: str
    requires_city_bounds: bool

    async def resolve_named_poi(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Resolve one named POI inside a known city, or return ``None``."""
        ...


@runtime_checkable
class FirstAddressNamedPoiResolver(Protocol):
    """Optional best-effort lookup used by routing endpoints."""

    async def resolve_first_address(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Return the provider-ranked first routable card with an address."""
        ...


class TextPlaceResolver:
    """Compose geocoding with an ordered place-search fallback chain."""

    def __init__(
        self,
        *,
        geocoded_place_resolver: GeocodedPlaceResolver,
        named_poi_resolvers: Sequence[NamedPoiResolver] = (),
        named_first: bool = False,
    ) -> None:
        self._geocoded_place_resolver = geocoded_place_resolver
        self._named_poi_resolvers = tuple(named_poi_resolvers)
        self._named_first = named_first

    async def resolve(
        self,
        *,
        query: str,
        city: str | None,
        context: ToolExecutionContext,
        area: PlaceResolutionArea | None = None,
        first_with_address: bool = False,
    ) -> ResolvedPlace | None:
        """Resolve text using the configured named-place/geocoder order."""

        if self._named_first and city is not None:
            named_place = await self._resolve_named_place(
                query=query,
                city=city,
                context=context,
                allow_geocoder_fallback=True,
                area=area,
                first_with_address=first_with_address,
            )
            if named_place is not None:
                return named_place

        geocoder_ambiguity: AmbiguousPlaceError | None = None
        try:
            resolved = await self._geocoded_place_resolver.geocode_place(
                query=query,
                city=city,
                context=context,
            )
        except AmbiguousPlaceError as exc:
            geocoder_ambiguity = exc
            resolved = None

        if resolved is not None:
            return resolved

        if not self._named_first and city is not None:
            named_place = await self._resolve_named_place(
                query=query,
                city=city,
                context=context,
                allow_geocoder_fallback=False,
                area=area,
                first_with_address=first_with_address,
            )
            if named_place is not None:
                return named_place

        if geocoder_ambiguity is not None:
            raise geocoder_ambiguity
        return None

    async def _resolve_named_place(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        allow_geocoder_fallback: bool,
        area: PlaceResolutionArea | None,
        first_with_address: bool,
    ) -> ResolvedPlace | None:
        """Try named providers in order, optionally falling back to geocoding."""

        city_bounds = area.bounds if area is not None else None
        city_bounds_resolved = city_bounds is not None
        for index, named_poi_resolver in enumerate(self._named_poi_resolvers):
            if named_poi_resolver.requires_city_bounds and not city_bounds_resolved:
                city_bounds = await self._resolve_city_bounds(city, context)
                city_bounds_resolved = True
            try:
                if first_with_address and isinstance(
                    named_poi_resolver,
                    FirstAddressNamedPoiResolver,
                ):
                    named_poi = await named_poi_resolver.resolve_first_address(
                        query=query,
                        city=city,
                        context=context,
                        city_bounds=city_bounds,
                        area=area,
                    )
                elif area is None:
                    named_poi = await named_poi_resolver.resolve_named_poi(
                        query=query,
                        city=city,
                        context=context,
                        city_bounds=city_bounds,
                    )
                else:
                    named_poi = await named_poi_resolver.resolve_named_poi(
                        query=query,
                        city=city,
                        context=context,
                        city_bounds=city_bounds,
                        area=area,
                    )
            except ToolExecutionError as exc:
                can_fallback = _can_try_next_named_provider(
                    exc,
                    provider=named_poi_resolver.provider,
                )
                if not can_fallback:
                    raise
                is_last = index == len(self._named_poi_resolvers) - 1
                if not is_last:
                    next_provider = self._named_poi_resolvers[index + 1]
                    context.add_warning(
                        f"{named_poi_resolver.provider} named-place lookup failed; "
                        f"retrying with {next_provider.provider}."
                    )
                    continue
                if allow_geocoder_fallback:
                    context.add_warning(
                        f"{named_poi_resolver.provider} named-place lookup failed; "
                        "retrying with geocoder."
                    )
                    return None
                raise
            if named_poi is not None:
                return named_poi
        return None

    async def _resolve_city_bounds(
        self,
        city: str,
        context: ToolExecutionContext,
    ) -> GeoBounds | None:
        """Resolve bounds lazily for providers that need geographic scoping."""

        try:
            resolved = await self._geocoded_place_resolver.geocode_bounded_locality(
                city,
                context,
            )
        except AmbiguousPlaceError:
            return None
        if resolved is None:
            return None
        return resolved.record.bounds


def _can_try_next_named_provider(error: ToolExecutionError, *, provider: str) -> bool:
    """Mirror coordinator fallback rules for the named-place provider chain."""

    if error.provider is None or not (
        error.provider == provider or error.provider.startswith(f"{provider}_")
    ):
        return False
    if error.status_code in {401, 403} or error.failure_kind in {
        ToolFailureKind.AUTHENTICATION,
        ToolFailureKind.INTERNAL_CONTRACT,
    }:
        return False
    if error.error_code is ToolErrorCode.NOT_FOUND or error.failure_kind in {
        ToolFailureKind.INVALID_JSON,
        ToolFailureKind.INVALID_SCHEMA,
    }:
        return True
    return error.retryable and (
        error.error_code in {ToolErrorCode.RATE_LIMITED, ToolErrorCode.TIMEOUT}
        or error.failure_kind in {ToolFailureKind.NETWORK, ToolFailureKind.TIMEOUT}
        or (
            error.failure_kind is ToolFailureKind.HTTP_STATUS
            and error.status_code is not None
            and (error.status_code == 429 or error.status_code >= 500)
        )
    )
