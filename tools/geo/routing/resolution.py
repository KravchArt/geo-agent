"""Prepare routing inputs and load their hidden place records."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence

from tools.base import ToolErrorCode, ToolExecutionError
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding.schemas import ToponymKind
from tools.geo.place_store import PlaceStore
from tools.geo.routing.schemas import (
    RoutingInput,
    RoutingMode,
    RoutingPlace,
    RoutingPlaceQuery,
)
from tools.geo.text_place_resolution import PlaceResolutionArea, TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import PLACE_REF_PATTERN, PlaceRef, ResolvedPlace

_MAX_CONCURRENT_GEOCODES = 5


class UnknownRoutingPlaceRefError(ToolExecutionError):
    """A place ref required for routing is absent from the place store."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.UNKNOWN_REF,
            f"Place ref {ref} was not found. "
            "Pass a structured {query, area} point or a fresh ref from places_search.",
        )


class UnknownRoutingAreaRefError(ToolExecutionError):
    """A locality ref used as a routing scope is absent from the place store."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.UNKNOWN_REF,
            f"Routing area ref {ref} was not found. Pass a locality name or a fresh area ref.",
        )


class InvalidRoutingAreaRefError(ToolExecutionError):
    """A routing scope ref does not describe a bounded locality."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.INVALID_INPUT,
            f"Routing area ref {ref} must point to a bounded locality.",
        )


class RoutingPlaceNotFoundError(ToolExecutionError):
    """Shared place resolution found no place for a routing text query."""

    def __init__(self, query: str) -> None:
        super().__init__(
            ToolErrorCode.NOT_FOUND,
            f"Could not resolve routing point: {query!r}",
        )


class RoutingPlaceResolver:
    """Resolve textual routing points into reusable place refs."""

    def __init__(
        self,
        text_place_resolver: TextPlaceResolver,
        place_store: PlaceStore,
    ) -> None:
        self._text_place_resolver = text_place_resolver
        self._place_store = place_store
        self._geocoding_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GEOCODES)

    async def resolve_to_refs(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingInput:
        """Replace every textual point once and preserve existing refs unchanged.

        The routing coordinator calls this before trying concrete providers. A
        provider fallback therefore reloads only hidden records by ref instead
        of repeating geocoding or named-POI resolution.
        """

        if args.mode is RoutingMode.ROUTE:
            waypoints = await self._replace_text_places(args.waypoints, context)
            if waypoints == args.waypoints:
                return args
            return args.model_copy(update={"waypoints": waypoints})

        points = [*args.origins, *args.candidates]
        point_refs = await self._replace_text_places(points, context)
        if point_refs == points:
            return args

        origin_count = len(args.origins)
        return args.model_copy(
            update={
                "origins": point_refs[:origin_count],
                "candidates": point_refs[origin_count:],
            }
        )

    async def _resolve_text_place(
        self,
        place: RoutingPlaceQuery,
        context: ToolExecutionContext,
        areas_by_ref: dict[PlaceRef, PlaceResolutionArea],
    ) -> ResolvedPlace:
        area = self._area_for_place(place, areas_by_ref)
        async with self._geocoding_semaphore:
            try:
                resolved = await self._text_place_resolver.resolve(
                    query=place.query,
                    city=area.name,
                    context=context,
                    area=area if area.ref is not None else None,
                    first_with_address=True,
                )
            except AmbiguousPlaceError as exc:
                resolved = await self._select_first_ambiguous_place(place, exc, context)
                if resolved is None:
                    raise
        if resolved is None:
            raise RoutingPlaceNotFoundError(place.query)
        return resolved

    async def _select_first_ambiguous_place(
        self,
        place: RoutingPlaceQuery,
        error: AmbiguousPlaceError,
        context: ToolExecutionContext,
    ) -> ResolvedPlace | None:
        """Load the provider-ranked first candidate for routing only."""

        clarification = error.clarification
        if clarification is None or not clarification.options:
            return None

        first = clarification.options[0]
        if re.fullmatch(PLACE_REF_PATTERN, first.value) is None:
            return None

        record = await self._place_store.get(first.value)
        if record is None:
            return None

        context.add_warning(
            f"Routing point {place.query!r} matched multiple places; using the "
            f"provider-ranked first result {first.label!r}."
        )
        return ResolvedPlace(ref=record.ref, record=record)

    async def _replace_text_places(
        self,
        points: Sequence[RoutingPlace],
        context: ToolExecutionContext,
    ) -> list[RoutingPlace]:
        unique_text_places: dict[tuple[str, str, str], RoutingPlaceQuery] = {}
        for place in points:
            if not isinstance(place, str):
                unique_text_places.setdefault(self._place_key(place), place)

        if not unique_text_places:
            return list(points)

        areas_by_ref = await self._load_area_refs(unique_text_places.values())

        resolved: list[ResolvedPlace] = []
        text_places = list(unique_text_places.values())
        for start in range(0, len(text_places), _MAX_CONCURRENT_GEOCODES):
            batch = text_places[start : start + _MAX_CONCURRENT_GEOCODES]
            with context.parallel_upstream_calls():
                resolved.extend(
                    await asyncio.gather(
                        *(self._resolve_text_place(place, context, areas_by_ref) for place in batch)
                    )
                )
        refs_by_key = {
            key: place.ref for key, place in zip(unique_text_places, resolved, strict=True)
        }
        return [
            place if isinstance(place, str) else refs_by_key[self._place_key(place)]
            for place in points
        ]

    @staticmethod
    def _place_key(place: RoutingPlace) -> tuple[str, str, str]:
        if isinstance(place, str):
            return "ref", place, ""

        query, city = place.resolution_key
        return "query", query, city

    async def _load_area_refs(
        self,
        places: Iterable[RoutingPlaceQuery],
    ) -> dict[PlaceRef, PlaceResolutionArea]:
        refs = list(
            dict.fromkeys(
                place.area for place in places if re.fullmatch(PLACE_REF_PATTERN, place.area)
            )
        )
        records = await asyncio.gather(*(self._place_store.get(ref) for ref in refs))
        areas: dict[PlaceRef, PlaceResolutionArea] = {}
        for ref, record in zip(refs, records, strict=True):
            if record is None:
                raise UnknownRoutingAreaRefError(ref)
            if record.kind != ToponymKind.LOCALITY.value or record.bounds is None:
                raise InvalidRoutingAreaRefError(ref)
            areas[ref] = PlaceResolutionArea(
                ref=ref,
                name=record.name,
                lat=record.lat,
                lon=record.lon,
                bounds=record.bounds,
            )
        return areas

    @staticmethod
    def _area_for_place(
        place: RoutingPlaceQuery,
        areas_by_ref: dict[PlaceRef, PlaceResolutionArea],
    ) -> PlaceResolutionArea:
        if re.fullmatch(PLACE_REF_PATTERN, place.area):
            return areas_by_ref[place.area]
        return PlaceResolutionArea(name=place.area)


class RoutingPlaceLoader:
    """Load coordinator-prepared routing refs directly from shared storage."""

    def __init__(self, place_store: PlaceStore) -> None:
        self._place_store = place_store

    async def load_places(
        self,
        places: Sequence[RoutingPlace],
    ) -> list[ResolvedPlace]:
        """Load prepared refs without invoking geocoding or POI resolution."""

        refs: list[PlaceRef] = []
        for place in places:
            if not isinstance(place, str):
                raise ValueError("routing provider input must contain only prepared place refs")
            refs.append(place)

        unique_refs = list(dict.fromkeys(refs))
        resolved = await asyncio.gather(*(self._load_place(ref) for ref in unique_refs))
        resolved_by_ref = dict(zip(unique_refs, resolved, strict=True))
        return [resolved_by_ref[ref] for ref in refs]

    async def _load_place(self, ref: PlaceRef) -> ResolvedPlace:
        record = await self._place_store.get(ref)
        if record is None:
            raise UnknownRoutingPlaceRefError(ref)
        return ResolvedPlace(ref=ref, record=record)
