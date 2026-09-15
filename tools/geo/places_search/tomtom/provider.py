"""TomTom implementation of the places-search provider contract."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.locality import compatible_locality
from tools.geo.places_search.resolution import (
    PlacesSearchScopeLoader,
    ResolvedArea,
)
from tools.geo.places_search.schemas import (
    Place,
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
    SearchMode,
)
from tools.geo.places_search.tomtom.categories import (
    TomTomCategorySpec,
    tomtom_category_spec,
)
from tools.geo.places_search.tomtom.client import (
    TomTomSearchClient,
    TomTomSearchEndpoint,
)
from tools.geo.places_search.tomtom.matching import (
    clean_optional_text,
    is_auxiliary,
    name_match_score,
    named_result_identity,
    normalized_tokens,
)
from tools.geo.places_search.tomtom.opening_hours import (
    TomTomOpeningHoursSummary,
    summarize_opening_hours,
)
from tools.geo.places_search.tomtom.schemas import TomTomSearchResult
from tools.geo.text_place_query import tomtom_response_language
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

_MAX_TOMTOM_RESULTS = 100
_HOUSE_NUMBER_COMPONENT = re.compile(r"^\s*(\d+)", flags=re.UNICODE)
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _TomTomMatch:
    result: TomTomSearchResult
    address: str
    name_score: tuple[int, int]
    distance_m: int | None
    opening_hours: TomTomOpeningHoursSummary


@dataclass(frozen=True, slots=True)
class _TomTomSearchPlan:
    """Endpoint and filters derived before loading the prepared search scope."""

    endpoint: TomTomSearchEndpoint
    query: str | None
    category_ids: tuple[int, ...]
    is_named_search: bool


class TomTomPlacesSearchProvider:
    """Search TomTom POIs and persist route-ready entrances under opaque refs."""

    provider = "tomtom"
    supports_open_now = True
    resolves_area_natively = False

    def __init__(
        self,
        *,
        client: TomTomSearchClient,
        place_store: PlaceStore,
        clock: Clock = _utc_now,
    ) -> None:
        self._client = client
        self._place_store = place_store
        self._clock = clock
        self._scope_loader = PlacesSearchScopeLoader(place_store=place_store)

    @staticmethod
    def _fetch_limit(args: PlacesSearchInput) -> int:
        # Local name, circle, and opening-hours filters can discard provider candidates.
        return min(_MAX_TOMTOM_RESULTS, max(20, args.result_limit * 4))

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        return await self._run_search(args, context, first_with_address=False)

    async def search_first_address(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Return the first provider-ranked card with an address for resolve."""

        return await self._run_search(args, context, first_with_address=True)

    async def _run_search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
        *,
        first_with_address: bool,
    ) -> PlacesSearchOutput:
        try:
            return await self._search(
                args,
                context,
                first_with_address=first_with_address,
            )
        except ToolExecutionError:
            raise
        except ValidationError as exc:
            raise self._internal_contract_error(
                "TomTom provider could not build a valid tool result"
            ) from exc
        except ValueError as exc:
            raise self._internal_contract_error(
                "TomTom provider produced an invalid search request"
            ) from exc

    async def _search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
        *,
        first_with_address: bool,
    ) -> PlacesSearchOutput:
        query = args.query
        if query is None:
            raise self._internal_contract_error("Prepared TomTom search input has no query")
        anchor_ref: PlaceRef | None = None
        anchor_record: PlaceRecord | None = None
        resolved_area: ResolvedArea | None = None
        area_bounds: GeoBounds | None = None
        search_locality: str | None = None
        fetch_limit = self._fetch_limit(args)
        category_spec = tomtom_category_spec(args.category) if args.category is not None else None
        search_plan = _build_search_plan(args, category_spec)

        if args.mode is SearchMode.AREA:
            if args.area_ref is None:
                raise self._internal_contract_error("Prepared area-search input has no area_ref")
            resolved_area = await self._scope_loader.load_area(args.area_ref)
            area_bounds = resolved_area.bounds
            response_language = tomtom_response_language(query, resolved_area.name)
            search_locality = resolved_area.localized_localities.get(
                response_language,
                resolved_area.name,
            )
            response = await self._client.search(
                endpoint=search_plan.endpoint,
                query=(
                    f"{search_plan.query}, {resolved_area.name}"
                    if search_plan.is_named_search and search_plan.query is not None
                    else search_plan.query
                ),
                category_ids=search_plan.category_ids,
                limit=fetch_limit,
                bbox=resolved_area.bounds,
                language=response_language,
                context=context,
            )
        else:
            if args.near_ref is None:
                raise self._internal_contract_error("Prepared nearby-search input has no near ref")
            if args.area_ref is not None:
                resolved_area = await self._scope_loader.load_area(args.area_ref)
            anchor_ref = args.near_ref
            anchor_record = await self._scope_loader.load_anchor(anchor_ref)
            search_locality = (
                resolved_area.name
                if resolved_area is not None
                else anchor_record.locality or args.city
            )
            response = await self._client.search(
                endpoint=search_plan.endpoint,
                query=search_plan.query,
                category_ids=search_plan.category_ids,
                limit=fetch_limit,
                center=(anchor_record.lon, anchor_record.lat),
                radius_m=args.radius_m,
                language=tomtom_response_language(query, search_locality or ""),
                context=context,
            )

        matches, unknown_open_now_count = _matches_from_results(
            response.results,
            args=args,
            search_plan=search_plan,
            category_spec=category_spec,
            search_locality=search_locality,
            area_bounds=area_bounds,
            anchor_record=anchor_record,
            hours_reference_time=self._clock(),
            first_with_address=first_with_address,
        )
        if args.open_now and unknown_open_now_count:
            context.add_warning(
                "TomTom did not provide enough schedule or time-zone data to verify "
                f"the current opening status of {unknown_open_now_count} result(s); "
                "those results were omitted."
            )

        if anchor_record is not None:
            matches.sort(
                key=lambda match: (
                    match.distance_m is None,
                    match.distance_m if match.distance_m is not None else 0,
                )
            )

        matches = _deduplicate_matches(matches)
        if search_plan.is_named_search:
            matches = _exclude_auxiliary_matches_for_exact_place(matches)
        matches, coordinate_conflicts = _exclude_coordinate_conflicts(matches)
        if coordinate_conflicts:
            context.add_warning(
                "TomTom returned "
                f"{coordinate_conflicts} places with incompatible house numbers at "
                "identical coordinates; those ambiguous places were omitted."
            )
        selected = matches[: args.result_limit]
        unverifiable_count = sum(
            match.opening_hours.text is None and not match.result.poi.phone for match in selected
        )
        if unverifiable_count:
            context.add_warning(
                "TomTom supplied neither opening hours nor a phone number for "
                f"{unverifiable_count} results; their current operation could "
                "not be verified from this response."
            )
        places, records = _materialize_matches(selected)

        if records:
            await self._place_store.save_many(records)

        return PlacesSearchOutput(
            places=places,
            area=(
                ResolvedSearchArea(
                    ref=resolved_area.ref,
                    name=resolved_area.name,
                    address=resolved_area.address,
                )
                if resolved_area is not None
                else None
            ),
            truncated=bool(places)
            and (
                response.summary.total_results > len(response.results)
                or len(matches) > args.result_limit
            ),
            anchor=anchor_ref,
        )

    def _internal_contract_error(self, message: str) -> ToolExecutionError:
        return ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=self._client.provider,
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        )


def _matches_from_results(
    results: list[TomTomSearchResult],
    *,
    args: PlacesSearchInput,
    search_plan: _TomTomSearchPlan,
    category_spec: TomTomCategorySpec | None,
    search_locality: str | None,
    area_bounds: GeoBounds | None,
    anchor_record: PlaceRecord | None,
    hours_reference_time: datetime,
    first_with_address: bool = False,
) -> tuple[list[_TomTomMatch], int]:
    """Normalize provider results and apply locally enforced search constraints."""

    query = args.query
    if query is None:
        raise ValueError("TomTom match filtering requires a search query")
    matches: list[_TomTomMatch] = []
    seen_ids: set[str] = set()
    unknown_open_now_count = 0

    for result in results:
        if result.id in seen_ids:
            continue
        seen_ids.add(result.id)

        address = clean_optional_text(result.address.freeform_address)
        if address is None:
            continue

        if first_with_address:
            matches.append(
                _TomTomMatch(
                    result=result,
                    address=address,
                    name_score=(0, 0),
                    distance_m=None,
                    opening_hours=summarize_opening_hours(
                        result.poi.opening_hours,
                        result.poi.time_zone,
                        now=hours_reference_time,
                    ),
                )
            )
            break

        # Enforce the geocoder-supplied bbox locally, then use an alias-aware
        # municipality check to exclude neighbouring cities inside its corners.
        if args.mode is SearchMode.AREA and not _is_inside_bounds(result, area_bounds):
            continue
        if (
            args.mode is SearchMode.AREA
            and result.address.municipality is not None
            and search_locality is not None
            and not compatible_locality(search_locality, result.address.municipality)
        ):
            continue

        # categorySet is the primary numeric constraint when available.
        # The response check also guards text-only category searches against
        # taxonomy drift and malformed provider cards.
        if category_spec is not None and not _matches_category(result, category_spec):
            continue

        name_score = (0, 0)
        if search_plan.is_named_search:
            match_score = name_match_score(
                result.poi.name,
                query,
                locality=search_locality,
            )
            if match_score is None:
                continue
            name_score = match_score

        opening_hours = summarize_opening_hours(
            result.poi.opening_hours,
            result.poi.time_zone,
            now=hours_reference_time,
        )
        point = result.routing_position
        distance_m: int | None = None
        if anchor_record is not None:
            distance_m = calculate_distance_m(
                from_lat=anchor_record.lat,
                from_lon=anchor_record.lon,
                to_lat=point.lat,
                to_lon=point.lon,
            )
            # TomTom already receives radius, but this enforces the public
            # contract against provider rounding and malformed responses.
            if distance_m > args.radius_m:
                continue

        if args.open_24h and opening_hours.open_24h is not True:
            continue
        if args.open_now and opening_hours.is_open_now is not True:
            if opening_hours.is_open_now is None:
                unknown_open_now_count += 1
            continue

        matches.append(
            _TomTomMatch(
                result=result,
                address=address,
                name_score=name_score,
                distance_m=distance_m,
                opening_hours=opening_hours,
            )
        )

    return matches, unknown_open_now_count


def _is_inside_bounds(
    result: TomTomSearchResult,
    bounds: GeoBounds | None,
) -> bool:
    if bounds is None:
        return False
    point = result.position
    return bounds.west <= point.lon <= bounds.east and bounds.south <= point.lat <= bounds.north


def _materialize_matches(
    matches: list[_TomTomMatch],
) -> tuple[list[Place], list[PlaceRecord]]:
    """Build public places and hidden route-ready records without performing I/O."""

    places: list[Place] = []
    records: list[PlaceRecord] = []

    for match in matches:
        result = match.result
        point = result.routing_position
        ref = mint_place_ref(f"tomtom:poi:{result.id}")

        records.append(
            PlaceRecord(
                ref=ref,
                name=result.poi.name,
                address=match.address,
                lat=point.lat,
                lon=point.lon,
                locality=result.address.municipality,
                provider=TomTomPlacesSearchProvider.provider,
                provider_id=result.id,
                origin=RecordOrigin.PLACES_SEARCH,
            )
        )
        places.append(
            Place(
                ref=ref,
                id=result.id,
                name=result.poi.name,
                address=match.address,
                categories=result.poi.category_names,
                phones=([result.poi.phone] if result.poi.phone else []),
                hours_text=match.opening_hours.text,
                open_24h=match.opening_hours.open_24h,
                is_open_now=match.opening_hours.is_open_now,
                distance_m=match.distance_m,
            )
        )

    return places, records


def _build_search_plan(
    args: PlacesSearchInput,
    category: TomTomCategorySpec | None,
) -> _TomTomSearchPlan:
    """Choose an endpoint from typed intent instead of interpreting query text."""

    if category is None:
        return _TomTomSearchPlan(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query=args.query,
            category_ids=(),
            is_named_search=True,
        )

    if (
        args.mode is SearchMode.NEAR
        and category.category_ids
        and category.supports_queryless_nearby
    ):
        return _TomTomSearchPlan(
            endpoint=TomTomSearchEndpoint.NEARBY,
            query=None,
            category_ids=category.category_ids,
            is_named_search=False,
        )

    if not category.category_ids:
        # Category Search can reinterpret an unsupported phrase (for example
        # "bus station" as railway stations). Fuzzy Search preserves the
        # model-extracted text while category post-filtering keeps discovery
        # results within the public contract.
        return _TomTomSearchPlan(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query=args.query,
            category_ids=(),
            is_named_search=False,
        )

    return _TomTomSearchPlan(
        endpoint=TomTomSearchEndpoint.CATEGORY,
        query=category.query,
        category_ids=category.category_ids,
        is_named_search=False,
    )


def _matches_category(
    result: TomTomSearchResult,
    category: TomTomCategorySpec,
) -> bool:
    category_ids = result.poi.category_ids
    if category_ids and category.category_ids:
        id_matches = any(
            _is_same_or_child_category(actual_id, requested_id)
            for actual_id in category_ids
            for requested_id in category.category_ids
        )
        if not id_matches:
            return False
        if category.category_names:
            returned_names = {
                _normalized_category_name(value) for value in result.poi.category_names
            }
            return bool(returned_names & category.category_names)
        return True

    # Older or incomplete TomTom records may omit categorySet. Keep the broad
    # classification guard only as compatibility fallback, never as the first
    # choice for a reviewed numeric category.
    codes = result.poi.classification_codes
    if codes and not codes & category.classification_codes:
        return False
    if category.category_names:
        returned_names = {_normalized_category_name(value) for value in result.poi.category_names}
        return bool(returned_names & category.category_names)
    return True


def _normalized_category_name(value: str) -> str:
    """Normalize provider taxonomy labels without interpreting user text."""

    return " ".join(value.casefold().split())


def _is_same_or_child_category(actual_id: int, requested_id: int) -> bool:
    """Match an exact TomTom category or a child of a four-digit parent."""

    return actual_id == requested_id or (
        requested_id < 10_000 and actual_id // 1000 == requested_id
    )


def _result_completeness(result: TomTomSearchResult) -> int:
    """Prefer the richer record when TomTom has duplicate IDs for one POI."""

    return sum(
        (
            bool(result.poi.phone),
            result.poi.opening_hours is not None,
            bool(result.entry_points),
            bool(result.poi.classifications),
        )
    )


def _exclude_auxiliary_matches_for_exact_place(
    matches: list[_TomTomMatch],
) -> list[_TomTomMatch]:
    """Hide a landmark's parking cards when the exact landmark is available."""

    has_exact_primary = any(
        match.name_score == (0, 0) and not is_auxiliary(match.result) for match in matches
    )
    if not has_exact_primary:
        return matches
    return [match for match in matches if not is_auxiliary(match.result)]


def _deduplicate_matches(matches: list[_TomTomMatch]) -> list[_TomTomMatch]:
    """Collapse duplicate provider cards without merging co-located businesses.

    TomTom sometimes publishes aliases for one facility under different IDs,
    for example ``P1`` and ``Covered parking P1`` at the exact same point and
    address. Different businesses merely sharing a building remain separate
    because their names are not aliases.
    """

    result: list[_TomTomMatch] = []

    for match in matches:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(result)
                if _matches_same_provider_place(existing, match)
            ),
            None,
        )
        if duplicate_index is None:
            result.append(match)
            continue

        existing = result[duplicate_index]
        if _result_completeness(match.result) > _result_completeness(existing.result):
            # Keep the original sorted position while retaining the richer
            # phone, opening-hours, or entrance data.
            result[duplicate_index] = match

    return result


def _matches_same_provider_place(left: _TomTomMatch, right: _TomTomMatch) -> bool:
    """Return true only for exact identities or same-point name aliases."""

    if named_result_identity(left.result, left.address) == named_result_identity(
        right.result,
        right.address,
    ):
        return True

    left_point = left.result.routing_position
    right_point = right.result.routing_position
    if (left_point.lat, left_point.lon) != (right_point.lat, right_point.lon):
        return False
    if normalized_tokens(left.address) != normalized_tokens(right.address):
        return False

    left_name = frozenset(normalized_tokens(left.result.poi.name))
    right_name = frozenset(normalized_tokens(right.result.poi.name))
    return bool(left_name and right_name) and (
        left_name.issubset(right_name) or right_name.issubset(left_name)
    )


def _exclude_coordinate_conflicts(
    matches: list[_TomTomMatch],
) -> tuple[list[_TomTomMatch], int]:
    """Omit coordinates assigned to incompatible explicit house numbers.

    Businesses inside one park, station, or shopping centre can legitimately
    share an entrance while TomTom renders sub-addresses such as ``9``, ``9/1``,
    and ``9/59``. Those all have the same root house number and remain valid.
    Coordinates shared by addresses such as ``Nevsky 47`` and ``Nevsky 136``
    are still rejected because the route-ready point is demonstrably unreliable.
    """

    house_numbers_by_point: dict[tuple[float, float], set[str]] = {}
    for match in matches:
        house_number = _root_house_number(match.address)
        if house_number is None:
            continue
        point = match.result.routing_position
        house_numbers_by_point.setdefault((point.lat, point.lon), set()).add(house_number)

    conflicted_points = {
        point for point, house_numbers in house_numbers_by_point.items() if len(house_numbers) > 1
    }
    if not conflicted_points:
        return matches, 0

    result = [
        match
        for match in matches
        if (
            match.result.routing_position.lat,
            match.result.routing_position.lon,
        )
        not in conflicted_points
    ]
    return result, len(matches) - len(result)


def _root_house_number(address: str) -> str | None:
    """Extract ``9`` from a TomTom component such as ``9/59`` or ``9A``."""

    # TomTom normally renders the street first and the house number as the next
    # comma-separated component. Skipping the first component avoids treating
    # an ordinal street name such as "9-я улица" as a house number.
    for component in address.split(",")[1:]:
        match = _HOUSE_NUMBER_COMPONENT.match(component)
        if match is None:
            continue
        value = match.group(1)
        # A standalone six-digit value is a Russian postal code, not a house.
        if len(value) == 6:
            continue
        return value
    return None
