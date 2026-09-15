"""2GIS implementation of the places-search provider contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.category_normalization import category_for_exact_query
from tools.geo.places_search.locality import same_locality
from tools.geo.places_search.resolution import PlacesSearchScopeLoader, ResolvedArea
from tools.geo.places_search.schemas import (
    Place,
    PlaceCategory,
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
    SearchMode,
)
from tools.geo.places_search.twogis.categories import twogis_rubric_lookup
from tools.geo.places_search.twogis.client import TwoGisSearchClient
from tools.geo.places_search.twogis.matching import (
    TwoGisCandidate,
    build_named_candidates,
    deduplicate_candidates,
    has_explicit_address,
    semantic_kind,
)
from tools.geo.places_search.twogis.opening_hours import (
    TwoGisOpeningHoursSummary,
    summarize_schedule,
)
from tools.geo.places_search.twogis.schemas import TwoGisItem
from tools.geo.text_place_query import twogis_response_locale, uses_cyrillic
from tools.geo.transliteration import transliterate_cyrillic
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _TwoGisMatch:
    candidate: TwoGisCandidate
    distance_m: int | None
    opening_hours: TwoGisOpeningHoursSummary


class TwoGisPlacesSearchProvider:
    """Search 2GIS POIs inside covered localities and persist opaque refs."""

    provider = "twogis"
    supports_open_now = True
    supports_min_rating = True
    # Area text must first be geocoded into a shared opaque ref.  2GIS then
    # checks coverage by that resolved point, avoiding fuzzy Region API text
    # ranking (for example, an unsupported Krakow being ranked as Moscow).
    resolves_area_natively = False

    @staticmethod
    def _fetch_limit(args: PlacesSearchInput) -> int:
        # Fetch extra candidates before local name, locality, distance, hours,
        # and duplicate filters. Keep requests compatible with demo keys, which
        # accept at most 10 results even though the public API reference allows
        # larger pages for other subscriptions.
        return 10

    def __init__(
        self,
        *,
        client: TwoGisSearchClient,
        place_store: PlaceStore,
        clock: Clock = _utc_now,
    ) -> None:
        self._client = client
        self._place_store = place_store
        self._clock = clock
        self._scope_loader = PlacesSearchScopeLoader(place_store=place_store)

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
        except (ValidationError, ValueError) as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS provider could not build a valid tool result",
                provider=self._client.provider,
                failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            ) from exc

    async def _search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
        *,
        first_with_address: bool,
    ) -> PlacesSearchOutput:
        if args.query is None:
            raise ValueError("2GIS provider received a search request without query")

        anchor_ref: PlaceRef | None = None
        anchor_record: PlaceRecord | None = None
        resolved_area: ResolvedArea | None = None
        city_id: str | None = None
        country_code: str | None = None
        region_id: str | None = None
        coverage_point: tuple[float, float] | None = None
        search_locality: str | None

        if args.mode is SearchMode.AREA:
            if args.area_ref is not None:
                resolved_area = await self._scope_loader.load_area(args.area_ref)
                search_locality = resolved_area.name
                coverage_point = (resolved_area.lon, resolved_area.lat)
            elif args.city is not None:
                # Kept for direct adapter use and narrow unit tests.  The
                # coordinator supplies area_ref in the production path.
                search_locality = args.city
            else:
                raise ValueError("2GIS area search has neither city nor area_ref")
        else:
            if args.near_ref is None:
                raise ValueError("prepared 2GIS nearby search has no near ref")
            if args.area_ref is not None:
                resolved_area = await self._scope_loader.load_area(args.area_ref)
            anchor_ref = args.near_ref
            anchor_record = await self._scope_loader.load_anchor(anchor_ref)
            search_locality = (
                resolved_area.name
                if resolved_area is not None
                else anchor_record.locality or args.city
            )
            coverage_point = (anchor_record.lon, anchor_record.lat)

        if args.min_rating is not None and search_locality is None:
            raise self._rating_coverage_error(None)

        if coverage_point is not None or search_locality is not None:
            try:
                if coverage_point is not None:
                    region = await self._client.find_region_at_point(
                        lon=coverage_point[0],
                        lat=coverage_point[1],
                        context=context,
                    )
                else:
                    assert search_locality is not None
                    region = await self._client.find_region(search_locality, context)
            except ToolExecutionError as exc:
                if (
                    args.min_rating is not None
                    and exc.failure_kind is ToolFailureKind.COVERAGE_MISS
                ):
                    raise self._rating_coverage_error(search_locality, cause=exc) from exc
                raise
            if region is None:
                if args.min_rating is not None:
                    raise self._rating_coverage_error(search_locality)
                return _empty_output(resolved_area=resolved_area, anchor_ref=anchor_ref)
            country_code = region.country_code
            region_id = region.id
            response_locale = twogis_response_locale(
                args.query,
                country_code=country_code,
            )
            if args.mode is SearchMode.AREA:
                if coverage_point is not None:
                    city = await self._client.find_city_at_point(
                        lon=coverage_point[0],
                        lat=coverage_point[1],
                        context=context,
                        expected_region_id=region.id,
                        locale=response_locale,
                    )
                else:
                    assert search_locality is not None
                    city = await self._client.find_city(
                        search_locality,
                        context=context,
                        expected_region_id=region.id,
                        country_code=country_code,
                    )
                if city is None:
                    if args.min_rating is not None:
                        raise self._rating_coverage_error(search_locality)
                    return _empty_output(resolved_area=resolved_area, anchor_ref=anchor_ref)
                city_id = city.id
                # A clarification retry may contain ``city, region, country``.
                # Once the region has selected the branch, use the canonical
                # city label for result-locality filtering.
                search_locality = city.name
            elif coverage_point is not None:
                # Nearby search cannot combine its point/radius restriction
                # with city_id. Resolve the anchor's containing city in the
                # same locale as the cards so the defensive locality check
                # compares like with like (for example, Ереван with Ереван).
                city = await self._client.find_city_at_point(
                    lon=coverage_point[0],
                    lat=coverage_point[1],
                    context=context,
                    expected_region_id=region.id,
                    locale=response_locale,
                )
                if city is not None:
                    search_locality = city.name

        response_locale = twogis_response_locale(args.query, country_code=country_code)
        rubric_ids: tuple[str, ...] = ()
        rubric_query: str | None = None
        if args.category is not None:
            if region_id is None:
                raise ValueError("2GIS category search has no resolved region")
            rubric_lookup = twogis_rubric_lookup(
                category=args.category,
                query=args.query,
                country_code=country_code,
            )
            if rubric_lookup.allowed_aliases is None:
                return _empty_output(resolved_area=resolved_area, anchor_ref=anchor_ref)
            rubric_query = rubric_lookup.query
            rubrics = await self._client.find_rubrics(
                query=rubric_lookup.query,
                region_id=region_id,
                locale=rubric_lookup.locale,
                context=context,
                allowed_aliases=rubric_lookup.allowed_aliases,
            )
            if not rubrics:
                return _empty_output(resolved_area=resolved_area, anchor_ref=anchor_ref)
            rubric_ids = tuple(rubric.id for rubric in rubrics)

        rubric_id = ",".join(rubric_ids) if rubric_ids else None

        catalog_query = args.query if rubric_id is None else None
        if anchor_record is not None and args.radius_m > 2_000:
            # 2GIS limits queryless point searches to a 2 km radius. Use the
            # reviewed rubric lookup phrase, not a possibly cross-script user
            # phrase, when a larger public-contract radius needs q.
            catalog_query = rubric_query

        response = await self._client.search_places(
            query=catalog_query,
            page_size=self._fetch_limit(args),
            center=(anchor_record.lon, anchor_record.lat) if anchor_record is not None else None,
            radius_m=args.radius_m if anchor_record is not None else None,
            city_id=city_id,
            region_id=region_id if rubric_id is not None else None,
            open_now=args.open_now,
            sort_by_rating=args.min_rating is not None,
            locale=response_locale,
            rubric_id=rubric_id,
            context=context,
        )
        if first_with_address:
            candidates = _first_address_candidates(response.result.items)
        else:
            candidates = _provider_candidates(
                response.result.items,
                args=args,
                search_locality=search_locality,
                rubric_ids=rubric_ids,
            )
        matches, unknown_open_now = _build_matches(
            candidates,
            args=args,
            anchor_record=anchor_record,
            now=self._clock(),
        )
        if args.open_now and unknown_open_now:
            context.add_warning(
                "2GIS did not provide a complete usable schedule for "
                f"{unknown_open_now} result(s); those results were omitted."
            )

        if anchor_record is not None:
            matches.sort(
                key=lambda match: (
                    match.distance_m is None,
                    match.distance_m if match.distance_m is not None else 0,
                    match.candidate.provider_rank,
                )
            )

        selected = matches[: args.result_limit]
        display_locality = (
            resolved_area.localized_localities.get("en-US", resolved_area.name)
            if resolved_area is not None
            else args.city
        )
        places, records = _materialize_matches(
            selected,
            transliterate_output=not uses_cyrillic(args.query),
            source_locality=search_locality,
            display_locality=display_locality,
            requested_category=args.category,
            rubric_ids=rubric_ids,
        )
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
                response.result.total > len(response.result.items)
                or len(matches) > args.result_limit
            ),
            anchor=anchor_ref,
        )

    def _rating_coverage_error(
        self,
        locality: str | None,
        *,
        cause: ToolExecutionError | None = None,
    ) -> ToolExecutionError:
        location = f" for {locality!r}" if locality is not None else ""
        return ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            "The requested minimum-rating filter requires 2GIS coverage"
            f"{location}. Use web_search for the same rated-place request.",
            status_code=cause.status_code if cause is not None else None,
            provider=self.provider,
            provider_code=cause.provider_code if cause is not None else None,
            failure_kind=ToolFailureKind.COVERAGE_MISS,
            retryable=False,
        )


def _provider_candidates(
    items: list[TwoGisItem],
    *,
    args: PlacesSearchInput,
    search_locality: str | None,
    rubric_ids: tuple[str, ...],
) -> list[TwoGisCandidate]:
    if args.query is None:
        raise ValueError("2GIS candidate filtering requires a search query")

    if args.category is None:
        return build_named_candidates(
            items,
            query=args.query,
            city=search_locality if args.mode is SearchMode.NEAR else None,
        )

    candidates: list[TwoGisCandidate] = []
    for provider_rank, item in enumerate(items):
        if item.point is None:
            continue
        if rubric_ids and not any(rubric.id in rubric_ids for rubric in item.rubrics):
            continue
        if (
            args.mode is SearchMode.NEAR
            and search_locality is not None
            and item.locality_names
            and not any(
                same_locality(search_locality, locality) for locality in item.locality_names
            )
        ):
            continue
        candidates.append(
            TwoGisCandidate(
                item=item,
                semantic_kind=semantic_kind(item),
                provider_rank=provider_rank,
            )
        )
    return deduplicate_candidates(candidates)


def _first_address_candidates(items: list[TwoGisItem]) -> list[TwoGisCandidate]:
    for provider_rank, item in enumerate(items):
        if item.point is not None and has_explicit_address(item):
            return [
                TwoGisCandidate(
                    item=item,
                    semantic_kind=semantic_kind(item),
                    provider_rank=provider_rank,
                )
            ]
    return []


def _build_matches(
    candidates: list[TwoGisCandidate],
    *,
    args: PlacesSearchInput,
    anchor_record: PlaceRecord | None,
    now: datetime,
) -> tuple[list[_TwoGisMatch], int]:
    matches: list[_TwoGisMatch] = []
    unknown_open_now = 0

    for candidate in candidates:
        item = candidate.item
        assert item.point is not None
        rating = item.reviews.display_rating if item.reviews is not None else None
        if args.min_rating is not None and (rating is None or rating < args.min_rating):
            continue
        hours = summarize_schedule(
            item.schedule,
            lat=item.point.lat,
            lon=item.point.lon,
            now=now,
        )
        if args.open_24h and hours.open_24h is not True:
            continue
        if args.open_now and hours.is_open_now is not True:
            if hours.is_open_now is None:
                unknown_open_now += 1
            continue

        distance_m: int | None = None
        if anchor_record is not None:
            distance_m = calculate_distance_m(
                from_lat=anchor_record.lat,
                from_lon=anchor_record.lon,
                to_lat=item.point.lat,
                to_lon=item.point.lon,
            )
            if distance_m > args.radius_m:
                continue

        matches.append(
            _TwoGisMatch(
                candidate=candidate,
                distance_m=distance_m,
                opening_hours=hours,
            )
        )
    return matches, unknown_open_now


def _materialize_matches(
    matches: list[_TwoGisMatch],
    *,
    transliterate_output: bool,
    source_locality: str | None,
    display_locality: str | None,
    requested_category: PlaceCategory | None,
    rubric_ids: tuple[str, ...],
) -> tuple[list[Place], list[PlaceRecord]]:
    places: list[Place] = []
    records: list[PlaceRecord] = []
    for match in matches:
        item = match.candidate.item
        assert item.point is not None
        ref = mint_place_ref(f"twogis:item:{item.id}")
        records.append(
            PlaceRecord(
                ref=ref,
                name=item.name,
                address=item.address,
                lat=item.point.lat,
                lon=item.point.lon,
                locality=item.locality,
                provider=TwoGisPlacesSearchProvider.provider,
                provider_id=item.id,
                origin=RecordOrigin.PLACES_SEARCH,
            )
        )
        places.append(
            Place(
                ref=ref,
                id=item.id,
                name=(transliterate_cyrillic(item.name) if transliterate_output else item.name),
                address=_display_address(
                    item.address,
                    transliterate_output=transliterate_output,
                    source_locality=source_locality,
                    display_locality=display_locality,
                ),
                categories=_display_categories(
                    item,
                    transliterate_output=transliterate_output,
                    requested_category=requested_category,
                    rubric_ids=rubric_ids,
                ),
                rating=item.reviews.display_rating if item.reviews is not None else None,
                review_count=(
                    item.reviews.display_review_count if item.reviews is not None else None
                ),
                hours_text=match.opening_hours.text,
                open_24h=match.opening_hours.open_24h,
                is_open_now=match.opening_hours.is_open_now,
                distance_m=match.distance_m,
            )
        )
    return places, records


def _display_address(
    address: str,
    *,
    transliterate_output: bool,
    source_locality: str | None,
    display_locality: str | None,
) -> str:
    if not transliterate_output:
        return address

    display_address = address
    if (
        source_locality is not None
        and display_locality is not None
        and display_address.startswith(source_locality)
    ):
        display_address = f"{display_locality}{display_address[len(source_locality) :]}"
    return transliterate_cyrillic(display_address)


def _display_categories(
    item: TwoGisItem,
    *,
    transliterate_output: bool,
    requested_category: PlaceCategory | None,
    rubric_ids: tuple[str, ...],
) -> list[str]:
    if not transliterate_output:
        return item.category_names

    categories: list[str] = []
    for rubric in item.rubrics:
        canonical = (
            requested_category.value
            if requested_category is not None and rubric.id in rubric_ids
            else category_for_exact_query(rubric.name)
        )
        display = (
            canonical.replace("_", " ")
            if canonical is not None
            else transliterate_cyrillic(rubric.name)
        )
        if display not in categories:
            categories.append(display)
    return categories


def _empty_output(
    *,
    resolved_area: ResolvedArea | None,
    anchor_ref: PlaceRef | None,
) -> PlacesSearchOutput:
    return PlacesSearchOutput(
        area=(
            ResolvedSearchArea(
                ref=resolved_area.ref,
                name=resolved_area.name,
                address=resolved_area.address,
            )
            if resolved_area is not None
            else None
        ),
        anchor=anchor_ref,
    )
