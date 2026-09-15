"""Yandex implementation of the organisation-search provider contract.

The provider sends ``query`` as Yandex ``text``; the API has no separate
provider-neutral category parameter. Area searches use the resolved locality's
bounding box. Near searches use ``ll`` plus a degree-based ``spn`` rectangle,
then enforce the requested circle locally. Opening-hours constraints and the
``distance_m`` value are also evaluated locally.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from math import cos, radians

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.resolution import PlacesSearchScopeLoader, ResolvedArea
from tools.geo.places_search.schemas import (
    Place,
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
    SearchMode,
)
from tools.geo.places_search.yandex.client import YandexOrganisationSearchClient
from tools.geo.places_search.yandex.opening_hours import current_opening_status
from tools.geo.places_search.yandex.schemas import (
    YandexOrganisation,
    YandexOrganisationSearchResponse,
)
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

_METRES_PER_LATITUDE_DEGREE = 111_320
_MAX_YANDEX_RESULTS = 50
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _span_for_radius(
    *,
    radius_m: int,
    center_lat: float,
) -> tuple[float, float]:
    """Return Yandex ``spn`` as ``(longitude_span, latitude_span)`` in degrees."""

    latitude_delta = radius_m / _METRES_PER_LATITUDE_DEGREE
    longitude_delta = radius_m / (_METRES_PER_LATITUDE_DEGREE * max(cos(radians(center_lat)), 0.01))

    return (
        min(360.0, longitude_delta * 2),
        min(180.0, latitude_delta * 2),
    )


class YandexPlacesSearchProvider:
    """Search organisations with Yandex and persist returned places by ref."""

    provider = "yandex"
    supports_open_now = True
    resolves_area_natively = False

    def __init__(
        self,
        *,
        client: YandexOrganisationSearchClient,
        place_store: PlaceStore,
        clock: Clock = _utc_now,
    ) -> None:
        """Wire the Yandex client to shared place storage and resolution."""

        self._client = client
        self._place_store = place_store
        self._clock = clock
        self._scope_loader = PlacesSearchScopeLoader(place_store=place_store)

    @staticmethod
    def _fetch_limit(args: PlacesSearchInput) -> int:
        """Over-fetch because radius and opening-hours filters are local."""

        if args.open_now:
            return min(_MAX_YANDEX_RESULTS, max(20, args.result_limit * 4))
        return min(_MAX_YANDEX_RESULTS, args.result_limit * 3)

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Run the search behind a boundary that returns only classified errors."""

        try:
            return await self._search(args, context)
        except ToolExecutionError:
            # Errors classified by the client, scope loader, or response validator
            # already contain the right metadata and safe message.
            raise
        except ValidationError as exc:
            raise self._internal_contract_error(
                "Places search provider could not build a valid tool result"
            ) from exc
        except ValueError as exc:
            raise self._internal_contract_error(
                "Places search provider produced an invalid request for Yandex"
            ) from exc

    async def _search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Load the prepared scope, call Yandex, filter results, and persist refs."""

        query = args.query
        if query is None:
            raise self._internal_contract_error("Prepared Yandex search input has no query")
        anchor_ref: PlaceRef | None = None
        anchor_record: PlaceRecord | None = None
        resolved_area: ResolvedArea | None = None
        fetch_limit = self._fetch_limit(args)
        reference_time = self._clock()
        if reference_time.tzinfo is None:
            raise ValueError("opening-hours reference time must be timezone-aware")

        if args.mode is SearchMode.AREA:
            if args.area_ref is None:
                raise self._internal_contract_error("Prepared area-search input has no area_ref")
            resolved_area = await self._scope_loader.load_area(args.area_ref)
            payload = await self._client.search(
                text=query,
                limit=fetch_limit,
                bbox=resolved_area.bounds,
                context=context,
            )
        else:
            if args.near_ref is None:
                raise self._internal_contract_error("Prepared nearby-search input has no near ref")
            if args.area_ref is not None:
                resolved_area = await self._scope_loader.load_area(args.area_ref)
            anchor_ref = args.near_ref
            anchor_record = await self._scope_loader.load_anchor(anchor_ref)

            payload = await self._client.search(
                text=query,
                limit=fetch_limit,
                center=(anchor_record.lon, anchor_record.lat),
                span=_span_for_radius(
                    radius_m=args.radius_m,
                    center_lat=anchor_record.lat,
                ),
                context=context,
            )

        response = self._validate_response(payload)

        matches: list[tuple[YandexOrganisation, int | None, bool | None]] = []
        seen_provider_ids: set[str] = set()
        unknown_open_now_count = 0

        for organisation in response.organisations:
            company = organisation.company

            # Duplicate provider ids would mint the same ref and make the model
            # see the same real organisation more than once.
            if company.id in seen_provider_ids:
                continue

            seen_provider_ids.add(company.id)

            # Yandex has no request parameter for this constraint. Keep only
            # explicitly confirmed 24/7 schedules; unknown hours are not enough.
            if args.open_24h and company.open_24h is not True:
                continue

            distance_m: int | None = None

            if anchor_record is not None:
                distance_m = calculate_distance_m(
                    from_lat=anchor_record.lat,
                    from_lon=anchor_record.lon,
                    to_lat=organisation.point.lat,
                    to_lon=organisation.point.lon,
                )

                # spn gives Yandex a rectangle; this restores the requested circle.
                if distance_m > args.radius_m:
                    continue

            is_open_now = (
                current_opening_status(
                    company.hours,
                    lat=organisation.point.lat,
                    lon=organisation.point.lon,
                    now=reference_time,
                )
                if args.open_now
                else None
            )
            if args.open_now and is_open_now is not True:
                if is_open_now is None:
                    unknown_open_now_count += 1
                continue

            matches.append((organisation, distance_m, is_open_now))

        if args.open_now and unknown_open_now_count:
            context.add_warning(
                "Yandex schedules could not verify the current opening status of "
                f"{unknown_open_now_count} result(s); those results were omitted."
            )

        if anchor_record is not None:
            # Yandex relevance order is not guaranteed to be distance order.
            matches.sort(key=lambda item: item[1] or 0)

        selected = matches[: args.result_limit]
        places: list[Place] = []
        records: list[PlaceRecord] = []

        for organisation, distance_m, is_open_now in selected:
            company = organisation.company
            ref = mint_place_ref(f"yandex:organisation:{company.id}")

            records.append(
                PlaceRecord(
                    ref=ref,
                    name=company.name,
                    address=company.address.formatted,
                    lat=organisation.point.lat,
                    lon=organisation.point.lon,
                    provider=self.provider,
                    provider_id=company.id,
                    origin=RecordOrigin.PLACES_SEARCH,
                )
            )

            places.append(
                Place(
                    ref=ref,
                    id=company.id,
                    name=company.name,
                    address=company.address.formatted,
                    categories=[category.name for category in company.categories],
                    phones=[phone.formatted for phone in company.phones],
                    hours_text=company.hours.text if company.hours is not None else None,
                    open_24h=company.open_24h,
                    is_open_now=is_open_now,
                    accessibility=[
                        feature.id for feature in company.features if feature.value is True
                    ],
                    distance_m=distance_m,
                )
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
            and (response.found > len(response.organisations) or len(matches) > args.result_limit),
            anchor=anchor_ref,
        )

    def _validate_response(
        self,
        payload: dict[str, object],
    ) -> YandexOrganisationSearchResponse:
        """Validate raw Yandex data and classify an incompatible response schema."""

        try:
            return YandexOrganisationSearchResponse.model_validate(payload)
        except ValidationError as exc:
            raise self._invalid_schema_error(
                "Yandex organisation search returned data in an unexpected format"
            ) from exc

    def _invalid_schema_error(self, message: str) -> ToolExecutionError:
        """Build a non-retryable error for an invalid Yandex response schema."""

        return ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=self._client.provider,
            failure_kind=ToolFailureKind.INVALID_SCHEMA,
            retryable=False,
        )

    def _internal_contract_error(self, message: str) -> ToolExecutionError:
        """Build a non-retryable error for a broken provider-side invariant."""

        return ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=self._client.provider,
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        )
