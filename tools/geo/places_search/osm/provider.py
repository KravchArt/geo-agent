"""OpenStreetMap implementation of the organisation-search provider contract."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.geocoding.schemas import GeocodePlaceInput, ToponymKind
from tools.geo.geocoding.service import GeocoderService
from tools.geo.opening_hours import is_open_at
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.locality import same_locality
from tools.geo.places_search.osm.client import OsmOverpassClient
from tools.geo.places_search.osm.schemas import OsmOverpassResponse
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
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

_CATEGORY_TAGS = ("amenity", "shop", "tourism", "office", "craft", "healthcare", "leisure")
_NAME_TAGS = ("name:ru", "name", "brand", "operator", "name:en")
_POSTCODE_TOKEN = re.compile(r"(?<!\w)\d{5,6}(?!\w)")
_NEAR_REVERSE_GEOCODE_LIMIT = 3
_LOCALITY_TAGS = ("addr:city", "addr:municipality", "is_in:city")
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _OsmMatch:
    provider_id: str
    name: str
    address: str
    categories: list[str]
    phones: list[str]
    hours_text: str | None
    open_24h: bool | None
    is_open_now: bool | None
    accessibility: list[str]
    lon: float
    lat: float
    distance_m: int | None


def _first_tag(tags: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = tags.get(key)
        if value is not None and value.strip():
            return " ".join(value.split())
    return None


def _split_tag_values(*values: str | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue
        for part in value.split(";"):
            cleaned = " ".join(part.split())
            normalized = cleaned.casefold()
            if cleaned and normalized not in seen:
                seen.add(normalized)
                result.append(cleaned)

    return result


def _address(tags: dict[str, str]) -> str:
    house_number = tags.get("addr:housenumber")
    if house_number is None or not house_number.strip():
        return ""

    full = tags.get("addr:full")
    if full is not None and full.strip():
        cleaned_full = _clean_address_text(full, postcode=tags.get("addr:postcode"))
        if cleaned_full:
            return cleaned_full

    street = tags.get("addr:street") or tags.get("addr:place")
    if street is None or not street.strip():
        return ""

    street_address = " ".join(
        part.strip() for part in (street, house_number) if part is not None and part.strip()
    )

    parts = [
        tags.get("addr:country"),
        tags.get("addr:city"),
        street_address or None,
    ]

    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part is None:
            continue
        cleaned = " ".join(part.split())
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)

    return ", ".join(result)


def _clean_address_text(value: str, *, postcode: str | None = None) -> str:
    cleaned = " ".join(value.split())
    if postcode is not None and postcode.strip():
        cleaned_postcode = " ".join(postcode.split())
        cleaned = re.sub(
            rf"(?<!\w){re.escape(cleaned_postcode)}(?!\w)",
            "",
            cleaned,
        )
    cleaned = _POSTCODE_TOKEN.sub("", cleaned)
    cleaned = re.sub(r"(?:\s*,\s*){2,}", ", ", cleaned).strip(" ,;-")
    return " ".join(cleaned.split())


def _categories(tags: dict[str, str], requested: str | None) -> list[str]:
    values: list[str | None] = [requested] if requested is not None else []
    values.extend(tags.get(key) for key in _CATEGORY_TAGS)
    return _split_tag_values(*values)


def _accessibility(tags: dict[str, str]) -> list[str]:
    wheelchair = tags.get("wheelchair")
    if wheelchair in {"yes", "designated"}:
        return ["wheelchair_access"]
    if wheelchair == "limited":
        return ["wheelchair_limited"]
    return []


def _is_open_24h(hours_text: str | None) -> bool | None:
    if hours_text is None or not hours_text.strip():
        return None
    return hours_text.strip() == "24/7"


class OsmPlacesSearchProvider:
    """Search OSM through Overpass and persist returned places by ref."""

    provider = "osm"
    supports_open_now = True
    resolves_area_natively = False

    def __init__(
        self,
        *,
        client: OsmOverpassClient,
        place_store: PlaceStore,
        geocoder: GeocoderService,
        max_elements: int = 300,
        clock: Clock = _utc_now,
    ) -> None:
        if not 20 <= max_elements <= 1_000:
            raise ValueError("max_elements must be between 20 and 1000")

        self._client = client
        self._place_store = place_store
        self._geocoder = geocoder
        self._clock = clock
        self._scope_loader = PlacesSearchScopeLoader(place_store=place_store)
        self._max_elements = max_elements

    def _fetch_limit(self, args: PlacesSearchInput) -> int:
        return min(self._max_elements, max(50, args.result_limit * 30))

    async def _reverse_geocode_nearest(
        self,
        matches: list[_OsmMatch],
        context: ToolExecutionContext,
    ) -> None:
        targets = [
            (index, match)
            for index, match in enumerate(matches[:_NEAR_REVERSE_GEOCODE_LIMIT])
            if not match.address
        ]
        if not targets:
            return

        # These lookups are one concurrent enrichment stage, so their
        # individual HTTP durations contribute only the slowest call.
        with context.parallel_upstream_calls():
            results = await asyncio.gather(
                *(
                    self._geocoder.geocode(
                        GeocodePlaceInput(
                            query=f"{match.lon:.6f},{match.lat:.6f}",
                            limit=1,
                        ),
                        context,
                    )
                    for _, match in targets
                )
            )

        approximate_addresses = 0
        for (index, match), result in zip(targets, results, strict=True):
            if result.best is None:
                continue
            address = _clean_address_text(result.best.address)
            if address:
                matches[index] = replace(match, address=address)
                if result.best.kind is not ToponymKind.HOUSE:
                    approximate_addresses += 1

        if approximate_addresses:
            context.add_warning(
                "The geocoder could only provide an approximate address for "
                f"{approximate_addresses} OpenStreetMap result(s); routing still "
                "uses the exact provider coordinates stored under each ref."
            )

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        try:
            return await self._search(args, context)
        except ToolExecutionError:
            raise
        except ValidationError as exc:
            raise self._internal_contract_error(
                "OpenStreetMap provider could not build a valid tool result"
            ) from exc
        except ValueError as exc:
            raise self._internal_contract_error(
                "OpenStreetMap provider produced an invalid Overpass request"
            ) from exc

    async def _search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        query = args.query
        if query is None:
            raise self._internal_contract_error("Prepared OpenStreetMap search input has no query")
        anchor_ref: PlaceRef | None = None
        anchor_record: PlaceRecord | None = None
        resolved_area: ResolvedArea | None = None
        requested_locality: str | None = None
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
                category=args.category,
                limit=fetch_limit,
                open_24h=args.open_24h,
                open_now=args.open_now,
                bbox=resolved_area.bounds,
                boundary_name=resolved_area.name,
                context=context,
            )
        else:
            if args.near_ref is None:
                raise self._internal_contract_error("Prepared nearby-search input has no near ref")
            if args.area_ref is not None:
                resolved_area = await self._scope_loader.load_area(args.area_ref)
            anchor_ref = args.near_ref
            anchor_record = await self._scope_loader.load_anchor(anchor_ref)
            requested_locality = (
                resolved_area.name
                if resolved_area is not None
                else args.city or anchor_record.locality
            )
            payload = await self._client.search(
                text=query,
                category=args.category,
                limit=fetch_limit,
                open_24h=args.open_24h,
                open_now=args.open_now,
                center=(anchor_record.lon, anchor_record.lat),
                radius_m=args.radius_m,
                context=context,
            )

        response = self._validate_response(payload)
        if resolved_area is not None and not any(
            element.type == "area" for element in response.elements
        ):
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND,
                f"OpenStreetMap city area was not found for {resolved_area.name!r}",
                provider=self._client.provider,
                retryable=False,
            )

        matches: list[_OsmMatch] = []
        seen_provider_ids: set[str] = set()
        unknown_open_now_count = 0

        for element in response.elements:
            if element.type in {"area", "count"}:
                continue

            provider_id = f"{element.type}/{element.id}"
            if provider_id in seen_provider_ids:
                continue
            seen_provider_ids.add(provider_id)

            coordinates = element.coordinates
            name = _first_tag(element.tags, _NAME_TAGS)
            if coordinates is None or name is None:
                continue

            lon, lat = coordinates
            result_locality = _first_tag(element.tags, _LOCALITY_TAGS)
            if (
                requested_locality is not None
                and result_locality is not None
                and not same_locality(requested_locality, result_locality)
            ):
                continue

            hours_text = _first_tag(element.tags, ("opening_hours",))
            open_24h = _is_open_24h(hours_text)

            if args.open_24h and open_24h is not True:
                continue

            distance_m: int | None = None
            if anchor_record is not None:
                distance_m = calculate_distance_m(
                    from_lat=anchor_record.lat,
                    from_lon=anchor_record.lon,
                    to_lat=lat,
                    to_lon=lon,
                )
                if distance_m > args.radius_m:
                    continue

            address = _address(element.tags)
            if args.mode is SearchMode.AREA and not address:
                continue

            is_open_now = (
                is_open_at(
                    hours_text,
                    lat=lat,
                    lon=lon,
                    now=reference_time,
                )
                if args.open_now
                else None
            )
            if args.open_now and is_open_now is not True:
                if is_open_now is None:
                    unknown_open_now_count += 1
                continue

            phones = _split_tag_values(
                element.tags.get("phone"),
                element.tags.get("contact:phone"),
            )
            categories = _categories(
                element.tags,
                args.category.value if args.category is not None else None,
            )

            matches.append(
                _OsmMatch(
                    provider_id=provider_id,
                    name=name,
                    address=address,
                    categories=categories,
                    phones=phones,
                    hours_text=hours_text,
                    open_24h=open_24h,
                    is_open_now=is_open_now,
                    accessibility=_accessibility(element.tags),
                    lon=lon,
                    lat=lat,
                    distance_m=distance_m,
                )
            )

        if args.open_now and unknown_open_now_count:
            context.add_warning(
                "OpenStreetMap schedules could not verify the current opening status of "
                f"{unknown_open_now_count} result(s); those results were omitted."
            )

        if args.mode is SearchMode.NEAR:
            matches.sort(
                key=lambda match: (
                    match.distance_m is None,
                    match.distance_m if match.distance_m is not None else 0,
                )
            )
            await self._reverse_geocode_nearest(matches, context)
            missing_addresses = sum(not match.address for match in matches)
            if missing_addresses:
                context.add_warning(
                    "OpenStreetMap supplied no address for "
                    f"{missing_addresses} result(s); routing still uses the exact "
                    "provider coordinates stored under each ref."
                )

        selected = matches[: args.result_limit]
        records: list[PlaceRecord] = []
        places: list[Place] = []

        for match in selected:
            ref = mint_place_ref(f"osm:{match.provider_id}")
            records.append(
                PlaceRecord(
                    ref=ref,
                    name=match.name,
                    address=match.address,
                    lat=match.lat,
                    lon=match.lon,
                    provider=self.provider,
                    provider_id=match.provider_id,
                    provider_uri=f"https://www.openstreetmap.org/{match.provider_id}",
                    origin=RecordOrigin.PLACES_SEARCH,
                )
            )
            places.append(
                Place(
                    ref=ref,
                    id=match.provider_id,
                    name=match.name,
                    address=match.address,
                    categories=match.categories,
                    phones=match.phones,
                    hours_text=match.hours_text,
                    open_24h=match.open_24h,
                    is_open_now=match.is_open_now,
                    accessibility=match.accessibility,
                    distance_m=match.distance_m,
                )
            )

        if records:
            await self._place_store.save_many(records)

        provider_total_found = response.total_found
        returned_count = sum(element.type not in {"area", "count"} for element in response.elements)

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
                (provider_total_found is not None and provider_total_found > returned_count)
                or len(matches) > args.result_limit
            ),
            anchor=anchor_ref,
        )

    def _validate_response(self, payload: dict[str, object]) -> OsmOverpassResponse:
        try:
            return OsmOverpassResponse.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "OpenStreetMap search returned data in an unexpected format",
                provider=self._client.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc

    def _internal_contract_error(self, message: str) -> ToolExecutionError:
        return ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=self._client.provider,
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        )
