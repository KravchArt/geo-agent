"""HTTP client for 2GIS Regions 2.0 and Places 3.0 APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar, cast

import httpx
from pydantic import BaseModel, ValidationError

from tools.base import (
    ToolClarification,
    ToolClarificationOption,
    ToolErrorCode,
    ToolExecutionError,
    ToolFailureKind,
)
from tools.geo.places_search.tomtom.matching import normalized_tokens
from tools.geo.places_search.twogis.schemas import (
    TwoGisItem,
    TwoGisItemsResponse,
    TwoGisRegion,
    TwoGisRegionsResponse,
    TwoGisRubricSearchItem,
    TwoGisRubricSearchResponse,
)
from tools.geo.text_place_query import twogis_response_locale
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class _CoverageCacheMiss:
    pass


_CACHE_MISS = _CoverageCacheMiss()


@dataclass(frozen=True, slots=True)
class _CachedCoverageNotFound:
    """Recreate a cached no-coverage error without another upstream call."""

    status_code: int | None
    provider_code: str | None

    def to_error(self, *, provider: str) -> ToolExecutionError:
        return ToolExecutionError(
            ToolErrorCode.NOT_FOUND,
            "2GIS does not cover the requested locality",
            status_code=self.status_code,
            provider=provider,
            provider_code=self.provider_code,
            failure_kind=ToolFailureKind.COVERAGE_MISS,
            retryable=False,
        )


_ITEM_FIELDS = ",".join(
    (
        "items.point",
        "items.address",
        "items.full_address_name",
        "items.adm_div",
        "items.rubrics",
        "items.org",
        "items.brand",
        "items.schedule",
        "items.reviews",
        "items.name_ex",
    )
)


class TwoGisSearchClient:
    """Execute bounded 2GIS searches and cache locality coverage decisions."""

    provider = "twogis_search"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://catalog.api.2gis.com",
        timeout_s: float = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("2GIS API key cannot be empty")
        if not base_url.strip():
            raise ValueError("2GIS catalog base URL cannot be empty")
        if timeout_s <= 0:
            raise ValueError("2GIS catalog timeout must be positive")

        self._api_key = api_key
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._coverage_cache: dict[
            tuple[str, ...],
            TwoGisRegion | None | _CachedCoverageNotFound,
        ] = {}
        self._coverage_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._city_cache: dict[tuple[str, ...], TwoGisItem | None] = {}
        self._city_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._rubric_cache: dict[tuple[str, ...], tuple[TwoGisRubricSearchItem, ...]] = {}
        self._rubric_locks: dict[tuple[str, ...], asyncio.Lock] = {}

    async def find_region(
        self,
        locality: str,
        context: ToolExecutionContext,
    ) -> TwoGisRegion | None:
        """Return a detailed 2GIS region covering a city or settlement."""

        cache_key = normalized_tokens(locality)
        if not cache_key:
            raise ValueError("2GIS coverage locality cannot be blank")
        cached = self._cached_coverage(cache_key)
        if not isinstance(cached, _CoverageCacheMiss):
            if isinstance(cached, _CachedCoverageNotFound):
                raise cached.to_error(provider=self.provider)
            return cached

        lock = self._coverage_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._cached_coverage(cache_key)
            if not isinstance(cached, _CoverageCacheMiss):
                if isinstance(cached, _CachedCoverageNotFound):
                    raise cached.to_error(provider=self.provider)
                return cached

            try:
                response = await self._request(
                    path="/2.0/region/search",
                    params={
                        "key": self._api_key,
                        "q": locality.strip(),
                        "type": "region",
                        "locale": twogis_response_locale(locality),
                        "page_size": "10",
                        "fields": "items.country_code,items.settlements,items.satellites",
                    },
                    operation="region_search",
                    response_model=TwoGisRegionsResponse,
                    context=context,
                )
            except ToolExecutionError as exc:
                if exc.error_code is ToolErrorCode.NOT_FOUND:
                    self._coverage_cache[cache_key] = _CachedCoverageNotFound(
                        status_code=exc.status_code,
                        provider_code=exc.provider_code,
                    )
                raise
            region = _select_region(response, locality)
            if region is None:
                clarification = _ambiguous_region_clarification(response, locality)
                if clarification is not None:
                    raise ToolExecutionError(
                        ToolErrorCode.INVALID_INPUT,
                        f"City is ambiguous in 2GIS coverage: {locality!r}.",
                        provider=self.provider,
                        retryable=False,
                        clarification=clarification,
                    )
            self._coverage_cache[cache_key] = region
            return region

    def _cached_coverage(
        self,
        cache_key: tuple[str, ...],
    ) -> TwoGisRegion | None | _CachedCoverageNotFound | _CoverageCacheMiss:
        return self._coverage_cache.get(cache_key, _CACHE_MISS)

    async def find_region_at_point(
        self,
        *,
        lon: float,
        lat: float,
        context: ToolExecutionContext,
    ) -> TwoGisRegion | None:
        """Return the 2GIS project that covers one already-resolved point.

        Unlike text Region Search, this query has no fuzzy locality ranking:
        the point comes from the shared geocoder record stored under an opaque
        ref.  A project returned for that point is therefore a safe coverage
        decision for the 2GIS catalog.
        """

        _validate_point(lon=lon, lat=lat)
        cache_key = _point_cache_key(lon=lon, lat=lat)
        cached = self._cached_coverage(cache_key)
        if not isinstance(cached, _CoverageCacheMiss):
            if isinstance(cached, _CachedCoverageNotFound):
                raise cached.to_error(provider=self.provider)
            return cached

        lock = self._coverage_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._cached_coverage(cache_key)
            if not isinstance(cached, _CoverageCacheMiss):
                if isinstance(cached, _CachedCoverageNotFound):
                    raise cached.to_error(provider=self.provider)
                return cached

            try:
                response = await self._request(
                    path="/2.0/region/search",
                    params={
                        "key": self._api_key,
                        "q": f"{lon:.6f},{lat:.6f}",
                        "type": "region",
                        "page_size": "2",
                        "fields": "items.country_code",
                    },
                    operation="region_point_search",
                    response_model=TwoGisRegionsResponse,
                    context=context,
                )
            except ToolExecutionError as exc:
                if exc.error_code is ToolErrorCode.NOT_FOUND:
                    self._coverage_cache[cache_key] = _CachedCoverageNotFound(
                        status_code=exc.status_code,
                        provider_code=exc.provider_code,
                    )
                raise

            region = _select_region_at_point(response)
            self._coverage_cache[cache_key] = region
            return region

    async def find_city(
        self,
        locality: str,
        context: ToolExecutionContext,
        *,
        expected_region_id: str,
        country_code: str | None = None,
    ) -> TwoGisItem | None:
        """Resolve a city from a 2GIS text-search result scoped to one project."""

        city_name = _base_locality(locality)
        locality_key = normalized_tokens(city_name)
        if not locality_key:
            raise ValueError("2GIS city locality cannot be blank")
        if not expected_region_id.strip():
            raise ValueError("2GIS expected city region id cannot be blank")
        # City labels are not globally unique.  The project id comes from the
        # Regions API and is used only to validate the returned item below;
        # passing it as ``region_id`` in the HTTP request causes 2GIS to reject
        # some otherwise valid city lookups.
        normalized_country_code = country_code.strip().casefold() if country_code else ""
        cache_key = (
            expected_region_id.strip(),
            normalized_country_code,
            *locality_key,
        )
        cached = self._city_cache.get(cache_key, _CACHE_MISS)
        if not isinstance(cached, _CoverageCacheMiss):
            return cached

        lock = self._city_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._city_cache.get(cache_key, _CACHE_MISS)
            if not isinstance(cached, _CoverageCacheMiss):
                return cached

            response = await self._request(
                path="/3.0/items",
                params={
                    "key": self._api_key,
                    "q": city_name,
                    # Keep the documented text-based city lookup, but exclude
                    # similarly named regions, districts, and POIs before the
                    # exact city-name check below. Do not pass ``region_id``:
                    # 2GIS rejects that combination for some covered cities.
                    "type": "adm_div.city",
                    "locale": twogis_response_locale(
                        locality,
                        country_code=country_code,
                    ),
                    "page_size": "10",
                    "fields": "items.region_id,items.adm_div,items.point",
                },
                operation="city_search",
                response_model=TwoGisItemsResponse,
                context=context,
            )
            city = _select_city(
                response,
                city_name,
                expected_region_id=expected_region_id,
            )
            self._city_cache[cache_key] = city
            return city

    async def find_city_at_point(
        self,
        *,
        lon: float,
        lat: float,
        expected_region_id: str,
        locale: str,
        context: ToolExecutionContext,
    ) -> TwoGisItem | None:
        """Resolve the city containing a point inside a confirmed 2GIS project."""

        _validate_point(lon=lon, lat=lat)
        if not expected_region_id.strip():
            raise ValueError("2GIS expected city region id cannot be blank")
        if not locale.strip():
            raise ValueError("2GIS city response locale cannot be blank")
        cache_key = (
            "point",
            expected_region_id.strip(),
            locale.strip(),
            f"{lon:.6f}",
            f"{lat:.6f}",
        )
        cached = self._city_cache.get(cache_key, _CACHE_MISS)
        if not isinstance(cached, _CoverageCacheMiss):
            return cached

        lock = self._city_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._city_cache.get(cache_key, _CACHE_MISS)
            if not isinstance(cached, _CoverageCacheMiss):
                return cached

            response = await self._request(
                path="/3.0/items",
                params={
                    "key": self._api_key,
                    "lon": f"{lon:.6f}",
                    "lat": f"{lat:.6f}",
                    "type": "adm_div.city",
                    "locale": locale.strip(),
                    "page_size": "10",
                    "fields": "items.region_id,items.adm_div,items.point",
                },
                operation="city_point_search",
                response_model=TwoGisItemsResponse,
                context=context,
            )
            city = _select_city_at_point(response, expected_region_id=expected_region_id)
            self._city_cache[cache_key] = city
            return city

    async def search_places(
        self,
        *,
        query: str | None,
        context: ToolExecutionContext,
        page_size: int = 10,
        center: tuple[float, float] | None = None,
        radius_m: int | None = None,
        bbox: GeoBounds | None = None,
        region_id: str | None = None,
        city_id: str | None = None,
        open_now: bool = False,
        sort_by_rating: bool = False,
        locale: str | None = None,
        rubric_id: str | None = None,
    ) -> TwoGisItemsResponse:
        """Search current 2GIS catalog data using one geographic restriction."""

        if query is not None and not query.strip():
            raise ValueError("2GIS search query cannot be blank")
        if query is None and rubric_id is None:
            raise ValueError("2GIS search requires query or rubric_id")
        if not 1 <= page_size <= 10:
            raise ValueError("2GIS page_size must be between 1 and 10")
        if (center is None) != (radius_m is None):
            raise ValueError("2GIS center and radius_m must be passed together")
        local_filters = sum(value is not None for value in (center, bbox, city_id))
        if local_filters > 1 or (region_id is not None and local_filters and rubric_id is None):
            raise ValueError("2GIS geographic restrictions cannot be combined")
        if radius_m is not None and not 1 <= radius_m <= 40_000:
            raise ValueError("2GIS radius_m must be between 1 and 40000")
        if center is not None and radius_m is not None and query is None and radius_m > 2_000:
            raise ValueError("2GIS queryless point search radius cannot exceed 2000")

        response_locale = locale or twogis_response_locale(query or "")
        if not response_locale.strip():
            raise ValueError("2GIS response locale cannot be blank")
        if rubric_id is not None and not rubric_id.strip():
            raise ValueError("2GIS rubric_id cannot be blank")
        if rubric_id is not None and region_id is None:
            raise ValueError("2GIS rubric search requires region_id")
        if region_id is not None and not region_id.strip():
            raise ValueError("2GIS region_id cannot be blank")

        params: dict[str, str] = {
            "key": self._api_key,
            "locale": response_locale.strip(),
            "page_size": str(page_size),
            "fields": _ITEM_FIELDS,
        }
        if query is not None:
            params["q"] = query.strip()
        if region_id is not None:
            params["region_id"] = region_id.strip()
        if center is not None and radius_m is not None:
            lon, lat = center
            params["point"] = f"{lon:.6f},{lat:.6f}"
            params["radius"] = str(radius_m)
            params["sort"] = "distance"
        elif bbox is not None:
            params["point1"] = f"{bbox.west:.6f},{bbox.north:.6f}"
            params["point2"] = f"{bbox.east:.6f},{bbox.south:.6f}"
        elif city_id is not None:
            params["city_id"] = city_id
        if rubric_id is not None:
            params["rubric_id"] = rubric_id.strip()
        if open_now:
            params["work_time"] = "now"
        if sort_by_rating:
            # 2GIS exposes rating sorting and a presence filter, but no numeric
            # threshold. The provider applies the requested min_rating locally.
            params["sort"] = "rating"
            params["has_rating"] = "true"

        return await self._request(
            path="/3.0/items",
            params=params,
            operation="places_search",
            response_model=TwoGisItemsResponse,
            context=context,
        )

    async def find_rubrics(
        self,
        *,
        query: str,
        region_id: str,
        locale: str,
        context: ToolExecutionContext,
        allowed_aliases: tuple[str, ...] = (),
    ) -> tuple[TwoGisRubricSearchItem, ...]:
        """Resolve a category to all explicitly accepted regional rubrics.

        Without ``allowed_aliases`` this retains the strict exact-name/alias
        behaviour used outside the reviewed 2GIS mapping. With aliases, fuzzy
        ranking is ignored and only mapped semantic equivalents are returned.
        """

        normalized_query = normalized_tokens(query)
        if not normalized_query:
            raise ValueError("2GIS rubric query cannot be blank")
        if not region_id.strip():
            raise ValueError("2GIS rubric region id cannot be blank")
        if not locale.strip():
            raise ValueError("2GIS rubric locale cannot be blank")

        normalized_aliases = tuple(
            alias.strip().casefold() for alias in allowed_aliases if alias.strip()
        )
        cache_key = (
            region_id.strip(),
            locale.strip(),
            *normalized_query,
            "__allowed_aliases__",
            *normalized_aliases,
        )
        cached = self._rubric_cache.get(cache_key, _CACHE_MISS)
        if not isinstance(cached, _CoverageCacheMiss):
            return cached

        lock = self._rubric_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._rubric_cache.get(cache_key, _CACHE_MISS)
            if not isinstance(cached, _CoverageCacheMiss):
                return cached

            response = await self._request(
                path="/2.0/catalog/rubric/search",
                params={
                    "key": self._api_key,
                    "region_id": region_id.strip(),
                    "q": query.strip(),
                    "locale": locale.strip(),
                    # Category search is fuzzy and its first result is not
                    # necessarily the requested rubric. Fetch enough results
                    # to select an exact name or alias locally.
                    "page_size": "50",
                },
                operation="rubric_search",
                response_model=TwoGisRubricSearchResponse,
                context=context,
            )
            rubrics = _select_rubrics(
                response.result.items,
                query=query,
                allowed_aliases=normalized_aliases,
            )
            self._rubric_cache[cache_key] = rubrics
            return rubrics

    async def _request(
        self,
        *,
        path: str,
        params: Mapping[str, str],
        operation: str,
        response_model: type[ResponseModelT],
        context: ToolExecutionContext,
    ) -> ResponseModelT:
        started = perf_counter()
        response: httpx.Response | None = None

        try:
            try:
                response = await self._http_client.get(
                    f"{self._base_url}{path}",
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=self._timeout_s,
                )
            except httpx.TimeoutException as exc:
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    "2GIS search timed out",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.TIMEOUT,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "2GIS search is unavailable",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.NETWORK,
                    retryable=True,
                ) from exc

            try:
                raw_payload: Any = response.json()
            except ValueError as exc:
                raise self._invalid_json_or_status(response) from exc
            if not isinstance(raw_payload, Mapping):
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "2GIS returned an unexpected response format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                )

            payload = cast(dict[str, Any], dict(raw_payload))
            meta_code = _meta_code(payload)
            effective_status = meta_code or response.status_code
            status_error = self._status_error(
                effective_status,
                http_status=response.status_code,
                provider_code=_provider_code(payload),
            )
            if (
                status_error is not None
                and operation in {"region_search", "region_point_search"}
                and status_error.error_code is ToolErrorCode.NOT_FOUND
            ):
                status_error = ToolExecutionError(
                    ToolErrorCode.NOT_FOUND,
                    "2GIS does not cover the requested locality",
                    status_code=status_error.status_code,
                    provider=self.provider,
                    provider_code=status_error.provider_code,
                    failure_kind=ToolFailureKind.COVERAGE_MISS,
                    retryable=False,
                )
            if status_error is not None:
                raise status_error

            try:
                typed_response = response_model.model_validate(payload)
            except ValidationError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "2GIS returned data in an unexpected format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                ) from exc
        except asyncio.CancelledError:
            context.record_cancelled_upstream_call(
                provider=self.provider,
                operation=operation,
                latency_ms=int((perf_counter() - started) * 1_000),
                status_code=response.status_code if response is not None else None,
            )
            raise
        except ToolExecutionError as exc:
            context.record_upstream_call(
                provider=self.provider,
                operation=operation,
                latency_ms=int((perf_counter() - started) * 1_000),
                outcome=UpstreamCallOutcome.FAILURE,
                status_code=exc.status_code
                or (response.status_code if response is not None else None),
                error_code=exc.error_code.value,
                failure_kind=exc.failure_kind.value if exc.failure_kind is not None else None,
                provider_code=exc.provider_code,
                retryable=exc.retryable,
            )
            raise

        context.record_upstream_call(
            provider=self.provider,
            operation=operation,
            latency_ms=int((perf_counter() - started) * 1_000),
            outcome=UpstreamCallOutcome.SUCCESS,
            status_code=response.status_code,
        )
        return typed_response

    def _invalid_json_or_status(self, response: httpx.Response) -> ToolExecutionError:
        status_error = self._status_error(
            response.status_code,
            http_status=response.status_code,
            provider_code=None,
        )
        return status_error or ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "2GIS returned invalid JSON",
            status_code=response.status_code,
            provider=self.provider,
            failure_kind=ToolFailureKind.INVALID_JSON,
        )

    def _status_error(
        self,
        effective_status: int,
        *,
        http_status: int,
        provider_code: str | None,
    ) -> ToolExecutionError | None:
        if effective_status < 400 and http_status < 400:
            return None
        status = max(effective_status, http_status)
        if (
            provider_code is not None
            and provider_code.casefold() == "backendexception"
            and (effective_status == 403 or http_status == 403)
        ):
            return ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS search backend temporarily failed",
                status_code=status,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=True,
            )
        if effective_status in {401, 403} or http_status in {401, 403}:
            return ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS search authentication failed",
                status_code=status,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.AUTHENTICATION,
            )
        if effective_status == 404 or http_status == 404:
            error_code = ToolErrorCode.NOT_FOUND
            message = "2GIS search returned no results"
            retryable = False
        elif effective_status in {408, 504} or http_status in {408, 504}:
            error_code = ToolErrorCode.TIMEOUT
            message = "2GIS search timed out"
            retryable = True
        elif effective_status == 429 or http_status == 429:
            error_code = ToolErrorCode.RATE_LIMITED
            message = "2GIS search rate limit exceeded"
            retryable = True
        else:
            error_code = ToolErrorCode.UPSTREAM_ERROR
            message = "2GIS search returned an HTTP error"
            retryable = status >= 500
        return ToolExecutionError(
            error_code,
            message,
            status_code=status,
            provider=self.provider,
            provider_code=provider_code,
            failure_kind=ToolFailureKind.HTTP_STATUS,
            retryable=retryable,
        )


def _select_region(response: TwoGisRegionsResponse, locality: str) -> TwoGisRegion | None:
    regions = [item for item in response.result.items if item.type == "region"]
    if not regions:
        return None

    exact = _exact_regions(regions, locality)
    qualifier_tokens = set(normalized_tokens(_locality_qualifier(locality)))
    if qualifier_tokens:
        qualified = [
            region
            for region in exact
            if qualifier_tokens
            <= set(
                normalized_tokens(
                    " ".join((region.name, region.country_code or "")),
                )
            )
        ]
        if len(qualified) == 1:
            return qualified[0]
    if len(exact) == 1:
        return exact[0]
    # Region Search ranking is not an identity guarantee: an unsupported
    # locality may still yield one unrelated catalog project (for example,
    # ``Krakow`` -> Moscow).  Satellites and settlements that 2GIS explicitly
    # associates with the project have already matched in ``_exact_regions``.
    # Do not turn a lone fuzzy result into a geographic restriction.
    return None


def _select_region_at_point(response: TwoGisRegionsResponse) -> TwoGisRegion | None:
    """Return a project only when the coordinate lookup is unambiguous."""

    regions = [item for item in response.result.items if item.type == "region"]
    return regions[0] if len(regions) == 1 and response.result.total == 1 else None


def _ambiguous_region_clarification(
    response: TwoGisRegionsResponse,
    locality: str,
) -> ToolClarification | None:
    """Build retryable city strings for several exact 2GIS coverage regions."""

    regions = [item for item in response.result.items if item.type == "region"]
    exact = _exact_regions(regions, locality)
    if len(exact) < 2:
        return None

    options: list[ToolClarificationOption] = []
    seen_values: set[str] = set()
    for region in exact:
        parts = [locality.strip()]
        if normalized_tokens(region.name) != normalized_tokens(locality):
            parts.append(region.name)
        if region.country_code is not None:
            parts.append(region.country_code.upper())
        value = ", ".join(parts)
        normalized_value = value.casefold()
        if normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        description = f"2GIS coverage region: {region.name}"
        if region.country_code is not None:
            description += f" ({region.country_code.upper()})"
        options.append(
            ToolClarificationOption(
                value=value,
                label=value,
                description=description,
            )
        )
        if len(options) == 3:
            break

    if len(options) < 2:
        return None
    return ToolClarification(
        kind="select_area_query",
        question=f"Which city matching {locality!r} do you mean?",
        options=options,
    )


def _exact_regions(
    regions: list[TwoGisRegion],
    locality: str,
) -> list[TwoGisRegion]:
    requested = normalized_tokens(_base_locality(locality))
    return [
        region
        for region in regions
        if normalized_tokens(region.name) == requested
        or any(normalized_tokens(settlement) == requested for settlement in region.settlements)
        or any(normalized_tokens(satellite.name) == requested for satellite in region.satellites)
    ]


def _base_locality(locality: str) -> str:
    """Return the city part of a tool-produced ``city, region, country`` value."""

    return locality.partition(",")[0].strip()


def _locality_qualifier(locality: str) -> str:
    """Return the optional region/country part used to select 2GIS coverage."""

    return locality.partition(",")[2].strip()


def _select_city(
    response: TwoGisItemsResponse,
    locality: str,
    *,
    expected_region_id: str,
) -> TwoGisItem | None:
    """Select a unique text-search city belonging to the covered project.

    ``locale`` controls matching but administrative city names may still be
    returned in their canonical catalog language (for example, ``Moscow`` →
    ``Москва``).  Therefore a string-equality check alone would discard a
    correct result.  The project-id check makes the unique provider candidate
    safe to use without treating an arbitrary city from another catalog as a
    match.
    """

    requested = normalized_tokens(locality)
    cities_in_expected_region = [
        item
        for item in response.result.items
        if item.type == "adm_div"
        and item.subtype == "city"
        and item.region_id == expected_region_id.strip()
    ]
    exact = [
        item for item in cities_in_expected_region if normalized_tokens(item.name) == requested
    ]
    if len(exact) == 1:
        return exact[0]
    return cities_in_expected_region[0] if len(cities_in_expected_region) == 1 else None


def _select_city_at_point(
    response: TwoGisItemsResponse,
    *,
    expected_region_id: str,
) -> TwoGisItem | None:
    """Return the sole city at a point belonging to the covered project."""

    cities = [
        item
        for item in response.result.items
        if item.type == "adm_div"
        and item.subtype == "city"
        and item.region_id == expected_region_id.strip()
    ]
    return cities[0] if len(cities) == 1 else None


def _select_exact_rubric(
    rubrics: list[TwoGisRubricSearchItem],
    *,
    query: str,
) -> TwoGisRubricSearchItem | None:
    """Select an exact category result instead of trusting fuzzy ranking."""

    requested = normalized_tokens(query.replace("_", " "))
    exact_name = [rubric for rubric in rubrics if normalized_tokens(rubric.name) == requested]
    if exact_name:
        return exact_name[0]

    exact_alias = [
        rubric
        for rubric in rubrics
        if rubric.alias is not None
        and normalized_tokens(rubric.alias.replace("_", " ")) == requested
    ]
    return exact_alias[0] if exact_alias else None


def _select_rubrics(
    rubrics: list[TwoGisRubricSearchItem],
    *,
    query: str,
    allowed_aliases: tuple[str, ...],
) -> tuple[TwoGisRubricSearchItem, ...]:
    if not allowed_aliases:
        exact = _select_exact_rubric(rubrics, query=query)
        return (exact,) if exact is not None else ()

    by_alias = {
        rubric.alias.strip().casefold(): rubric
        for rubric in rubrics
        if rubric.alias is not None and rubric.alias.strip()
    }
    selected: list[TwoGisRubricSearchItem] = []
    selected_ids: set[str] = set()
    for alias in allowed_aliases:
        rubric = by_alias.get(alias)
        if rubric is not None and rubric.id not in selected_ids:
            selected.append(rubric)
            selected_ids.add(rubric.id)
    return tuple(selected)


def _validate_point(*, lon: float, lat: float) -> None:
    if not -180 <= lon <= 180:
        raise ValueError("2GIS longitude must be between -180 and 180")
    if not -90 <= lat <= 90:
        raise ValueError("2GIS latitude must be between -90 and 90")


def _point_cache_key(*, lon: float, lat: float) -> tuple[str, str, str]:
    return ("point", f"{lon:.6f}", f"{lat:.6f}")


def _meta_code(payload: Mapping[str, Any]) -> int | None:
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        return None
    code = meta.get("code")
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _provider_code(payload: Mapping[str, Any]) -> str | None:
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        return None
    for key in ("error", "error_type", "code"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            for nested_key in ("type", "code"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return None
