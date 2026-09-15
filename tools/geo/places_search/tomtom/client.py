"""HTTP client for TomTom POI Search v2 endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from time import perf_counter
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.places_search.tomtom.schemas import TomTomSearchResponse
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds


class TomTomSearchEndpoint(StrEnum):
    """TomTom endpoint selected from the already resolved search intent."""

    FUZZY = "search"
    CATEGORY = "categorySearch"
    NEARBY = "nearbySearch"


class TomTomSearchClient:
    """Execute bounded POI searches while preserving provider error metadata."""

    provider = "tomtom_search"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://api.tomtom.com/search/2",
    ) -> None:
        if not api_key.strip():
            raise ValueError("TomTom API key cannot be empty")
        if not base_url.strip():
            raise ValueError("TomTom Search base URL cannot be empty")

        self._api_key = api_key
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    async def search(
        self,
        *,
        endpoint: TomTomSearchEndpoint,
        query: str | None,
        category_ids: tuple[int, ...] = (),
        limit: int,
        context: ToolExecutionContext,
        language: str = "NGT",
        center: tuple[float, float] | None = None,
        radius_m: int | None = None,
        bbox: GeoBounds | None = None,
    ) -> TomTomSearchResponse:
        """Execute the endpoint chosen by the provider's search-intent logic."""

        if endpoint is TomTomSearchEndpoint.NEARBY:
            if query is not None:
                raise ValueError("TomTom Nearby Search is queryless")
            if not category_ids:
                raise ValueError("TomTom Nearby Search requires category IDs")
            if center is None:
                raise ValueError("TomTom Nearby Search requires a search center")
        elif query is None or not query.strip():
            raise ValueError("TomTom Fuzzy and Category Search require a query")
        if any(category_id < 1 for category_id in category_ids):
            raise ValueError("TomTom category IDs must be positive")
        if not 1 <= limit <= 100:
            raise ValueError("TomTom search limit must be between 1 and 100")
        if not language.strip():
            raise ValueError("TomTom response language cannot be blank")
        if (center is None) != (radius_m is None):
            raise ValueError("center and radius_m must be passed together")
        if bbox is not None and center is not None:
            raise ValueError("bbox cannot be combined with center and radius")
        if radius_m is not None and radius_m < 1:
            raise ValueError("radius_m must be positive")

        params: dict[str, str] = {
            "key": self._api_key,
            "typeahead": "false",
            "limit": str(limit),
            # Callers resolving a textual anchor can request the query's
            # language so conservative lexical matching remains possible
            # across translated place names. Other searches keep NGT.
            "language": language.strip(),
            "view": "RU",
            # These fields are returned in the same search request and therefore
            # do not consume an additional TomTom transaction.
            "openingHours": "nextSevenDays",
            "timeZone": "iana",
        }
        if endpoint is TomTomSearchEndpoint.FUZZY:
            # Fuzzy Search can return addresses and geographies; named-POI
            # searches need only POIs. Category Search is already POI-only and
            # does not expose idxSet.
            params["idxSet"] = "POI"
        if category_ids:
            params["categorySet"] = ",".join(str(category_id) for category_id in category_ids)
        if bbox is not None:
            params["topLeft"] = f"{bbox.north:.6f},{bbox.west:.6f}"
            params["btmRight"] = f"{bbox.south:.6f},{bbox.east:.6f}"
        if center is not None and radius_m is not None:
            lon, lat = center
            params["lat"] = f"{lat:.6f}"
            params["lon"] = f"{lon:.6f}"
            params["radius"] = str(radius_m)

        if endpoint is TomTomSearchEndpoint.NEARBY:
            url = f"{self._base_url}/nearbySearch/.json"
        else:
            assert query is not None  # Guarded by validation above.
            url = f"{self._base_url}/{endpoint.value}/{quote(query.strip(), safe='')}.json"
        return await self._request(url=url, params=params, context=context)

    async def _request(
        self,
        *,
        url: str,
        params: Mapping[str, str],
        context: ToolExecutionContext,
    ) -> TomTomSearchResponse:
        started = perf_counter()
        response: httpx.Response | None = None

        try:
            try:
                response = await self._http_client.get(
                    url,
                    params=params,
                    headers={
                        "Accept": "application/json",
                    },
                )
            except httpx.TimeoutException as exc:
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    "TomTom search timed out",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.TIMEOUT,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom search is unavailable",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.NETWORK,
                    retryable=True,
                ) from exc

            try:
                payload: Any = response.json()
            except ValueError as exc:
                status_error = self._http_error(
                    response.status_code,
                    provider_code=None,
                    quota_exceeded=False,
                )
                if status_error is not None:
                    raise status_error from exc
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom search returned invalid JSON",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_JSON,
                    retryable=False,
                ) from exc

            if not isinstance(payload, Mapping):
                status_error = self._http_error(
                    response.status_code,
                    provider_code=None,
                    quota_exceeded=False,
                )
                if status_error is not None:
                    raise status_error
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom search returned an unexpected response format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                    retryable=False,
                )

            result = cast(dict[str, Any], dict(payload))
            provider_code = self._provider_code(result)
            status_error = self._http_error(
                response.status_code,
                provider_code=provider_code,
                quota_exceeded=self._is_quota_error(result),
            )
            if status_error is not None:
                raise status_error

            result, omitted_blank_names = _without_blank_poi_names(result)
            if omitted_blank_names:
                if not result["results"]:
                    raise ToolExecutionError(
                        ToolErrorCode.UPSTREAM_ERROR,
                        "TomTom search returned only malformed results",
                        status_code=response.status_code,
                        provider=self.provider,
                        failure_kind=ToolFailureKind.INVALID_SCHEMA,
                        retryable=False,
                    )
                context.add_warning(
                    f"TomTom omitted {omitted_blank_names} malformed result(s) with blank names."
                )
            try:
                typed_result = TomTomSearchResponse.model_validate(result)
            except ValidationError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom search returned data in an unexpected format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                    retryable=False,
                ) from exc
        except asyncio.CancelledError:
            context.record_cancelled_upstream_call(
                provider=self.provider,
                operation="search",
                latency_ms=int((perf_counter() - started) * 1_000),
                status_code=response.status_code if response is not None else None,
            )
            raise
        except ToolExecutionError as exc:
            context.record_upstream_call(
                provider=self.provider,
                operation="search",
                latency_ms=int((perf_counter() - started) * 1_000),
                outcome=UpstreamCallOutcome.FAILURE,
                status_code=exc.status_code
                or (response.status_code if response is not None else None),
                error_code=exc.error_code.value,
                failure_kind=(exc.failure_kind.value if exc.failure_kind is not None else None),
                provider_code=exc.provider_code,
                retryable=exc.retryable,
            )
            raise

        context.record_upstream_call(
            provider=self.provider,
            operation="search",
            latency_ms=int((perf_counter() - started) * 1_000),
            outcome=UpstreamCallOutcome.SUCCESS,
            status_code=response.status_code,
        )
        return typed_result

    def _http_error(
        self,
        status_code: int,
        *,
        provider_code: str | None,
        quota_exceeded: bool,
    ) -> ToolExecutionError | None:
        if status_code < 400:
            return None
        # TomTom documents 403 for both invalid authorization and exhausted
        # rate/volume quota. Preserve authentication failures, but classify an
        # explicitly identified quota response as retryable so OSM can take over.
        if status_code == 403 and quota_exceeded:
            return ToolExecutionError(
                ToolErrorCode.RATE_LIMITED,
                "TomTom search rate limit exceeded",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=True,
            )
        if status_code in {401, 403}:
            return ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "TomTom search authentication failed",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.AUTHENTICATION,
                retryable=False,
            )
        if status_code in {408, 504}:
            error_code = ToolErrorCode.TIMEOUT
            message = "TomTom search timed out"
            retryable = True
        elif status_code == 429:
            error_code = ToolErrorCode.RATE_LIMITED
            message = "TomTom search rate limit exceeded"
            retryable = True
        else:
            error_code = ToolErrorCode.UPSTREAM_ERROR
            message = "TomTom search returned an HTTP error"
            retryable = status_code >= 500

        return ToolExecutionError(
            error_code,
            message,
            status_code=status_code,
            provider=self.provider,
            provider_code=provider_code,
            failure_kind=ToolFailureKind.HTTP_STATUS,
            retryable=retryable,
        )

    @staticmethod
    def _provider_code(payload: Mapping[str, Any]) -> str | None:
        detailed_error = payload.get("detailedError")
        if isinstance(detailed_error, Mapping):
            code = detailed_error.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip()
        return None

    @staticmethod
    def _is_quota_error(payload: Mapping[str, Any]) -> bool:
        """Recognize only explicit quota wording in TomTom's ambiguous 403."""

        candidates: list[str] = []
        message = payload.get("message")
        if isinstance(message, str):
            candidates.append(message)

        detailed_error = payload.get("detailedError")
        if isinstance(detailed_error, Mapping):
            for key in ("code", "message"):
                value = detailed_error.get(key)
                if isinstance(value, str):
                    candidates.append(value)

        normalized = " ".join(candidates).casefold()
        return any(
            marker in normalized
            for marker in (
                "rate limit",
                "volume limit",
                "quota",
                "over qps",
                "over qpd",
            )
        )


def _without_blank_poi_names(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Drop only TomTom's known blank-name POI cards before strict validation."""

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return payload, 0

    usable_results: list[Any] = []
    omitted = 0
    for result in raw_results:
        if isinstance(result, Mapping):
            poi = result.get("poi")
            if isinstance(poi, Mapping):
                name = poi.get("name")
                if isinstance(name, str) and not name.strip():
                    omitted += 1
                    continue
        usable_results.append(result)

    if omitted == 0:
        return payload, 0

    normalized = dict(payload)
    normalized["results"] = usable_results
    return normalized, omitted
