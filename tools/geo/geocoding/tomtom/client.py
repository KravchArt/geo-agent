"""HTTP client for TomTom Geocoding API v2."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any
from urllib.parse import quote

import httpx

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.text_place_query import tomtom_response_language
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


class TomTomGeocoderClient:
    """Execute forward-geocoding requests and preserve provider diagnostics."""

    provider = "tomtom_geocoder"

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
            raise ValueError("TomTom Geocoding base URL cannot be empty")
        self._api_key = api_key
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        limit: int,
        context: ToolExecutionContext,
        entity_type_set: str | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("TomTom geocoding query cannot be empty")
        if not 1 <= limit <= 100:
            raise ValueError("TomTom geocoding limit must be between 1 and 100")

        url = f"{self._base_url}/geocode/{quote(query.strip(), safe='')}.json"
        params = {
            "key": self._api_key,
            "limit": str(limit),
            "language": tomtom_response_language(query),
            "view": "RU",
        }
        if entity_type_set is not None:
            params["entityTypeSet"] = entity_type_set
        return await self._request(
            url=url,
            params=params,
            operation="search",
            context=context,
        )

    async def reverse_search(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one coordinate into the nearest address or requested geography."""

        if not -90 <= lat <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= lon <= 180:
            raise ValueError("longitude must be between -180 and 180")
        if not language.strip():
            raise ValueError("TomTom reverse-geocoding language cannot be empty")

        url = f"{self._base_url}/reverseGeocode/{lat:.6f},{lon:.6f}.json"
        params = {
            "key": self._api_key,
            "language": language.strip(),
            "view": "RU",
        }
        if entity_type is not None:
            params["entityType"] = entity_type
        return await self._request(
            url=url,
            params=params,
            operation="reverse_geocode",
            context=context,
        )

    async def _request(
        self,
        *,
        url: str,
        params: Mapping[str, str],
        operation: str,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        started = perf_counter()
        response: httpx.Response | None = None
        try:
            try:
                response = await self._http_client.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                )
            except httpx.TimeoutException as exc:
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    "TomTom geocoder timed out",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.TIMEOUT,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom geocoder is unavailable",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.NETWORK,
                    retryable=True,
                ) from exc

            try:
                payload: Any = response.json()
            except ValueError as exc:
                error = self._http_error(response.status_code)
                if error is not None:
                    raise error from exc
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom geocoder returned invalid JSON",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_JSON,
                    retryable=False,
                ) from exc

            error = self._http_error(response.status_code)
            if error is not None:
                raise error
            if not isinstance(payload, Mapping):
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "TomTom geocoder returned an unexpected response format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                    retryable=False,
                )
            result = dict(payload)
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
        return result

    def _http_error(self, status_code: int) -> ToolExecutionError | None:
        if status_code < 400:
            return None
        if status_code in {408, 504}:
            error_code = ToolErrorCode.TIMEOUT
            message = "TomTom geocoder timed out"
            retryable = True
        elif status_code == 429:
            error_code = ToolErrorCode.RATE_LIMITED
            message = "TomTom geocoder rate limit exceeded"
            retryable = True
        elif status_code in {401, 403}:
            error_code = ToolErrorCode.UPSTREAM_ERROR
            message = "TomTom geocoder authentication failed"
            retryable = False
        else:
            error_code = ToolErrorCode.UPSTREAM_ERROR
            message = "TomTom geocoder returned an HTTP error"
            retryable = status_code >= 500
        return ToolExecutionError(
            error_code,
            message,
            status_code=status_code,
            provider=self.provider,
            failure_kind=(
                ToolFailureKind.AUTHENTICATION
                if status_code in {401, 403}
                else ToolFailureKind.HTTP_STATUS
            ),
            retryable=retryable,
        )
