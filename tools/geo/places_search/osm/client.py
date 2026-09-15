"""HTTP client for an OpenStreetMap Overpass API instance."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any, cast

import httpx

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.places_search.osm.query import build_overpass_query
from tools.geo.places_search.schemas import PlaceCategory
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds


class OsmOverpassClient:
    """Execute bounded place searches against Overpass."""

    provider = "osm_overpass"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        endpoint: str,
        user_agent: str,
        http_timeout_s: int = 5,
        retries: int = 0,
        query_timeout_s: int = 12,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("Overpass endpoint cannot be empty")
        if not user_agent.strip():
            raise ValueError("Overpass User-Agent cannot be empty")
        if http_timeout_s < 1:
            raise ValueError("Overpass HTTP timeout must be positive")
        if not 0 <= retries <= 3:
            raise ValueError("Overpass retries must be between 0 and 3")
        if not 1 <= query_timeout_s <= 180:
            raise ValueError("Overpass query timeout must be between 1 and 180 seconds")

        self._http_client = http_client
        self._endpoint = endpoint
        self._user_agent = user_agent
        self._http_timeout_s = http_timeout_s
        self._retries = retries
        self._query_timeout_s = query_timeout_s

    async def search(
        self,
        *,
        text: str,
        category: PlaceCategory | None,
        limit: int,
        open_24h: bool,
        open_now: bool = False,
        context: ToolExecutionContext,
        center: tuple[float, float] | None = None,
        radius_m: int | None = None,
        bbox: GeoBounds | None = None,
        boundary_name: str | None = None,
    ) -> dict[str, Any]:
        """Build safe Overpass QL and return the raw JSON object."""

        query = build_overpass_query(
            text=text,
            category=category,
            limit=limit,
            timeout_s=self._query_timeout_s,
            open_24h=open_24h,
            open_now=open_now,
            center=center,
            radius_m=radius_m,
            bbox=bbox,
            boundary_name=boundary_name,
        )
        for attempt in range(self._retries + 1):
            try:
                return await self._post_query(query=query, context=context)
            except ToolExecutionError as exc:
                if not exc.retryable or attempt == self._retries:
                    raise

        raise RuntimeError("Unreachable Overpass retry state")

    async def _post_query(
        self,
        *,
        query: str,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        started = perf_counter()
        response: httpx.Response | None = None

        try:
            try:
                response = await self._http_client.post(
                    self._endpoint,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self._user_agent,
                    },
                    data={"data": query},
                    timeout=httpx.Timeout(float(self._http_timeout_s)),
                )
            except httpx.TimeoutException as exc:
                raise ToolExecutionError(
                    ToolErrorCode.TIMEOUT,
                    "OpenStreetMap search timed out",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.TIMEOUT,
                    retryable=True,
                ) from exc
            except httpx.HTTPError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "OpenStreetMap search is unavailable",
                    provider=self.provider,
                    failure_kind=ToolFailureKind.NETWORK,
                    retryable=True,
                ) from exc

            self._raise_for_status(response.status_code)

            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "OpenStreetMap search returned invalid JSON",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_JSON,
                    retryable=False,
                ) from exc

            if not isinstance(payload, Mapping):
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "OpenStreetMap search returned an unexpected response format",
                    status_code=response.status_code,
                    provider=self.provider,
                    failure_kind=ToolFailureKind.INVALID_SCHEMA,
                    retryable=False,
                )

            result = cast(dict[str, Any], dict(payload))
        except asyncio.CancelledError:
            context.record_cancelled_upstream_call(
                provider=self.provider,
                operation="search",
                latency_ms=int((perf_counter() - started) * 1000),
                status_code=response.status_code if response is not None else None,
            )
            raise
        except ToolExecutionError as exc:
            context.record_upstream_call(
                provider=self.provider,
                operation="search",
                latency_ms=int((perf_counter() - started) * 1000),
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
            latency_ms=int((perf_counter() - started) * 1000),
            outcome=UpstreamCallOutcome.SUCCESS,
            status_code=response.status_code,
        )
        return result

    def _raise_for_status(self, status_code: int) -> None:
        if status_code < 400:
            return
        if status_code == 429:
            raise ToolExecutionError(
                ToolErrorCode.RATE_LIMITED,
                "OpenStreetMap search rate limit exceeded",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=True,
            )
        if status_code in {408, 504}:
            raise ToolExecutionError(
                ToolErrorCode.TIMEOUT,
                "OpenStreetMap search timed out",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=True,
            )

        raise ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "OpenStreetMap search returned an HTTP error",
            status_code=status_code,
            provider=self.provider,
            failure_kind=ToolFailureKind.HTTP_STATUS,
            retryable=status_code >= 500,
        )
