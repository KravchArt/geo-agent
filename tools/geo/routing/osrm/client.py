"""HTTP client for OSRM Route and Table services."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing.http_transport import (
    common_routing_http_error,
    request_routing_json,
)
from tools.geo.routing.osrm.schemas import OsrmRouteResponse, OsrmTableResponse
from tools.geo.routing.schemas import MAX_MATRIX_ELEMENTS, TransportMode
from tools.observability import ToolExecutionContext

Coordinates = tuple[float, float]
_NOT_FOUND_CODES = frozenset({"NoRoute", "NoTable", "NoSegment"})
_INTERNAL_CONTRACT_CODES = frozenset(
    {
        "InvalidUrl",
        "InvalidService",
        "InvalidVersion",
        "InvalidOptions",
        "InvalidQuery",
        "InvalidValue",
    }
)
_UNSUPPORTED_CODES = frozenset({"DisabledDataset", "NotImplemented"})


class OsrmRoutingClient:
    """Call an OSRM HTTP server without requiring an API key."""

    provider = "osrm_routing"
    service_name = "OSRM"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str = "https://router.project-osrm.org",
        walking_base_url: str = "https://routing.openstreetmap.de/routed-foot",
        user_agent: str = "GeoAgent/0.0.0",
    ) -> None:
        if not base_url.strip():
            raise ValueError("OSRM base URL cannot be empty")
        if not walking_base_url.strip():
            raise ValueError("OSRM walking base URL cannot be empty")
        if not user_agent.strip():
            raise ValueError("OSRM user agent cannot be empty")

        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._walking_base_url = walking_base_url.rstrip("/")
        self._walking_request_lock = asyncio.Lock()
        self._last_walking_request_started = 0.0
        self._headers = {"User-Agent": user_agent}

    async def build_route(
        self,
        *,
        waypoints: Sequence[Coordinates],
        transport: TransportMode,
        avoid_tolls: bool,
        optimize_waypoints: bool,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        if optimize_waypoints:
            raise ValueError("OSRM does not support waypoint optimization")

        params: dict[str, str] = {
            "steps": "true",
            "overview": "false",
            "alternatives": "false",
        }
        if avoid_tolls:
            params["exclude"] = "toll"

        return await self._get_json(
            base_url=self._transport_base_url(transport),
            endpoint=f"route/v1/driving/{self._encode_coordinates(waypoints)}",
            params=params,
            operation="build_route",
            context=context,
            rate_limited=transport is TransportMode.WALKING,
            response_model=OsrmRouteResponse,
        )

    async def distance_matrix(
        self,
        *,
        origins: Sequence[Coordinates],
        destinations: Sequence[Coordinates],
        transport: TransportMode,
        avoid_tolls: bool,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if not origins:
            raise ValueError("at least one origin is required")
        if not destinations:
            raise ValueError("at least one destination is required")
        if len(origins) * len(destinations) > MAX_MATRIX_ELEMENTS:
            raise ValueError(f"distance matrix must not exceed {MAX_MATRIX_ELEMENTS} elements")

        points = [*origins, *destinations]
        destination_offset = len(origins)
        params = {
            "annotations": "duration,distance",
            "sources": ";".join(str(index) for index in range(len(origins))),
            "destinations": ";".join(
                str(index) for index in range(destination_offset, len(points))
            ),
        }
        if avoid_tolls:
            params["exclude"] = "toll"

        return await self._get_json(
            base_url=self._transport_base_url(transport),
            endpoint=f"table/v1/driving/{self._encode_coordinates(points)}",
            params=params,
            operation="distance_matrix",
            context=context,
            rate_limited=transport is TransportMode.WALKING,
            response_model=OsrmTableResponse,
        )

    async def _get_json(
        self,
        *,
        base_url: str,
        endpoint: str,
        params: Mapping[str, str],
        operation: str,
        context: ToolExecutionContext,
        rate_limited: bool,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        if not rate_limited:
            return await self._send_get_json(
                base_url=base_url,
                endpoint=endpoint,
                params=params,
                operation=operation,
                context=context,
                response_model=response_model,
            )

        # FOSSGIS permits at most one request per second to its public routing
        # service. Serialize walking requests and space their start times.
        async with self._walking_request_lock:
            wait_s = 1.0 - (monotonic() - self._last_walking_request_started)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_walking_request_started = monotonic()
            return await self._send_get_json(
                base_url=base_url,
                endpoint=endpoint,
                params=params,
                operation=operation,
                context=context,
                response_model=response_model,
            )

    async def _send_get_json(
        self,
        *,
        base_url: str,
        endpoint: str,
        params: Mapping[str, str],
        operation: str,
        context: ToolExecutionContext,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        return await request_routing_json(
            http_client=self._http_client,
            method="GET",
            url=f"{base_url}/{endpoint}",
            params=params,
            operation=operation,
            provider=self.provider,
            service_name=self.service_name,
            context=context,
            headers=self._headers,
            response_parser=lambda status_code, payload: self._parse_response(
                status_code,
                payload,
                response_model=response_model,
            ),
        )

    def _parse_response(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        provider_code_value = payload.get("code")
        provider_code = provider_code_value if isinstance(provider_code_value, str) else None

        common_error = common_routing_http_error(
            status_code=status_code,
            provider=self.provider,
            service_name=self.service_name,
            provider_code=provider_code,
        )
        if common_error is not None:
            raise common_error

        if provider_code in _NOT_FOUND_CODES:
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND,
                "OSRM could not build a route between all requested points",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                retryable=False,
            )
        if provider_code == "TooBig":
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT,
                "The routing request exceeds the OSRM service limit",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=False,
            )
        if provider_code in _UNSUPPORTED_CODES:
            raise ToolExecutionError(
                ToolErrorCode.UNSUPPORTED_FILTER,
                "The configured OSRM service does not support this routing request",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                retryable=False,
            )
        if provider_code in _INTERNAL_CONTRACT_CODES:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "GeoAgent generated a request that OSRM rejected",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
                retryable=False,
            )
        if status_code >= 400 or provider_code != "Ok":
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "OSRM returned an unrecognized routing response",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            )

        try:
            response_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "OSRM routing returned data in an unexpected format",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc
        return payload

    def _transport_base_url(self, transport: TransportMode) -> str:
        if transport is TransportMode.DRIVING:
            return self._base_url
        if transport is TransportMode.WALKING:
            return self._walking_base_url
        raise ValueError(f"unsupported OSRM transport: {transport}")

    @staticmethod
    def _encode_coordinates(points: Sequence[Coordinates]) -> str:
        encoded: list[str] = []
        for latitude, longitude in points:
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError("routing coordinates are outside WGS84 bounds")
            encoded.append(f"{longitude:.6f},{latitude:.6f}")
        return ";".join(encoded)
