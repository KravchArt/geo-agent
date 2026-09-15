"""HTTP client for Yandex Route Details and synchronous Distance Matrix APIs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing.http_transport import (
    common_routing_http_error,
    request_routing_json,
)
from tools.geo.routing.schemas import MAX_MATRIX_ELEMENTS, TransportMode
from tools.geo.routing.yandex.schemas import (
    YandexDistanceMatrixResponse,
    YandexRouteResponse,
)
from tools.observability import ToolExecutionContext

Coordinates = tuple[float, float]


class YandexRoutingClient:
    """Call both synchronous Yandex routing endpoints with one API key."""

    _ROUTE_URL = "https://api.routing.yandex.net/v2/route"
    _MATRIX_URL = "https://api.routing.yandex.net/v2/distancematrix"
    provider = "yandex_routing"
    service_name = "Yandex"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Yandex Routing API key cannot be empty")

        self._api_key = api_key
        self._http_client = http_client

    async def build_route(
        self,
        *,
        waypoints: Sequence[Coordinates],
        transport: TransportMode,
        use_traffic: bool,
        avoid_tolls: bool,
        departure_time: datetime | None,
        optimize_waypoints: bool,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")

        params = self._common_params(
            transport=transport,
            use_traffic=use_traffic,
            avoid_tolls=avoid_tolls,
            departure_time=departure_time,
        )
        params["waypoints"] = self._encode_points(waypoints)

        if optimize_waypoints:
            params["optimize"] = "true"

        return await self._request_json(
            url=self._ROUTE_URL,
            params=params,
            operation="build_route",
            context=context,
            response_model=YandexRouteResponse,
        )

    async def distance_matrix(
        self,
        *,
        origins: Sequence[Coordinates],
        destinations: Sequence[Coordinates],
        transport: TransportMode,
        use_traffic: bool,
        avoid_tolls: bool,
        departure_time: datetime | None,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if not origins:
            raise ValueError("at least one origin is required")
        if not destinations:
            raise ValueError("at least one destination is required")
        if len(origins) * len(destinations) > MAX_MATRIX_ELEMENTS:
            raise ValueError(f"distance matrix must not exceed {MAX_MATRIX_ELEMENTS} elements")

        params = self._common_params(
            transport=transport,
            use_traffic=use_traffic,
            avoid_tolls=avoid_tolls,
            departure_time=departure_time,
        )
        params["origins"] = self._encode_points(origins)
        params["destinations"] = self._encode_points(destinations)

        return await self._request_json(
            url=self._MATRIX_URL,
            params=params,
            operation="distance_matrix",
            context=context,
            response_model=YandexDistanceMatrixResponse,
        )

    async def _request_json(
        self,
        *,
        url: str,
        params: dict[str, str],
        operation: str,
        context: ToolExecutionContext,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        return await request_routing_json(
            http_client=self._http_client,
            method="GET",
            url=url,
            params=params,
            operation=operation,
            provider=self.provider,
            service_name=self.service_name,
            context=context,
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
        common_error = common_routing_http_error(
            status_code=status_code,
            provider=self.provider,
            service_name=self.service_name,
        )
        if common_error is not None:
            raise common_error

        if status_code == 400:
            # RoutingInput and the client have already validated all required
            # parameters. A rejected generated request is an adapter/config
            # defect and must not be hidden by provider fallback.
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "GeoAgent generated a request that Yandex rejected",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
                retryable=False,
            )
        if status_code >= 400:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex rejected the routing request",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=False,
            )
        if "errors" in payload:
            # The documented synchronous APIs report request errors with an
            # unsuccessful HTTP status. An errors-only 2xx body violates that
            # contract and is therefore an upstream schema incident.
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex routing returned an unexpected error payload",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            )

        try:
            response_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex routing returned data in an unexpected format",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc
        return payload

    def _common_params(
        self,
        *,
        transport: TransportMode,
        use_traffic: bool,
        avoid_tolls: bool,
        departure_time: datetime | None,
    ) -> dict[str, str]:
        params = {
            "apikey": self._api_key,
            "mode": transport.value,
        }

        if transport is TransportMode.DRIVING:
            if not use_traffic:
                params["traffic"] = "disabled"
            if avoid_tolls:
                params["avoid_tolls"] = "true"

        if departure_time is not None and use_traffic:
            params["departure_time"] = str(int(departure_time.timestamp()))

        return params

    @staticmethod
    def _encode_points(points: Sequence[Coordinates]) -> str:
        encoded: list[str] = []

        for lat, lon in points:
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("routing coordinates are outside WGS84 bounds")
            encoded.append(f"{lat:.6f},{lon:.6f}")

        return "|".join(encoded)
