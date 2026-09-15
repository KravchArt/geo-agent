"""HTTP client for 2GIS Routing and synchronous Distance Matrix APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing.http_transport import common_routing_http_error, request_routing_json
from tools.geo.routing.schemas import MAX_MATRIX_ELEMENTS, TransportMode
from tools.geo.routing.twogis.schemas import (
    TwoGisDetailedRouteResponse,
    TwoGisDistanceMatrixResponse,
    TwoGisPublicTransportResponse,
)
from tools.observability import ToolExecutionContext

Coordinates = tuple[float, float]
_MAX_MATRIX_AXIS = 25
_MAX_PARALLEL_ROUTE_LEGS = 5


class TwoGisRoutingClient:
    """Call 2GIS route, public-transport, and distance-matrix endpoints."""

    provider = "twogis_routing"
    service_name = "2GIS"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://routing.api.2gis.com",
    ) -> None:
        if not api_key.strip():
            raise ValueError("2GIS Routing API key cannot be empty")
        if not base_url.strip():
            raise ValueError("2GIS Routing base URL cannot be empty")

        self._api_key = api_key
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    async def build_route(
        self,
        *,
        waypoints: Sequence[Coordinates],
        transport: TransportMode,
        use_traffic: bool,
        avoid_tolls: bool,
        departure_time: datetime | None,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        self._validate_coordinates(waypoints)

        if transport is TransportMode.TRANSIT:
            return await self._build_public_transport_route(
                waypoints=waypoints,
                departure_time=departure_time,
                context=context,
            )
        if transport is TransportMode.DRIVING and not use_traffic:
            raise ValueError("2GIS does not provide fastest driving times with traffic disabled")
        return await self._build_detailed_route(
            waypoints=waypoints,
            transport=transport,
            avoid_tolls=avoid_tolls,
            departure_time=departure_time,
            context=context,
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
        self._validate_coordinates([*origins, *destinations])
        if transport is TransportMode.DRIVING and not use_traffic:
            raise ValueError("2GIS does not provide fastest driving times with traffic disabled")

        routes: list[dict[str, Any]] = []
        for origin_start in range(0, len(origins), _MAX_MATRIX_AXIS):
            origin_chunk = origins[origin_start : origin_start + _MAX_MATRIX_AXIS]
            for destination_start in range(0, len(destinations), _MAX_MATRIX_AXIS):
                destination_chunk = destinations[
                    destination_start : destination_start + _MAX_MATRIX_AXIS
                ]
                batch = await self._distance_matrix_batch(
                    origins=origin_chunk,
                    destinations=destination_chunk,
                    transport=transport,
                    use_traffic=use_traffic,
                    avoid_tolls=avoid_tolls,
                    departure_time=departure_time,
                    context=context,
                )
                for route in batch.routes or []:
                    routes.append(
                        route.model_copy(
                            update={
                                "source_id": origin_start + route.source_id,
                                "target_id": (
                                    destination_start + route.target_id - len(origin_chunk)
                                ),
                            }
                        ).model_dump()
                    )
        return {"routes": routes}

    async def _distance_matrix_batch(
        self,
        *,
        origins: Sequence[Coordinates],
        destinations: Sequence[Coordinates],
        transport: TransportMode,
        use_traffic: bool,
        avoid_tolls: bool,
        departure_time: datetime | None,
        context: ToolExecutionContext,
    ) -> TwoGisDistanceMatrixResponse:
        points = [*origins, *destinations]
        payload: dict[str, Any] = {
            "points": [self._coordinate(point) for point in points],
            "sources": list(range(len(origins))),
            "targets": list(range(len(origins), len(points))),
            "transport": (
                "public_transport" if transport is TransportMode.TRANSIT else transport.value
            ),
        }
        if avoid_tolls:
            payload["filters"] = ["toll_road"]
        if departure_time is not None:
            payload["start_time"] = departure_time.isoformat()
            payload["type"] = "statistics"
        elif transport is TransportMode.DRIVING and use_traffic:
            payload["type"] = "jam"

        raw = await request_routing_json(
            http_client=self._http_client,
            method="POST",
            url=f"{self._base_url}/get_dist_matrix",
            params=httpx.QueryParams({"key": self._api_key, "version": "2.0"}),
            payload=payload,
            operation="distance_matrix",
            provider=self.provider,
            service_name=self.service_name,
            context=context,
            response_parser=lambda status_code, response_payload: self._parse_response(
                status_code,
                response_payload,
                response_model=TwoGisDistanceMatrixResponse,
            ),
        )
        return TwoGisDistanceMatrixResponse.model_validate(raw)

    async def _build_public_transport_route(
        self,
        *,
        waypoints: Sequence[Coordinates],
        departure_time: datetime | None,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(_MAX_PARALLEL_ROUTE_LEGS)

        async def build_leg(start: Coordinates, end: Coordinates) -> list[dict[str, Any]]:
            payload: dict[str, Any] = {
                "source": {"point": self._coordinate(start)},
                "target": {"point": self._coordinate(end)},
                "transport": [
                    "metro",
                    "light_metro",
                    "suburban_train",
                    "aeroexpress",
                    "tram",
                    "bus",
                    "trolleybus",
                    "shuttle_bus",
                    "monorail",
                    "funicular_railway",
                    "river_transport",
                    "cable_car",
                    "light_rail",
                    "premetro",
                    "mcc",
                    "mcd",
                ],
                "locale": "en",
            }
            if departure_time is not None:
                payload["start_time"] = int(departure_time.timestamp())
            async with semaphore:
                return await request_routing_json(
                    http_client=self._http_client,
                    method="POST",
                    url=f"{self._base_url}/public_transport/2.0",
                    params=httpx.QueryParams({"key": self._api_key}),
                    payload=payload,
                    operation="build_public_transport_route",
                    provider=self.provider,
                    service_name=self.service_name,
                    context=context,
                    response_body="object_or_array",
                    response_parser=lambda status_code, response_payload: self._parse_response(
                        status_code,
                        response_payload,
                        response_model=TwoGisPublicTransportResponse,
                    ),
                )

        with context.parallel_upstream_calls():
            legs = await asyncio.gather(
                *(build_leg(start, end) for start, end in pairwise(waypoints))
            )
        return {"legs": legs}

    async def _build_detailed_route(
        self,
        *,
        waypoints: Sequence[Coordinates],
        transport: TransportMode,
        avoid_tolls: bool,
        departure_time: datetime | None,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(_MAX_PARALLEL_ROUTE_LEGS)

        async def build_leg(start: Coordinates, end: Coordinates) -> dict[str, Any]:
            point_type = "walking" if transport is TransportMode.WALKING else "stop"
            payload: dict[str, Any] = {
                "points": [
                    self._point(start, point_type=point_type),
                    self._point(end, point_type=point_type),
                ],
                "transport": transport.value,
                "route_mode": "fastest",
                "output": "detailed",
                "locale": "en",
            }
            if avoid_tolls:
                payload["filters"] = ["toll_road"]
            if departure_time is not None:
                payload["traffic_mode"] = "statistics"
                payload["utc"] = int(departure_time.timestamp())
            elif transport is TransportMode.DRIVING:
                payload["traffic_mode"] = "jam"

            async with semaphore:
                return await request_routing_json(
                    http_client=self._http_client,
                    method="POST",
                    url=f"{self._base_url}/routing/7.0.0/global",
                    params=httpx.QueryParams({"key": self._api_key}),
                    payload=payload,
                    operation="build_route",
                    provider=self.provider,
                    service_name=self.service_name,
                    context=context,
                    response_body="object_or_array",
                    response_parser=lambda status_code, response_payload: self._parse_response(
                        status_code,
                        response_payload,
                        response_model=TwoGisDetailedRouteResponse,
                    ),
                )

        with context.parallel_upstream_calls():
            legs = await asyncio.gather(
                *(build_leg(start, end) for start, end in pairwise(waypoints))
            )
        return {"legs": legs}

    def _parse_response(
        self,
        status_code: int,
        payload: Any,
        *,
        response_model: type[Any],
    ) -> Any:
        common_error = common_routing_http_error(
            status_code=status_code,
            provider=self.provider,
            service_name=self.service_name,
        )
        if common_error is not None:
            raise common_error
        if status_code == 204:
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND,
                "2GIS could not build a route between the requested points",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                retryable=False,
            )
        if status_code in {400, 422}:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "GeoAgent generated a request that 2GIS rejected",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
                retryable=False,
            )
        if status_code >= 400:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS rejected the routing request",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=False,
            )

        try:
            validated = TypeAdapter(response_model).validate_python(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS routing returned data in an unexpected format",
                status_code=status_code,
                provider=self.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc
        return validated.model_dump()

    @classmethod
    def _validate_coordinates(cls, points: Sequence[Coordinates]) -> None:
        for point in points:
            cls._coordinate(point)

    @staticmethod
    def _coordinate(point: Coordinates) -> dict[str, float]:
        latitude, longitude = point
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("routing coordinates are outside WGS84 bounds")
        return {"lat": latitude, "lon": longitude}

    @classmethod
    def _point(
        cls,
        point: Coordinates,
        *,
        point_type: str = "stop",
    ) -> dict[str, float | str]:
        return {**cls._coordinate(point), "type": point_type}
