"""HTTP client for GraphHopper Routing and synchronous Matrix APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing.graphhopper.schemas import (
    GraphHopperMatrixResponse,
    GraphHopperRouteResponse,
)
from tools.geo.routing.http_transport import (
    common_routing_http_error,
    request_routing_json,
)
from tools.geo.routing.schemas import MAX_MATRIX_ELEMENTS, TransportMode
from tools.observability import ToolExecutionContext

Coordinates = tuple[float, float]
DEFAULT_GRAPHHOPPER_HTTP_TIMEOUT_S = 2.0
_NOT_FOUND_HINTS = frozenset({"ConnectionNotFound", "PointNotFound"})


class GraphHopperRoutingClient:
    """Call GraphHopper's route and matrix endpoints with one API key."""

    provider = "graphhopper_routing"
    service_name = "GraphHopper"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        base_url: str = "https://graphhopper.com/api/1",
        http_timeout_s: float = DEFAULT_GRAPHHOPPER_HTTP_TIMEOUT_S,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GraphHopper API key cannot be empty")
        if not base_url.strip():
            raise ValueError("GraphHopper base URL cannot be empty")
        if http_timeout_s <= 0:
            raise ValueError("GraphHopper HTTP timeout must be positive")

        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_timeout_s = http_timeout_s

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

        params: list[tuple[str, str | int | float | bool | None]] = [
            ("key", self._api_key),
            *[("point", self._encode_route_point(point)) for point in waypoints],
            ("profile", self._profile(transport, avoid_tolls=avoid_tolls)),
            # The hosted API currently stalls for this route when the `toll`
            # path detail is requested, while both leg details return normally.
            ("details", "leg_distance"),
            ("details", "leg_time"),
            # Instructions can span several road names without emitting a
            # separate maneuver (for example Bolshaya Pecherskaya continuing
            # onto Rodionova Street).  These edge-level details let the adapter
            # restore the omitted named-road transition with exact costs.
            ("details", "street_name"),
            ("details", "distance"),
            ("details", "time"),
            ("instructions", "true"),
            ("locale", "en"),
            ("points_encoded", "true"),
        ]
        if optimize_waypoints:
            params.append(("optimize", "true"))

        # GET is intentionally used for routes: from some Russian networks the
        # hosted GraphHopper endpoint accepts POST but never returns its body.
        return await request_routing_json(
            http_client=self._http_client,
            method="GET",
            url=f"{self._base_url}/route",
            params=httpx.QueryParams(params),
            operation="build_route",
            provider=self.provider,
            service_name=self.service_name,
            context=context,
            http_timeout_s=self._http_timeout_s,
            response_parser=lambda status_code, payload: self._parse_response(
                status_code,
                payload,
                response_model=GraphHopperRouteResponse,
            ),
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

        payload = {
            "from_points": self._encode_points(origins),
            "to_points": self._encode_points(destinations),
            "profile": self._profile(transport, avoid_tolls=avoid_tolls),
            "out_arrays": ["distances", "times"],
            # Preserve partial results: null means that one pair is unreachable.
            "fail_fast": False,
        }
        return await request_routing_json(
            http_client=self._http_client,
            method="POST",
            url=f"{self._base_url}/matrix",
            params=httpx.QueryParams({"key": self._api_key}),
            payload=payload,
            operation="distance_matrix",
            provider=self.provider,
            service_name=self.service_name,
            context=context,
            http_timeout_s=self._http_timeout_s,
            response_parser=lambda status_code, response_payload: self._parse_response(
                status_code,
                response_payload,
                response_model=GraphHopperMatrixResponse,
            ),
        )

    def _parse_response(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        provider_code = self._error_hint(payload)
        common_error = common_routing_http_error(
            status_code=status_code,
            provider=self.provider,
            service_name=self.service_name,
            provider_code=provider_code,
        )
        if common_error is not None:
            raise common_error

        if status_code >= 400:
            if provider_code is not None and any(
                hint in provider_code for hint in _NOT_FOUND_HINTS
            ):
                raise ToolExecutionError(
                    ToolErrorCode.NOT_FOUND,
                    "GraphHopper could not build a route between all requested points",
                    status_code=status_code,
                    provider=self.provider,
                    provider_code=provider_code,
                    failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                    retryable=False,
                )
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "GraphHopper rejected the routing request",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=False,
            )

        try:
            response_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "GraphHopper routing returned data in an unexpected format",
                status_code=status_code,
                provider=self.provider,
                provider_code=provider_code,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc
        return payload

    @staticmethod
    def _error_hint(payload: dict[str, Any]) -> str | None:
        hints = payload.get("hints")
        if not isinstance(hints, list):
            return None
        for hint in hints:
            if isinstance(hint, dict):
                details = hint.get("details")
                if isinstance(details, str):
                    return details
        return None

    @staticmethod
    def _profile(transport: TransportMode, *, avoid_tolls: bool) -> str:
        profiles = {
            TransportMode.DRIVING: "car_avoid_toll" if avoid_tolls else "car",
            TransportMode.WALKING: "foot",
            TransportMode.BICYCLE: "bike",
            TransportMode.SCOOTER: "scooter",
        }
        try:
            return profiles[transport]
        except KeyError as exc:
            raise ValueError(f"unsupported GraphHopper transport: {transport}") from exc

    @staticmethod
    def _encode_points(points: Sequence[Coordinates]) -> list[list[float]]:
        # GraphHopper POST endpoints use GeoJSON order: longitude, latitude.
        encoded: list[list[float]] = []
        for point in points:
            latitude, longitude = GraphHopperRoutingClient._validate_coordinate(point)
            encoded.append([longitude, latitude])
        return encoded

    @staticmethod
    def _encode_route_point(point: Coordinates) -> str:
        latitude, longitude = GraphHopperRoutingClient._validate_coordinate(point)
        return f"{latitude},{longitude}"

    @staticmethod
    def _validate_coordinate(point: Coordinates) -> Coordinates:
        latitude, longitude = point
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("routing coordinates are outside WGS84 bounds")
        return point
