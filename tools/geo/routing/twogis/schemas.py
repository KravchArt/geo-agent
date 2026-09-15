"""Validated subsets of 2GIS Routing and Distance Matrix responses."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, RootModel

from tools.geo.routing.upstream_schemas import RoutingUpstreamModel

TwoGisRouteStatus = Literal[
    "OK",
    "FAIL",
    "POINT_EXCLUDED",
    "ROUTE_NOT_FOUND",
    "ROUTE_DOES_NOT_EXISTS",
    "ATTRACT_FAIL",
    "PLATFORMS_NOT_FOUND",
]


class TwoGisDetailedPath(RoutingUpstreamModel):
    distance: int = Field(ge=0)
    duration: int = Field(ge=0)
    names: list[str] = Field(default_factory=list)


class TwoGisDetailedManeuver(RoutingUpstreamModel):
    comment: str = Field(min_length=1)
    outcoming_path_comment: str | None = None
    outcoming_path: TwoGisDetailedPath | None = None
    type: str = Field(min_length=1)


class TwoGisDetailedRoute(RoutingUpstreamModel):
    total_distance: int = Field(ge=0)
    total_duration: int = Field(ge=0)
    maneuvers: list[TwoGisDetailedManeuver] = Field(default_factory=list)


class TwoGisDetailedRouteResponse(RoutingUpstreamModel):
    status: TwoGisRouteStatus
    type: Literal["result", "error"]
    message: str | None = None
    result: list[TwoGisDetailedRoute] | None = None


class TwoGisMatrixRoute(RoutingUpstreamModel):
    status: TwoGisRouteStatus
    source_id: int = Field(ge=0)
    target_id: int = Field(ge=0)
    distance: int = Field(ge=0)
    duration: int = Field(ge=0)


class TwoGisDistanceMatrixResponse(RoutingUpstreamModel):
    routes: list[TwoGisMatrixRoute] | None = None


class TwoGisPublicTransportMovement(RoutingUpstreamModel):
    distance: int = Field(ge=0)
    moving_duration: int = Field(ge=0)
    waiting_duration: int = Field(default=0, ge=0)
    type: str = Field(min_length=1)


class TwoGisPublicTransportRoute(RoutingUpstreamModel):
    total_distance: int = Field(ge=0)
    total_duration: int = Field(ge=0)
    movements: list[TwoGisPublicTransportMovement] = Field(default_factory=list)


class TwoGisPublicTransportResponse(RootModel[list[TwoGisPublicTransportRoute]]):
    """Public Transport API returns route alternatives as a top-level array."""
