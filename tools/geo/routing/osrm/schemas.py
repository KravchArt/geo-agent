"""Validated subsets of OSRM Route and Table API responses."""

from __future__ import annotations

from pydantic import Field

from tools.geo.routing.upstream_schemas import (
    NonNegativeNumber,
    RoutingUpstreamModel,
)


class OsrmStepManeuver(RoutingUpstreamModel):
    type: str = Field(min_length=1)
    modifier: str | None = None
    exit: int | None = Field(default=None, ge=1)


class OsrmRouteStep(RoutingUpstreamModel):
    distance: NonNegativeNumber
    duration: NonNegativeNumber
    name: str
    maneuver: OsrmStepManeuver


class OsrmRouteLeg(RoutingUpstreamModel):
    distance: NonNegativeNumber
    duration: NonNegativeNumber
    steps: list[OsrmRouteStep] = Field(default_factory=list)


class OsrmRoute(RoutingUpstreamModel):
    distance: NonNegativeNumber
    duration: NonNegativeNumber
    legs: list[OsrmRouteLeg] = Field(default_factory=list)


class OsrmWaypoint(RoutingUpstreamModel):
    distance: NonNegativeNumber
    name: str


class OsrmRouteResponse(RoutingUpstreamModel):
    code: str
    message: str | None = None
    routes: list[OsrmRoute] = Field(default_factory=list)
    waypoints: list[OsrmWaypoint] = Field(default_factory=list)


class OsrmTableResponse(RoutingUpstreamModel):
    code: str
    message: str | None = None
    durations: list[list[NonNegativeNumber | None]] | None = None
    distances: list[list[NonNegativeNumber | None]] | None = None
