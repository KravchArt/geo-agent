"""Validated subsets of GraphHopper Routing and Matrix API responses."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from tools.geo.routing.upstream_schemas import (
    NonNegativeNumber,
    RoutingUpstreamModel,
)


class GraphHopperInstruction(RoutingUpstreamModel):
    distance: NonNegativeNumber
    time: int = Field(ge=0)
    text: str = Field(min_length=1)
    street_name: str | None = None
    sign: int
    #: Inclusive geometry-point bounds used to align instructions with path
    #: details.  Older/self-hosted responses may omit them; route instructions
    #: remain usable in that case, only street-name enrichment is skipped.
    interval: list[int] | None = Field(default=None, min_length=2, max_length=2)


class GraphHopperRoutePath(RoutingUpstreamModel):
    distance: NonNegativeNumber
    #: GraphHopper Routing API returns route and leg times in milliseconds.
    time: int = Field(ge=0)
    #: Encoded input points after GraphHopper snapped them to its routing graph.
    snapped_waypoints: str = Field(min_length=1)
    details: dict[str, list[list[Any]]] = Field(default_factory=dict)
    instructions: list[GraphHopperInstruction] = Field(default_factory=list)
    points_order: list[int] | None = None


class GraphHopperRouteResponse(RoutingUpstreamModel):
    paths: list[GraphHopperRoutePath] = Field(default_factory=list)


class GraphHopperMatrixResponse(RoutingUpstreamModel):
    #: Matrix times are seconds, unlike the millisecond values from /route.
    times: list[list[NonNegativeNumber | None]]
    distances: list[list[NonNegativeNumber | None]]
