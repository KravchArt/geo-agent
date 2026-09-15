"""Deterministic stub handlers for model-facing tools.

These handlers do not call external APIs. They let the tool registry execute
validated tool calls before real adapters are implemented.
"""

from __future__ import annotations

from tools.geo import (
    PlacesSearchInput,
    PlacesSearchOutput,
    RouteInfo,
    RoutingInput,
    RoutingMode,
    RoutingOutput,
    SearchMode,
)
from tools.observability import ToolExecutionContext
from tools.web import WebSearchInput, WebSearchOutput


async def places_search_stub(
    args: PlacesSearchInput,
    _context: ToolExecutionContext,
) -> PlacesSearchOutput:
    """Return an empty successful places_search result."""

    return PlacesSearchOutput(
        anchor=args.near if args.mode is SearchMode.NEAR else None,
    )


async def routing_tool_stub(
    args: RoutingInput,
    _context: ToolExecutionContext,
) -> RoutingOutput:
    if args.mode is RoutingMode.ROUTE:
        return RoutingOutput(
            mode=args.mode,
            transport=args.transport,
            route=RouteInfo(
                length_m=0,
                duration_s=0,
                waypoint_order=list(range(len(args.waypoints))),
            ),
        )

    return RoutingOutput(
        mode=args.mode,
        transport=args.transport,
        optimize_by=args.optimize_by,
        aggregate=args.aggregate,
    )


async def web_search_stub(
    args: WebSearchInput,
    _context: ToolExecutionContext,
) -> WebSearchOutput:
    """Return an empty successful web_search result echoing the normalized query."""

    return WebSearchOutput(query=args.query)
