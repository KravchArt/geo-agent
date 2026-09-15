"""Contract implemented by concrete routing providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tools.geo.routing.schemas import RoutingInput, RoutingOutput
from tools.observability import ToolExecutionContext


@runtime_checkable
class RoutingProvider(Protocol):
    """One concrete backend that consumes coordinator-prepared place refs."""

    provider: str

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        """Build a route or matrix result from prepared place refs."""
        ...
