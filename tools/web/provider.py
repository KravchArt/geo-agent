"""Contract implemented by concrete web-search providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tools.observability import ToolExecutionContext
from tools.web.search import WebSearchInput, WebSearchOutput


@runtime_checkable
class WebSearchProvider(Protocol):
    """One concrete web-search backend."""

    provider: str

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> WebSearchOutput: ...
