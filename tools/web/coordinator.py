from __future__ import annotations

from collections.abc import Sequence

from tools.coordination import ProviderExecutionStrategy
from tools.observability import ToolExecutionContext
from tools.web.provider import WebSearchProvider
from tools.web.search import WebSearchInput, WebSearchOutput


class WebSearchCoordinator:
    def __init__(
        self,
        *,
        providers: Sequence[WebSearchProvider],
        strategy: ProviderExecutionStrategy = ProviderExecutionStrategy.FIRST,
    ) -> None:
        if not providers:
            raise ValueError("web search coordinator requires at least one provider")

        provider_names = [provider.provider for provider in providers]

        if len(provider_names) != len(set(provider_names)):
            raise ValueError(f"duplicate web search providers: {provider_names}")

        if strategy is ProviderExecutionStrategy.PARALLEL:
            raise NotImplementedError("parallel web search strategy is not implemented")

        self._providers = tuple(providers)
        self._strategy = strategy

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> WebSearchOutput:
        if self._strategy is ProviderExecutionStrategy.FIRST:
            return await self._providers[0].search(args, context)

        raise AssertionError(f"unsupported strategy: {self._strategy}")
