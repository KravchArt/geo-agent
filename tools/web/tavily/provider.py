"""Tavily implementation of the web-search provider contract."""

from tools.observability import ToolExecutionContext
from tools.web.adapter import (
    WebSearchCandidate,
    materialize_search_results,
    parse_iso_date,
    validate_search_response,
)
from tools.web.search import MAX_RESULTS, WebSearchInput, WebSearchOutput
from tools.web.source_store import SourceStore
from tools.web.tavily.client import TavilySearchClient
from tools.web.tavily.schemas import TavilySearchResponse


class TavilyWebSearchProvider:
    provider = "tavily"

    def __init__(
        self,
        *,
        client: TavilySearchClient,
        source_store: SourceStore,
    ) -> None:
        self._client = client
        self._source_store = source_store

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> WebSearchOutput:
        payload = await self._client.search(args, context)

        response = validate_search_response(
            payload,
            response_model=TavilySearchResponse,
            provider=self.provider,
        )

        candidates = (
            WebSearchCandidate(
                title=item.title,
                url=item.url,
                snippet=item.content,
                published_date=parse_iso_date(item.published_date),
            )
            for item in response.results
        )
        return await materialize_search_results(
            provider=self.provider,
            query=args.query,
            max_results=MAX_RESULTS,
            candidates=candidates,
            source_store=self._source_store,
        )
