from typing import Any

import httpx

from tools.observability import ToolExecutionContext
from tools.web.http_transport import post_search_json
from tools.web.search import MAX_RESULTS, WebSearchInput


class TavilySearchClient:
    _SEARCH_URL = "https://api.tavily.com/search"
    provider = "tavily"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": args.query,
            "topic": args.topic.value,
            "max_results": MAX_RESULTS,
            "search_depth": "advanced",
            "chunks_per_source": 3,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        if args.time_range is not None:
            payload["time_range"] = args.time_range.value

        if args.include_domains:
            payload["include_domains"] = args.include_domains

        if args.exclude_domains:
            payload["exclude_domains"] = args.exclude_domains

        return await post_search_json(
            http_client=self._http_client,
            url=self._SEARCH_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload=payload,
            provider=self.provider,
            context=context,
        )
