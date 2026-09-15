"""HTTP client for the Firecrawl v2 Search API."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from tools.observability import ToolExecutionContext
from tools.web.http_transport import post_search_json
from tools.web.search import MAX_RESULTS, TimeRange, WebSearchInput, WebTopic


class FirecrawlSearchClient:
    _SEARCH_URL = "https://api.firecrawl.dev/v2/search"
    _TIME_RANGES: ClassVar[dict[TimeRange, str]] = {
        TimeRange.DAY: "qdr:d",
        TimeRange.WEEK: "qdr:w",
        TimeRange.MONTH: "qdr:m",
        TimeRange.YEAR: "qdr:y",
    }
    provider = "firecrawl"

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
            "limit": MAX_RESULTS,
            "sources": ["news" if args.topic is WebTopic.NEWS else "web"],
        }

        if args.time_range is not None:
            payload["tbs"] = self._TIME_RANGES[args.time_range]

        # Firecrawl makes these parameters mutually exclusive. When callers use
        # both, narrow upstream with includeDomains and let the provider adapter
        # enforce exclusions locally.
        if args.include_domains:
            payload["includeDomains"] = args.include_domains
        elif args.exclude_domains:
            payload["excludeDomains"] = args.exclude_domains

        return await post_search_json(
            http_client=self._http_client,
            url=self._SEARCH_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload=payload,
            provider=self.provider,
            context=context,
            timeout_statuses=frozenset({408}),
        )
