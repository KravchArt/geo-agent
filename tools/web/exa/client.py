"""HTTP client for the current Exa Search API."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx

from tools.observability import ToolExecutionContext
from tools.web.http_transport import post_search_json
from tools.web.search import MAX_RESULTS, MAX_SNIPPET_CHARS, TimeRange, WebSearchInput, WebTopic


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ExaSearchClient:
    _SEARCH_URL = "https://api.exa.ai/search"
    _TIME_RANGE_DELTAS: ClassVar[dict[TimeRange, timedelta]] = {
        TimeRange.DAY: timedelta(days=1),
        TimeRange.WEEK: timedelta(days=7),
        TimeRange.MONTH: timedelta(days=30),
        TimeRange.YEAR: timedelta(days=365),
    }
    provider = "exa"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client
        self._clock = clock

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": args.query,
            "type": "auto",
            "numResults": MAX_RESULTS,
            "contents": {
                "highlights": {
                    "maxCharacters": MAX_SNIPPET_CHARS,
                }
            },
        }

        if args.topic is WebTopic.NEWS:
            payload["category"] = "news"

        if args.time_range is not None:
            payload["startPublishedDate"] = self._start_published_date(args.time_range)

        if args.include_domains:
            payload["includeDomains"] = args.include_domains

        if args.exclude_domains:
            payload["excludeDomains"] = args.exclude_domains

        return await post_search_json(
            http_client=self._http_client,
            url=self._SEARCH_URL,
            headers={"x-api-key": self._api_key},
            payload=payload,
            provider=self.provider,
            context=context,
        )

    def _start_published_date(self, time_range: TimeRange) -> str:
        now = self._clock()

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Exa client clock must return a timezone-aware datetime")

        start = now.astimezone(UTC) - self._TIME_RANGE_DELTAS[time_range]
        return start.isoformat(timespec="seconds").replace("+00:00", "Z")
