"""Firecrawl implementation of the provider-agnostic web-search contract."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit

from tools.observability import ToolExecutionContext
from tools.web.adapter import (
    WebSearchCandidate,
    materialize_search_results,
    parse_iso_date,
    validate_search_response,
)
from tools.web.firecrawl.client import FirecrawlSearchClient
from tools.web.firecrawl.schemas import FirecrawlSearchResponse
from tools.web.search import MAX_RESULTS, WebSearchInput, WebSearchOutput, WebTopic
from tools.web.source_store import SourceStore


class FirecrawlWebSearchProvider:
    provider = "firecrawl"

    def __init__(
        self,
        *,
        client: FirecrawlSearchClient,
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
            response_model=FirecrawlSearchResponse,
            provider=self.provider,
        )

        if args.topic is WebTopic.NEWS:
            candidates: Iterable[WebSearchCandidate] = (
                WebSearchCandidate(
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    published_date=parse_iso_date(item.date),
                )
                for item in response.data.news
            )
        else:
            candidates = (
                WebSearchCandidate(
                    title=item.title,
                    url=item.url,
                    snippet=item.description,
                )
                for item in response.data.web
            )

        if args.include_domains and args.exclude_domains:
            candidates = (
                candidate
                for candidate in candidates
                if not _url_matches_any_domain(candidate.url, args.exclude_domains)
            )

        return await materialize_search_results(
            provider=self.provider,
            query=args.query,
            max_results=MAX_RESULTS,
            candidates=candidates,
            source_store=self._source_store,
        )


def _url_matches_any_domain(url: str, domains: list[str]) -> bool:
    hostname = urlsplit(url).hostname
    if hostname is None:
        return False

    normalized_hostname = hostname.lower()
    return any(
        normalized_hostname == domain or normalized_hostname.endswith(f".{domain}")
        for domain in domains
    )
