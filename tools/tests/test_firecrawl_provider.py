"""Tests for the Firecrawl-backed web search provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext
from tools.refs import SourceRecord, SourceRef
from tools.web.firecrawl.client import FirecrawlSearchClient
from tools.web.firecrawl.provider import FirecrawlWebSearchProvider
from tools.web.search import WebSearchInput, WebTopic


class FakeFirecrawlClient:
    provider = "firecrawl"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return self._payload


class FakeSourceStore:
    def __init__(self) -> None:
        self.records: dict[str, SourceRecord] = {}
        self.save_many_calls: list[list[SourceRecord]] = []

    async def save_many(self, sources: Sequence[SourceRecord]) -> None:
        batch = list(sources)
        self.save_many_calls.append(batch)
        self.records.update({record.ref: record for record in batch})

    async def get(self, ref: SourceRef) -> SourceRecord | None:
        return self.records.get(ref)


def make_provider(payload: dict[str, Any], store: FakeSourceStore) -> FirecrawlWebSearchProvider:
    return FirecrawlWebSearchProvider(
        client=cast(FirecrawlSearchClient, FakeFirecrawlClient(payload)),
        source_store=store,
    )


async def test_general_search_materializes_web_descriptions() -> None:
    store = FakeSourceStore()
    provider = make_provider(
        {
            "success": True,
            "id": "search-1",
            "creditsUsed": 1,
            "data": {
                "web": [
                    {
                        "title": " Coffee guide ",
                        "description": " Useful search excerpt. ",
                        "url": "https://Example.COM/guide#places",
                    }
                ]
            },
        },
        store,
    )

    result = await provider.search(
        WebSearchInput(query="coffee in moscow"),
        ToolExecutionContext(),
    )

    assert result.query == "coffee in moscow"
    assert len(result.results) == 1
    assert result.results[0].title == "Coffee guide"
    assert result.results[0].domain == "example.com"
    assert result.results[0].snippet == "Useful search excerpt."
    assert result.results[0].published_date is None
    assert store.save_many_calls[0][0].url == "https://example.com/guide"


async def test_news_search_materializes_snippet_and_date() -> None:
    store = FakeSourceStore()
    provider = make_provider(
        {
            "success": True,
            "id": "search-1",
            "data": {
                "news": [
                    {
                        "title": "Museum reopens",
                        "snippet": "The museum reopened after renovation.",
                        "url": "https://news.example.org/story",
                        "date": "2026-08-12T09:00:00Z",
                    }
                ]
            },
        },
        store,
    )

    result = await provider.search(
        WebSearchInput(query="museum reopening", topic=WebTopic.NEWS),
        ToolExecutionContext(),
    )

    assert result.results[0].snippet == "The museum reopened after renovation."
    assert result.results[0].published_date is not None
    assert result.results[0].published_date.isoformat() == "2026-08-12"


async def test_combined_domain_filters_are_completed_locally() -> None:
    store = FakeSourceStore()
    provider = make_provider(
        {
            "success": True,
            "id": "search-1",
            "data": {
                "web": [
                    {
                        "title": "Excluded subdomain",
                        "description": "Do not return this.",
                        "url": "https://private.example.com/story",
                    },
                    {
                        "title": "Allowed",
                        "description": "Return this result.",
                        "url": "https://public.example.com/story",
                    },
                ]
            },
        },
        store,
    )

    result = await provider.search(
        WebSearchInput(
            query="story",
            include_domains=["example.com"],
            exclude_domains=["private.example.com"],
        ),
        ToolExecutionContext(),
    )

    assert [item.title for item in result.results] == ["Allowed"]
    assert [record.url for record in store.save_many_calls[0]] == [
        "https://public.example.com/story"
    ]


async def test_unsuccessful_or_malformed_response_is_an_upstream_schema_error() -> None:
    store = FakeSourceStore()
    provider = make_provider(
        {"success": False, "error": "Request failed"},
        store,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.search(WebSearchInput(query="coffee"), ToolExecutionContext())

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.provider == "firecrawl"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
    assert store.save_many_calls == []
