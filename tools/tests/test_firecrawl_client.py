"""Tests for the Firecrawl v2 web search HTTP client."""

from __future__ import annotations

import json

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.web.firecrawl.client import FirecrawlSearchClient
from tools.web.search import TimeRange, WebSearchInput, WebTopic


async def test_search_sends_documented_firecrawl_news_request() -> None:
    expected_response: dict[str, object] = {
        "success": True,
        "data": {"news": []},
        "id": "search-1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://api.firecrawl.dev/v2/search")
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert json.loads(request.content) == {
            "query": "coffee in moscow",
            "limit": 5,
            "sources": ["news"],
            "tbs": "qdr:w",
            "includeDomains": ["mos.ru"],
        }
        return httpx.Response(200, json=expected_response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = FirecrawlSearchClient(api_key="test-api-key", http_client=http_client)
        context = ToolExecutionContext()
        result = await client.search(
            WebSearchInput(
                query="coffee in moscow",
                topic=WebTopic.NEWS,
                time_range=TimeRange.WEEK,
                include_domains=["mos.ru"],
                exclude_domains=["example.com"],
            ),
            context,
        )

    assert result == expected_response
    assert len(context.upstream_calls) == 1
    upstream = context.upstream_calls[0]
    assert upstream.provider == "firecrawl"
    assert upstream.operation == "search"
    assert upstream.outcome is UpstreamCallOutcome.SUCCESS
    assert upstream.status_code == 200


async def test_general_search_uses_web_source_and_native_exclusions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "query": "coffee in moscow",
            "limit": 5,
            "sources": ["web"],
            "excludeDomains": ["example.com"],
        }
        return httpx.Response(
            200,
            json={"success": True, "data": {"web": []}, "id": "search-1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = FirecrawlSearchClient(api_key="test-api-key", http_client=http_client)
        await client.search(
            WebSearchInput(query="coffee in moscow", exclude_domains=["example.com"]),
            ToolExecutionContext(),
        )


@pytest.mark.parametrize(
    ("status_code", "error_code", "failure_kind", "retryable"),
    [
        (400, ToolErrorCode.INVALID_INPUT, ToolFailureKind.HTTP_STATUS, False),
        (401, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.HTTP_STATUS, False),
        (408, ToolErrorCode.TIMEOUT, ToolFailureKind.TIMEOUT, True),
        (429, ToolErrorCode.RATE_LIMITED, ToolFailureKind.HTTP_STATUS, True),
        (500, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.HTTP_STATUS, True),
    ],
)
async def test_search_maps_http_errors(
    status_code: int,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = FirecrawlSearchClient(api_key="test-api-key", http_client=http_client)
        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(WebSearchInput(query="coffee"), context)

    assert exc_info.value.error_code is error_code
    assert exc_info.value.provider == "firecrawl"
    assert exc_info.value.status_code == status_code
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is retryable
