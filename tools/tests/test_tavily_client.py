"""Tests for the Tavily web search HTTP client."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.web.search import TimeRange, WebSearchInput, WebTopic
from tools.web.tavily.client import TavilySearchClient


async def test_search_sends_safe_tavily_request() -> None:
    """Verify that search sends safe Tavily request."""

    expected_payload: dict[str, object] = {
        "query": "coffee in moscow",
        "results": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://api.tavily.com/search")
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert json.loads(request.content) == {
            "query": "coffee in moscow",
            "topic": "news",
            "max_results": 5,
            "search_depth": "advanced",
            "chunks_per_source": 3,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "time_range": "week",
            "include_domains": ["mos.ru"],
            "exclude_domains": ["example.com"],
        }
        return httpx.Response(200, json=expected_payload)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(
            api_key="test-api-key",
            http_client=http_client,
        )

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

    assert result == expected_payload
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "tavily"
    assert context.upstream_calls[0].operation == "search"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.upstream_calls[0].status_code == 200


@pytest.mark.parametrize(
    ("status_code", "error_code", "retryable"),
    [
        (400, ToolErrorCode.INVALID_INPUT, False),
        (403, ToolErrorCode.UPSTREAM_ERROR, False),
        (429, ToolErrorCode.RATE_LIMITED, True),
        (500, ToolErrorCode.UPSTREAM_ERROR, True),
    ],
)
async def test_search_maps_http_errors_to_tool_errors(
    status_code: int,
    error_code: ToolErrorCode,
    retryable: bool,
) -> None:
    """Verify that search maps HTTP errors to tool errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)
        context = ToolExecutionContext()

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                context,
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.status_code == status_code
    assert exc_info.value.provider == "tavily"
    assert exc_info.value.failure_kind is ToolFailureKind.HTTP_STATUS
    assert exc_info.value.retryable is retryable
    assert len(context.upstream_calls) == 1
    upstream = context.upstream_calls[0]
    assert upstream.provider == "tavily"
    assert upstream.outcome is UpstreamCallOutcome.FAILURE
    assert upstream.status_code == status_code
    assert upstream.error_code == error_code.value
    assert upstream.failure_kind == ToolFailureKind.HTTP_STATUS.value
    assert upstream.retryable is retryable


async def test_search_maps_timeout_to_tool_error() -> None:
    """Verify that search maps timeout to tool error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)
        context = ToolExecutionContext()

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                context,
            )

    assert exc_info.value.error_code is ToolErrorCode.TIMEOUT
    assert str(exc_info.value) == "web search timed out"
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "tavily"
    assert exc_info.value.failure_kind is ToolFailureKind.TIMEOUT
    assert exc_info.value.retryable is True
    assert len(context.upstream_calls) == 1
    upstream = context.upstream_calls[0]
    assert upstream.provider == "tavily"
    assert upstream.outcome is UpstreamCallOutcome.FAILURE
    assert upstream.status_code is None
    assert upstream.error_code == ToolErrorCode.TIMEOUT.value
    assert upstream.failure_kind == ToolFailureKind.TIMEOUT.value
    assert upstream.retryable is True


async def test_cancelled_search_records_failed_upstream_call() -> None:
    request_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await never_finishes.wait()
        return httpx.Response(200, json={}, request=request)

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)
        task = asyncio.create_task(client.search(WebSearchInput(query="coffee"), context))
        await request_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(context.upstream_calls) == 1
    upstream = context.upstream_calls[0]
    assert upstream.provider == "tavily"
    assert upstream.operation == "search"
    assert upstream.outcome is UpstreamCallOutcome.FAILURE
    assert upstream.error_code == ToolErrorCode.TIMEOUT.value
    assert upstream.failure_kind == ToolFailureKind.TIMEOUT.value
    assert upstream.retryable is True


async def test_search_maps_network_error_to_retryable_provider_error() -> None:
    """Verify that search maps network error to retryable provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "tavily"
    assert exc_info.value.failure_kind is ToolFailureKind.NETWORK
    assert exc_info.value.retryable is True


async def test_search_rejects_invalid_json_response() -> None:
    """Verify that search rejects invalid JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "web search provider returned invalid data"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "tavily"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_JSON
    assert exc_info.value.retryable is False


async def test_search_rejects_non_object_json_response() -> None:
    """Verify that search rejects non object JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[], request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = TavilySearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "web search provider returned invalid data"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "tavily"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
