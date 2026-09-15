"""Tests for the Exa web search HTTP client."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext
from tools.web.exa.client import ExaSearchClient
from tools.web.search import MAX_SNIPPET_CHARS, TimeRange, WebSearchInput, WebTopic


async def test_search_sends_current_cost_bounded_exa_request() -> None:
    """Verify that search sends current cost bounded Exa request."""

    expected_payload: dict[str, object] = {
        "requestId": "request-1",
        "results": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://api.exa.ai/search")
        assert request.headers["x-api-key"] == "test-api-key"
        assert json.loads(request.content) == {
            "query": "coffee in moscow",
            "type": "auto",
            "numResults": 5,
            "category": "news",
            "startPublishedDate": "2026-07-12T12:30:00Z",
            "includeDomains": ["mos.ru"],
            "excludeDomains": ["example.com"],
            "contents": {
                "highlights": {
                    "maxCharacters": MAX_SNIPPET_CHARS,
                }
            },
        }
        return httpx.Response(200, json=expected_payload)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = ExaSearchClient(
            api_key="test-api-key",
            http_client=http_client,
            clock=lambda: datetime(2026, 7, 19, 12, 30, tzinfo=UTC),
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
    assert context.upstream_calls[0].provider == "exa"
    assert context.upstream_calls[0].operation == "search"


async def test_general_search_omits_optional_exa_filters() -> None:
    """Verify that general search omits optional Exa filters."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "query": "coffee in moscow",
            "type": "auto",
            "numResults": 5,
            "contents": {
                "highlights": {
                    "maxCharacters": MAX_SNIPPET_CHARS,
                }
            },
        }
        return httpx.Response(200, json={"requestId": "request-1", "results": []})

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = ExaSearchClient(api_key="test-api-key", http_client=http_client)
        await client.search(
            WebSearchInput(query="coffee in moscow"),
            ToolExecutionContext(),
        )


@pytest.mark.parametrize(
    ("status_code", "error_code", "retryable"),
    [
        (400, ToolErrorCode.INVALID_INPUT, False),
        (401, ToolErrorCode.UPSTREAM_ERROR, False),
        (402, ToolErrorCode.UPSTREAM_ERROR, False),
        (422, ToolErrorCode.INVALID_INPUT, False),
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
        client = ExaSearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.status_code == status_code
    assert exc_info.value.provider == "exa"
    assert exc_info.value.failure_kind is ToolFailureKind.HTTP_STATUS
    assert exc_info.value.retryable is retryable


@pytest.mark.parametrize(
    ("upstream_error", "error_code", "failure_kind", "public_message"),
    [
        (
            httpx.ReadTimeout("timed out"),
            ToolErrorCode.TIMEOUT,
            ToolFailureKind.TIMEOUT,
            "web search timed out",
        ),
        (
            httpx.ConnectError("connection failed"),
            ToolErrorCode.UPSTREAM_ERROR,
            ToolFailureKind.NETWORK,
            "web search provider is unavailable",
        ),
    ],
)
async def test_search_maps_transport_errors(
    upstream_error: httpx.HTTPError,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    public_message: str,
) -> None:
    """Verify that search maps transport errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_error.request = request
        raise upstream_error

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = ExaSearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is error_code
    assert str(exc_info.value) == public_message
    assert exc_info.value.provider == "exa"
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is True


@pytest.mark.parametrize(
    ("response", "failure_kind"),
    [
        (httpx.Response(200, content=b"not json"), ToolFailureKind.INVALID_JSON),
        (httpx.Response(200, json=[]), ToolFailureKind.INVALID_SCHEMA),
    ],
)
async def test_search_rejects_invalid_response(
    response: httpx.Response,
    failure_kind: ToolFailureKind,
) -> None:
    """Verify that search rejects invalid response."""

    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = ExaSearchClient(api_key="test-api-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                WebSearchInput(query="coffee in moscow"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "web search provider returned invalid data"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "exa"
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is False
