"""Tests for the TomTom geocoding HTTP client."""

from __future__ import annotations

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.tomtom import TomTomGeocoderClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


async def test_search_sends_expected_tomtom_request() -> None:
    payload: dict[str, object] = {"summary": {"totalResults": 0}, "results": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/search/2/geocode/Berlin.json"
        assert dict(request.url.params) == {
            "key": "test-key",
            "limit": "5",
            "language": "en-US",
            "view": "RU",
        }
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomGeocoderClient(api_key="test-key", http_client=http_client)
        context = ToolExecutionContext()
        result = await client.search("Berlin", limit=5, context=context)

    assert result == payload
    assert len(context.upstream_calls) == 1
    call = context.upstream_calls[0]
    assert call.provider == "tomtom_geocoder"
    assert call.operation == "search"
    assert call.outcome is UpstreamCallOutcome.SUCCESS
    assert call.status_code == 200


async def test_reverse_search_sends_coordinate_and_municipality_filter() -> None:
    payload: dict[str, object] = {"summary": {"numResults": 0}, "addresses": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search/2/reverseGeocode/52.520000,13.405000.json"
        assert dict(request.url.params) == {
            "key": "test-key",
            "language": "de-DE",
            "view": "RU",
            "entityType": "Municipality",
        }
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomGeocoderClient(api_key="test-key", http_client=http_client)
        context = ToolExecutionContext()
        result = await client.reverse_search(
            lat=52.52,
            lon=13.405,
            language="de-DE",
            entity_type="Municipality",
            context=context,
        )

    assert result == payload
    assert context.upstream_calls[0].operation == "reverse_geocode"


async def test_reverse_search_validates_coordinates_before_http() -> None:
    async with httpx.AsyncClient() as http_client:
        client = TomTomGeocoderClient(api_key="test-key", http_client=http_client)
        with pytest.raises(ValueError, match="latitude"):
            await client.reverse_search(
                lat=91,
                lon=13.405,
                context=ToolExecutionContext(),
            )


@pytest.mark.parametrize(
    ("status", "error_code", "failure_kind", "retryable"),
    [
        (403, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.AUTHENTICATION, False),
        (408, ToolErrorCode.TIMEOUT, ToolFailureKind.HTTP_STATUS, True),
        (429, ToolErrorCode.RATE_LIMITED, ToolFailureKind.HTTP_STATUS, True),
        (503, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.HTTP_STATUS, True),
    ],
)
async def test_search_maps_http_errors(
    status: int,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomGeocoderClient(api_key="test-key", http_client=http_client)
        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search("Berlin", limit=5, context=context)

    error = exc_info.value
    assert error.error_code is error_code
    assert error.failure_kind is failure_kind
    assert error.retryable is retryable
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE


async def test_search_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TomTomGeocoderClient(api_key="test-key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search("Berlin", limit=5, context=ToolExecutionContext())

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_JSON
    assert exc_info.value.retryable is False
