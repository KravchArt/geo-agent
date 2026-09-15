"""HTTP client tests for OpenStreetMap Overpass search."""

from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.places_search.osm.client import OsmOverpassClient
from tools.geo.places_search.schemas import PlaceCategory
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds

_RESPONSE = {
    "version": 0.6,
    "generator": "Overpass API",
    "osm3s": {},
    "elements": [],
}


async def test_posts_form_encoded_query_with_identifying_user_agent():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["User-Agent"] == "GeoAgent/test (test@example.test)"
        assert request.headers["Accept"] == "application/json"
        assert request.extensions["timeout"] == {
            "connect": 5.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }

        form = parse_qs(request.content.decode())
        query = form["data"][0]
        assert 'nwr["amenity"="cafe"]["cuisine"~"(^|;)coffee_shop(;|$)"]' in query
        assert (
            'rel["boundary"="administrative"]["name"="Москва"]'
            "(55.100000,36.800000,56.000000,38.000000);" in query
        )
        assert ".searchBoundaries map_to_area ->.searchArea;" in query
        assert "(area.searchArea)" in query
        assert "out body geom" not in query
        assert '["addr:city"' not in query
        assert '["addr:place"' not in query
        return httpx.Response(200, json=_RESPONSE)

    context = ToolExecutionContext()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test (test@example.test)",
        )
        payload = await client.search(
            text="кофейни",
            category=PlaceCategory.COFFEE_SHOP,
            limit=50,
            open_24h=False,
            bbox=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
            boundary_name="Москва",
            context=context,
        )

    assert payload == _RESPONSE
    assert context.upstream_calls[0].provider == "osm_overpass"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.upstream_calls[0].status_code == 200


async def test_retries_retryable_failure_once():
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json=_RESPONSE)

    context = ToolExecutionContext()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test",
            retries=1,
        )
        payload = await client.search(
            text="аптека",
            category=PlaceCategory.PHARMACY,
            limit=20,
            open_24h=False,
            center=(37.62, 55.75),
            radius_m=1_000,
            context=context,
        )

    assert payload == _RESPONSE
    assert request_count == 2
    assert len(context.upstream_calls) == 2
    assert [call.outcome for call in context.upstream_calls] == [
        UpstreamCallOutcome.FAILURE,
        UpstreamCallOutcome.SUCCESS,
    ]
    assert context.upstream_calls[0].status_code == 503
    assert context.upstream_calls[0].retryable is True


async def test_does_not_retry_non_retryable_failure():
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, text="invalid query")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test",
        )
        with pytest.raises(ToolExecutionError):
            await client.search(
                text="аптека",
                category=PlaceCategory.PHARMACY,
                limit=20,
                open_24h=False,
                center=(37.62, 55.75),
                radius_m=1_000,
                context=ToolExecutionContext(),
            )

    assert request_count == 1


@pytest.mark.parametrize(
    ("status_code", "error_code", "failure_kind", "retryable"),
    [
        (400, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.HTTP_STATUS, False),
        (429, ToolErrorCode.RATE_LIMITED, ToolFailureKind.HTTP_STATUS, True),
        (504, ToolErrorCode.TIMEOUT, ToolFailureKind.HTTP_STATUS, True),
        (503, ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.HTTP_STATUS, True),
    ],
)
async def test_classifies_http_errors(
    status_code: int,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    retryable: bool,
):
    transport = httpx.MockTransport(lambda _: httpx.Response(status_code, text="error"))

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test",
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="аптека",
                category=PlaceCategory.PHARMACY,
                limit=20,
                open_24h=False,
                center=(37.62, 55.75),
                radius_m=1_000,
                context=context,
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is retryable
    call = context.upstream_calls[0]
    assert call.outcome is UpstreamCallOutcome.FAILURE
    assert call.status_code == status_code
    assert call.error_code == error_code.value
    assert call.failure_kind == failure_kind.value
    assert call.retryable is retryable


@pytest.mark.parametrize(
    ("response", "failure_kind"),
    [
        (httpx.Response(200, text="not json"), ToolFailureKind.INVALID_JSON),
        (httpx.Response(200, json=[]), ToolFailureKind.INVALID_SCHEMA),
    ],
)
async def test_classifies_invalid_success_payloads_as_failed_calls(
    response: httpx.Response,
    failure_kind: ToolFailureKind,
) -> None:
    context = ToolExecutionContext()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response),
    ) as http_client:
        client = OsmOverpassClient(
            http_client=http_client,
            endpoint="https://overpass.example.test/api/interpreter",
            user_agent="GeoAgent/test",
        )

        with pytest.raises(ToolExecutionError):
            await client.search(
                text="аптека",
                category=PlaceCategory.PHARMACY,
                limit=20,
                open_24h=False,
                center=(37.62, 55.75),
                radius_m=1_000,
                context=context,
            )

    call = context.upstream_calls[0]
    assert call.outcome is UpstreamCallOutcome.FAILURE
    assert call.status_code == 200
    assert call.error_code == ToolErrorCode.UPSTREAM_ERROR.value
    assert call.failure_kind == failure_kind.value
    assert call.retryable is False
