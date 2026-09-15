from __future__ import annotations

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.places_search.yandex.client import (
    YandexOrganisationSearchClient,
)
from tools.observability import ToolExecutionContext, UpstreamCallOutcome
from tools.refs import GeoBounds


async def test_search_sends_expected_bounded_business_request():
    """Verify that search sends expected bounded business request."""

    expected_payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL(
            "https://search-maps.yandex.ru/v1/",
            params={
                "apikey": "test-key",
                "text": "кофейни",
                "type": "biz",
                "lang": "ru_RU",
                "results": "10",
                "ll": "37.621202,55.753544",
                "spn": "0.015000,0.009000",
                "rspn": "1",
            },
        )
        return httpx.Response(200, json=expected_payload)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        context = ToolExecutionContext()
        result = await client.search(
            text="кофейни",
            limit=10,
            center=(37.621202, 55.753544),
            span=(0.015, 0.009),
            context=context,
        )

    assert result == expected_payload
    assert len(context.upstream_calls) == 1
    call = context.upstream_calls[0]
    assert call.provider == "yandex_organisation_search"
    assert call.operation == "search"
    assert call.outcome is UpstreamCallOutcome.SUCCESS
    assert call.status_code == 200


async def test_search_sends_expected_bbox_request():
    """Verify that search sends expected bbox request."""

    expected_payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["bbox"] == ("36.803101,55.142174~37.967427,56.021251")
        assert request.url.params["rspn"] == "1"
        assert "ll" not in request.url.params
        assert "spn" not in request.url.params
        return httpx.Response(200, json=expected_payload)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        result = await client.search(
            text="кофейни",
            limit=10,
            bbox=GeoBounds(
                west=36.803101,
                south=55.142174,
                east=37.967427,
                north=56.021251,
            ),
            context=ToolExecutionContext(),
        )

    assert result == expected_payload


async def test_search_rejects_bbox_combined_with_center_and_span():
    """Verify that search rejects bbox combined with center and span."""

    async with httpx.AsyncClient() as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        with pytest.raises(ValueError, match="bbox cannot be combined"):
            await client.search(
                text="кофейни",
                limit=10,
                center=(37.621202, 55.753544),
                span=(0.015, 0.009),
                bbox=GeoBounds(
                    west=36.803101,
                    south=55.142174,
                    east=37.967427,
                    north=56.021251,
                ),
                context=ToolExecutionContext(),
            )


@pytest.mark.parametrize(
    ("status", "error_code", "retryable"),
    [
        (400, ToolErrorCode.UPSTREAM_ERROR, False),
        (403, ToolErrorCode.UPSTREAM_ERROR, False),
        (408, ToolErrorCode.TIMEOUT, True),
        (429, ToolErrorCode.RATE_LIMITED, True),
        (503, ToolErrorCode.UPSTREAM_ERROR, True),
        (504, ToolErrorCode.TIMEOUT, True),
    ],
)
async def test_search_maps_http_error_metadata(
    status: int,
    error_code: ToolErrorCode,
    retryable: bool,
):
    """Verify that search maps HTTP error metadata."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="кофейни",
                limit=10,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.failure_kind is ToolFailureKind.HTTP_STATUS
    assert exc_info.value.retryable is retryable


async def test_search_maps_timeout_to_retryable_provider_error():
    """Verify that search maps timeout to retryable provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request timed out", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="кофейни",
                limit=10,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.TIMEOUT
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.failure_kind is ToolFailureKind.TIMEOUT
    assert exc_info.value.retryable is True


async def test_search_maps_network_error_to_retryable_provider_error():
    """Verify that search maps network error to retryable provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(
            api_key="test-key",
            http_client=http_client,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="кофейни",
                limit=10,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.failure_kind is ToolFailureKind.NETWORK
    assert exc_info.value.retryable is True


async def test_search_maps_invalid_json_to_safe_upstream_error():
    """Verify that search maps invalid JSON to safe upstream error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(api_key="test-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="кофейни",
                limit=10,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "Yandex returned invalid JSON"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_JSON
    assert exc_info.value.retryable is False


async def test_search_rejects_non_object_json_response():
    """Verify that search rejects non object JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexOrganisationSearchClient(api_key="test-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                text="кофейни",
                limit=10,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "Yandex returned an unexpected response format"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "yandex_organisation_search"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
