"""Tests for the Yandex geocoding HTTP client."""

from __future__ import annotations

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.yandex.client import YandexGeocoderClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


def _assert_failure_metrics(
    context: ToolExecutionContext,
    *,
    status_code: int | None,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    retryable: bool,
) -> None:
    assert len(context.upstream_calls) == 1
    call = context.upstream_calls[0]
    assert call.provider == "yandex_geocoder"
    assert call.operation == "search"
    assert call.outcome is UpstreamCallOutcome.FAILURE
    assert call.status_code == status_code
    assert call.error_code == error_code.value
    assert call.failure_kind == failure_kind.value
    assert call.provider_code is None
    assert call.retryable is retryable


async def test_search_sends_expected_yandex_request():
    """Verify that search sends expected Yandex request."""

    expected_payload: dict[str, object] = {
        "response": {
            "GeoObjectCollection": {
                "featureMember": [],
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL(
            "https://geocode-maps.yandex.ru/v1/",
            params={
                "apikey": "test-key",
                "geocode": "Красная площадь, Москва",
                "lang": "ru_RU",
                "results": "3",
                "format": "json",
            },
        )
        return httpx.Response(200, json=expected_payload)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(
            api_key="test-key",
            http_client=http_client,
        )

        context = ToolExecutionContext()
        result = await client.search(
            "Красная площадь, Москва",
            limit=3,
            context=context,
        )

    assert result == expected_payload
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "yandex_geocoder"


async def test_search_localizes_latin_query_response_to_english():
    expected_payload: dict[str, object] = {
        "response": {"GeoObjectCollection": {"featureMember": []}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["geocode"] == "Frankfurt am Main, Germany"
        assert request.url.params["lang"] == "en_US"
        return httpx.Response(200, json=expected_payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexGeocoderClient(api_key="test-key", http_client=http_client)
        result = await client.search(
            "Frankfurt am Main, Germany",
            limit=5,
            context=ToolExecutionContext(),
        )

    assert result == expected_payload


@pytest.mark.parametrize(
    ("status", "error_code", "retryable"),
    [
        (400, ToolErrorCode.UPSTREAM_ERROR, False),
        (404, ToolErrorCode.UPSTREAM_ERROR, False),
        (403, ToolErrorCode.UPSTREAM_ERROR, False),
        (408, ToolErrorCode.TIMEOUT, True),
        (429, ToolErrorCode.RATE_LIMITED, True),
        (503, ToolErrorCode.UPSTREAM_ERROR, True),
        (504, ToolErrorCode.TIMEOUT, True),
    ],
)
async def test_search_maps_yandex_http_error_metadata(
    status: int,
    error_code: ToolErrorCode,
    retryable: bool,
):
    """Verify that search maps Yandex HTTP error metadata."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(
            api_key="test-key",
            http_client=http_client,
        )

        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                "Красная площадь",
                limit=1,
                context=context,
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.status_code == status
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.failure_kind is ToolFailureKind.HTTP_STATUS
    assert exc_info.value.retryable is retryable
    _assert_failure_metrics(
        context,
        status_code=status,
        error_code=error_code,
        failure_kind=ToolFailureKind.HTTP_STATUS,
        retryable=retryable,
    )


async def test_search_maps_timeout_to_retryable_provider_error():
    """Verify that search maps timeout to retryable provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request timed out", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(api_key="test-key", http_client=http_client)

        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                "Красная площадь",
                limit=1,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.TIMEOUT
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.failure_kind is ToolFailureKind.TIMEOUT
    assert exc_info.value.retryable is True
    _assert_failure_metrics(
        context,
        status_code=None,
        error_code=ToolErrorCode.TIMEOUT,
        failure_kind=ToolFailureKind.TIMEOUT,
        retryable=True,
    )


async def test_search_maps_network_error_to_retryable_provider_error():
    """Verify that search maps network error to retryable provider error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(api_key="test-key", http_client=http_client)

        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                "Красная площадь",
                limit=1,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.status_code is None
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.failure_kind is ToolFailureKind.NETWORK
    assert exc_info.value.retryable is True
    _assert_failure_metrics(
        context,
        status_code=None,
        error_code=ToolErrorCode.UPSTREAM_ERROR,
        failure_kind=ToolFailureKind.NETWORK,
        retryable=True,
    )


async def test_search_maps_invalid_json_to_safe_upstream_error():
    """Verify that search maps invalid JSON to safe upstream error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(api_key="test-key", http_client=http_client)

        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                "Красная площадь",
                limit=1,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "Yandex returned invalid JSON"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_JSON
    assert exc_info.value.retryable is False
    _assert_failure_metrics(
        context,
        status_code=200,
        error_code=ToolErrorCode.UPSTREAM_ERROR,
        failure_kind=ToolFailureKind.INVALID_JSON,
        retryable=False,
    )


async def test_search_rejects_non_object_json_response():
    """Verify that search rejects non object JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(api_key="test-key", http_client=http_client)

        context = ToolExecutionContext()
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.search(
                "Красная площадь",
                limit=1,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "Yandex returned an unexpected response format"
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
    _assert_failure_metrics(
        context,
        status_code=200,
        error_code=ToolErrorCode.UPSTREAM_ERROR,
        failure_kind=ToolFailureKind.INVALID_SCHEMA,
        retryable=False,
    )
