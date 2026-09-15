"""Tests for the Yandex route-details and distance-matrix HTTP client."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing import TransportMode
from tools.geo.routing.yandex.client import YandexRoutingClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


async def test_build_route_sends_documented_yandex_parameters() -> None:
    """Verify that build route sends documented Yandex parameters."""

    departure_time = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.copy_with(query=None) == httpx.URL(
            "https://api.routing.yandex.net/v2/route"
        )
        assert dict(request.url.params) == {
            "apikey": "routing-key",
            "mode": "driving",
            "avoid_tolls": "true",
            "departure_time": str(int(departure_time.timestamp())),
            "waypoints": "55.753930,37.620795|55.760000,37.630000",
            "optimize": "true",
        }
        return httpx.Response(200, json={"route": {"legs": []}})

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)
        result = await client.build_route(
            waypoints=[(55.75393, 37.620795), (55.76, 37.63)],
            transport=TransportMode.DRIVING,
            use_traffic=True,
            avoid_tolls=True,
            departure_time=departure_time,
            optimize_waypoints=True,
            context=context,
        )

    assert result == {"route": {"legs": []}}
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "yandex_routing"
    assert context.upstream_calls[0].operation == "build_route"


async def test_distance_matrix_disables_traffic_for_driving() -> None:
    """Verify that distance matrix disables traffic for driving."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.copy_with(query=None) == httpx.URL(
            "https://api.routing.yandex.net/v2/distancematrix"
        )
        assert dict(request.url.params) == {
            "apikey": "routing-key",
            "mode": "driving",
            "traffic": "disabled",
            "origins": "55.700000,37.500000|55.800000,37.600000",
            "destinations": "55.900000,37.700000",
        }
        return httpx.Response(200, json={"rows": []})

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)
        result = await client.distance_matrix(
            origins=[(55.7, 37.5), (55.8, 37.6)],
            destinations=[(55.9, 37.7)],
            transport=TransportMode.DRIVING,
            use_traffic=False,
            avoid_tolls=False,
            departure_time=None,
            context=context,
        )

    assert result == {"rows": []}
    assert context.upstream_calls[0].operation == "distance_matrix"


@pytest.mark.parametrize(
    (
        "status_code",
        "message",
        "error_code",
        "failure_kind",
        "retryable",
    ),
    [
        (
            400,
            "parameter 'waypoints' is missing",
            ToolErrorCode.UPSTREAM_ERROR,
            ToolFailureKind.INTERNAL_CONTRACT,
            False,
        ),
        (
            401,
            "Key not found",
            ToolErrorCode.UPSTREAM_ERROR,
            ToolFailureKind.AUTHENTICATION,
            False,
        ),
        (
            408,
            "Request timeout",
            ToolErrorCode.TIMEOUT,
            ToolFailureKind.TIMEOUT,
            True,
        ),
        (
            429,
            "Counter total limit exceeded",
            ToolErrorCode.RATE_LIMITED,
            ToolFailureKind.HTTP_STATUS,
            True,
        ),
        (
            500,
            "Internal server error",
            ToolErrorCode.UPSTREAM_ERROR,
            ToolFailureKind.HTTP_STATUS,
            True,
        ),
        (
            504,
            "Gateway timeout",
            ToolErrorCode.TIMEOUT,
            ToolFailureKind.TIMEOUT,
            True,
        ),
    ],
)
async def test_http_errors_are_mapped_to_typed_tool_errors(
    status_code: int,
    message: str,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    retryable: bool,
) -> None:
    """Verify documented Yandex error payloads remain classifiable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"errors": [message]},
            request=request,
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.WALKING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                optimize_waypoints=False,
                context=context,
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.status_code == status_code
    assert exc_info.value.provider == "yandex_routing"
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.provider_code is None
    assert exc_info.value.retryable is retryable
    assert len(context.upstream_calls) == 1
    call = context.upstream_calls[0]
    assert call.outcome is UpstreamCallOutcome.FAILURE
    assert call.status_code == status_code
    assert call.error_code == error_code.value
    assert call.failure_kind == failure_kind.value
    assert call.provider_code is None
    assert call.retryable is retryable


async def test_timeout_is_retryable_and_recorded() -> None:
    """Verify that timeout is retryable and recorded."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.distance_matrix(
                origins=[(55.7, 37.5)],
                destinations=[(55.8, 37.6)],
                transport=TransportMode.TRANSIT,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.TIMEOUT
    assert exc_info.value.failure_kind is ToolFailureKind.TIMEOUT
    assert exc_info.value.retryable is True
    assert context.upstream_calls[0].operation == "distance_matrix"


@pytest.mark.parametrize(
    ("content", "failure_kind"),
    [
        (b"not-json", ToolFailureKind.INVALID_JSON),
        (b"[]", ToolFailureKind.INVALID_SCHEMA),
    ],
)
async def test_invalid_response_body_is_rejected(
    content: bytes,
    failure_kind: ToolFailureKind,
) -> None:
    """Verify that invalid response body is rejected."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "application/json"},
            request=request,
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.WALKING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                optimize_waypoints=False,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is failure_kind
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].failure_kind == failure_kind.value


async def test_errors_field_in_success_response_is_rejected_and_recorded() -> None:
    """Verify an undocumented 2xx error payload cannot look successful."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"errors": ["unexpected routing failure"]},
            request=request,
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)

        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.WALKING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                optimize_waypoints=False,
                context=context,
            )

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.provider_code is None
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE


async def test_client_rejects_empty_api_key() -> None:
    async with httpx.AsyncClient() as http_client:
        with pytest.raises(ValueError, match="API key cannot be empty"):
            YandexRoutingClient(api_key=" ", http_client=http_client)


async def test_matrix_element_cap_is_defended_by_the_client() -> None:
    """Verify that matrix element cap is defended by the client."""

    async with httpx.AsyncClient() as http_client:
        client = YandexRoutingClient(api_key="routing-key", http_client=http_client)

        with pytest.raises(ValueError, match="must not exceed 100 elements"):
            await client.distance_matrix(
                origins=[(55.7, 37.5)] * 11,
                destinations=[(55.8, 37.6)] * 10,
                transport=TransportMode.DRIVING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                context=ToolExecutionContext(),
            )
