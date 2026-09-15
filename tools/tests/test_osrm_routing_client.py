"""Tests for the OSRM Route and Table HTTP client."""

from __future__ import annotations

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing import TransportMode
from tools.geo.routing.osrm import OsrmRoutingClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


async def test_build_route_sends_osrm_get_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == ("/route/v1/driving/37.620795,55.753930;37.632778,55.829813")
        assert request.url.params["steps"] == "true"
        assert request.url.params["overview"] == "false"
        assert request.url.params["alternatives"] == "false"
        assert request.url.params["exclude"] == "toll"
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["user-agent"] == "GeoAgent/test"
        return httpx.Response(200, json={"code": "Ok", "routes": []})

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(
            http_client=http_client,
            user_agent="GeoAgent/test",
        )
        result = await client.build_route(
            waypoints=[(55.75393, 37.620795), (55.829813, 37.632778)],
            transport=TransportMode.DRIVING,
            avoid_tolls=True,
            optimize_waypoints=False,
            context=context,
        )

    assert result == {"code": "Ok", "routes": []}
    assert context.upstream_calls[0].provider == "osrm_routing"
    assert context.upstream_calls[0].operation == "build_route"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.upstream_calls[0].status_code == 200


async def test_walking_route_uses_dedicated_foot_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "routing.openstreetmap.de"
        assert request.url.path.startswith("/routed-foot/route/v1/driving/")
        return httpx.Response(200, json={"code": "Ok", "routes": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(http_client=http_client)
        await client.build_route(
            waypoints=[(55.75393, 37.620795), (55.829813, 37.632778)],
            transport=TransportMode.WALKING,
            avoid_tolls=False,
            optimize_waypoints=False,
            context=ToolExecutionContext(),
        )


async def test_distance_matrix_uses_source_and_destination_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            "/table/v1/driving/37.500000,55.700000;37.600000,55.800000;37.700000,55.900000"
        )
        assert request.url.params["annotations"] == "duration,distance"
        assert request.url.params["sources"] == "0"
        assert request.url.params["destinations"] == "1;2"
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[100.0, None]],
                "distances": [[1_000.0, None]],
            },
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(http_client=http_client)
        result = await client.distance_matrix(
            origins=[(55.7, 37.5)],
            destinations=[(55.8, 37.6), (55.9, 37.7)],
            transport=TransportMode.DRIVING,
            avoid_tolls=False,
            context=context,
        )

    assert result["durations"] == [[100.0, None]]
    assert context.upstream_calls[0].operation == "distance_matrix"


@pytest.mark.parametrize(
    ("status_code", "error_code", "retryable"),
    [
        (400, ToolErrorCode.UPSTREAM_ERROR, False),
        (429, ToolErrorCode.RATE_LIMITED, True),
        (500, ToolErrorCode.UPSTREAM_ERROR, True),
        (504, ToolErrorCode.TIMEOUT, True),
    ],
)
async def test_http_errors_are_mapped(
    status_code: int,
    error_code: ToolErrorCode,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.provider == "osrm_routing"
    assert exc_info.value.retryable is retryable


@pytest.mark.parametrize(
    ("provider_code", "error_code", "failure_kind"),
    [
        ("NoSegment", ToolErrorCode.NOT_FOUND, ToolFailureKind.PROVIDER_RESPONSE),
        ("NoRoute", ToolErrorCode.NOT_FOUND, ToolFailureKind.PROVIDER_RESPONSE),
        ("TooBig", ToolErrorCode.INVALID_INPUT, ToolFailureKind.HTTP_STATUS),
        ("InvalidOptions", ToolErrorCode.UPSTREAM_ERROR, ToolFailureKind.INTERNAL_CONTRACT),
    ],
)
async def test_osrm_http_400_payload_is_classified(
    provider_code: str,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
) -> None:
    context = ToolExecutionContext()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"code": provider_code, "message": "OSRM error"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=context,
            )

    error = exc_info.value
    assert error.error_code is error_code
    assert error.failure_kind is failure_kind
    assert error.provider_code == provider_code
    assert error.status_code == 400
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].provider_code == provider_code
    assert context.upstream_calls[0].error_code == error_code.value
    assert context.upstream_calls[0].failure_kind == failure_kind.value


@pytest.mark.parametrize(
    ("upstream_error", "error_code", "failure_kind", "public_message"),
    [
        (
            httpx.ReadTimeout("timed out"),
            ToolErrorCode.TIMEOUT,
            ToolFailureKind.TIMEOUT,
            "OSRM request timed out",
        ),
        (
            httpx.ConnectError("connection failed"),
            ToolErrorCode.UPSTREAM_ERROR,
            ToolFailureKind.NETWORK,
            "OSRM service is unavailable",
        ),
    ],
)
async def test_transport_errors_are_mapped_and_measured(
    upstream_error: httpx.HTTPError,
    error_code: ToolErrorCode,
    failure_kind: ToolFailureKind,
    public_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        upstream_error.request = request
        raise upstream_error

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OsrmRoutingClient(http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=context,
            )

    assert exc_info.value.error_code is error_code
    assert str(exc_info.value) == public_message
    assert exc_info.value.provider == "osrm_routing"
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is True
    assert context.upstream_calls[0].operation == "build_route"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].error_code == error_code.value
    assert context.upstream_calls[0].failure_kind == failure_kind.value
