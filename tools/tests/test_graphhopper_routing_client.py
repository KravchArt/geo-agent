"""Tests for the GraphHopper route and matrix HTTP client."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing import TransportMode
from tools.geo.routing.graphhopper import GraphHopperRoutingClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


async def test_build_route_sends_graphhopper_get_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.copy_with(query=None) == httpx.URL("https://graphhopper.com/api/1/route")
        assert request.url.params["key"] == "graphhopper-key"
        assert request.url.params.get_list("point") == [
            "55.75393,37.620795",
            "55.76,37.63",
        ]
        assert request.url.params["profile"] == "car_avoid_toll"
        assert request.url.params.get_list("details") == [
            "leg_distance",
            "leg_time",
            "street_name",
            "distance",
            "time",
        ]
        assert request.url.params["instructions"] == "true"
        assert request.url.params["locale"] == "en"
        assert request.url.params["points_encoded"] == "true"
        assert request.url.params["optimize"] == "true"
        assert request.headers["accept-encoding"] == "identity"
        assert request.content == b""
        return httpx.Response(200, json={"paths": []})

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
        )
        result = await client.build_route(
            waypoints=[(55.75393, 37.620795), (55.76, 37.63)],
            transport=TransportMode.DRIVING,
            avoid_tolls=True,
            optimize_waypoints=True,
            context=context,
        )

    assert result == {"paths": []}
    assert context.upstream_calls[0].provider == "graphhopper_routing"
    assert context.upstream_calls[0].operation == "build_route"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    assert context.upstream_calls[0].status_code == 200


async def test_distance_matrix_requests_partial_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.copy_with(query=None) == httpx.URL(
            "https://graphhopper.com/api/1/matrix"
        )
        assert request.headers["accept-encoding"] == "identity"
        assert json.loads(request.content) == {
            "from_points": [[37.5, 55.7]],
            "to_points": [[37.6, 55.8], [37.7, 55.9]],
            "profile": "foot",
            "out_arrays": ["distances", "times"],
            "fail_fast": False,
        }
        return httpx.Response(
            200,
            json={"distances": [[1000, None]], "times": [[720, None]]},
        )

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
        )
        result = await client.distance_matrix(
            origins=[(55.7, 37.5)],
            destinations=[(55.8, 37.6), (55.9, 37.7)],
            transport=TransportMode.WALKING,
            avoid_tolls=False,
            context=context,
        )

    assert result["times"] == [[720, None]]
    assert context.upstream_calls[0].operation == "distance_matrix"


@pytest.mark.parametrize("operation", ["route", "matrix"])
async def test_http_timeout_is_limited_to_graphhopper_request(operation: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"paths": []}, request=request)

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
            http_timeout_s=0.001,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            if operation == "route":
                await client.build_route(
                    waypoints=[(55.7, 37.5), (55.8, 37.6)],
                    transport=TransportMode.DRIVING,
                    avoid_tolls=False,
                    optimize_waypoints=False,
                    context=context,
                )
            else:
                await client.distance_matrix(
                    origins=[(55.7, 37.5)],
                    destinations=[(55.8, 37.6)],
                    transport=TransportMode.DRIVING,
                    avoid_tolls=False,
                    context=context,
                )

    assert exc_info.value.error_code is ToolErrorCode.TIMEOUT
    assert exc_info.value.provider == "graphhopper_routing"
    assert exc_info.value.failure_kind is ToolFailureKind.TIMEOUT
    assert exc_info.value.retryable is True
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].operation == (
        "build_route" if operation == "route" else "distance_matrix"
    )
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].error_code == ToolErrorCode.TIMEOUT.value
    assert context.upstream_calls[0].failure_kind == ToolFailureKind.TIMEOUT.value


@pytest.mark.parametrize(
    ("status_code", "error_code", "retryable"),
    [
        (400, ToolErrorCode.UPSTREAM_ERROR, False),
        (401, ToolErrorCode.UPSTREAM_ERROR, False),
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
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.BICYCLE,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is error_code
    assert exc_info.value.provider == "graphhopper_routing"
    assert exc_info.value.retryable is retryable


async def test_authentication_error_is_preserved_in_metrics() -> None:
    context = ToolExecutionContext()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"message": "Invalid API key"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="invalid-key",
            http_client=http_client,
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=context,
            )

    error = exc_info.value
    assert error.status_code == 401
    assert error.failure_kind is ToolFailureKind.AUTHENTICATION
    assert error.retryable is False
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert context.upstream_calls[0].status_code == 401
    assert context.upstream_calls[0].failure_kind == ToolFailureKind.AUTHENTICATION.value


async def test_point_not_found_hint_is_preserved() -> None:
    context = ToolExecutionContext()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "message": "Cannot find point",
                "hints": [{"details": "PointNotFound"}],
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert exc_info.value.provider_code == "PointNotFound"
    assert context.upstream_calls[0].provider_code == "PointNotFound"
    assert context.upstream_calls[0].error_code == ToolErrorCode.NOT_FOUND.value


@pytest.mark.parametrize(
    ("response", "failure_kind", "public_message"),
    [
        (
            httpx.Response(200, content=b"not json"),
            ToolFailureKind.INVALID_JSON,
            "GraphHopper returned invalid JSON",
        ),
        (
            httpx.Response(200, json=[]),
            ToolFailureKind.INVALID_SCHEMA,
            "GraphHopper returned an unexpected response format",
        ),
    ],
)
async def test_invalid_response_is_rejected(
    response: httpx.Response,
    failure_kind: ToolFailureKind,
    public_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GraphHopperRoutingClient(
            api_key="graphhopper-key",
            http_client=http_client,
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.DRIVING,
                avoid_tolls=False,
                optimize_waypoints=False,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == public_message
    assert exc_info.value.status_code == 200
    assert exc_info.value.provider == "graphhopper_routing"
    assert exc_info.value.failure_kind is failure_kind
    assert exc_info.value.retryable is False
