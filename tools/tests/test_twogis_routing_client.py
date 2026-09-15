"""Tests for the 2GIS Routing and Distance Matrix HTTP client."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing import TransportMode
from tools.geo.routing.twogis import TwoGisRoutingClient
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


def _detailed_response(distance: int, duration: int, street: str) -> dict[str, object]:
    return {
        "status": "OK",
        "type": "result",
        "message": None,
        "result": [
            {
                "total_distance": distance,
                "total_duration": duration,
                "maneuvers": [
                    {
                        "comment": "start",
                        "outcoming_path_comment": f"Continue on {street}",
                        "type": "begin",
                        "outcoming_path": {
                            "distance": distance,
                            "duration": duration,
                            "names": [street],
                        },
                    },
                    {
                        "comment": "finish",
                        "outcoming_path_comment": "You have arrived!",
                        "type": "end",
                    },
                ],
            }
        ],
    }


async def test_build_route_uses_parallel_detailed_requests_for_ordered_legs() -> None:
    departure = datetime(2030, 5, 1, 9, 30, tzinfo=UTC)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.copy_with(query=None) == httpx.URL(
            "https://routing.api.2gis.com/routing/7.0.0/global"
        )
        assert request.url.params["key"] == "dgis-key"
        body = json.loads(request.content)
        requests.append(body)
        assert body["transport"] == "driving"
        assert body["route_mode"] == "fastest"
        assert body["traffic_mode"] == "statistics"
        assert body["output"] == "detailed"
        assert body["locale"] == "en"
        assert body["filters"] == ["toll_road"]
        assert body["utc"] == int(departure.timestamp())
        start = body["points"][0]["lat"]
        return httpx.Response(200, json=_detailed_response(1000, 120, f"Street {start}"))

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        result = await client.build_route(
            waypoints=[(55.7, 37.5), (55.8, 37.6), (55.9, 37.7)],
            transport=TransportMode.DRIVING,
            use_traffic=True,
            avoid_tolls=True,
            departure_time=departure,
            context=context,
        )

    assert [request["points"] for request in requests] == [
        [
            {"lat": 55.7, "lon": 37.5, "type": "stop"},
            {"lat": 55.8, "lon": 37.6, "type": "stop"},
        ],
        [
            {"lat": 55.8, "lon": 37.6, "type": "stop"},
            {"lat": 55.9, "lon": 37.7, "type": "stop"},
        ],
    ]
    assert len(result["legs"]) == 2
    assert result["legs"][1]["result"][0]["total_distance"] == 1000
    assert len(context.upstream_calls) == 2
    assert context.upstream_calls[0].provider == "twogis_routing"
    assert context.upstream_calls[0].operation == "build_route"
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.SUCCESS
    parallel_groups = {call.parallel_group for call in context.upstream_calls}
    assert len(parallel_groups) == 1
    assert None not in parallel_groups


async def test_current_route_uses_detailed_driving_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "points": [
                {"lat": 56.3287, "lon": 44.002, "type": "stop"},
                {"lat": 56.2891, "lon": 43.9792, "type": "stop"},
            ],
            "transport": "driving",
            "route_mode": "fastest",
            "traffic_mode": "jam",
            "output": "detailed",
            "locale": "en",
        }
        return httpx.Response(200, json=_detailed_response(7100, 900, "Gagarin Avenue"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        result = await client.build_route(
            waypoints=[(56.3287, 44.002), (56.2891, 43.9792)],
            transport=TransportMode.DRIVING,
            use_traffic=True,
            avoid_tolls=False,
            departure_time=None,
            context=ToolExecutionContext(),
        )

    assert result["legs"][0]["result"][0]["total_distance"] == 7100


@pytest.mark.parametrize(
    ("transport", "expected_transport"),
    [
        (TransportMode.DRIVING, "driving"),
        (TransportMode.WALKING, "walking"),
        (TransportMode.BICYCLE, "bicycle"),
        (TransportMode.SCOOTER, "scooter"),
    ],
)
async def test_detailed_route_uses_shared_transport_values(
    transport: TransportMode,
    expected_transport: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["transport"] == expected_transport
        expected_point_type = "walking" if transport is TransportMode.WALKING else "stop"
        assert {point["type"] for point in body["points"]} == {expected_point_type}
        return httpx.Response(200, json=_detailed_response(100, 20, "Test street"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        await client.build_route(
            waypoints=[(56.3287, 44.002), (56.2891, 43.9792)],
            transport=transport,
            use_traffic=True,
            avoid_tolls=False,
            departure_time=None,
            context=ToolExecutionContext(),
        )


async def test_route_preserves_object_shaped_2gis_error_classification() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "invalid pairs request"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(56.3287, 44.002), (56.2891, 43.9792)],
                transport=TransportMode.DRIVING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT


async def test_distance_matrix_maps_point_indexes_to_candidate_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.copy_with(query=None) == httpx.URL(
            "https://routing.api.2gis.com/get_dist_matrix"
        )
        assert request.url.params["version"] == "2.0"
        assert json.loads(request.content) == {
            "points": [
                {"lat": 55.7, "lon": 37.5},
                {"lat": 55.8, "lon": 37.6},
                {"lat": 55.9, "lon": 37.7},
            ],
            "sources": [0],
            "targets": [1, 2],
            "transport": "walking",
        }
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "status": "OK",
                        "source_id": 0,
                        "target_id": 1,
                        "distance": 800,
                        "duration": 600,
                    },
                    {
                        "status": "ROUTE_NOT_FOUND",
                        "source_id": 0,
                        "target_id": 2,
                        "distance": 0,
                        "duration": 0,
                    },
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        result = await client.distance_matrix(
            origins=[(55.7, 37.5)],
            destinations=[(55.8, 37.6), (55.9, 37.7)],
            transport=TransportMode.WALKING,
            use_traffic=True,
            avoid_tolls=False,
            departure_time=None,
            context=ToolExecutionContext(),
        )

    assert [(route["source_id"], route["target_id"]) for route in result["routes"]] == [
        (0, 0),
        (0, 1),
    ]


async def test_distance_matrix_chunks_axes_larger_than_sync_limit() -> None:
    request_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        source_count = len(body["sources"])
        request_sizes.append(source_count)
        target_id = source_count
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "status": "OK",
                        "source_id": source_id,
                        "target_id": target_id,
                        "distance": source_id + 1,
                        "duration": source_id + 2,
                    }
                    for source_id in range(source_count)
                ]
            },
        )

    origins = [(55.0 + index / 100, 37.0) for index in range(26)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        result = await client.distance_matrix(
            origins=origins,
            destinations=[(56.0, 38.0)],
            transport=TransportMode.WALKING,
            use_traffic=True,
            avoid_tolls=False,
            departure_time=None,
            context=ToolExecutionContext(),
        )

    assert request_sizes == [25, 1]
    assert len(result["routes"]) == 26
    assert result["routes"][-1]["source_id"] == 25
    assert {route["target_id"] for route in result["routes"]} == {0}


async def test_no_content_is_a_not_found_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, request=request)

    context = ToolExecutionContext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.build_route(
                waypoints=[(55.7, 37.5), (55.8, 37.6)],
                transport=TransportMode.WALKING,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                context=context,
            )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert exc_info.value.failure_kind is ToolFailureKind.PROVIDER_RESPONSE
    assert context.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE


async def test_bad_generated_request_is_not_fallback_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": "bad request"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TwoGisRoutingClient(api_key="dgis-key", http_client=http_client)
        with pytest.raises(ToolExecutionError) as exc_info:
            await client.distance_matrix(
                origins=[(55.7, 37.5)],
                destinations=[(55.8, 37.6)],
                transport=TransportMode.BICYCLE,
                use_traffic=True,
                avoid_tolls=False,
                departure_time=None,
                context=ToolExecutionContext(),
            )

    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT


def test_client_rejects_empty_configuration() -> None:
    with pytest.raises(ValueError, match="key cannot be empty"):
        TwoGisRoutingClient(api_key=" ", http_client=httpx.AsyncClient())
