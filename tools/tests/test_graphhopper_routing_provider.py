"""Tests for the GraphHopper-backed routing provider."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tools.base import ToolErrorCode, ToolExecutionError
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.routing import RoutingInput, TrafficType
from tools.geo.routing.graphhopper import (
    GraphHopperRoutingClient,
    GraphHopperRoutingProvider,
)
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"
C = "plc_c3d4e5f6a7"


def _encode_polyline(points: list[tuple[float, float]]) -> str:
    encoded: list[str] = []
    previous_latitude = 0
    previous_longitude = 0

    for latitude, longitude in points:
        scaled_latitude = round(latitude * 100_000)
        scaled_longitude = round(longitude * 100_000)
        for delta in (
            scaled_latitude - previous_latitude,
            scaled_longitude - previous_longitude,
        ):
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                encoded.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            encoded.append(chr(value + 63))
        previous_latitude = scaled_latitude
        previous_longitude = scaled_longitude

    return "".join(encoded)


def _record(ref: PlaceRef, name: str, lat: float, lon: float) -> PlaceRecord:
    return PlaceRecord(
        ref=ref,
        name=name,
        address=f"Адрес: {name}",
        lat=lat,
        lon=lon,
        origin=RecordOrigin.GEOCODE,
    )


class FakeGraphHopperClient:
    provider = "graphhopper_routing"

    def __init__(
        self,
        *,
        route_payload: dict[str, Any] | None = None,
        matrix_payload: dict[str, Any] | None = None,
    ) -> None:
        self.route_payload = route_payload or {}
        self.matrix_payload = matrix_payload or {}
        self.route_calls: list[dict[str, Any]] = []
        self.matrix_calls: list[dict[str, Any]] = []

    async def build_route(self, **kwargs: Any) -> dict[str, Any]:
        self.route_calls.append(kwargs)
        return self.route_payload

    async def distance_matrix(self, **kwargs: Any) -> dict[str, Any]:
        self.matrix_calls.append(kwargs)
        return self.matrix_payload


async def _provider(
    *,
    route_payload: dict[str, Any] | None = None,
    matrix_payload: dict[str, Any] | None = None,
) -> tuple[GraphHopperRoutingProvider, FakeGraphHopperClient]:
    store = InMemoryPlaceStore()
    await store.save_many(
        [
            _record(A, "Старт", 55.7, 37.5),
            _record(B, "Точка B", 55.8, 37.6),
            _record(C, "Точка C", 55.9, 37.7),
        ]
    )
    client = FakeGraphHopperClient(
        route_payload=route_payload,
        matrix_payload=matrix_payload,
    )
    provider = GraphHopperRoutingProvider(
        client=cast(GraphHopperRoutingClient, client),
        place_store=store,
    )
    return provider, client


async def test_route_maps_leg_details_and_optimized_order() -> None:
    provider, client = await _provider(
        route_payload={
            "paths": [
                {
                    "distance": 3000.0,
                    "time": 600000,
                    "snapped_waypoints": _encode_polyline(
                        [(55.7, 37.5), (55.9, 37.7), (55.8, 37.6)]
                    ),
                    "points_order": [0, 2, 1],
                    "details": {
                        "leg_distance": [[0, 10, 1000.4], [10, 20, 1999.6]],
                        "leg_time": [[0, 10, 240000], [10, 20, 360000]],
                    },
                    "instructions": [
                        {
                            "distance": 1000.4,
                            "time": 240000,
                            "text": "Continue straight",
                            "street_name": "Первая улица",
                            "sign": 0,
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "Waypoint reached",
                            "sign": 5,
                        },
                        {
                            "distance": 1999.6,
                            "time": 360000,
                            "text": "Turn right",
                            "street_name": "Вторая улица",
                            "sign": 2,
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "Destination reached",
                            "sign": 4,
                        },
                    ],
                }
            ]
        }
    )

    context = ToolExecutionContext()
    result = await provider.route(
        RoutingInput(
            mode="route",
            waypoints=[A, B, C],
            optimize_waypoints=True,
        ),
        context,
    )

    assert result.route is not None
    assert result.route.length_m == 3000
    assert result.route.duration_s == 600
    assert result.route.waypoint_order == [0, 2, 1]
    assert [point.ref for point in result.route.waypoints] == [A, C, B]
    assert [point.snap_distance_m for point in result.route.waypoints] == [0, 0, 0]
    assert result.route.has_tolls is None
    assert result.route.traffic_type is TrafficType.DISABLED
    assert [step.instruction for step in result.route.legs[0].steps] == [
        "Continue straight",
        "Waypoint reached",
    ]
    assert [step.instruction for step in result.route.legs[1].steps] == [
        "Turn right",
        "Destination reached",
    ]
    assert context.warnings == ("GraphHopper route times do not include live traffic.",)
    assert client.route_calls[0]["waypoints"] == [
        (55.7, 37.5),
        (55.8, 37.6),
        (55.9, 37.7),
    ]


async def test_route_reports_far_snapping_and_merges_duplicate_continue_steps() -> None:
    provider, _ = await _provider(
        route_payload={
            "paths": [
                {
                    "distance": 1000.0,
                    "time": 60000,
                    "snapped_waypoints": _encode_polyline([(55.71, 37.5), (55.8, 37.6)]),
                    "details": {
                        "leg_distance": [[0, 2, 1000.0]],
                        "leg_time": [[0, 2, 60000]],
                    },
                    "instructions": [
                        {
                            "distance": 400.0,
                            "time": 24000,
                            "text": "Continue on Test Street",
                            "street_name": "Test Street",
                            "sign": 0,
                        },
                        {
                            "distance": 600.0,
                            "time": 36000,
                            "text": "Continue on Test Street",
                            "street_name": "Test Street",
                            "sign": 0,
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "Destination reached",
                            "sign": 4,
                        },
                    ],
                }
            ]
        }
    )
    context = ToolExecutionContext()

    result = await provider.route(
        RoutingInput(
            mode="route",
            waypoints=[A, B],
            use_traffic=False,
        ),
        context,
    )

    assert result.route is not None
    assert result.route.waypoints[0].snap_distance_m is not None
    assert result.route.waypoints[0].snap_distance_m > 1_000
    assert len(result.route.legs[0].steps) == 2
    assert result.route.legs[0].steps[0].length_m == 1_000
    assert result.route.legs[0].steps[0].duration_s == 60
    assert context.warnings[0].startswith("GraphHopper snapped waypoint 1 by ")


async def test_route_restores_a_named_street_transition_inside_one_instruction() -> None:
    provider, _ = await _provider(
        route_payload={
            "paths": [
                {
                    "distance": 1_000.0,
                    "time": 100_000,
                    "snapped_waypoints": _encode_polyline([(55.7, 37.5), (55.8, 37.6)]),
                    "details": {
                        "leg_distance": [[0, 4, 1_000.0]],
                        "leg_time": [[0, 4, 100_000]],
                        "street_name": [
                            [0, 2, "Большая Печёрская улица"],
                            [2, 4, "улица Родионова"],
                        ],
                        "distance": [
                            [0, 2, 400.4],
                            [2, 3, 299.6],
                            [3, 4, 300.0],
                        ],
                        "time": [
                            [0, 2, 40_400],
                            [2, 3, 29_600],
                            [3, 4, 30_000],
                        ],
                    },
                    "instructions": [
                        {
                            "distance": 700.0,
                            "time": 70_000,
                            "text": "Turn right onto Большая Печёрская улица",
                            "street_name": "Большая Печёрская улица",
                            "sign": 2,
                            "interval": [0, 3],
                        },
                        {
                            "distance": 300.0,
                            "time": 30_000,
                            "text": "Continue onto улица Родионова",
                            "street_name": "улица Родионова",
                            "sign": 0,
                            "interval": [3, 4],
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "Arrive at destination",
                            "street_name": "",
                            "sign": 4,
                            "interval": [4, 4],
                        },
                    ],
                }
            ]
        }
    )

    result = await provider.route(
        RoutingInput(mode="route", waypoints=[A, B], use_traffic=False),
        ToolExecutionContext(),
    )

    assert result.route is not None
    steps = result.route.legs[0].steps
    assert [step.instruction for step in steps] == [
        "Turn right onto Большая Печёрская улица",
        "Continue onto улица Родионова",
        "Arrive at destination",
    ]
    assert [step.street_name for step in steps] == [
        "Большая Печёрская улица",
        "улица Родионова",
        None,
    ]
    assert [step.length_m for step in steps] == [400, 600, 0]
    assert [step.duration_s for step in steps] == [40, 60, 0]
    assert sum(step.length_m for step in steps) == result.route.length_m
    assert sum(step.duration_s for step in steps) == result.route.duration_s


async def test_route_rejects_leg_totals_that_disagree_with_path() -> None:
    provider, _ = await _provider(
        route_payload={
            "paths": [
                {
                    "distance": 1000.0,
                    "time": 60000,
                    "snapped_waypoints": _encode_polyline([(55.7, 37.5), (55.8, 37.6)]),
                    "details": {
                        "leg_distance": [[0, 1, 900.0]],
                        "leg_time": [[0, 1, 60000]],
                    },
                    "instructions": [
                        {
                            "distance": 1000.0,
                            "time": 60000,
                            "text": "Continue",
                            "sign": 0,
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "Destination reached",
                            "sign": 4,
                        },
                    ],
                }
            ]
        }
    )

    with pytest.raises(ToolExecutionError, match="distance does not match"):
        await provider.route(
            RoutingInput(mode="route", waypoints=[A, B]),
            ToolExecutionContext(),
        )


async def test_avoid_tolls_adds_non_guarantee_warning_before_routing() -> None:
    provider, client = await _provider()
    context = ToolExecutionContext()

    with pytest.raises(ToolExecutionError):
        await provider.route(
            RoutingInput(
                mode="route",
                waypoints=[A, B],
                avoid_tolls=True,
            ),
            context,
        )

    assert client.route_calls
    assert (
        "GraphHopper penalizes toll roads but cannot guarantee a toll-free route."
        in context.warnings
    )


async def test_matrix_uses_shared_ranking_and_preserves_unreachable_pairs() -> None:
    provider, _ = await _provider(
        matrix_payload={
            "distances": [[2000, None], [1000, 3000]],
            "times": [[200, None], [100, 300]],
        }
    )

    result = await provider.route(
        RoutingInput(
            mode="rank",
            origins=[A, B],
            candidates=[B, C],
            aggregate="min",
            limit=2,
        ),
        ToolExecutionContext(),
    )

    assert [candidate.candidate_index for candidate in result.ranked] == [0, 1]
    assert result.ranked[0].duration_s == 100
    assert result.ranked[1].duration_s == 300
    assert result.ranked[1].per_origin[0].reachable is False
    assert result.unreachable_count == 0


async def test_transit_is_rejected_before_an_http_call() -> None:
    provider, client = await _provider()

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(
                mode="route",
                transport="transit",
                waypoints=[A, B],
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert not client.route_calls


async def test_departure_time_is_rejected() -> None:
    provider, _ = await _provider()

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(
                mode="route",
                waypoints=[A, B],
                departure_time=datetime(2030, 1, 1, tzinfo=UTC),
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
