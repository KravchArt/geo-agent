"""Tests for the 2GIS-backed shared routing provider."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tools.base import ToolErrorCode, ToolExecutionError
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.routing import RoutingInput, TrafficType
from tools.geo.routing.twogis import TwoGisRoutingClient, TwoGisRoutingProvider
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"
C = "plc_c3d4e5f6a7"


def _record(ref: PlaceRef, name: str, lat: float, lon: float) -> PlaceRecord:
    return PlaceRecord(
        ref=ref,
        name=name,
        address=f"Address: {name}",
        lat=lat,
        lon=lon,
        origin=RecordOrigin.GEOCODE,
    )


class FakeTwoGisClient:
    provider = "twogis_routing"

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
) -> tuple[TwoGisRoutingProvider, FakeTwoGisClient]:
    store = InMemoryPlaceStore()
    await store.save_many(
        [
            _record(A, "Start", 55.7, 37.5),
            _record(B, "Candidate B", 55.8, 37.6),
            _record(C, "Candidate C", 55.9, 37.7),
        ]
    )
    client = FakeTwoGisClient(route_payload=route_payload, matrix_payload=matrix_payload)
    return (
        TwoGisRoutingProvider(
            client=cast(TwoGisRoutingClient, client),
            place_store=store,
        ),
        client,
    )


def _detailed_leg(
    *,
    distance: int,
    duration: int,
    street: str,
    turn: str,
) -> dict[str, Any]:
    first_distance = distance // 2
    first_duration = duration // 2
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
                            "distance": first_distance,
                            "duration": first_duration,
                            "names": [street],
                        },
                    },
                    {
                        "comment": turn,
                        "outcoming_path_comment": "Continue straight",
                        "type": "crossroad",
                        "outcoming_path": {
                            "distance": distance - first_distance,
                            "duration": duration - first_duration,
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


async def test_route_materializes_ordered_2gis_legs() -> None:
    provider, client = await _provider(
        route_payload={
            "legs": [
                _detailed_leg(
                    distance=1000,
                    duration=120,
                    street="Bolshaya Pokrovskaya Street",
                    turn="Turn right onto Bolshaya Pokrovskaya Street",
                ),
                _detailed_leg(
                    distance=2000,
                    duration=240,
                    street="Gagarin Avenue",
                    turn="Keep left on Gagarin Avenue",
                ),
            ]
        }
    )

    result = await provider.route(
        RoutingInput(mode="route", waypoints=[A, B, C]),
        ToolExecutionContext(),
    )

    assert result.route is not None
    assert result.route.length_m == 3000
    assert result.route.duration_s == 360
    assert result.route.waypoint_order == [0, 1, 2]
    assert [point.ref for point in result.route.waypoints] == [A, B, C]
    assert result.route.traffic_type is TrafficType.REALTIME
    assert [leg.length_m for leg in result.route.legs] == [1000, 2000]
    assert result.route.legs[0].segments[0].transport.value == "driving"
    assert [step.instruction for step in result.route.legs[0].steps] == [
        "Continue on Bolshaya Pokrovskaya Street",
        "Turn right onto Bolshaya Pokrovskaya Street",
        "You have arrived!",
    ]
    assert result.route.legs[1].steps[1].street_name == "Gagarin Avenue"
    assert client.route_calls[0]["waypoints"] == [
        (55.7, 37.5),
        (55.8, 37.6),
        (55.9, 37.7),
    ]


async def test_public_transport_route_collapses_movement_segments() -> None:
    provider, _ = await _provider(
        route_payload={
            "legs": [
                [
                    {
                        "total_distance": 3500,
                        "total_duration": 1260,
                        "movements": [
                            {
                                "distance": 500,
                                "moving_duration": 360,
                                "waiting_duration": 0,
                                "type": "walkway",
                            },
                            {
                                "distance": 3000,
                                "moving_duration": 600,
                                "waiting_duration": 300,
                                "type": "passage",
                            },
                        ],
                    }
                ]
            ]
        }
    )

    result = await provider.route(
        RoutingInput(mode="route", transport="transit", waypoints=[A, B]),
        ToolExecutionContext(),
    )

    assert result.route is not None
    assert result.route.duration_s == 1260
    assert [segment.transport.value for segment in result.route.legs[0].segments] == [
        "walking",
        "transit",
    ]
    assert result.route.traffic_type is TrafficType.DISABLED


async def test_rank_preserves_unreachable_cells_and_sorts_candidates() -> None:
    provider, _ = await _provider(
        matrix_payload={
            "routes": [
                {
                    "status": "OK",
                    "source_id": 0,
                    "target_id": 0,
                    "distance": 2000,
                    "duration": 600,
                },
                {
                    "status": "ROUTE_NOT_FOUND",
                    "source_id": 0,
                    "target_id": 1,
                    "distance": 0,
                    "duration": 0,
                },
            ]
        }
    )

    result = await provider.route(
        RoutingInput(
            mode="rank",
            transport="walking",
            origins=[A],
            candidates=[B, C],
        ),
        ToolExecutionContext(),
    )

    assert [candidate.point.ref for candidate in result.ranked] == [B, C]
    assert result.ranked[0].duration_s == 600
    assert result.ranked[1].reachable is False
    assert result.unreachable_count == 1


async def test_rank_all_unreachable_becomes_not_found_for_fallback() -> None:
    provider, _ = await _provider(
        matrix_payload={
            "routes": [
                {
                    "status": "ATTRACT_FAIL",
                    "source_id": 0,
                    "target_id": 0,
                    "distance": 0,
                    "duration": 0,
                }
            ]
        }
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(mode="rank", origins=[A], candidates=[B]),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert exc_info.value.provider == "twogis_routing"


async def test_provider_rejects_traffic_disabled_for_coordinator_fallback() -> None:
    provider, client = await _provider()

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(
                mode="route",
                waypoints=[A, B],
                use_traffic=False,
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert exc_info.value.provider == "twogis_routing"
    assert client.route_calls == []
