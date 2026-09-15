"""Tests for the OSRM-backed routing provider."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tools.base import ToolErrorCode, ToolExecutionError
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.routing import RoutingInput, TrafficType
from tools.geo.routing.osrm import OsrmRoutingClient, OsrmRoutingProvider
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"
C = "plc_c3d4e5f6a7"


def _record(ref: PlaceRef, name: str, lat: float, lon: float) -> PlaceRecord:
    return PlaceRecord(
        ref=ref,
        name=name,
        address=f"Адрес: {name}",
        lat=lat,
        lon=lon,
        origin=RecordOrigin.GEOCODE,
    )


class FakeOsrmClient:
    provider = "osrm_routing"

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
) -> tuple[OsrmRoutingProvider, FakeOsrmClient]:
    store = InMemoryPlaceStore()
    await store.save_many(
        [
            _record(A, "Старт", 55.7, 37.5),
            _record(B, "Точка B", 55.8, 37.6),
            _record(C, "Точка C", 55.9, 37.7),
        ]
    )
    client = FakeOsrmClient(
        route_payload=route_payload,
        matrix_payload=matrix_payload,
    )
    provider = OsrmRoutingProvider(
        client=cast(OsrmRoutingClient, client),
        place_store=store,
    )
    return provider, client


async def test_route_maps_legs_and_generates_english_steps() -> None:
    provider, client = await _provider(
        route_payload={
            "code": "Ok",
            "waypoints": [
                {"distance": 0.0, "name": "Тверская улица"},
                {"distance": 0.0, "name": "Садовая улица"},
            ],
            "routes": [
                {
                    "distance": 2_000.4,
                    "duration": 300.4,
                    "legs": [
                        {
                            "distance": 2_000.4,
                            "duration": 300.4,
                            "steps": [
                                {
                                    "distance": 1_200.0,
                                    "duration": 180.0,
                                    "name": "Тверская улица",
                                    "maneuver": {
                                        "type": "depart",
                                        "modifier": "straight",
                                    },
                                },
                                {
                                    "distance": 800.4,
                                    "duration": 120.4,
                                    "name": "Садовая улица",
                                    "maneuver": {
                                        "type": "turn",
                                        "modifier": "right",
                                    },
                                },
                                {
                                    "distance": 0.0,
                                    "duration": 0.0,
                                    "name": "",
                                    "maneuver": {
                                        "type": "arrive",
                                        "modifier": "right",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )

    context = ToolExecutionContext()
    result = await provider.route(
        RoutingInput(mode="route", waypoints=[A, B]),
        context,
    )

    assert result.route is not None
    assert result.route.length_m == 2_000
    assert result.route.duration_s == 300
    assert result.route.waypoint_order == [0, 1]
    assert [point.snap_distance_m for point in result.route.waypoints] == [0, 0]
    assert result.route.traffic_type is TrafficType.DISABLED
    assert result.route.legs[0].steps[0].instruction == "Depart on Тверская улица"
    assert result.route.legs[0].steps[1].instruction == "Turn right onto Садовая улица"
    assert result.route.legs[0].steps[2].instruction == "You have arrived at your destination"
    assert context.warnings == ("OSRM route times do not include live traffic.",)
    assert client.route_calls[0]["waypoints"] == [(55.7, 37.5), (55.8, 37.6)]


async def test_route_reports_far_waypoint_snapping() -> None:
    provider, _ = await _provider(
        route_payload={
            "code": "Ok",
            "waypoints": [
                {"distance": 420.4, "name": ""},
                {"distance": 10.0, "name": ""},
            ],
            "routes": [
                {
                    "distance": 1_000.0,
                    "duration": 100.0,
                    "legs": [
                        {
                            "distance": 1_000.0,
                            "duration": 100.0,
                            "steps": [
                                {
                                    "distance": 400.0,
                                    "duration": 40.0,
                                    "name": "Test Street",
                                    "maneuver": {
                                        "type": "continue",
                                        "modifier": "straight",
                                    },
                                },
                                {
                                    "distance": 600.0,
                                    "duration": 60.0,
                                    "name": "Test Street",
                                    "maneuver": {
                                        "type": "continue",
                                        "modifier": "straight",
                                    },
                                },
                                {
                                    "distance": 0.0,
                                    "duration": 0.0,
                                    "name": "",
                                    "maneuver": {"type": "arrive"},
                                },
                            ],
                        }
                    ],
                }
            ],
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
    assert [point.snap_distance_m for point in result.route.waypoints] == [420, 10]
    assert len(result.route.legs[0].steps) == 2
    assert result.route.legs[0].steps[0].length_m == 1_000
    assert result.route.legs[0].steps[0].duration_s == 100
    assert context.warnings == (
        "OSRM snapped waypoint 1 by 420 m; "
        "verify that the resolved place matches the intended access point.",
    )


async def test_route_rejects_leg_totals_that_disagree_with_route() -> None:
    provider, _ = await _provider(
        route_payload={
            "code": "Ok",
            "waypoints": [
                {"distance": 0.0, "name": ""},
                {"distance": 0.0, "name": ""},
            ],
            "routes": [
                {
                    "distance": 1_000.0,
                    "duration": 100.0,
                    "legs": [
                        {
                            "distance": 900.0,
                            "duration": 100.0,
                            "steps": [],
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ToolExecutionError, match="distance does not match"):
        await provider.route(
            RoutingInput(mode="route", waypoints=[A, B]),
            ToolExecutionContext(),
        )


async def test_table_uses_shared_ranking_and_preserves_unreachable_pairs() -> None:
    provider, _ = await _provider(
        matrix_payload={
            "code": "Ok",
            "distances": [[2_000.0, None], [1_000.0, 3_000.0]],
            "durations": [[200.0, None], [100.0, 300.0]],
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


async def test_walking_route_is_forwarded_to_osrm_client() -> None:
    provider, client = await _provider(
        route_payload={
            "code": "Ok",
            "waypoints": [
                {"distance": 0.0, "name": ""},
                {"distance": 0.0, "name": ""},
            ],
            "routes": [
                {
                    "distance": 1_000.0,
                    "duration": 720.0,
                    "legs": [
                        {
                            "distance": 1_000.0,
                            "duration": 720.0,
                            "steps": [
                                {
                                    "distance": 1_000.0,
                                    "duration": 720.0,
                                    "name": "Walking path",
                                    "maneuver": {
                                        "type": "continue",
                                        "modifier": "straight",
                                    },
                                },
                                {
                                    "distance": 0.0,
                                    "duration": 0.0,
                                    "name": "",
                                    "maneuver": {"type": "arrive"},
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )

    result = await provider.route(
        RoutingInput(
            mode="route",
            transport="walking",
            waypoints=[A, B],
        ),
        ToolExecutionContext(),
    )

    assert result.route is not None
    assert result.transport.value == "walking"
    assert client.route_calls[0]["transport"].value == "walking"


@pytest.mark.parametrize("transport", ["bicycle", "scooter", "transit"])
async def test_non_driving_transport_is_rejected(transport: str) -> None:
    provider, client = await _provider()

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(
                mode="route",
                transport=transport,
                waypoints=[A, B],
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UNSUPPORTED_FILTER
    assert not client.route_calls
