"""Cross-provider contract tests for the OSRM and GraphHopper adapters."""

from __future__ import annotations

from typing import Any, cast

from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.routing import RoutingInput
from tools.geo.routing.graphhopper import (
    GraphHopperRoutingClient,
    GraphHopperRoutingProvider,
)
from tools.geo.routing.osrm import OsrmRoutingClient, OsrmRoutingProvider
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, RecordOrigin

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"


class FakeRoutingClient:
    def __init__(
        self,
        *,
        provider: str,
        route_payload: dict[str, Any],
        matrix_payload: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.route_payload = route_payload
        self.matrix_payload = matrix_payload
        self.route_calls: list[dict[str, Any]] = []
        self.matrix_calls: list[dict[str, Any]] = []

    async def build_route(self, **kwargs: Any) -> dict[str, Any]:
        self.route_calls.append(kwargs)
        return self.route_payload

    async def distance_matrix(self, **kwargs: Any) -> dict[str, Any]:
        self.matrix_calls.append(kwargs)
        return self.matrix_payload


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


async def _providers() -> tuple[
    tuple[OsrmRoutingProvider, FakeRoutingClient],
    tuple[GraphHopperRoutingProvider, FakeRoutingClient],
]:
    store = InMemoryPlaceStore()
    await store.save_many(
        [
            PlaceRecord(
                ref=A,
                name="Start",
                address="Start address",
                lat=55.7,
                lon=37.5,
                origin=RecordOrigin.GEOCODE,
            ),
            PlaceRecord(
                ref=B,
                name="Destination",
                address="Destination address",
                lat=55.8,
                lon=37.6,
                origin=RecordOrigin.GEOCODE,
            ),
        ]
    )
    osrm_client = FakeRoutingClient(
        provider="osrm_routing",
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
                            "distance": 1_000.0,
                            "duration": 100.0,
                            "steps": [
                                {
                                    "distance": 1_000.0,
                                    "duration": 100.0,
                                    "name": "Shared Street",
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
        },
        matrix_payload={
            "code": "Ok",
            "distances": [[1_000.0]],
            "durations": [[100.0]],
        },
    )
    graphhopper_client = FakeRoutingClient(
        provider="graphhopper_routing",
        route_payload={
            "paths": [
                {
                    "distance": 1_000.0,
                    "time": 100_000,
                    "snapped_waypoints": _encode_polyline([(55.7, 37.5), (55.8, 37.6)]),
                    "details": {
                        "leg_distance": [[0, 1, 1_000.0]],
                        "leg_time": [[0, 1, 100_000]],
                    },
                    "instructions": [
                        {
                            "distance": 1_000.0,
                            "time": 100_000,
                            "text": "Continue straight on Shared Street",
                            "street_name": "Shared Street",
                            "sign": 0,
                        },
                        {
                            "distance": 0.0,
                            "time": 0,
                            "text": "You have arrived at your destination",
                            "sign": 4,
                        },
                    ],
                }
            ]
        },
        matrix_payload={
            "distances": [[1_000.0]],
            "times": [[100.0]],
        },
    )
    return (
        (
            OsrmRoutingProvider(
                client=cast(OsrmRoutingClient, osrm_client),
                place_store=store,
            ),
            osrm_client,
        ),
        (
            GraphHopperRoutingProvider(
                client=cast(GraphHopperRoutingClient, graphhopper_client),
                place_store=store,
            ),
            graphhopper_client,
        ),
    )


async def test_equivalent_route_payloads_produce_the_same_tool_contract() -> None:
    (osrm, osrm_client), (graphhopper, graphhopper_client) = await _providers()
    args = RoutingInput(
        mode="route",
        waypoints=[A, B],
        use_traffic=False,
    )

    osrm_result = await osrm.route(args, ToolExecutionContext())
    graphhopper_result = await graphhopper.route(args, ToolExecutionContext())

    assert osrm_result == graphhopper_result
    assert set(osrm_client.route_calls[0]) == set(graphhopper_client.route_calls[0])
    assert osrm_client.route_calls[0]["waypoints"] == graphhopper_client.route_calls[0]["waypoints"]


async def test_equivalent_matrix_payloads_produce_the_same_ranking_contract() -> None:
    (osrm, osrm_client), (graphhopper, graphhopper_client) = await _providers()
    args = RoutingInput(
        mode="rank",
        origins=[A],
        candidates=[B],
    )

    osrm_result = await osrm.route(args, ToolExecutionContext())
    graphhopper_result = await graphhopper.route(args, ToolExecutionContext())

    assert osrm_result == graphhopper_result
    assert set(osrm_client.matrix_calls[0]) == set(graphhopper_client.matrix_calls[0])
