"""Tests for the Yandex-backed routing provider."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding import (
    GeocodedPlaceResolver,
    GeocodePlaceInput,
    GeocodePlaceOutput,
    GeocoderService,
    PlaceMatch,
)
from tools.geo.place_store import InMemoryPlaceStore
from tools.geo.routing import (
    Aggregate,
    RoutingInput,
    RoutingOutput,
    RoutingPlaceResolver,
    TransportMode,
)
from tools.geo.routing.yandex.client import YandexRoutingClient
from tools.geo.routing.yandex.provider import YandexRoutingProvider
from tools.geo.text_place_resolution import TextPlaceResolver
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"
C = "plc_c3d4e5f6a7"
D = "plc_d4e5f6a7b8"
E = "plc_e5f6a7b8c9"
KREMLIN = mint_place_ref("yandex:geocoder:kremlin")
VDNH = mint_place_ref("yandex:geocoder:vdnh")


def _record(ref: PlaceRef, name: str, lat: float, lon: float) -> PlaceRecord:
    return PlaceRecord(
        ref=ref,
        name=name,
        address=f"Адрес: {name}",
        lat=lat,
        lon=lon,
        origin=RecordOrigin.GEOCODE,
    )


class FakeYandexRoutingClient:
    provider = "yandex_routing"

    def __init__(
        self,
        *,
        route_payload: dict[str, Any] | None = None,
        matrix_payload: dict[str, Any] | None = None,
        route_error: ValueError | None = None,
        matrix_error: ValueError | None = None,
    ) -> None:
        self.route_payload = route_payload or {}
        self.matrix_payload = matrix_payload or {}
        self.route_error = route_error
        self.matrix_error = matrix_error
        self.route_calls: list[dict[str, Any]] = []
        self.matrix_calls: list[dict[str, Any]] = []

    async def build_route(self, **kwargs: Any) -> dict[str, Any]:
        self.route_calls.append(kwargs)
        if self.route_error is not None:
            raise self.route_error
        return self.route_payload

    async def distance_matrix(self, **kwargs: Any) -> dict[str, Any]:
        self.matrix_calls.append(kwargs)
        if self.matrix_error is not None:
            raise self.matrix_error
        return self.matrix_payload


class UnexpectedGeocoder:
    provider = "unexpected_geocoder"

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        raise AssertionError(f"unexpected geocoder call: {args}")


class RecordingGeocoder:
    provider = "fake_geocoder"

    def __init__(
        self,
        *,
        store: InMemoryPlaceStore,
        records: dict[tuple[str, str | None], PlaceRecord],
        persist: bool = True,
    ) -> None:
        self._store = store
        self._records = records
        self._persist = persist
        self.calls: list[GeocodePlaceInput] = []

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        record = self._records.get((args.query, args.city))
        if record is None:
            return GeocodePlaceOutput()

        if self._persist:
            await self._store.save(record)

        match = PlaceMatch(
            ref=record.ref,
            name=record.name,
            address=(
                f"Россия, {args.city}, {record.name}" if args.city is not None else record.address
            ),
        )
        return GeocodePlaceOutput(best=match, matches=[match])


class AmbiguousGeocoder:
    provider = "fake_geocoder"

    def __init__(self, store: InMemoryPlaceStore) -> None:
        self._store = store
        self.calls: list[GeocodePlaceInput] = []

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        self.calls.append(args)
        records = [
            _record(KREMLIN, args.query, 55.752023, 37.617499),
            _record(VDNH, args.query, 55.829813, 37.632778),
        ]
        await self._store.save_many(records)
        matches = [
            PlaceMatch(
                ref=record.ref,
                name=args.query,
                address=f"Россия, {args.city or f'город {index}'}, {args.query}, вариант {index}",
            )
            for index, record in enumerate(records, start=1)
        ]
        return GeocodePlaceOutput(
            best=matches[0],
            matches=matches,
            ambiguous=True,
        )


async def _service(
    *,
    records: list[PlaceRecord],
    route_payload: dict[str, Any] | None = None,
    matrix_payload: dict[str, Any] | None = None,
    route_error: ValueError | None = None,
    matrix_error: ValueError | None = None,
    geocoder: GeocoderService | None = None,
    store: InMemoryPlaceStore | None = None,
) -> tuple[YandexRoutingProvider, FakeYandexRoutingClient]:
    place_store = store or InMemoryPlaceStore()
    await place_store.save_many(records)
    fake_client = FakeYandexRoutingClient(
        route_payload=route_payload,
        matrix_payload=matrix_payload,
        route_error=route_error,
        matrix_error=matrix_error,
    )
    provider = YandexRoutingProvider(
        client=cast(YandexRoutingClient, fake_client),
        place_store=place_store,
    )
    return provider, fake_client


async def _route_with_resolved_input(
    provider: YandexRoutingProvider,
    args: RoutingInput,
    context: ToolExecutionContext,
    *,
    store: InMemoryPlaceStore,
    geocoder: GeocoderService,
) -> RoutingOutput:
    resolver = RoutingPlaceResolver(
        TextPlaceResolver(
            geocoded_place_resolver=GeocodedPlaceResolver(
                geocoder=geocoder,
                place_store=store,
            )
        ),
        store,
    )
    resolved_args = await resolver.resolve_to_refs(args, context)
    return await provider.route(resolved_args, context)


async def test_route_loads_refs_reorders_waypoints_and_collapses_steps() -> None:
    """Verify that route loads refs, reorders waypoints, and collapses steps."""

    payload: dict[str, Any] = {
        "traffic_type": "forecast",
        "route": {
            "legs": [
                {
                    "status": "OK",
                    "steps": [
                        {"length": 100.4, "duration": 60.2, "mode": "walking"},
                        {"length": 199.6, "duration": 119.8, "mode": "walking"},
                        {"length": 1_000.0, "duration": 300.0, "mode": "transit"},
                    ],
                },
                {
                    "status": "OK",
                    "steps": [
                        {"length": 500.0, "duration": 240.0, "mode": "transit"},
                    ],
                },
            ],
            "flags": {"hasTolls": False},
        },
        "optimization": {"waypoints_order": [0, 2, 1]},
    }
    provider, client = await _service(
        records=[
            _record(A, "Старт", 55.7, 37.5),
            _record(B, "Финиш", 55.8, 37.6),
            _record(C, "Пересадка", 55.9, 37.7),
        ],
        route_payload=payload,
    )

    result = await provider.route(
        RoutingInput(
            mode="route",
            transport="transit",
            waypoints=[A, B, C],
            optimize_waypoints=True,
        ),
        ToolExecutionContext(),
    )

    assert result.route is not None
    assert result.route.length_m == 1_800
    assert result.route.duration_s == 720
    assert result.route.waypoint_order == [0, 2, 1]
    assert [point.ref for point in result.route.waypoints] == [A, C, B]
    assert [segment.transport for segment in result.route.legs[0].segments] == [
        TransportMode.WALKING,
        TransportMode.TRANSIT,
    ]
    assert result.route.legs[0].segments[0].length_m == 300
    assert client.route_calls[0]["waypoints"] == [
        (55.7, 37.5),
        (55.8, 37.6),
        (55.9, 37.7),
    ]


async def test_first_route_request_resolves_text_points_before_provider() -> None:
    """Verify that the routing input resolver prepares text points for the provider."""

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(
        store=store,
        records={
            ("Московский Кремль", "Москва"): _record(
                KREMLIN,
                "Московский Кремль",
                55.752023,
                37.617499,
            ),
            ("ВДНХ", "Москва"): _record(
                VDNH,
                "ВДНХ",
                55.829813,
                37.632778,
            ),
        },
    )
    provider, client = await _service(
        records=[],
        store=store,
        geocoder=geocoder,
        route_payload={
            "route": {
                "legs": [
                    {
                        "status": "OK",
                        "steps": [{"length": 12_000.0, "duration": 1_800.0, "mode": "driving"}],
                    }
                ]
            }
        },
    )

    context = ToolExecutionContext()
    result = await _route_with_resolved_input(
        provider,
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [
                    {"query": "  Московский   Кремль ", "city": " Москва "},
                    {"query": "ВДНХ", "city": "Москва"},
                ],
            }
        ),
        context,
        store=store,
        geocoder=geocoder,
    )

    assert [(call.query, call.city, call.limit) for call in geocoder.calls] == [
        ("Московский Кремль", "Москва", 5),
        ("ВДНХ", "Москва", 5),
    ]
    assert client.route_calls[0]["waypoints"] == [
        (55.752023, 37.617499),
        (55.829813, 37.632778),
    ]
    assert result.route is not None
    assert [point.ref for point in result.route.waypoints] == [KREMLIN, VDNH]


async def test_route_geocodes_case_variants_only_once() -> None:
    """Verify that normalized duplicate text points share one geocoder call."""

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(
        store=store,
        records={
            ("ВДНХ", "Москва"): _record(
                VDNH,
                "ВДНХ",
                55.829813,
                37.632778,
            ),
        },
    )
    provider, client = await _service(
        records=[],
        store=store,
        geocoder=geocoder,
        route_payload={
            "route": {
                "legs": [
                    {
                        "status": "OK",
                        "steps": [],
                    }
                ]
            }
        },
    )

    context = ToolExecutionContext()
    result = await _route_with_resolved_input(
        provider,
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [
                    {"query": "ВДНХ", "city": "Москва"},
                    {"query": "вднх", "city": "МОСКВА"},
                ],
            }
        ),
        context,
        store=store,
        geocoder=geocoder,
    )

    assert [(call.query, call.city) for call in geocoder.calls] == [("ВДНХ", "Москва")]
    assert client.route_calls[0]["waypoints"] == [
        (55.829813, 37.632778),
        (55.829813, 37.632778),
    ]
    assert result.route is not None
    assert [point.ref for point in result.route.waypoints] == [VDNH, VDNH]


async def test_failed_route_leg_returns_not_found() -> None:
    """Verify that failed route leg returns not found."""

    provider, _ = await _service(
        records=[
            _record(A, "Старт", 55.7, 37.5),
            _record(B, "Финиш", 55.8, 37.6),
        ],
        route_payload={
            "route": {
                "legs": [
                    {
                        "status": "FAIL",
                        "steps": [],
                    }
                ]
            }
        },
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(mode="route", waypoints=[A, B]),
            ToolExecutionContext(),
        )

    error = exc_info.value
    assert error.error_code is ToolErrorCode.NOT_FOUND
    assert str(error) == (
        "The routing provider could not build a route between all requested points"
    )
    assert error.provider == "yandex_routing"
    assert error.failure_kind is None
    assert error.retryable is False


async def test_text_point_not_found_stops_before_routing_call() -> None:
    """Verify that text point not found stops before routing call."""

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(store=store, records={})
    provider, client = await _service(
        records=[_record(A, "Старт", 55.7, 37.5)],
        store=store,
        geocoder=geocoder,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await _route_with_resolved_input(
            provider,
            RoutingInput.model_validate(
                {
                    "mode": "route",
                    "waypoints": [A, {"query": "Несуществующее место", "city": "Москва"}],
                }
            ),
            ToolExecutionContext(),
            store=store,
            geocoder=geocoder,
        )

    assert exc_info.value.error_code is ToolErrorCode.NOT_FOUND
    assert "Несуществующее место" in str(exc_info.value)
    assert client.route_calls == []


async def test_ambiguous_text_point_uses_first_match_for_routing() -> None:
    """Verify routing sends the provider-ranked first text match to Route API."""

    store = InMemoryPlaceStore()
    geocoder = AmbiguousGeocoder(store)
    provider, client = await _service(
        records=[_record(B, "Финиш", 55.76, 37.63)],
        store=store,
        geocoder=geocoder,
        route_payload={"route": {"legs": [{"status": "OK", "steps": []}]}},
    )

    context = ToolExecutionContext()
    await _route_with_resolved_input(
        provider,
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [
                    {"query": "Центральный парк", "city": "Москва"},
                    B,
                ],
            }
        ),
        context,
        store=store,
        geocoder=geocoder,
    )

    assert geocoder.calls == [GeocodePlaceInput(query="Центральный парк", city="Москва", limit=5)]
    assert client.route_calls[0]["waypoints"][0] == (55.752023, 37.617499)
    assert context.warnings == (
        "Routing point 'Центральный парк' matched multiple places; using the provider-ranked "
        "first result 'Центральный парк'.",
    )


async def test_geocoder_must_persist_resolved_text_point() -> None:
    """Verify that geocoder must persist resolved text point."""

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(
        store=store,
        records={
            ("ВДНХ", "Москва"): _record(VDNH, "ВДНХ", 55.829813, 37.632778),
        },
        persist=False,
    )
    provider, client = await _service(
        records=[_record(A, "Старт", 55.7, 37.5)],
        store=store,
        geocoder=geocoder,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await _route_with_resolved_input(
            provider,
            RoutingInput.model_validate(
                {
                    "mode": "route",
                    "waypoints": [A, {"query": "ВДНХ", "city": "Москва"}],
                }
            ),
            ToolExecutionContext(),
            store=store,
            geocoder=geocoder,
        )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert exc_info.value.provider == "fake_geocoder"
    assert client.route_calls == []


def _matrix_payload() -> dict[str, Any]:
    return {
        "rows": [
            {
                "elements": [
                    {
                        "status": "OK",
                        "distance": {"value": 1_000},
                        "duration": {"value": 600},
                    },
                    {"status": "FAIL"},
                    {
                        "status": "OK",
                        "distance": {"value": 500},
                        "duration": {"value": 900},
                    },
                ]
            },
            {
                "elements": [
                    {
                        "status": "OK",
                        "distance": {"value": 1_200},
                        "duration": {"value": 700},
                    },
                    {
                        "status": "OK",
                        "distance": {"value": 300},
                        "duration": {"value": 400},
                    },
                    {
                        "status": "OK",
                        "distance": {"value": 600},
                        "duration": {"value": 800},
                    },
                ]
            },
        ]
    }


async def test_matrix_ranks_candidates_and_preserves_per_origin_costs() -> None:
    """Verify that matrix ranks candidates and preserves per origin costs."""

    records = [
        _record(A, "Дом", 55.7, 37.5),
        _record(B, "Офис", 55.8, 37.6),
        _record(C, "Кафе C", 55.9, 37.7),
        _record(D, "Кафе D", 56.0, 37.8),
        _record(E, "Кафе E", 56.1, 37.9),
    ]
    provider, client = await _service(records=records, matrix_payload=_matrix_payload())

    result = await provider.route(
        RoutingInput(
            mode="rank",
            transport="driving",
            origins=[A, B],
            candidates=[C, D, E],
            aggregate="sum",
            optimize_by="duration",
        ),
        ToolExecutionContext(),
    )

    assert [candidate.point.ref for candidate in result.ranked] == [C, E, D]
    assert [candidate.rank for candidate in result.ranked] == [1, 2, 3]
    assert result.ranked[0].duration_s == 1_300
    assert result.ranked[0].length_m == 2_200
    assert len(result.ranked[0].per_origin) == 2
    assert result.ranked[2].reachable is False
    assert result.unreachable_count == 1
    assert result.aggregate is Aggregate.SUM
    assert [point.ref for point in result.origins] == [A, B]
    assert client.matrix_calls[0]["origins"] == [(55.7, 37.5), (55.8, 37.6)]
    assert client.matrix_calls[0]["destinations"] == [
        (55.9, 37.7),
        (56.0, 37.8),
        (56.1, 37.9),
    ]


async def test_rank_mode_resolves_text_origin_before_provider() -> None:
    """Verify that the routing input resolver prepares a text origin."""

    store = InMemoryPlaceStore()
    geocoder = RecordingGeocoder(
        store=store,
        records={
            ("Московский Кремль", "Москва"): _record(
                KREMLIN,
                "Московский Кремль",
                55.752023,
                37.617499,
            ),
        },
    )
    provider, client = await _service(
        records=[_record(C, "Кафе", 55.76, 37.63)],
        store=store,
        geocoder=geocoder,
        matrix_payload={
            "rows": [
                {
                    "elements": [
                        {
                            "status": "OK",
                            "distance": {"value": 1_500},
                            "duration": {"value": 420},
                        }
                    ]
                }
            ]
        },
    )

    context = ToolExecutionContext()
    result = await _route_with_resolved_input(
        provider,
        RoutingInput.model_validate(
            {
                "mode": "rank",
                "origins": [{"query": "Московский Кремль", "city": "Москва"}],
                "candidates": [C],
            }
        ),
        context,
        store=store,
        geocoder=geocoder,
    )

    assert len(geocoder.calls) == 1
    assert client.matrix_calls[0]["origins"] == [(55.752023, 37.617499)]
    assert result.origins[0].ref == KREMLIN
    assert result.ranked[0].point.ref == C


async def test_min_aggregation_accepts_candidate_reachable_from_any_origin() -> None:
    """Verify that min aggregation accepts candidate reachable from any origin."""

    provider, _ = await _service(
        records=[
            _record(A, "Дом", 55.7, 37.5),
            _record(B, "Офис", 55.8, 37.6),
            _record(C, "Кафе C", 55.9, 37.7),
            _record(D, "Кафе D", 56.0, 37.8),
            _record(E, "Кафе E", 56.1, 37.9),
        ],
        matrix_payload=_matrix_payload(),
    )

    result = await provider.route(
        RoutingInput(
            mode="rank",
            origins=[A, B],
            candidates=[C, D, E],
            aggregate="min",
            optimize_by="duration",
        ),
        ToolExecutionContext(),
    )

    assert result.ranked[0].point.ref == D
    assert result.ranked[0].reachable is True
    assert result.ranked[0].duration_s == 400
    assert result.unreachable_count == 0


async def test_unknown_place_ref_stops_before_upstream_call() -> None:
    """Verify that unknown place ref stops before upstream call."""

    provider, client = await _service(
        records=[_record(A, "Старт", 55.7, 37.5)],
        route_payload={"route": {"legs": []}},
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(mode="route", waypoints=[A, B]),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UNKNOWN_REF
    assert client.route_calls == []


async def test_invalid_matrix_shape_is_an_upstream_schema_error() -> None:
    """Verify that invalid matrix shape is an upstream schema error."""

    provider, _ = await _service(
        records=[
            _record(A, "Дом", 55.7, 37.5),
            _record(C, "Кафе", 55.9, 37.7),
        ],
        matrix_payload={"rows": []},
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(mode="rank", origins=[A], candidates=[C]),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.provider == "yandex_routing"


@pytest.mark.parametrize("mode", ["route", "rank"])
async def test_invalid_yandex_response_is_a_typed_schema_error(mode: str) -> None:
    """Verify that invalid Yandex response is a typed schema error."""

    if mode == "route":
        provider, _ = await _service(
            records=[
                _record(A, "Старт", 55.7, 37.5),
                _record(B, "Финиш", 55.8, 37.6),
            ],
            route_payload={"route": {}},
        )
        args = RoutingInput(mode="route", waypoints=[A, B])
    else:
        provider, _ = await _service(
            records=[
                _record(A, "Старт", 55.7, 37.5),
                _record(B, "Кандидат", 55.8, 37.6),
            ],
            matrix_payload={"rows": "not-a-list"},
        )
        args = RoutingInput(mode="rank", origins=[A], candidates=[B])

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(args, ToolExecutionContext())

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.provider == "yandex_routing"
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ValidationError)


async def test_invalid_materialized_output_is_an_internal_contract_error() -> None:
    """Verify that invalid materialized output is an internal contract error."""

    provider, _ = await _service(
        records=[
            _record(A, "", 55.7, 37.5),
            _record(B, "Финиш", 55.8, 37.6),
        ],
        route_payload={
            "route": {
                "legs": [
                    {
                        "status": "OK",
                        "steps": [{"length": 100.0, "duration": 60.0, "mode": "driving"}],
                    }
                ]
            }
        },
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(mode="route", waypoints=[A, B]),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert exc_info.value.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert exc_info.value.provider == "yandex_routing"
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.parametrize("mode", ["route", "rank"])
async def test_client_value_error_is_an_internal_contract_error(mode: str) -> None:
    """Verify that client value error is an internal contract error."""

    client_error = ValueError("defensive client validation failed")

    if mode == "route":
        provider, _ = await _service(
            records=[
                _record(A, "Старт", 55.7, 37.5),
                _record(B, "Финиш", 55.8, 37.6),
            ],
            route_error=client_error,
        )
        args = RoutingInput(mode="route", waypoints=[A, B])
    else:
        provider, _ = await _service(
            records=[
                _record(A, "Старт", 55.7, 37.5),
                _record(B, "Кандидат", 55.8, 37.6),
            ],
            matrix_error=client_error,
        )
        args = RoutingInput(mode="rank", origins=[A], candidates=[B])

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(args, ToolExecutionContext())

    error = exc_info.value
    assert error.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert error.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert error.provider == "yandex_routing"
    assert error.retryable is False
    assert error.__cause__ is client_error
    assert "defensive client validation failed" not in str(error)


async def test_invalid_optimized_waypoint_order_is_rejected() -> None:
    """Verify that invalid optimized waypoint order is rejected."""

    provider, _ = await _service(
        records=[
            _record(A, "Старт", 55.7, 37.5),
            _record(B, "Финиш", 55.8, 37.6),
        ],
        route_payload={
            "route": {"legs": [{"status": "OK", "steps": []}]},
            "optimization": {"waypoints_order": [0, 0]},
        },
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await provider.route(
            RoutingInput(
                mode="route",
                transport="walking",
                waypoints=[A, B],
                optimize_waypoints=True,
            ),
            ToolExecutionContext(),
        )

    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
