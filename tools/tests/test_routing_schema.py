"""routing_tool — schema-level tests (no tool, no API)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tools.geo.routing import (
    ROUTING_TOOL_SPEC,
    Aggregate,
    OptimizeBy,
    OriginCost,
    RankedCandidate,
    RouteInfo,
    RouteLeg,
    RoutePoint,
    RouteSegment,
    RoutingInput,
    RoutingMode,
    RoutingOutput,
    RoutingPlaceQuery,
    TrafficType,
    TransportMode,
)
from tools.geo.routing.schemas import MAX_TEXT_POINTS_PER_ROUTING_CALL, ROUTING_LLM_PARAMETERS

A = "plc_a1b2c3d4e5"
B = "plc_b2c3d4e5f6"
C = "plc_c3d4e5f6a7"


def _refs(n: int) -> list[str]:
    return [f"plc_{i:010x}" for i in range(n)]


def _text_points(n: int, *, city: str = "Москва") -> list[dict[str, str]]:
    return [{"query": f"Точка {index}", "area": city} for index in range(n)]


def test_json_schema_is_generated_for_backend_validation():
    """The complete Pydantic schema remains the authoritative validator."""

    schema = RoutingInput.model_json_schema()
    assert set(schema["required"]) == {"mode"}
    # Truck is out of scope: the model must not be able to ask for it.
    transports = schema["$defs"]["TransportMode"]["enum"]
    assert "truck" not in transports
    assert set(transports) == {"driving", "walking", "transit", "bicycle", "scooter"}
    assert "final destination" in schema["properties"]["optimize_waypoints"]["description"]


def test_json_schema_contains_routing_mode_constraints() -> None:
    """Verify that structured output exposes the route/rank field contract."""

    schema = RoutingInput.model_json_schema()
    rules = schema["allOf"]
    route_rule = next(
        rule["then"] for rule in rules if rule["if"]["properties"]["mode"] == {"const": "route"}
    )
    rank_rule = next(
        rule["then"] for rule in rules if rule["if"]["properties"]["mode"] == {"const": "rank"}
    )

    assert route_rule["required"] == ["waypoints"]
    assert route_rule["properties"]["waypoints"]["minItems"] == 2
    assert route_rule["properties"]["origins"]["maxItems"] == 0
    assert route_rule["properties"]["candidates"]["maxItems"] == 0
    assert route_rule["properties"]["optimize_by"]["const"] == "duration"
    assert route_rule["properties"]["aggregate"]["const"] == "min"
    assert route_rule["properties"]["limit"]["const"] == 5

    assert set(rank_rule["required"]) == {"origins", "candidates"}
    assert rank_rule["properties"]["origins"]["minItems"] == 1
    assert rank_rule["properties"]["candidates"]["minItems"] == 1
    assert rank_rule["properties"]["waypoints"]["maxItems"] == 0
    assert rank_rule["properties"]["optimize_waypoints"]["const"] is False


def test_tool_spec_points_to_routing_contract():
    """Verify that tool spec points to routing contract."""

    assert ROUTING_TOOL_SPEC.name == "routing_tool"
    assert ROUTING_TOOL_SPEC.input_model is RoutingInput
    assert ROUTING_TOOL_SPEC.output_model is RoutingOutput
    assert ROUTING_TOOL_SPEC.description
    assert "call routing_tool directly" in ROUTING_TOOL_SPEC.description.lower()
    assert "never call places_search first solely" in ROUTING_TOOL_SPEC.description.lower()
    assert "provider-ranked first address-bearing POI card" in ROUTING_TOOL_SPEC.description
    assert "meeting point" in ROUTING_TOOL_SPEC.description
    assert "materially equivalent parameters" not in ROUTING_TOOL_SPEC.description
    assert "transport_mode_accuracy" in ROUTING_TOOL_SPEC.eval_metrics
    assert "route_success_rate" in ROUTING_TOOL_SPEC.eval_metrics
    assert ROUTING_TOOL_SPEC.answer_fields == ("route", "ranked")
    assert ROUTING_TOOL_SPEC.output_exclude_none is True

    schema = ROUTING_TOOL_SPEC.input_model.model_json_schema()
    assert set(schema["required"]) == {"mode"}
    assert "transport" in schema["properties"]
    assert "waypoints" in schema["properties"]
    assert "origins" in schema["properties"]
    assert "candidates" in schema["properties"]


def test_llm_contract_is_compact_but_keeps_the_calling_convention():
    """The model gets concise instructions; Pydantic keeps strict validation."""

    assert ROUTING_TOOL_SPEC.llm_parameters == ROUTING_LLM_PARAMETERS
    assert set(ROUTING_LLM_PARAMETERS["properties"]) == {
        "mode",
        "transport",
        "waypoints",
        "optimize_waypoints",
        "origins",
        "candidates",
        "optimize_by",
        "aggregate",
        "limit",
        "use_traffic",
        "avoid_tolls",
    }
    assert ROUTING_LLM_PARAMETERS["required"] == ["mode"]
    assert "allOf" not in ROUTING_LLM_PARAMETERS
    assert "$defs" not in ROUTING_LLM_PARAMETERS
    assert ROUTING_LLM_PARAMETERS["properties"]["transport"]["enum"] == [
        transport.value for transport in TransportMode
    ]
    point_schema = ROUTING_LLM_PARAMETERS["properties"]["waypoints"]["items"]
    assert "anyOf" not in point_schema
    assert point_schema["oneOf"][0]["pattern"] == r"^plc_[0-9a-f]{10}$"
    assert "without query or area" in point_schema["oneOf"][0]["description"]
    point_object = point_schema["oneOf"][1]
    assert point_object["required"] == ["query", "area"]
    assert "only when no matching plc_ ref" in point_object["description"]
    assert "departure_time" not in ROUTING_LLM_PARAMETERS["properties"]
    assert "departure_time" in RoutingInput.model_json_schema()["properties"]
    assert (
        "When route waypoints are optimised"
        in (ROUTING_LLM_PARAMETERS["properties"]["optimize_waypoints"]["description"])
    )


# --- points are refs or structured queries, never coordinates ------------


def test_points_must_be_refs():
    """Verify that points must be refs."""

    RoutingInput.model_validate({"mode": "route", "waypoints": [A, B]})

    # A coordinate pair, a bare name, a truncated ref — all rejected by the schema,
    # so a mistyped point can never reach the API.
    for bad in ("55.7539,37.6208", "Красная площадь", "plc_a1b2c3d4e"):
        with pytest.raises(ValidationError):
            RoutingInput.model_validate({"mode": "route", "waypoints": [A, bad]})


def test_a_web_source_ref_is_not_a_place():
    """Verify that a web source ref is not a place."""

    with pytest.raises(ValidationError):
        RoutingInput.model_validate({"mode": "route", "waypoints": [A, "src_9f8e7d6c5b"]})


def test_route_accepts_structured_text_points_and_mixed_refs():
    """Verify that route accepts structured text points and mixed refs."""

    params = RoutingInput.model_validate(
        {
            "mode": "route",
            "waypoints": [
                A,
                {"query": "  Красная   площадь ", "area": " Москва "},
            ],
        }
    )

    assert params.waypoints[0] == A
    assert params.waypoints[1] == RoutingPlaceQuery(
        query="Красная площадь",
        area="Москва",
    )


def test_every_text_point_requires_a_separate_area() -> None:
    with pytest.raises(ValidationError, match="area"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [A, {"query": "Красная площадь, Москва"}],
            }
        )


def test_route_text_point_budget_ignores_refs() -> None:
    """Verify that refs do not consume the internal geocoder request budget."""

    RoutingInput.model_validate(
        {
            "mode": "route",
            "waypoints": [
                *_text_points(MAX_TEXT_POINTS_PER_ROUTING_CALL),
                *_refs(40),
            ],
        }
    )


def test_route_rejects_too_many_unique_text_points() -> None:
    """Verify that excessive route geocoding is rejected before provider execution."""

    with pytest.raises(ValidationError, match="at most 10 unique text points"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": _text_points(MAX_TEXT_POINTS_PER_ROUTING_CALL + 1),
            }
        )


def test_text_point_budget_deduplicates_case_variants() -> None:
    """Verify that case-only query differences consume one geocoder budget slot."""

    points = _text_points(MAX_TEXT_POINTS_PER_ROUTING_CALL)
    points.append({"query": "ТОЧКА 0", "area": "МОСКВА"})

    RoutingInput.model_validate(
        {
            "mode": "route",
            "waypoints": points,
        }
    )


def test_text_point_budget_keeps_same_query_in_different_cities_distinct() -> None:
    """Verify that identical names in different cities require separate resolution."""

    points = _text_points(MAX_TEXT_POINTS_PER_ROUTING_CALL - 1)
    points.extend(
        [
            {"query": "улица Ленина", "area": "Москва"},
            {"query": "улица Ленина", "area": "Омск"},
        ]
    )

    with pytest.raises(ValidationError, match="at most 10 unique text points"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": points,
            }
        )


def test_rank_accepts_text_origins_and_candidates():
    """Verify that rank accepts text origins and candidates."""

    params = RoutingInput.model_validate(
        {
            "mode": "rank",
            "origins": [{"query": "Кремль", "area": "Москва"}],
            "candidates": [B, {"query": "ВДНХ", "area": "Москва"}],
        }
    )

    assert isinstance(params.origins[0], RoutingPlaceQuery)
    assert isinstance(params.candidates[1], RoutingPlaceQuery)


def test_rank_applies_one_text_point_budget_across_both_sides() -> None:
    """Verify that rank shares one geocoder budget across origins and candidates."""

    with pytest.raises(ValidationError, match="at most 10 unique text points"):
        RoutingInput.model_validate(
            {
                "mode": "rank",
                "origins": [{"query": "Старт", "area": "Москва"}],
                "candidates": _text_points(MAX_TEXT_POINTS_PER_ROUTING_CALL),
            }
        )


def test_text_point_rejects_coordinates_and_unknown_fields():
    """Verify that text point rejects coordinates and unknown fields."""

    with pytest.raises(ValidationError, match="coordinates are not allowed"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [A, {"query": "55.7539, 37.6208", "area": "Москва"}],
            }
        )

    with pytest.raises(ValidationError):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "waypoints": [A, {"query": "ВДНХ", "area": "Москва", "lat": 55.8298}],
            }
        )


# --- mode=route -----------------------------------------------------------


def test_route_mode_requires_two_waypoints():
    """Verify that route mode requires two waypoints."""

    RoutingInput.model_validate({"mode": "route", "waypoints": [A, B]})
    with pytest.raises(ValidationError, match="at least 2 waypoints"):
        RoutingInput.model_validate({"mode": "route", "waypoints": [A]})


def test_route_mode_rejects_rank_fields():
    """Verify that route mode rejects rank fields."""

    with pytest.raises(ValidationError, match="not used in mode=route"):
        RoutingInput.model_validate({"mode": "route", "waypoints": [A, B], "candidates": [C]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimize_by", "distance"),
        ("aggregate", "max"),
        ("limit", 20),
    ],
)
def test_route_mode_rejects_non_default_rank_options(field: str, value: str | int) -> None:
    """Verify that meaningful ranking options cannot silently affect route mode."""

    with pytest.raises(ValidationError, match="used only in mode=rank"):
        RoutingInput.model_validate({"mode": "route", "waypoints": [A, B], field: value})


def test_waypoint_cap_depends_on_transport():
    """Verify that waypoint cap depends on transport."""

    # Driving allows 50 waypoints, everything else only 25.
    RoutingInput.model_validate({"mode": "route", "transport": "driving", "waypoints": _refs(50)})
    with pytest.raises(ValidationError, match="at most 25 waypoints"):
        RoutingInput.model_validate(
            {"mode": "route", "transport": "walking", "waypoints": _refs(26)}
        )


@pytest.mark.parametrize("transport", ["bicycle", "scooter"])
def test_waypoint_optimization_rejects_unsupported_transport(transport: str):
    """Verify that waypoint optimization rejects unsupported transport."""

    with pytest.raises(ValidationError, match="supported only"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "transport": transport,
                "waypoints": [A, B],
                "optimize_waypoints": True,
            }
        )


# --- mode=rank ------------------------------------------------------------


def test_rank_mode_requires_origins_and_candidates():
    """Verify that rank mode requires origins and candidates."""

    RoutingInput.model_validate({"mode": "rank", "origins": [A], "candidates": [B]})
    with pytest.raises(ValidationError, match="at least 1 candidate"):
        RoutingInput.model_validate({"mode": "rank", "origins": [A]})
    with pytest.raises(ValidationError, match="at least 1 origin"):
        RoutingInput.model_validate({"mode": "rank", "candidates": [B]})


def test_matrix_element_cap_is_enforced_before_the_call():
    """Verify that matrix element cap is enforced before the call."""

    # 1 matrix element = 1 BILLED request. 10x10 is exactly the 100-element cap;
    # 11x10 must be rejected here rather than by a 400 (and a bill).
    RoutingInput.model_validate({"mode": "rank", "origins": _refs(10), "candidates": _refs(10)})
    with pytest.raises(ValidationError, match="110 elements; maximum is 100"):
        RoutingInput.model_validate({"mode": "rank", "origins": _refs(11), "candidates": _refs(10)})


def test_rank_defaults_answer_nearest_object():
    """Verify that rank defaults answer nearest object."""

    params = RoutingInput.model_validate({"mode": "rank", "origins": [A], "candidates": [B]})
    assert params.optimize_by is OptimizeBy.DURATION
    assert params.aggregate is Aggregate.MIN
    assert params.transport is TransportMode.DRIVING


def test_rank_mode_rejects_waypoint_optimization():
    """Verify that rank mode rejects waypoint optimization."""

    with pytest.raises(ValidationError, match="not used in mode=rank"):
        RoutingInput.model_validate(
            {
                "mode": "rank",
                "origins": [A],
                "candidates": [B],
                "optimize_waypoints": True,
            }
        )


# --- common ---------------------------------------------------------------


def test_departure_time_must_be_aware_and_in_the_future():
    """Verify that departure time must be aware and in the future."""

    future = datetime.now(UTC) + timedelta(hours=1)
    RoutingInput.model_validate(
        {"mode": "route", "waypoints": [A, B], "departure_time": future.isoformat()}
    )
    with pytest.raises(ValidationError, match="must not be in the past"):
        RoutingInput.model_validate(
            {"mode": "route", "waypoints": [A, B], "departure_time": "2020-01-01T10:00:00Z"}
        )
    with pytest.raises(ValidationError, match="must include a timezone"):
        RoutingInput.model_validate(
            {"mode": "route", "waypoints": [A, B], "departure_time": "2999-01-01T10:00:00"}
        )


@pytest.mark.parametrize("transport", ["walking", "bicycle", "scooter"])
def test_departure_time_rejects_modes_where_yandex_ignores_it(transport: str):
    """Verify that departure time rejects modes where Yandex ignores it."""

    future = datetime.now(UTC) + timedelta(hours=1)
    with pytest.raises(ValidationError, match="departure_time is not used"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "transport": transport,
                "waypoints": [A, B],
                "departure_time": future.isoformat(),
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("use_traffic", False), ("avoid_tolls", True)],
)
def test_driving_only_options_are_rejected_for_other_transports(field: str, value: bool):
    """Verify that driving only options are rejected for other transports."""

    with pytest.raises(ValidationError, match="supported only for driving"):
        RoutingInput.model_validate(
            {
                "mode": "route",
                "transport": "walking",
                "waypoints": [A, B],
                field: value,
            }
        )


def test_unknown_field_is_rejected():
    """Verify that unknown field is rejected."""

    with pytest.raises(ValidationError):
        RoutingInput.model_validate({"mode": "route", "waypoints": [A, B], "avoid_unpaved": True})


# --- output ---------------------------------------------------------------


def test_route_output_round_trips_and_reads_as_segments():
    """Verify that route output round trips and reads as segments."""

    output = RoutingOutput(
        mode=RoutingMode.ROUTE,
        transport=TransportMode.TRANSIT,
        route=RouteInfo(
            length_m=12_400,
            duration_s=1_680,
            waypoints=[
                RoutePoint(
                    ref=A,
                    name="Красная площадь",
                    snap_distance_m=12,
                )
            ],
            waypoint_order=[0, 1],
            traffic_type=TrafficType.REALTIME,
            legs=[
                RouteLeg(
                    from_index=0,
                    to_index=1,
                    length_m=12_400,
                    duration_s=1_680,
                    segments=[
                        RouteSegment(transport=TransportMode.WALKING, length_m=400, duration_s=360),
                        RouteSegment(
                            transport=TransportMode.TRANSIT, length_m=11_600, duration_s=1_140
                        ),
                        RouteSegment(transport=TransportMode.WALKING, length_m=400, duration_s=180),
                    ],
                )
            ],
        ),
    )
    serialized = output.model_dump(mode="json", exclude_none=True)
    assert RoutingOutput.model_validate(serialized) == output
    assert serialized["route"]["waypoints"][0]["snap_distance_m"] == 12
    assert set(serialized) == {"mode", "transport", "route"}


def test_rank_output_keeps_the_per_origin_breakdown():
    """Verify that rank output keeps the per origin breakdown."""

    # The "convenient for everyone" answer must be able to justify itself.
    output = RoutingOutput(
        mode=RoutingMode.RANK,
        transport=TransportMode.TRANSIT,
        optimize_by=OptimizeBy.DURATION,
        aggregate=Aggregate.SUM,
        ranked=[
            RankedCandidate(
                candidate_index=2,
                point=RoutePoint(ref=C, name="Кофемания"),
                rank=1,
                score=1_500.0,
                duration_s=1_500,
                length_m=6_200,
                per_origin=[
                    OriginCost(origin_index=0, duration_s=840, length_m=3_100),
                    OriginCost(origin_index=1, duration_s=660, length_m=3_100),
                ],
            ),
            RankedCandidate(
                candidate_index=0,
                point=RoutePoint(ref=B, name="Далёкое место"),
                rank=2,
                reachable=False,
                per_origin=[OriginCost(origin_index=0, reachable=False)],
            ),
        ],
        unreachable_count=1,
    )
    serialized = output.model_dump(mode="json", exclude_none=True)
    restored = RoutingOutput.model_validate(serialized)
    assert restored == output
    assert "route" not in serialized
    # An unreachable candidate is DATA (ok=True), not a failed call.
    assert restored.ranked[1].reachable is False
    assert restored.ranked[1].score is None


def test_output_shape_must_match_mode():
    """Verify that output shape must match mode."""

    with pytest.raises(ValidationError, match="requires route"):
        RoutingOutput(mode="route", transport="driving")

    with pytest.raises(ValidationError, match="not used in mode=rank output"):
        RoutingOutput(
            mode="rank",
            transport="driving",
            route=RouteInfo(length_m=100, duration_s=60),
        )
