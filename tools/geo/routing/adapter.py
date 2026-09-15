"""Shared routing validation, normalization, and result materialization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.routing.schemas import (
    Aggregate,
    OptimizeBy,
    OriginCost,
    RankedCandidate,
    RouteInfo,
    RouteLeg,
    RoutePoint,
    RouteStep,
    RoutingInput,
    RoutingOutput,
    TrafficType,
    TransportMode,
)
from tools.observability import ToolExecutionContext
from tools.refs import ResolvedPlace

Coordinates = tuple[float, float]
ResponseT = TypeVar("ResponseT", bound=BaseModel)
NonNegativeNumber = int | float

_LEG_DISTANCE_TOLERANCE_M = 1.0
_LEG_DURATION_TOLERANCE_S = 1.0
_STEP_DISTANCE_MIN_TOLERANCE_M = 5.0
_STEP_DURATION_MIN_TOLERANCE_S = 5.0
_STEP_RELATIVE_TOLERANCE = 0.01


class RouteNotFoundError(ToolExecutionError):
    """The provider could not build every required leg of a route."""

    def __init__(self, *, provider: str | None = None) -> None:
        super().__init__(
            ToolErrorCode.NOT_FOUND,
            "The routing provider could not build a route between all requested points",
            provider=provider,
            retryable=False,
        )


@dataclass(frozen=True, slots=True)
class MatrixCost:
    """One normalized, reachable origin-to-candidate matrix element."""

    length_m: int
    duration_s: int


@dataclass(frozen=True, slots=True)
class NormalizedRoute:
    """Provider-neutral route data before place refs are attached."""

    legs: Sequence[RouteLeg]
    waypoint_order: Sequence[int]
    snap_distances_m: Sequence[int]


@dataclass(frozen=True, slots=True)
class _IndexedMatrixCost:
    origin_index: int
    length_m: int
    duration_s: int


@contextmanager
def map_routing_contract_errors(
    *,
    provider: str,
    service_name: str,
) -> Iterator[None]:
    """Map adapter/request construction failures to the public tool contract."""

    try:
        yield
    except ToolExecutionError:
        # Provider/client errors already satisfy the public tool contract.
        raise
    except ValidationError as exc:
        # At this point upstream payloads have already passed their own strict
        # schemas, so a new Pydantic error is a bug in our normalization code.
        raise internal_contract_error(
            f"{service_name} provider could not build a valid tool result",
            provider=provider,
        ) from exc
    except ValueError as exc:
        raise internal_contract_error(
            f"{service_name} provider produced an invalid upstream request",
            provider=provider,
        ) from exc


def validate_snap_warning_distance(distance_m: int) -> int:
    if distance_m < 1:
        raise ValueError("snap warning distance must be positive")
    return distance_m


def validate_static_capabilities(
    args: RoutingInput,
    *,
    supported_transports: frozenset[TransportMode],
    supports_waypoint_optimization: bool,
    provider: str,
    service_name: str,
) -> None:
    """Validate capabilities shared by static OSRM/GraphHopper adapters."""

    if args.transport not in supported_transports:
        supported = ", ".join(sorted(mode.value for mode in supported_transports))
        raise ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            f"{service_name} routing supports these transport modes: {supported}",
            provider=provider,
            retryable=False,
        )
    if args.optimize_waypoints and not supports_waypoint_optimization:
        raise ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            f"{service_name} route construction does not support waypoint optimization",
            provider=provider,
            retryable=False,
        )
    if args.departure_time is not None:
        raise ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            f"{service_name} routing does not support departure-time traffic forecasts",
            provider=provider,
            retryable=False,
        )


def record_static_traffic_warning(
    args: RoutingInput,
    *,
    service_name: str,
    context: ToolExecutionContext,
) -> None:
    if args.transport is TransportMode.DRIVING and args.use_traffic:
        context.add_warning(f"{service_name} route times do not include live traffic.")


def validate_routing_response(
    payload: object,
    *,
    response_model: type[ResponseT],
    provider: str,
    service_name: str,
) -> ResponseT:
    """Validate a provider payload with a strict provider-specific schema."""

    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise invalid_schema_error(
            f"{service_name} routing returned data in an unexpected format",
            provider=provider,
        ) from exc


def materialize_static_route(
    *,
    args: RoutingInput,
    resolved_places: Sequence[ResolvedPlace],
    route: NormalizedRoute,
    provider: str,
    service_name: str,
    snap_warning_distance_m: int,
    context: ToolExecutionContext,
) -> RoutingOutput:
    """Attach place refs and apply common static-route output invariants."""

    waypoint_count = len(resolved_places)
    natural_order = list(range(waypoint_count))
    if sorted(route.waypoint_order) != natural_order:
        raise invalid_schema_error(
            f"{service_name} route response contains an invalid waypoint order",
            provider=provider,
        )
    if len(route.snap_distances_m) != waypoint_count:
        raise invalid_schema_error(
            f"{service_name} route response contains an unexpected number of snapped waypoints",
            provider=provider,
        )
    if len(route.legs) != waypoint_count - 1:
        raise invalid_schema_error(
            f"{service_name} route response contains an unexpected number of legs",
            provider=provider,
        )
    if any(
        leg.from_index != index or leg.to_index != index + 1 for index, leg in enumerate(route.legs)
    ):
        raise invalid_schema_error(
            f"{service_name} route response contains invalid leg indexes",
            provider=provider,
        )

    for index, snap_distance_m in enumerate(route.snap_distances_m):
        if snap_distance_m <= snap_warning_distance_m:
            continue
        context.add_warning(
            f"{service_name} snapped waypoint {index + 1} by {snap_distance_m} m; "
            "verify that the resolved place matches the intended access point."
        )

    # waypoint_order contains original input indexes in travelled order.
    # snap_distances_m stays input-indexed, so optimized routes can safely
    # reorder displayed points without attaching a snap distance to the wrong ref.
    waypoint_order = list(route.waypoint_order)
    legs = list(route.legs)
    ordered_places = [resolved_places[index] for index in waypoint_order]
    return RoutingOutput(
        mode=args.mode,
        transport=args.transport,
        route=RouteInfo(
            length_m=sum(leg.length_m for leg in legs),
            duration_s=sum(leg.duration_s for leg in legs),
            legs=legs,
            waypoints=[
                RoutePoint(
                    ref=place.ref,
                    name=place.record.name,
                    snap_distance_m=route.snap_distances_m[original_index],
                )
                for original_index, place in zip(
                    waypoint_order,
                    ordered_places,
                    strict=True,
                )
            ],
            waypoint_order=waypoint_order,
            has_tolls=None,
            traffic_type=TrafficType.DISABLED,
        ),
    )


def validate_route_totals(
    *,
    route_distance_m: float,
    route_duration_s: float,
    leg_distances_m: Sequence[float],
    leg_durations_s: Sequence[float],
    step_distances_m: Sequence[float],
    step_durations_s: Sequence[float],
    provider: str,
    service_name: str,
    context: ToolExecutionContext,
) -> None:
    """Validate provider totals and warn for non-authoritative step rounding."""

    if abs(sum(leg_distances_m) - route_distance_m) > _LEG_DISTANCE_TOLERANCE_M:
        raise invalid_schema_error(
            f"{service_name} route distance does not match its leg details",
            provider=provider,
        )
    if abs(sum(leg_durations_s) - route_duration_s) > _LEG_DURATION_TOLERANCE_S:
        raise invalid_schema_error(
            f"{service_name} route duration does not match its leg details",
            provider=provider,
        )

    distance_tolerance = max(
        _STEP_DISTANCE_MIN_TOLERANCE_M,
        route_distance_m * _STEP_RELATIVE_TOLERANCE,
    )
    duration_tolerance = max(
        _STEP_DURATION_MIN_TOLERANCE_S,
        route_duration_s * _STEP_RELATIVE_TOLERANCE,
    )
    if (
        abs(sum(step_distances_m) - route_distance_m) > distance_tolerance
        or abs(sum(step_durations_s) - route_duration_s) > duration_tolerance
    ):
        context.add_warning(
            f"{service_name} navigation steps do not fully match the route totals; "
            "use the route distance and duration as authoritative."
        )


def normalize_matrix_values(
    *,
    distances: Sequence[Sequence[NonNegativeNumber | None]] | None,
    durations: Sequence[Sequence[NonNegativeNumber | None]] | None,
    origin_count: int,
    candidate_count: int,
    provider: str,
    service_name: str,
) -> list[list[MatrixCost | None]]:
    """Validate and normalize a provider distance matrix."""

    if distances is None or durations is None:
        raise invalid_schema_error(
            f"{service_name} matrix response omitted distances or durations",
            provider=provider,
        )
    if len(distances) != origin_count or len(durations) != origin_count:
        raise invalid_schema_error(
            f"{service_name} matrix response contains an unexpected number of rows",
            provider=provider,
        )
    if any(len(row) != candidate_count for row in distances) or any(
        len(row) != candidate_count for row in durations
    ):
        raise invalid_schema_error(
            f"{service_name} matrix response contains an unexpected number of elements",
            provider=provider,
        )

    matrix: list[list[MatrixCost | None]] = []
    for distance_row, duration_row in zip(distances, durations, strict=True):
        normalized_row: list[MatrixCost | None] = []
        for distance, duration in zip(distance_row, duration_row, strict=True):
            if distance is None and duration is None:
                normalized_row.append(None)
            elif distance is None or duration is None:
                raise invalid_schema_error(
                    f"{service_name} matrix response contains a partial cost element",
                    provider=provider,
                )
            else:
                normalized_row.append(
                    MatrixCost(
                        length_m=round(distance),
                        duration_s=round(duration),
                    )
                )
        matrix.append(normalized_row)
    return matrix


def append_route_step(
    steps: list[RouteStep],
    step: RouteStep,
    *,
    merge_with_previous: bool,
) -> None:
    """Append a normalized step, merging an identical continuation when allowed."""

    if (
        merge_with_previous
        and steps
        and steps[-1].instruction == step.instruction
        and steps[-1].street_name == step.street_name
    ):
        steps[-1].length_m += step.length_m
        steps[-1].duration_s += step.duration_s
        return
    steps.append(step)


def rank_matrix_candidates(
    *,
    args: RoutingInput,
    origin_places: Sequence[ResolvedPlace],
    candidate_places: Sequence[ResolvedPlace],
    matrix: Sequence[Sequence[MatrixCost | None]],
) -> RoutingOutput:
    """Apply the shared aggregation/ranking contract to a normalized matrix."""

    ranked: list[RankedCandidate] = []
    unreachable_count = 0

    for candidate_index, candidate_place in enumerate(candidate_places):
        per_origin: list[OriginCost] = []
        reachable_costs: list[_IndexedMatrixCost] = []

        for origin_index, row in enumerate(matrix):
            cost = row[candidate_index]
            if cost is None:
                per_origin.append(
                    OriginCost(
                        origin_index=origin_index,
                        reachable=False,
                    )
                )
                continue

            per_origin.append(
                OriginCost(
                    origin_index=origin_index,
                    length_m=cost.length_m,
                    duration_s=cost.duration_s,
                )
            )
            reachable_costs.append(
                _IndexedMatrixCost(
                    origin_index=origin_index,
                    length_m=cost.length_m,
                    duration_s=cost.duration_s,
                )
            )

        # "min" means reachable from any origin. Group aggregates are only
        # meaningful when every origin can reach the candidate.
        requires_all_origins = len(origin_places) > 1 and args.aggregate is not Aggregate.MIN
        reachable = bool(reachable_costs) and (
            not requires_all_origins or len(reachable_costs) == len(origin_places)
        )

        length_m: int | None = None
        duration_s: int | None = None
        score: float | None = None
        if reachable:
            length_m, duration_s, score = _aggregate_costs(
                reachable_costs,
                aggregate=args.aggregate,
                optimize_by=args.optimize_by,
            )
        else:
            unreachable_count += 1

        ranked.append(
            RankedCandidate(
                candidate_index=candidate_index,
                point=RoutePoint(
                    ref=candidate_place.ref,
                    name=candidate_place.record.name,
                ),
                rank=1,
                reachable=reachable,
                score=score,
                length_m=length_m,
                duration_s=duration_s,
                per_origin=per_origin,
            )
        )

    ranked.sort(
        key=lambda candidate: (
            not candidate.reachable,
            candidate.score if candidate.score is not None else float("inf"),
            candidate.candidate_index,
        )
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate.rank = rank

    return RoutingOutput(
        mode=args.mode,
        transport=args.transport,
        optimize_by=args.optimize_by,
        aggregate=args.aggregate,
        ranked=ranked[: args.limit],
        unreachable_count=unreachable_count,
        origins=[RoutePoint(ref=place.ref, name=place.record.name) for place in origin_places],
    )


def invalid_schema_error(message: str, *, provider: str) -> ToolExecutionError:
    return ToolExecutionError(
        ToolErrorCode.UPSTREAM_ERROR,
        message,
        provider=provider,
        failure_kind=ToolFailureKind.INVALID_SCHEMA,
        retryable=False,
    )


def internal_contract_error(message: str, *, provider: str) -> ToolExecutionError:
    return ToolExecutionError(
        ToolErrorCode.UPSTREAM_ERROR,
        message,
        provider=provider,
        failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
        retryable=False,
    )


def _aggregate_costs(
    costs: Sequence[_IndexedMatrixCost],
    *,
    aggregate: Aggregate,
    optimize_by: OptimizeBy,
) -> tuple[int, int, float]:
    def criterion(cost: _IndexedMatrixCost) -> int:
        if optimize_by is OptimizeBy.DISTANCE:
            return cost.length_m
        return cost.duration_s

    if len(costs) == 1 or aggregate in {Aggregate.MIN, Aggregate.MAX}:
        selector = min if aggregate is Aggregate.MIN or len(costs) == 1 else max
        selected = selector(costs, key=criterion)
        return selected.length_m, selected.duration_s, float(criterion(selected))

    total_length = sum(cost.length_m for cost in costs)
    total_duration = sum(cost.duration_s for cost in costs)

    if aggregate is Aggregate.AVG:
        total_length = round(total_length / len(costs))
        total_duration = round(total_duration / len(costs))

    score = total_length if optimize_by is OptimizeBy.DISTANCE else total_duration
    return total_length, total_duration, float(score)
