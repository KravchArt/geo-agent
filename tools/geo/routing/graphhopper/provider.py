"""GraphHopper response adapter for the shared routing provider pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import Any

from tools.base import ToolExecutionError
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.place_store import PlaceStore
from tools.geo.routing.adapter import (
    Coordinates,
    MatrixCost,
    NormalizedRoute,
    RouteNotFoundError,
    append_route_step,
    invalid_schema_error,
    map_routing_contract_errors,
    materialize_static_route,
    normalize_matrix_values,
    rank_matrix_candidates,
    record_static_traffic_warning,
    validate_route_totals,
    validate_routing_response,
    validate_snap_warning_distance,
    validate_static_capabilities,
)
from tools.geo.routing.graphhopper.client import GraphHopperRoutingClient
from tools.geo.routing.graphhopper.schemas import (
    GraphHopperInstruction,
    GraphHopperMatrixResponse,
    GraphHopperRoutePath,
    GraphHopperRouteResponse,
)
from tools.geo.routing.resolution import RoutingPlaceLoader
from tools.geo.routing.schemas import (
    RouteLeg,
    RouteSegment,
    RouteStep,
    RoutingInput,
    RoutingMode,
    RoutingOutput,
    TransportMode,
)
from tools.observability import ToolExecutionContext


@dataclass(frozen=True, slots=True)
class _NumericPathDetail:
    start: int
    end: int
    value: float


@dataclass(frozen=True, slots=True)
class _StreetPathDetail:
    start: int
    end: int
    street_name: str | None


@dataclass(frozen=True, slots=True)
class _PathCostSpan:
    start: int
    end: int
    distance_m: float
    duration_ms: float
    street_name: str | None


@dataclass(slots=True)
class _StreetCostGroup:
    street_name: str | None
    distance_m: float = 0.0
    duration_ms: float = 0.0


class GraphHopperRoutingProvider:
    """Adapt GraphHopper route/matrix responses to the shared routing contract."""

    provider = "graphhopper"
    service_name = "GraphHopper"
    supported_transports = frozenset(
        {
            TransportMode.DRIVING,
            TransportMode.WALKING,
            TransportMode.BICYCLE,
            TransportMode.SCOOTER,
        }
    )
    supports_waypoint_optimization = True

    def __init__(
        self,
        *,
        client: GraphHopperRoutingClient,
        place_store: PlaceStore,
        snap_warning_distance_m: int = 250,
    ) -> None:
        self._client = client
        self._places = RoutingPlaceLoader(place_store)
        self._snap_warning_distance_m = validate_snap_warning_distance(snap_warning_distance_m)

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        validate_static_capabilities(
            args,
            supported_transports=self.supported_transports,
            supports_waypoint_optimization=self.supports_waypoint_optimization,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        record_static_traffic_warning(
            args,
            service_name=self.service_name,
            context=context,
        )
        self._record_provider_warnings(args, context)

        with map_routing_contract_errors(
            provider=self._client.provider,
            service_name=self.service_name,
        ):
            if args.mode is RoutingMode.ROUTE:
                return await self._build_route(args, context)
            return await self._rank_candidates(args, context)

    async def _build_route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        resolved_places = await self._places.load_places(args.waypoints)
        coordinates = [(place.record.lat, place.record.lon) for place in resolved_places]
        payload = await self._client.build_route(
            waypoints=coordinates,
            transport=args.transport,
            avoid_tolls=args.avoid_tolls,
            optimize_waypoints=args.optimize_waypoints,
            context=context,
        )
        route = self._normalize_route_response(
            payload,
            resolved_coordinates=coordinates,
            transport=args.transport,
            optimized=args.optimize_waypoints,
            context=context,
        )
        return materialize_static_route(
            args=args,
            resolved_places=resolved_places,
            route=route,
            provider=self._client.provider,
            service_name=self.service_name,
            snap_warning_distance_m=self._snap_warning_distance_m,
            context=context,
        )

    async def _rank_candidates(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        # Load both groups together so repeated refs share one store lookup
        # while the slices preserve caller-provided ordering.
        resolved_places = await self._places.load_places([*args.origins, *args.candidates])
        origin_places = resolved_places[: len(args.origins)]
        candidate_places = resolved_places[len(args.origins) :]
        payload = await self._client.distance_matrix(
            origins=[(place.record.lat, place.record.lon) for place in origin_places],
            destinations=[(place.record.lat, place.record.lon) for place in candidate_places],
            transport=args.transport,
            avoid_tolls=args.avoid_tolls,
            context=context,
        )
        matrix = self._normalize_matrix_response(
            payload,
            origin_count=len(origin_places),
            candidate_count=len(candidate_places),
        )
        return rank_matrix_candidates(
            args=args,
            origin_places=origin_places,
            candidate_places=candidate_places,
            matrix=matrix,
        )

    @staticmethod
    def _record_provider_warnings(
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> None:
        if args.avoid_tolls:
            context.add_warning(
                "GraphHopper penalizes toll roads but cannot guarantee a toll-free route."
            )

    def _normalize_route_response(
        self,
        payload: Mapping[str, object],
        *,
        resolved_coordinates: Sequence[Coordinates],
        transport: TransportMode,
        optimized: bool,
        context: ToolExecutionContext,
    ) -> NormalizedRoute:
        response = validate_routing_response(
            payload,
            response_model=GraphHopperRouteResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        if not response.paths:
            raise RouteNotFoundError(provider=self._client.provider)

        path = response.paths[0]
        waypoint_order = self._waypoint_order(
            path=path,
            waypoint_count=len(resolved_coordinates),
            optimized=optimized,
        )
        # Unlike OSRM, GraphHopper encodes snapped coordinates in a polyline;
        # snap distances therefore have to be calculated locally.
        snapped_waypoints = self._decode_polyline(path.snapped_waypoints)
        if len(snapped_waypoints) != len(resolved_coordinates):
            raise invalid_schema_error(
                "GraphHopper route response contains an unexpected number of snapped waypoints",
                provider=self._client.provider,
            )
        snap_distances = self._snap_distances_by_input_order(
            resolved_coordinates=resolved_coordinates,
            snapped_waypoints=snapped_waypoints,
            waypoint_order=waypoint_order,
        )
        legs = self._route_legs(
            path=path,
            transport=transport,
            expected_count=len(resolved_coordinates) - 1,
        )

        distance_entries = path.details["leg_distance"]
        time_entries = path.details["leg_time"]
        validate_route_totals(
            route_distance_m=path.distance,
            route_duration_s=path.time / 1_000,
            leg_distances_m=[
                self._detail_value(entry, name="leg_distance") for entry in distance_entries
            ],
            leg_durations_s=[
                self._detail_value(entry, name="leg_time") / 1_000 for entry in time_entries
            ],
            step_distances_m=[instruction.distance for instruction in path.instructions],
            step_durations_s=[instruction.time / 1_000 for instruction in path.instructions],
            provider=self._client.provider,
            service_name=self.service_name,
            context=context,
        )
        return NormalizedRoute(
            legs=legs,
            waypoint_order=waypoint_order,
            snap_distances_m=snap_distances,
        )

    def _normalize_matrix_response(
        self,
        payload: Mapping[str, object],
        *,
        origin_count: int,
        candidate_count: int,
    ) -> list[list[MatrixCost | None]]:
        response = validate_routing_response(
            payload,
            response_model=GraphHopperMatrixResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        return normalize_matrix_values(
            distances=response.distances,
            # GraphHopper Matrix uses seconds, although Route uses milliseconds.
            durations=response.times,
            origin_count=origin_count,
            candidate_count=candidate_count,
            provider=self._client.provider,
            service_name=self.service_name,
        )

    def _waypoint_order(
        self,
        *,
        path: GraphHopperRoutePath,
        waypoint_count: int,
        optimized: bool,
    ) -> list[int]:
        natural_order = list(range(waypoint_count))
        if not optimized:
            return natural_order
        if path.points_order is None or sorted(path.points_order) != natural_order:
            raise self._invalid_schema_error(
                "GraphHopper route response contains an invalid optimized waypoint order"
            )
        return path.points_order

    def _route_legs(
        self,
        *,
        path: GraphHopperRoutePath,
        transport: TransportMode,
        expected_count: int,
    ) -> list[RouteLeg]:
        distance_entries = path.details.get("leg_distance")
        time_entries = path.details.get("leg_time")
        if (
            distance_entries is None
            or time_entries is None
            or len(distance_entries) != expected_count
            or len(time_entries) != expected_count
        ):
            raise self._invalid_schema_error(
                "GraphHopper route response omitted the expected leg details"
            )

        legs: list[RouteLeg] = []
        steps_by_leg = self._route_steps(path, expected_count=expected_count)
        for index, (distance_entry, time_entry) in enumerate(
            zip(distance_entries, time_entries, strict=True)
        ):
            length_m = round(self._detail_value(distance_entry, name="leg_distance"))
            duration_s = round(self._detail_value(time_entry, name="leg_time") / 1_000)
            legs.append(
                RouteLeg(
                    from_index=index,
                    to_index=index + 1,
                    length_m=length_m,
                    duration_s=duration_s,
                    segments=[
                        RouteSegment(
                            transport=transport,
                            length_m=length_m,
                            duration_s=duration_s,
                        )
                    ],
                    steps=steps_by_leg[index],
                )
            )
        return legs

    def _route_steps(
        self,
        path: GraphHopperRoutePath,
        *,
        expected_count: int,
    ) -> list[list[RouteStep]]:
        if not path.instructions:
            raise self._invalid_schema_error(
                "GraphHopper route response omitted navigation instructions"
            )
        final_instruction = path.instructions[-1]
        if (
            final_instruction.sign != 4
            or final_instruction.distance != 0
            or final_instruction.time != 0
        ):
            raise self._invalid_schema_error(
                "GraphHopper route response omitted the final arrival instruction"
            )

        steps_by_leg: list[list[RouteStep]] = [[] for _ in range(expected_count)]
        previous_sign_by_leg: list[int | None] = [None for _ in range(expected_count)]
        path_cost_spans = self._path_cost_spans(path)
        leg_index = 0
        for instruction in path.instructions:
            if leg_index >= expected_count:
                raise self._invalid_schema_error(
                    "GraphHopper returned navigation instructions for an unexpected route leg"
                )

            normalized_steps = self._instruction_steps(
                instruction,
                path_cost_spans=path_cost_spans,
            )
            for split_index, step in enumerate(normalized_steps):
                append_route_step(
                    steps_by_leg[leg_index],
                    step,
                    merge_with_previous=(
                        split_index == 0
                        and instruction.sign == 0
                        and previous_sign_by_leg[leg_index] == 0
                    ),
                )
            # Every additional path-detail group is a synthetic continuation.
            # Treat the final one as sign=0 so an identical continuation from
            # the next upstream instruction is merged instead of duplicated.
            previous_sign_by_leg[leg_index] = 0 if len(normalized_steps) > 1 else instruction.sign

            # GraphHopper sign 5 marks the end of an intermediate route leg.
            if instruction.sign == 5:
                leg_index += 1

        if leg_index != expected_count - 1:
            raise self._invalid_schema_error(
                "GraphHopper route response omitted an intermediate waypoint instruction"
            )
        return steps_by_leg

    def _instruction_steps(
        self,
        instruction: GraphHopperInstruction,
        *,
        path_cost_spans: Sequence[_PathCostSpan] | None,
    ) -> list[RouteStep]:
        """Restore named-road transitions omitted by GraphHopper instructions."""

        original_street = self._clean_street_name(instruction.street_name)
        original_step = RouteStep(
            instruction=instruction.text,
            street_name=original_street,
            length_m=round(instruction.distance),
            duration_s=round(instruction.time / 1_000),
        )
        if path_cost_spans is None or instruction.interval is None:
            return [original_step]

        start, end = instruction.interval
        if start < 0 or end < start:
            raise self._invalid_schema_error("GraphHopper instruction contains an invalid interval")
        if start == end:
            return [original_step]

        spans = [span for span in path_cost_spans if start <= span.start and span.end <= end]
        if (
            not spans
            or spans[0].start != start
            or spans[-1].end != end
            or any(left.end != right.start for left, right in pairwise(spans))
        ):
            raise self._invalid_schema_error(
                "GraphHopper path details do not cover a navigation instruction"
            )

        if abs(sum(span.distance_m for span in spans) - instruction.distance) > 1.0:
            raise self._invalid_schema_error(
                "GraphHopper distance details do not match a navigation instruction"
            )
        if abs(sum(span.duration_ms for span in spans) - instruction.time) > 1_000:
            raise self._invalid_schema_error(
                "GraphHopper time details do not match a navigation instruction"
            )

        effective_names = self._effective_street_names(
            spans,
            fallback=original_street,
        )
        groups: list[_StreetCostGroup] = []
        for span, street_name in zip(spans, effective_names, strict=True):
            if not groups or groups[-1].street_name != street_name:
                groups.append(_StreetCostGroup(street_name=street_name))
            groups[-1].distance_m += span.distance_m
            groups[-1].duration_ms += span.duration_ms

        distinct_named_streets = {
            group.street_name for group in groups if group.street_name is not None
        }
        if len(distinct_named_streets) <= 1:
            return [original_step]

        lengths = self._round_preserving_total(
            [group.distance_m for group in groups],
            total=original_step.length_m,
        )
        durations = self._round_preserving_total(
            [group.duration_ms / 1_000 for group in groups],
            total=original_step.duration_s,
        )
        result: list[RouteStep] = []
        for index, (group, length_m, duration_s) in enumerate(
            zip(groups, lengths, durations, strict=True)
        ):
            if index == 0:
                text = instruction.text
            elif group.street_name is not None:
                text = f"Continue onto {group.street_name}"
            else:
                text = "Continue"
            result.append(
                RouteStep(
                    instruction=text,
                    street_name=group.street_name,
                    length_m=length_m,
                    duration_s=duration_s,
                )
            )
        return result

    def _path_cost_spans(
        self,
        path: GraphHopperRoutePath,
    ) -> list[_PathCostSpan] | None:
        street_entries = path.details.get("street_name")
        distance_entries = path.details.get("distance")
        time_entries = path.details.get("time")
        if street_entries is None or distance_entries is None or time_entries is None:
            return None

        streets = [self._street_path_detail(entry, name="street_name") for entry in street_entries]
        distances = [
            self._numeric_path_detail(entry, name="distance") for entry in distance_entries
        ]
        times = [self._numeric_path_detail(entry, name="time") for entry in time_entries]
        if len(distances) != len(times) or any(
            (distance.start, distance.end) != (duration.start, duration.end)
            for distance, duration in zip(distances, times, strict=False)
        ):
            raise self._invalid_schema_error(
                "GraphHopper distance and time details contain different intervals"
            )

        spans: list[_PathCostSpan] = []
        street_index = 0
        for distance, duration in zip(distances, times, strict=True):
            while street_index < len(streets) and streets[street_index].end <= distance.start:
                street_index += 1
            if street_index >= len(streets):
                raise self._invalid_schema_error(
                    "GraphHopper street-name details do not cover route costs"
                )
            street = streets[street_index]
            if street.start > distance.start or street.end < distance.end:
                raise self._invalid_schema_error(
                    "GraphHopper street-name and cost details contain incompatible intervals"
                )
            spans.append(
                _PathCostSpan(
                    start=distance.start,
                    end=distance.end,
                    distance_m=distance.value,
                    duration_ms=duration.value,
                    street_name=street.street_name,
                )
            )

        if abs(sum(span.distance_m for span in spans) - path.distance) > 1.0:
            raise self._invalid_schema_error(
                "GraphHopper distance details do not match the route total"
            )
        if abs(sum(span.duration_ms for span in spans) - path.time) > 1_000:
            raise self._invalid_schema_error(
                "GraphHopper time details do not match the route total"
            )
        return spans

    def _numeric_path_detail(
        self,
        entry: Sequence[Any],
        *,
        name: str,
    ) -> _NumericPathDetail:
        start, end = self._detail_interval(entry, name=name)
        value = entry[2]
        if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
            raise self._invalid_schema_error(f"GraphHopper {name} detail contains an invalid value")
        return _NumericPathDetail(start=start, end=end, value=float(value))

    def _street_path_detail(
        self,
        entry: Sequence[Any],
        *,
        name: str,
    ) -> _StreetPathDetail:
        start, end = self._detail_interval(entry, name=name)
        value = entry[2]
        if value is not None and not isinstance(value, str):
            raise self._invalid_schema_error(f"GraphHopper {name} detail contains an invalid value")
        return _StreetPathDetail(
            start=start,
            end=end,
            street_name=self._clean_street_name(value),
        )

    def _detail_interval(self, entry: Sequence[Any], *, name: str) -> tuple[int, int]:
        if len(entry) != 3:
            raise self._invalid_schema_error(f"GraphHopper {name} detail has an unexpected shape")
        start, end = entry[:2]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise self._invalid_schema_error(
                f"GraphHopper {name} detail contains an invalid interval"
            )
        return start, end

    @staticmethod
    def _effective_street_names(
        spans: Sequence[_PathCostSpan],
        *,
        fallback: str | None,
    ) -> list[str | None]:
        names: list[str | None] = []
        current = fallback
        for span in spans:
            if span.street_name is not None:
                current = span.street_name
            names.append(current)

        next_name: str | None = None
        for index in range(len(names) - 1, -1, -1):
            if names[index] is not None:
                next_name = names[index]
            elif next_name is not None:
                names[index] = next_name
        return names

    def _round_preserving_total(self, values: Sequence[float], *, total: int) -> list[int]:
        rounded = [round(value) for value in values]
        difference = total - sum(rounded)
        if difference == 0:
            return rounded

        for index in range(len(rounded) - 1, -1, -1):
            if rounded[index] + difference >= 0:
                rounded[index] += difference
                return rounded
        raise self._invalid_schema_error(
            "GraphHopper path details cannot be rounded to the instruction total"
        )

    @staticmethod
    def _clean_street_name(value: str | None) -> str | None:
        return (value or "").strip() or None

    def _snap_distances_by_input_order(
        self,
        *,
        resolved_coordinates: Sequence[Coordinates],
        snapped_waypoints: Sequence[Coordinates],
        waypoint_order: Sequence[int],
    ) -> list[int]:
        """Handle APIs that return snapped points in input or optimized order."""

        input_order_distances = [
            self._coordinate_distance(resolved, snapped)
            for resolved, snapped in zip(
                resolved_coordinates,
                snapped_waypoints,
                strict=True,
            )
        ]
        if list(waypoint_order) == list(range(len(resolved_coordinates))):
            return input_order_distances

        optimized_order_distances = [
            self._coordinate_distance(
                resolved_coordinates[original_index],
                snapped_waypoints[travelled_index],
            )
            for travelled_index, original_index in enumerate(waypoint_order)
        ]
        if sum(input_order_distances) <= sum(optimized_order_distances):
            return input_order_distances

        distances_by_input = [0 for _ in resolved_coordinates]
        for original_index, snap_distance_m in zip(
            waypoint_order,
            optimized_order_distances,
            strict=True,
        ):
            distances_by_input[original_index] = snap_distance_m
        return distances_by_input

    @staticmethod
    def _coordinate_distance(
        resolved: Coordinates,
        snapped: Coordinates,
    ) -> int:
        return calculate_distance_m(
            from_lat=resolved[0],
            from_lon=resolved[1],
            to_lat=snapped[0],
            to_lon=snapped[1],
        )

    def _decode_polyline(self, encoded: str) -> list[Coordinates]:
        """Decode GraphHopper's two-dimensional polyline (precision 1e-5)."""

        coordinates: list[Coordinates] = []
        index = 0
        latitude = 0
        longitude = 0

        while index < len(encoded):
            latitude_delta, index = self._decode_polyline_value(encoded, index)
            longitude_delta, index = self._decode_polyline_value(encoded, index)
            latitude += latitude_delta
            longitude += longitude_delta
            decoded = (latitude / 100_000, longitude / 100_000)
            if not -90 <= decoded[0] <= 90 or not -180 <= decoded[1] <= 180:
                raise self._invalid_schema_error(
                    "GraphHopper returned invalid snapped waypoint coordinates"
                )
            coordinates.append(decoded)

        if not coordinates:
            raise self._invalid_schema_error(
                "GraphHopper returned empty snapped waypoint coordinates"
            )
        return coordinates

    def _decode_polyline_value(self, encoded: str, start: int) -> tuple[int, int]:
        result = 0
        shift = 0
        index = start

        while True:
            if index >= len(encoded) or shift > 30:
                raise self._invalid_schema_error(
                    "GraphHopper returned an invalid encoded snapped waypoint"
                )
            value = ord(encoded[index]) - 63
            if value < 0 or value > 63:
                raise self._invalid_schema_error(
                    "GraphHopper returned an invalid encoded snapped waypoint"
                )
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break

        decoded = ~(result >> 1) if result & 1 else result >> 1
        return decoded, index

    def _detail_value(self, entry: Sequence[Any], *, name: str) -> float:
        if len(entry) != 3:
            raise self._invalid_schema_error(f"GraphHopper {name} detail has an unexpected shape")
        value = entry[2]
        if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
            raise self._invalid_schema_error(f"GraphHopper {name} detail contains an invalid value")
        return float(value)

    def _invalid_schema_error(self, message: str) -> ToolExecutionError:
        return invalid_schema_error(message, provider=self._client.provider)
