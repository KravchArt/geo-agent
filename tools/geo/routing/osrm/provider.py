"""OSRM response adapter for the shared routing provider pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
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
from tools.geo.routing.osrm.client import OsrmRoutingClient
from tools.geo.routing.osrm.schemas import (
    OsrmRouteResponse,
    OsrmRouteStep,
    OsrmStepManeuver,
    OsrmTableResponse,
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


class OsrmRoutingProvider:
    """Adapt OSRM route/table responses to the shared routing contract."""

    provider = "osrm"
    service_name = "OSRM"
    supported_transports = frozenset(
        {
            TransportMode.DRIVING,
            TransportMode.WALKING,
        }
    )

    def __init__(
        self,
        *,
        client: OsrmRoutingClient,
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
            supports_waypoint_optimization=False,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        record_static_traffic_warning(
            args,
            service_name=self.service_name,
            context=context,
        )

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

    def _normalize_route_response(
        self,
        payload: Mapping[str, object],
        *,
        resolved_coordinates: Sequence[Coordinates],
        transport: TransportMode,
        optimized: bool,
        context: ToolExecutionContext,
    ) -> NormalizedRoute:
        if optimized:
            raise AssertionError("OSRM optimization must be rejected before the upstream call")

        response = validate_routing_response(
            payload,
            response_model=OsrmRouteResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        # OSRM reports routing failures in a successful HTTP response via code.
        self._check_response_code(response.code)
        if not response.routes:
            raise RouteNotFoundError(provider=self._client.provider)

        route = response.routes[0]
        validate_route_totals(
            route_distance_m=float(route.distance),
            route_duration_s=float(route.duration),
            leg_distances_m=[float(leg.distance) for leg in route.legs],
            leg_durations_s=[float(leg.duration) for leg in route.legs],
            step_distances_m=[float(step.distance) for leg in route.legs for step in leg.steps],
            step_durations_s=[float(step.duration) for leg in route.legs for step in leg.steps],
            provider=self._client.provider,
            service_name=self.service_name,
            context=context,
        )
        if any(not leg.steps for leg in route.legs):
            raise invalid_schema_error(
                "OSRM route response omitted navigation instructions",
                provider=self._client.provider,
            )
        if any(
            leg.steps[-1].maneuver.type != "arrive"
            or leg.steps[-1].distance != 0
            or leg.steps[-1].duration != 0
            for leg in route.legs
        ):
            raise invalid_schema_error(
                "OSRM route response omitted a final arrival instruction",
                provider=self._client.provider,
            )

        legs = [
            RouteLeg(
                from_index=index,
                to_index=index + 1,
                length_m=round(leg.distance),
                duration_s=round(leg.duration),
                segments=[
                    RouteSegment(
                        transport=transport,
                        length_m=round(leg.distance),
                        duration_s=round(leg.duration),
                    )
                ],
                steps=self._route_steps(leg.steps),
            )
            for index, leg in enumerate(route.legs)
        ]
        return NormalizedRoute(
            legs=legs,
            # OSRM Route preserves input order and reports snap distance
            # directly for every input waypoint.
            waypoint_order=list(range(len(resolved_coordinates))),
            snap_distances_m=[round(waypoint.distance) for waypoint in response.waypoints],
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
            response_model=OsrmTableResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        self._check_response_code(response.code)
        return normalize_matrix_values(
            distances=response.distances,
            # OSRM Table durations are already expressed in seconds.
            durations=response.durations,
            origin_count=origin_count,
            candidate_count=candidate_count,
            provider=self._client.provider,
            service_name=self.service_name,
        )

    def _check_response_code(self, code: str) -> None:
        if code == "Ok":
            return
        if code in {"NoRoute", "NoTable"}:
            raise RouteNotFoundError(provider=self._client.provider)
        raise ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "OSRM could not process the routing request",
            provider=self._client.provider,
            failure_kind=ToolFailureKind.INVALID_SCHEMA,
            retryable=False,
        )

    def _route_step(self, step: OsrmRouteStep) -> RouteStep:
        street_name = step.name.strip() or None
        # OSRM returns maneuver primitives rather than ready-to-display text,
        # so English instructions are assembled locally and deterministically.
        return RouteStep(
            instruction=self._instruction_text(step.maneuver, street_name=street_name),
            street_name=street_name,
            length_m=round(step.distance),
            duration_s=round(step.duration),
        )

    def _route_steps(self, steps: list[OsrmRouteStep]) -> list[RouteStep]:
        normalized: list[RouteStep] = []
        previous_type: str | None = None

        for source_step in steps:
            step = self._route_step(source_step)
            append_route_step(
                normalized,
                step,
                merge_with_previous=(
                    source_step.maneuver.type == "continue" and previous_type == "continue"
                ),
            )
            previous_type = source_step.maneuver.type

        return normalized

    @classmethod
    def _instruction_text(
        cls,
        maneuver: OsrmStepManeuver,
        *,
        street_name: str | None,
    ) -> str:
        maneuver_type = maneuver.type
        direction = cls._direction_text(maneuver.modifier)

        if maneuver_type == "depart":
            text = "Depart"
        elif maneuver_type == "arrive":
            text = "You have arrived at your destination"
        elif maneuver_type in {"roundabout", "rotary"}:
            text = "Enter the roundabout"
            if maneuver.exit is not None:
                text += f" and take exit {maneuver.exit}"
        elif maneuver_type == "roundabout turn":
            text = direction or "Continue around the roundabout"
        elif maneuver_type == "merge":
            side = cls._side_text(maneuver.modifier)
            text = f"Merge {side}" if side else "Merge"
        elif maneuver_type == "fork":
            side = cls._side_text(maneuver.modifier)
            text = f"Keep {side} at the fork" if side else "Continue at the fork"
        elif maneuver_type == "on ramp":
            side = cls._side_text(maneuver.modifier)
            text = f"Take the ramp on the {side}" if side else "Take the ramp"
        elif maneuver_type == "off ramp":
            side = cls._side_text(maneuver.modifier)
            text = f"Take the exit on the {side}" if side else "Take the exit"
        elif maneuver_type == "end of road":
            text = (
                f"At the end of the road, {direction}"
                if direction
                else "Continue at the end of the road"
            )
        elif direction:
            text = direction.capitalize()
        else:
            text = "Continue"

        if street_name and maneuver_type != "arrive":
            connector = (
                "on" if maneuver_type == "depart" or maneuver.modifier == "straight" else "onto"
            )
            text += f" {connector} {street_name}"
        return text

    @staticmethod
    def _direction_text(modifier: str | None) -> str | None:
        if modifier is None:
            return None
        return {
            "uturn": "make a U-turn",
            "sharp right": "turn sharply right",
            "right": "turn right",
            "slight right": "turn slightly right",
            "straight": "continue straight",
            "slight left": "turn slightly left",
            "left": "turn left",
            "sharp left": "turn sharply left",
        }.get(modifier)

    @staticmethod
    def _side_text(modifier: str | None) -> str | None:
        if modifier is None:
            return None
        if "right" in modifier:
            return "right"
        if "left" in modifier:
            return "left"
        if modifier == "straight":
            return "straight"
        if modifier == "uturn":
            return "for a U-turn"
        return None
