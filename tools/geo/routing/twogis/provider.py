"""2GIS response adapter for the shared routing provider pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.place_store import PlaceStore
from tools.geo.routing.adapter import (
    MatrixCost,
    RouteNotFoundError,
    invalid_schema_error,
    map_routing_contract_errors,
    rank_matrix_candidates,
    validate_route_totals,
    validate_routing_response,
)
from tools.geo.routing.resolution import RoutingPlaceLoader
from tools.geo.routing.schemas import (
    RouteInfo,
    RouteLeg,
    RoutePoint,
    RouteSegment,
    RouteStep,
    RoutingInput,
    RoutingMode,
    RoutingOutput,
    TrafficType,
    TransportMode,
)
from tools.geo.routing.twogis.client import TwoGisRoutingClient
from tools.geo.routing.twogis.schemas import (
    TwoGisDetailedManeuver,
    TwoGisDetailedRouteResponse,
    TwoGisDistanceMatrixResponse,
    TwoGisPublicTransportMovement,
    TwoGisPublicTransportResponse,
)
from tools.observability import ToolExecutionContext

_UNREACHABLE_STATUSES = frozenset(
    {
        "POINT_EXCLUDED",
        "ROUTE_NOT_FOUND",
        "ROUTE_DOES_NOT_EXISTS",
        "ATTRACT_FAIL",
        "PLATFORMS_NOT_FOUND",
    }
)


class TwoGisRoutingProvider:
    """Load prepared refs and adapt 2GIS route/matrix results."""

    provider = "twogis"
    service_name = "2GIS"

    def __init__(
        self,
        *,
        client: TwoGisRoutingClient,
        place_store: PlaceStore,
    ) -> None:
        self._client = client
        self._places = RoutingPlaceLoader(place_store)

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        self._validate_capabilities(args)
        if args.avoid_tolls:
            context.add_warning(
                "2GIS avoids toll roads when possible but may use one if no practical route exists."
            )

        with map_routing_contract_errors(
            provider=self._client.provider,
            service_name=self.service_name,
        ):
            if args.mode is RoutingMode.ROUTE:
                return await self._build_route(args, context)
            return await self._rank_candidates(args, context)

    async def _rank_candidates(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        resolved = await self._places.load_places([*args.origins, *args.candidates])
        origin_places = resolved[: len(args.origins)]
        candidate_places = resolved[len(args.origins) :]
        payload = await self._client.distance_matrix(
            origins=[(place.record.lat, place.record.lon) for place in origin_places],
            destinations=[(place.record.lat, place.record.lon) for place in candidate_places],
            transport=args.transport,
            use_traffic=args.use_traffic,
            avoid_tolls=args.avoid_tolls,
            departure_time=args.departure_time,
            context=context,
        )
        matrix = self._matrix(
            payload,
            origin_count=len(origin_places),
            candidate_count=len(candidate_places),
        )
        if not any(cost is not None for row in matrix for cost in row):
            raise RouteNotFoundError(provider=self._client.provider)
        return rank_matrix_candidates(
            args=args,
            origin_places=origin_places,
            candidate_places=candidate_places,
            matrix=matrix,
        )

    def _routing_legs(
        self,
        payload: Mapping[str, object],
        *,
        expected_count: int,
        transport: TransportMode,
        context: ToolExecutionContext,
    ) -> list[RouteLeg]:
        raw_legs = payload.get("legs")
        if not isinstance(raw_legs, list) or len(raw_legs) != expected_count:
            raise self._invalid_schema_error(
                "2GIS routing returned an unexpected number of route legs"
            )

        legs: list[RouteLeg] = []
        for index, raw_leg in enumerate(raw_legs):
            response = validate_routing_response(
                raw_leg if isinstance(raw_leg, Mapping) else {},
                response_model=TwoGisDetailedRouteResponse,
                provider=self._client.provider,
                service_name=self.service_name,
            )
            self._raise_for_route_status(response.status)
            if response.type != "result" or not response.result:
                raise self._invalid_schema_error("2GIS detailed route response omitted a route")
            route = response.result[0]
            steps = self._route_steps(route.maneuvers)
            legs.append(
                RouteLeg(
                    from_index=index,
                    to_index=index + 1,
                    length_m=route.total_distance,
                    duration_s=route.total_duration,
                    segments=[
                        RouteSegment(
                            transport=transport,
                            length_m=route.total_distance,
                            duration_s=route.total_duration,
                        )
                    ],
                    steps=steps,
                )
            )
        validate_route_totals(
            route_distance_m=sum(leg.length_m for leg in legs),
            route_duration_s=sum(leg.duration_s for leg in legs),
            leg_distances_m=[leg.length_m for leg in legs],
            leg_durations_s=[leg.duration_s for leg in legs],
            step_distances_m=[step.length_m for leg in legs for step in leg.steps],
            step_durations_s=[step.duration_s for leg in legs for step in leg.steps],
            provider=self._client.provider,
            service_name=self.service_name,
            context=context,
        )
        return legs

    def _route_steps(self, maneuvers: Sequence[TwoGisDetailedManeuver]) -> list[RouteStep]:
        if not maneuvers or maneuvers[-1].type != "end":
            raise self._invalid_schema_error(
                "2GIS detailed route response omitted navigation instructions"
            )

        steps: list[RouteStep] = []
        for maneuver in maneuvers:
            path = maneuver.outcoming_path
            instruction = maneuver.comment.strip()
            if maneuver.type in {"begin", "end"} and maneuver.outcoming_path_comment:
                instruction = maneuver.outcoming_path_comment.strip() or instruction
            names = path.names if path is not None else []
            steps.append(
                RouteStep(
                    instruction=instruction,
                    street_name=next((name for name in names if name.strip()), None),
                    length_m=path.distance if path is not None else 0,
                    duration_s=path.duration if path is not None else 0,
                )
            )
        return steps

    def _public_transport_legs(
        self,
        payload: Mapping[str, object],
    ) -> list[RouteLeg]:
        raw_legs = payload.get("legs")
        if not isinstance(raw_legs, list):
            raise self._invalid_schema_error("2GIS public transport response omitted route legs")

        legs: list[RouteLeg] = []
        for index, alternatives in enumerate(raw_legs):
            response = validate_routing_response(
                alternatives if isinstance(alternatives, list) else [],
                response_model=TwoGisPublicTransportResponse,
                provider=self._client.provider,
                service_name=self.service_name,
            )
            if not response.root:
                raise RouteNotFoundError(provider=self._client.provider)
            route = response.root[0]
            segments = self._transit_segments(route.movements)
            if (
                sum(segment.length_m for segment in segments) != route.total_distance
                or sum(segment.duration_s for segment in segments) != route.total_duration
            ):
                raise self._invalid_schema_error(
                    "2GIS public transport movements do not match the route totals"
                )
            legs.append(
                RouteLeg(
                    from_index=index,
                    to_index=index + 1,
                    length_m=route.total_distance,
                    duration_s=route.total_duration,
                    segments=segments,
                )
            )
        return legs

    def _matrix(
        self,
        payload: Mapping[str, object],
        *,
        origin_count: int,
        candidate_count: int,
    ) -> list[list[MatrixCost | None]]:
        response = validate_routing_response(
            payload,
            response_model=TwoGisDistanceMatrixResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        if response.routes is None:
            raise RouteNotFoundError(provider=self._client.provider)

        matrix: list[list[MatrixCost | None]] = [
            [None for _ in range(candidate_count)] for _ in range(origin_count)
        ]
        seen: set[tuple[int, int]] = set()
        for route in response.routes:
            cell = (route.source_id, route.target_id)
            if (
                route.source_id >= origin_count
                or route.target_id >= candidate_count
                or cell in seen
            ):
                raise self._invalid_schema_error(
                    "2GIS distance matrix contains invalid or duplicate point indexes"
                )
            seen.add(cell)
            if route.status == "OK":
                matrix[route.source_id][route.target_id] = MatrixCost(
                    length_m=route.distance,
                    duration_s=route.duration,
                )
            elif route.status == "FAIL":
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    "2GIS distance matrix failed to calculate a route",
                    provider=self._client.provider,
                    provider_code=route.status,
                    failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                    retryable=True,
                )
            elif route.status not in _UNREACHABLE_STATUSES:
                raise self._invalid_schema_error(
                    "2GIS distance matrix returned an unknown route status"
                )

        if len(seen) != origin_count * candidate_count:
            raise self._invalid_schema_error(
                "2GIS distance matrix omitted one or more requested routes"
            )
        return matrix

    @staticmethod
    def _transit_segments(
        movements: Sequence[TwoGisPublicTransportMovement],
    ) -> list[RouteSegment]:
        segments: list[RouteSegment] = []
        for movement in movements:
            transport = (
                TransportMode.TRANSIT if movement.type == "passage" else TransportMode.WALKING
            )
            duration = movement.moving_duration + movement.waiting_duration
            if segments and segments[-1].transport is transport:
                segments[-1].length_m += movement.distance
                segments[-1].duration_s += duration
            else:
                segments.append(
                    RouteSegment(
                        transport=transport,
                        length_m=movement.distance,
                        duration_s=duration,
                    )
                )
        return segments

    def _raise_for_route_status(self, status: str) -> None:
        if status == "OK":
            return
        if status in _UNREACHABLE_STATUSES:
            raise RouteNotFoundError(provider=self._client.provider)
        if status == "FAIL":
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "2GIS failed to calculate a route",
                provider=self._client.provider,
                provider_code=status,
                failure_kind=ToolFailureKind.PROVIDER_RESPONSE,
                retryable=True,
            )
        raise self._invalid_schema_error("2GIS routing returned an unknown route status")

    def _validate_capabilities(self, args: RoutingInput) -> None:
        if args.optimize_waypoints:
            raise ToolExecutionError(
                ToolErrorCode.UNSUPPORTED_FILTER,
                "2GIS routing does not support waypoint optimization",
                provider=self._client.provider,
                retryable=False,
            )
        if args.transport is TransportMode.DRIVING and not args.use_traffic:
            raise ToolExecutionError(
                ToolErrorCode.UNSUPPORTED_FILTER,
                "2GIS routing cannot calculate fastest driving time with traffic disabled",
                provider=self._client.provider,
                retryable=False,
            )

    @staticmethod
    def _traffic_type(args: RoutingInput) -> TrafficType:
        if args.transport is not TransportMode.DRIVING:
            return TrafficType.DISABLED
        return TrafficType.FORECAST if args.departure_time is not None else TrafficType.REALTIME

    def _invalid_schema_error(self, message: str) -> ToolExecutionError:
        return invalid_schema_error(message, provider=self._client.provider)

    async def _build_route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        resolved = await self._places.load_places(args.waypoints)
        payload = await self._client.build_route(
            waypoints=[(place.record.lat, place.record.lon) for place in resolved],
            transport=args.transport,
            use_traffic=args.use_traffic,
            avoid_tolls=args.avoid_tolls,
            departure_time=args.departure_time,
            context=context,
        )
        if args.transport is TransportMode.TRANSIT:
            legs = self._public_transport_legs(payload)
        else:
            legs = self._routing_legs(
                payload,
                expected_count=len(resolved) - 1,
                transport=args.transport,
                context=context,
            )
        return RoutingOutput(
            mode=args.mode,
            transport=args.transport,
            route=RouteInfo(
                length_m=sum(leg.length_m for leg in legs),
                duration_s=sum(leg.duration_s for leg in legs),
                legs=legs,
                waypoints=[RoutePoint(ref=place.ref, name=place.record.name) for place in resolved],
                waypoint_order=list(range(len(resolved))),
                has_tolls=None,
                traffic_type=self._traffic_type(args),
            ),
        )
