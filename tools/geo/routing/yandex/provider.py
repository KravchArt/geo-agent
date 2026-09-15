"""Yandex implementation of route construction and distance-based ranking."""

from __future__ import annotations

from collections.abc import Sequence

from tools.base import ToolExecutionError
from tools.geo.place_store import PlaceStore
from tools.geo.routing.adapter import (
    MatrixCost,
    RouteNotFoundError,
    invalid_schema_error,
    map_routing_contract_errors,
    rank_matrix_candidates,
    validate_routing_response,
)
from tools.geo.routing.resolution import RoutingPlaceLoader
from tools.geo.routing.schemas import (
    RouteInfo,
    RouteLeg,
    RoutePoint,
    RouteSegment,
    RoutingInput,
    RoutingMode,
    RoutingOutput,
)
from tools.geo.routing.yandex.client import YandexRoutingClient
from tools.geo.routing.yandex.schemas import (
    YandexDistanceMatrixResponse,
    YandexMatrixElement,
    YandexRouteResponse,
    YandexRouteStep,
)
from tools.observability import ToolExecutionContext


class YandexRoutingProvider:
    """Load prepared place refs and adapt Yandex routing responses."""

    provider = "yandex"
    service_name = "Yandex"

    def __init__(
        self,
        *,
        client: YandexRoutingClient,
        place_store: PlaceStore,
    ) -> None:
        self._client = client
        self._places = RoutingPlaceLoader(place_store)

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
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
        resolved = await self._places.load_places(args.waypoints)
        payload = await self._client.build_route(
            waypoints=[(place.record.lat, place.record.lon) for place in resolved],
            transport=args.transport,
            use_traffic=args.use_traffic,
            avoid_tolls=args.avoid_tolls,
            departure_time=args.departure_time,
            optimize_waypoints=args.optimize_waypoints,
            context=context,
        )
        response = validate_routing_response(
            payload,
            response_model=YandexRouteResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )

        # Yandex returns legs in the optimized visit order. Reorder our stored
        # places the same way so refs, names and leg indexes describe one route.
        waypoint_order = self._waypoint_order(
            response=response,
            waypoint_count=len(args.waypoints),
            optimized=args.optimize_waypoints,
        )
        ordered_places = [resolved[index] for index in waypoint_order]

        if any(leg.status == "FAIL" for leg in response.route.legs):
            raise RouteNotFoundError(provider=self._client.provider)

        expected_leg_count = len(ordered_places) - 1
        if len(response.route.legs) != expected_leg_count:
            raise self._invalid_schema_error(
                "Yandex route response contains an unexpected number of legs"
            )

        legs: list[RouteLeg] = []
        for leg_index, provider_leg in enumerate(response.route.legs):
            segments = self._collapse_steps(provider_leg.steps)
            legs.append(
                RouteLeg(
                    from_index=leg_index,
                    to_index=leg_index + 1,
                    length_m=round(sum(step.length for step in provider_leg.steps)),
                    duration_s=round(sum(step.duration for step in provider_leg.steps)),
                    segments=segments,
                )
            )

        route = RouteInfo(
            length_m=sum(leg.length_m for leg in legs),
            duration_s=sum(leg.duration_s for leg in legs),
            legs=legs,
            waypoints=[
                RoutePoint(ref=place.ref, name=place.record.name) for place in ordered_places
            ],
            waypoint_order=waypoint_order,
            has_tolls=response.route.flags.has_tolls,
            traffic_type=response.traffic_type,
        )
        return RoutingOutput(
            mode=args.mode,
            transport=args.transport,
            route=route,
        )

    async def _rank_candidates(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        places = [*args.origins, *args.candidates]
        resolved = await self._places.load_places(places)

        # Loading preserves the input order, so this boundary remains valid
        # even when the same ref occurs in both groups.
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

    def _normalize_matrix_response(
        self,
        payload: dict[str, object],
        *,
        origin_count: int,
        candidate_count: int,
    ) -> list[list[MatrixCost | None]]:
        response = validate_routing_response(
            payload,
            response_model=YandexDistanceMatrixResponse,
            provider=self._client.provider,
            service_name=self.service_name,
        )
        if len(response.rows) != origin_count:
            raise self._invalid_schema_error(
                "Yandex distance matrix contains an unexpected number of rows"
            )
        if any(len(row.elements) != candidate_count for row in response.rows):
            raise self._invalid_schema_error(
                "Yandex distance matrix contains an unexpected number of elements"
            )
        return [[self._matrix_cost(element) for element in row.elements] for row in response.rows]

    def _invalid_schema_error(self, message: str) -> ToolExecutionError:
        return invalid_schema_error(message, provider=self._client.provider)

    def _waypoint_order(
        self,
        *,
        response: YandexRouteResponse,
        waypoint_count: int,
        optimized: bool,
    ) -> list[int]:
        natural_order = list(range(waypoint_count))
        if not optimized:
            return natural_order
        if response.optimization is None:
            raise self._invalid_schema_error(
                "Yandex route response omitted requested waypoint optimization"
            )

        # Accept the upstream order only if it is a complete permutation. This
        # prevents missing or repeated indexes from corrupting route attribution.
        order = response.optimization.waypoints_order
        if sorted(order) != natural_order:
            raise self._invalid_schema_error(
                "Yandex route response contains an invalid waypoint order"
            )
        return order

    @staticmethod
    def _collapse_steps(steps: Sequence[YandexRouteStep]) -> list[RouteSegment]:
        if not steps:
            return []

        segments: list[RouteSegment] = []
        current_mode = steps[0].mode
        current_length = 0.0
        current_duration = 0.0

        # The public contract needs transport-level segments, not every Yandex
        # navigation instruction, so merge only adjacent steps with the same mode.
        for step in steps:
            if step.mode is not current_mode:
                segments.append(
                    RouteSegment(
                        transport=current_mode,
                        length_m=round(current_length),
                        duration_s=round(current_duration),
                    )
                )
                current_mode = step.mode
                current_length = 0.0
                current_duration = 0.0

            current_length += step.length
            current_duration += step.duration

        segments.append(
            RouteSegment(
                transport=current_mode,
                length_m=round(current_length),
                duration_s=round(current_duration),
            )
        )
        return segments

    def _matrix_cost(self, element: YandexMatrixElement) -> MatrixCost | None:
        if element.status == "FAIL":
            return None
        if element.distance is None or element.duration is None:
            raise self._invalid_schema_error(
                "Yandex distance matrix contains a successful element without costs"
            )
        return MatrixCost(
            length_m=element.distance.value,
            duration_s=element.duration.value,
        )
