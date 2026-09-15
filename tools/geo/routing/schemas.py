"""Model-facing schemas for ``routing_tool``.

Two modes, chosen by the model via ``mode``:

* ``route`` — build an actual route through 2..N waypoints and report distance,
  travel time and the leg/segment breakdown.
* ``rank`` — score candidate places against one or more reference points and
  rank them by distance or travel time.

It is deliberately one provider-neutral tool with two modes. 2GIS, Yandex,
GraphHopper, and OSRM adapters implement this contract:

* ``route`` uses 2GIS Routing, Yandex Route Details, GraphHopper Routing, or OSRM Route.
* ``rank`` uses 2GIS Distance Matrix, Yandex Distance Matrix, GraphHopper Matrix,
  or OSRM Table.

Every point is either an existing ``plc_...`` ref or ``{"query": ..., "area":
...}``. ``area`` is mandatory for every text point and accepts either a locality
name or a bounded-locality ``plc_...`` ref. Text queries pass through the shared
configured place resolver.
Resolved points are persisted as refs before provider attempts. Unknown or
expired refs fail with ``UNKNOWN_REF``.

Providers use different coordinate orders: Yandex parameters use
``latitude,longitude`` while GraphHopper POST bodies use GeoJSON
``[longitude, latitude]``. The adapters perform this conversion; the model
never passes coordinates.

Provider-independent post-processing:

* ``optimize_by`` and ``aggregate`` are applied locally to raw matrix costs.
* unreachable matrix pairs remain successful data with ``reachable=False``.
* a required route that cannot be completed returns ``NOT_FOUND``.
* distances are normalized to metres and durations to seconds.

GraphHopper's default OSM profiles and OSRM are static, so their route outputs
report ``traffic_type=disabled`` and emit a warning when live traffic was
requested. They reject public transit and departure-time forecasting
explicitly. Yandex can provide those capabilities.

Route geometry is deliberately absent from the model result. Turn-by-turn steps
are normalized to English, while provider adapters may merge only adjacent
duplicate "continue" instructions to avoid wasting model context. API keys,
provider URLs, profiles and authentication are configuration, never model input.

The shared matrix cap stays at 100 elements: this is Yandex's hard API cap and a
conservative billing guard. GraphHopper additionally applies plan-specific
location and credit limits.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from tools.base import ToolSpec
from tools.refs import PLACE_REF_PATTERN, PlaceRef

#: Distance Matrix: origins x destinations must not exceed this. 1 element = 1 request.
MAX_MATRIX_ELEMENTS = 100
#: Route API: waypoint cap. Driving allows 50, every other transport 25.
MAX_WAYPOINTS_DRIVING = 50
MAX_WAYPOINTS_OTHER = 25
#: Product-side quota guard: refs are free, but every unique text point may
#: consume one internal geocoder request before routing provider execution.
MAX_TEXT_POINTS_PER_ROUTING_CALL = 10
_COORDINATE_QUERY_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*,\s*"
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
)


# ---------------------------------------------------------------------------
# INPUT — what the MODEL fills in
# ---------------------------------------------------------------------------


class RoutingMode(StrEnum):
    ROUTE = "route"
    RANK = "rank"


class TransportMode(StrEnum):
    DRIVING = "driving"
    WALKING = "walking"
    TRANSIT = "transit"
    BICYCLE = "bicycle"
    SCOOTER = "scooter"


class OptimizeBy(StrEnum):
    """The criterion the ranking is done by. Also picks the unit of ``score``."""

    DURATION = "duration"
    DISTANCE = "distance"


class Aggregate(StrEnum):
    """How to collapse several reference points into ONE score per candidate.

    Only meaningful when there is more than one origin:

    * ``min`` — closest to ANY of them (default: "nearest object").
    * ``sum`` / ``avg`` — convenient for EVERYONE (the meeting-point case).
    * ``max`` — fairest worst case: minimise the longest trip anybody makes.
    """

    MIN = "min"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"


class RoutingPlaceQuery(BaseModel):
    """A free-text point resolved before routing provider attempts."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Address, geographic toponym, named organisation, or POI without the city; pass the "
            "locality separately in area"
        ),
    )
    area: str = Field(
        min_length=1,
        max_length=120,
        description=("Required locality scope: its name or a plc_ ref for a bounded locality"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_city(cls, value: object) -> object:
        """Accept stored/internal legacy payloads without exposing city to the model."""

        if not isinstance(value, dict) or "city" not in value:
            return value
        migrated = dict(value)
        city = migrated.pop("city")
        if "area" in migrated and migrated["area"] != city:
            raise ValueError("area and legacy city must match")
        migrated.setdefault("area", city)
        return migrated

    @field_validator("query", "area")
    @classmethod
    def _clean(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("query")
    @classmethod
    def _reject_coordinates(cls, value: str) -> str:
        if _COORDINATE_QUERY_PATTERN.fullmatch(value):
            raise ValueError("coordinates are not allowed; use a place name or address")
        return value

    @property
    def resolution_key(self) -> tuple[str, str]:
        """Return the identity shared by quota counting and provider deduplication."""

        return (
            self.query.casefold(),
            self.area.casefold(),
        )


RoutingPlace: TypeAlias = PlaceRef | RoutingPlaceQuery


class RoutingInput(BaseModel):
    """Filled in by the MODEL. This class *is* the tool's input schema."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"mode": {"const": "route"}},
                        "required": ["mode"],
                    },
                    "then": {
                        "required": ["waypoints"],
                        "properties": {
                            "waypoints": {"minItems": 2},
                            "origins": {"maxItems": 0},
                            "candidates": {"maxItems": 0},
                            # Some structured-output backends materialize
                            # defaults, so accept only the harmless rank defaults.
                            "optimize_by": {"const": "duration"},
                            "aggregate": {"const": "min"},
                            "limit": {"const": 5},
                        },
                    },
                },
                {
                    "if": {
                        "properties": {"mode": {"const": "rank"}},
                        "required": ["mode"],
                    },
                    "then": {
                        "required": ["origins", "candidates"],
                        "properties": {
                            "origins": {"minItems": 1},
                            "candidates": {"minItems": 1},
                            "waypoints": {"maxItems": 0},
                            "optimize_waypoints": {"const": False},
                        },
                    },
                },
            ],
        },
    )

    mode: RoutingMode = Field(
        description=(
            "route builds a route through ordered waypoints and returns measured distance/time; "
            "rank compares candidates by route cost from one or more origins and returns them "
            "best-first"
        ),
    )
    transport: TransportMode = Field(
        default=TransportMode.DRIVING,
        description=(
            "Transport mode: driving, walking, transit, bicycle, or scooter. When the user does "
            "not specify one, use the driving default without asking a clarifying question"
        ),
    )

    # --- mode=route ---
    waypoints: list[RoutingPlace] = Field(
        default_factory=list,
        description=(
            "Ordered route points: start, intermediate stops, and destination. Each point is "
            "either a stored plc_ ref or {query, area}. Always use a matching stored ref when one "
            "is available; use {query, area} only for a point without a ref. A ref is passed by "
            "itself and has no area. area is required for every text point, accepts a locality "
            "name or bounded-locality ref, and must be separate from query. Required in "
            f"mode=route with at least 2 points. At most {MAX_TEXT_POINTS_PER_ROUTING_CALL} "
            "unique text points are allowed per call; use plc_ refs for the rest. "
            "Coordinates are not allowed"
        ),
    )
    optimize_waypoints: bool = Field(
        default=False,
        description=(
            "Allow the provider to reorder every waypoint after the fixed start, including the "
            "point that would otherwise be the final destination. Use only when the user permits "
            "a flexible visit order. Supported for driving, walking, and transit in mode=route"
        ),
    )

    # --- mode=rank ---
    origins: list[RoutingPlace] = Field(
        default_factory=list,
        description=(
            "Origins from which candidates are compared: plc_ refs or {query, area}. "
            f"Required in mode=rank with at least 1 point. Origins and candidates share a "
            f"limit of {MAX_TEXT_POINTS_PER_ROUTING_CALL} unique text points"
        ),
    )
    candidates: list[RoutingPlace] = Field(
        default_factory=list,
        description=(
            "Candidate places to compare: plc_ refs or {query, area}. Required in mode=rank "
            "with at least 1 point. Use plc_ refs for large candidate lists"
        ),
    )
    optimize_by: OptimizeBy = Field(
        default=OptimizeBy.DURATION,
        description=(
            "Ranking criterion: duration for travel time or distance for route length. "
            "Used only in mode=rank"
        ),
    )
    aggregate: Aggregate = Field(
        default=Aggregate.MIN,
        description=(
            "How to combine several origin costs: min means closest to any origin; sum/avg "
            "optimizes for the group; max minimizes the longest trip. Ignored for one origin. "
            "Used only in mode=rank"
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidates to return. Used only in mode=rank",
    )

    # --- common ---
    use_traffic: bool = Field(
        default=True,
        description=(
            "Account for traffic in driving routes. Leave true for non-driving transports; "
            "use_traffic=false is accepted only for driving"
        ),
    )
    avoid_tolls: bool = Field(
        default=False,
        description="Avoid toll roads; avoid_tolls=true is accepted only for driving",
    )
    departure_time: datetime | None = Field(
        default=None,
        description=(
            "Timezone-aware departure time (ISO 8601) for traffic forecasting. Set it only when "
            "the user explicitly supplies a future departure date or time; otherwise omit it. It "
            "must not be in the past, cannot be combined with use_traffic=false, and is not used "
            "for walking, bicycle, or scooter"
        ),
    )

    @model_validator(mode="after")
    def _check_mode(self) -> RoutingInput:
        if self.mode is RoutingMode.ROUTE:
            if len(self.waypoints) < 2:
                raise ValueError("mode=route requires at least 2 waypoints")
            if self.origins or self.candidates:
                raise ValueError("origins and candidates are not used in mode=route")

            # json_schema_extra guides structured output, while this validator
            # enforces the same contract for every non-LLM caller at runtime.
            if (
                self.optimize_by is not OptimizeBy.DURATION
                or self.aggregate is not Aggregate.MIN
                or self.limit != 5
            ):
                raise ValueError("optimize_by, aggregate, and limit are used only in mode=rank")

            cap = (
                MAX_WAYPOINTS_DRIVING
                if self.transport is TransportMode.DRIVING
                else MAX_WAYPOINTS_OTHER
            )
            if len(self.waypoints) > cap:
                raise ValueError(
                    f"transport={self.transport} allows at most {cap} waypoints; "
                    f"received {len(self.waypoints)}"
                )

            if self.optimize_waypoints and self.transport in {
                TransportMode.BICYCLE,
                TransportMode.SCOOTER,
            }:
                raise ValueError(
                    "optimize_waypoints is supported only for driving, walking, and transit"
                )

        if self.mode is RoutingMode.RANK:
            if not self.origins:
                raise ValueError("mode=rank requires at least 1 origin")
            if not self.candidates:
                raise ValueError("mode=rank requires at least 1 candidate")
            if self.waypoints:
                raise ValueError("waypoints are not used in mode=rank")
            if self.optimize_waypoints:
                raise ValueError("optimize_waypoints is not used in mode=rank")

            # Hard API cap AND the billing unit: 1 element = 1 paid request.
            elements = len(self.origins) * len(self.candidates)
            if elements > MAX_MATRIX_ELEMENTS:
                raise ValueError(
                    f"matrix {len(self.origins)}x{len(self.candidates)} contains {elements} "
                    f"elements; maximum is {MAX_MATRIX_ELEMENTS}"
                )

        active_places = (
            self.waypoints if self.mode is RoutingMode.ROUTE else [*self.origins, *self.candidates]
        )
        self._check_text_point_budget(active_places)

        if self.transport is not TransportMode.DRIVING:
            if not self.use_traffic:
                raise ValueError("use_traffic=false is supported only for driving")
            if self.avoid_tolls:
                raise ValueError("avoid_tolls is supported only for driving")

        return self

    @staticmethod
    def _check_text_point_budget(places: list[RoutingPlace]) -> None:
        """Reject a call that could spend too much internal geocoder quota."""

        # Count the same normalized identities that the input resolver resolves once.
        # Refs are local store lookups and therefore do not consume this budget.
        unique_text_points = {
            place.resolution_key for place in places if isinstance(place, RoutingPlaceQuery)
        }
        if len(unique_text_points) > MAX_TEXT_POINTS_PER_ROUTING_CALL:
            raise ValueError(
                "routing accepts at most "
                f"{MAX_TEXT_POINTS_PER_ROUTING_CALL} unique text points per call; "
                "pass the remaining points as plc_ refs"
            )

    @model_validator(mode="after")
    def _check_departure_time(self) -> RoutingInput:
        if self.departure_time is None:
            return self
        if self.departure_time.tzinfo is None:
            raise ValueError("departure_time must include a timezone")
        if self.departure_time < datetime.now(UTC):
            raise ValueError("departure_time must not be in the past")
        if not self.use_traffic:
            raise ValueError("departure_time is not used when use_traffic=false")
        if self.transport in {
            TransportMode.WALKING,
            TransportMode.BICYCLE,
            TransportMode.SCOOTER,
        }:
            raise ValueError("departure_time is not used for walking, bicycle, or scooter")
        return self


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------


class TrafficType(StrEnum):
    """Which traffic model the answer was actually computed with."""

    REALTIME = "realtime"
    FORECAST = "forecast"
    DISABLED = "disabled"


class RoutePoint(BaseModel):
    """A point echoed back so the answer is checkable — by ref and name, not coordinates.

    The coordinates the API actually used are in the ref's PlaceRecord in Redis;
    the map UI reads them from there.
    """

    ref: PlaceRef
    name: str = Field(min_length=1)
    #: Distance between the resolved input coordinate and the routable graph.
    snap_distance_m: int | None = Field(default=None, ge=0)


class RouteSegment(BaseModel):
    """A run of consecutive steps sharing one transport mode.

    This is what makes a transit route readable: "walk 6 min, transit 14 min,
    walk 3 min" instead of the raw turn-by-turn step list.
    """

    transport: TransportMode
    length_m: int = Field(ge=0)
    duration_s: int = Field(ge=0)


class RouteStep(BaseModel):
    """One provider-supplied navigation instruction within a route leg."""

    instruction: str = Field(
        min_length=1,
        description=(
            "Provider navigation instruction. Its embedded distance may be rounded; use length_m "
            "as the authoritative distance when presenting the step"
        ),
    )
    street_name: str | None = Field(
        default=None,
        description=(
            "Provider-supplied street name. When absent, describe the step as an unnamed road or "
            "path and never invent a name"
        ),
    )
    length_m: int = Field(
        ge=0,
        description=(
            "Authoritative step distance in metres; include it for every non-arrival step"
        ),
    )
    duration_s: int = Field(ge=0, description="Step duration in seconds")


class RouteLeg(BaseModel):
    """The stretch between two consecutive waypoints."""

    from_index: int = Field(ge=0)
    to_index: int = Field(ge=0)
    #: Returned route legs are complete. A provider maps a failed required leg
    #: to NOT_FOUND before constructing RouteInfo.
    ok: bool = True
    length_m: int = Field(default=0, ge=0)
    duration_s: int = Field(default=0, ge=0)
    segments: list[RouteSegment] = Field(default_factory=list)
    steps: list[RouteStep] = Field(default_factory=list)


class RouteInfo(BaseModel):
    """Result of ``mode=route``."""

    length_m: int = Field(ge=0)
    duration_s: int = Field(ge=0)
    legs: list[RouteLeg] = Field(default_factory=list)
    #: The waypoints in the order actually travelled.
    waypoints: list[RoutePoint] = Field(default_factory=list)
    #: Maps travelled order -> the index the model passed in. Differs from
    #: [0,1,2,...] only when `optimize_waypoints` reordered them.
    waypoint_order: list[int] = Field(default_factory=list)
    #: None when the selected provider cannot determine toll usage reliably.
    has_tolls: bool | None = None
    traffic_type: TrafficType | None = None


class OriginCost(BaseModel):
    """Cost from ONE reference point to one candidate (a single matrix element)."""

    origin_index: int = Field(ge=0)
    #: elements[].status == "OK". False = no route; length/duration stay None.
    reachable: bool = True
    length_m: int | None = Field(default=None, ge=0)
    duration_s: int | None = Field(default=None, ge=0)


class RankedCandidate(BaseModel):
    """Result of ``mode=rank`` — one candidate, scored and placed."""

    #: Index into the `candidates` the model passed in.
    candidate_index: int = Field(ge=0)
    point: RoutePoint
    #: 1 = best. Unreachable candidates are ranked last.
    rank: int = Field(ge=1)
    reachable: bool = True
    #: The value ranked by, in the unit of `optimize_by`
    #: (seconds for duration, metres for distance), after `aggregate`.
    score: float | None = Field(default=None, ge=0)
    #: Aggregated cost, both units — so the answer can say "12 min, 3.4 km".
    length_m: int | None = Field(default=None, ge=0)
    duration_s: int | None = Field(default=None, ge=0)
    #: Per-reference-point breakdown, so a "convenient for everyone" answer can
    #: justify itself ("14 min for you, 9 for Ann").
    per_origin: list[OriginCost] = Field(default_factory=list)


class RoutingOutput(BaseModel):
    mode: RoutingMode
    transport: TransportMode
    optimize_by: OptimizeBy | None = None
    aggregate: Aggregate | None = None
    #: Set only in `route` mode.
    route: RouteInfo | None = None
    #: Set only in `rank` mode, best first, already cut to `limit`.
    ranked: list[RankedCandidate] = Field(default_factory=list)
    #: Candidates with no route at all — dropped from `ranked`'s head, not hidden.
    unreachable_count: int = Field(default=0, ge=0)
    origins: list[RoutePoint] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize_active_mode(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        """Keep only fields that belong to the active mode."""

        data = cast(dict[str, object], handler(self))
        if self.mode is RoutingMode.ROUTE:
            for field_name in (
                "optimize_by",
                "aggregate",
                "ranked",
                "unreachable_count",
                "origins",
            ):
                data.pop(field_name, None)
        else:
            data.pop("route", None)
        return data

    @model_validator(mode="after")
    def _check_mode_shape(self) -> RoutingOutput:
        if self.mode is RoutingMode.ROUTE:
            if self.route is None:
                raise ValueError("mode=route requires route in the output")
            if self.ranked or self.origins:
                raise ValueError("ranked and origins are not used in mode=route output")
            return self

        if self.route is not None:
            raise ValueError("route is not used in mode=rank output")

        return self


_ROUTING_LLM_POINT: dict[str, Any] = {
    "oneOf": [
        {
            "type": "string",
            "pattern": PLACE_REF_PATTERN,
            "description": (
                "Existing plc_ ref passed by itself, without query or area. Always use this form "
                "when a matching ref is available in conversation context or a tool result."
            ),
        },
        {
            "type": "object",
            "additionalProperties": False,
            "description": "Use only when no matching plc_ ref is available.",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Address, toponym, organisation, or POI without the locality.",
                },
                "area": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 120,
                    "description": (
                        "Required locality scope: its name or a plc_ ref for a bounded locality."
                    ),
                },
            },
            "required": ["query", "area"],
        },
    ]
}


ROUTING_LLM_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {
            "type": "string",
            "enum": [mode.value for mode in RoutingMode],
            "description": "route: ordered waypoints; rank: compare candidates from origins.",
        },
        "transport": {
            "type": "string",
            "enum": [mode.value for mode in TransportMode],
            "description": "Default driving when omitted; do not ask the user to choose a mode.",
        },
        "waypoints": {
            "type": "array",
            "items": _ROUTING_LLM_POINT,
            "description": (
                "Route mode only: ordered start to destination, at least two points. Reuse every "
                "available plc_ ref; otherwise pass exactly {query, area}."
            ),
        },
        "optimize_waypoints": {
            "type": "boolean",
            "description": (
                "Route mode only; reorder after the start only when the user allows it. When route "
                "waypoints are optimised, report the travelled order returned by the tool, not the "
                "input order."
            ),
        },
        "origins": {
            "type": "array",
            "items": _ROUTING_LLM_POINT,
            "description": "Rank mode only: one or more reference points.",
        },
        "candidates": {
            "type": "array",
            "items": _ROUTING_LLM_POINT,
            "description": "Rank mode only: one or more places to compare.",
        },
        "optimize_by": {
            "type": "string",
            "enum": [criterion.value for criterion in OptimizeBy],
            "description": "Rank mode only; default duration.",
        },
        "aggregate": {
            "type": "string",
            "enum": [aggregate.value for aggregate in Aggregate],
            "description": "Rank mode only; combine multiple-origin costs, default min.",
        },
        "limit": {"type": "integer", "description": "Rank mode result count; default 5."},
        "use_traffic": {
            "type": "boolean",
            "description": "Driving only; default true. Leave true for other transport.",
        },
        "avoid_tolls": {"type": "boolean", "description": "Driving only."},
    },
    "required": ["mode"],
}


ROUTING_TOOL_SPEC = ToolSpec[RoutingInput, RoutingOutput](
    name="routing_tool",
    description=(
        "Calculate measured route distance and travel time, rank places by route cost, or find a "
        "meeting point. Call routing_tool directly for route requests; never call places_search "
        "first solely to prepare route endpoints. routing_tool resolves textual endpoints and "
        "uses the provider-ranked first address-bearing POI card without semantic reranking or "
        "clarification."
    ),
    input_model=RoutingInput,
    llm_parameters=ROUTING_LLM_PARAMETERS,
    output_model=RoutingOutput,
    eval_metrics=[
        "input_validation_accuracy",
        "origin_accuracy",
        "destination_accuracy",
        "transport_mode_accuracy",
        "route_success_rate",
        "start_snap_distance",
        "end_snap_distance",
        "distance_accuracy",
        "duration_accuracy",
        "pair_success_rate",
        "average_intra_day_travel_time",
        "max_leg_duration",
    ],
    answer_fields=("route", "ranked"),
    output_exclude_none=True,
)
