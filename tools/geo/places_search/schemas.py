"""``places_search`` — SCHEMAS ONLY (no API client, no tool logic).

Three modes, chosen by the model via ``mode``:

* ``area`` — search across a locality. Pass its name for the first request or
  reuse the ``plc_`` ref returned in ``PlacesSearchOutput.area``.
* ``near`` — search around a place within ``radius_m``. Pass either its full
  textual name/address or an existing ``plc_`` ref. A textual anchor also needs
  ``area`` (again either locality text or a reusable ref).
* ``resolve`` — identify one geographic record for each concrete organisation
  already named by the user, conversation, or web evidence. It supports a
  singular factual lookup such as "the phone of Starbucks on Oxford Street", but
  is not used to discover or list branches.

The model never supplies coordinates. Area resolution is lazy: providers such
as 2GIS use their native region lookup, while providers that require portable
bounds receive a geocoder-backed area ref only when fallback reaches them.
When textual ``near`` and ``area`` are
available, the service tries the configured city-scoped place catalogs first
(normally 2GIS, then TomTom) and falls back to the internal geocoder. Area/city
resolution for those providers remains geocoder-backed. The selected place is stored under a ref.
That keeps both input classification and coordinates out of the model's hands
— see :mod:`tools.refs`.

    places_search(mode="area", area="Chicago", query="restaurants")
    # subsequent request in the same resolved city:
    places_search(mode="area", area="plc_a1b2c3d4e5", query="museums")

    places_search(mode="near", near="Millennium Park", area="Chicago", query="coffee shops")
    # a named organisation used only as the nearby-search anchor is resolved internally:
    places_search(mode="near", near="cafe Molot", area="Gorodets", query="pharmacies")
    # subsequent request around a different anchor in the same resolved city:
    places_search(mode="near", near="Clark/Lake station", area="plc_a1b2c3d4e5", query="cafes")
    # subsequent request around the same resolved anchor:
    places_search(mode="near", near="plc_a1b2c3d4e5", query="coffee shops")

Every place returned gets a ref of its own, so it can be fed straight into
``routing_tool`` without the model ever retyping a coordinate.

``category`` communicates search intent as well as a provider-neutral taxonomy:
set it only when the user wants any places of that category, and omit it when
the user names a particular organisation or chain. Providers can therefore
choose category discovery or full-text name search without guessing intent
from the wording of ``query``.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from tools.base import ToolSpec
from tools.geo.places_search.category_normalization import normalized_category_value
from tools.refs import PLACE_REF_PATTERN, PlaceRef

MAX_NAMED_PLACE_RESULTS = 5
MAX_RESOLVE_ORGANISATIONS = 10
MAX_RESOLVE_OPTIONS = 3
_COORDINATE_QUERY_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*,\s*"
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
)


def _is_place_ref(value: str) -> bool:
    return re.fullmatch(PLACE_REF_PATTERN, value) is not None


class SearchMode(StrEnum):
    AREA = "area"
    NEAR = "near"
    RESOLVE = "resolve"


class OrganisationCandidate(BaseModel):
    """One already-known organisation that needs a geographic identity."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Caller-defined stable correlation key, for example candidate_1. It links the "
            "resolved place back to web facts and src_ evidence; it is not a place ref"
        ),
    )
    name: str = Field(min_length=1, max_length=200)
    address_hint: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional address, district, street, or branch hint from source evidence. "
            "Use it only to distinguish same-name locations"
        ),
    )

    @field_validator("address_hint", mode="before")
    @classmethod
    def _empty_address_hint_is_none(cls, value: object) -> object:
        """Models often serialize an omitted optional hint as an empty string."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("client_id", "name", "address_hint")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class PlaceCategory(StrEnum):
    """Canonical category hint selected by the model for structured providers."""

    AIRPORT = "airport"
    ALCOHOL_STORE = "alcohol_store"
    ARTS_CENTRE = "arts_centre"
    ATM = "atm"
    ATTRACTION = "attraction"
    BAKERY = "bakery"
    BANK = "bank"
    BAR = "bar"
    BEAUTY_SALON = "beauty_salon"
    BICYCLE_RENTAL = "bicycle_rental"
    BICYCLE_STORE = "bicycle_store"
    BOOKSTORE = "bookstore"
    BUS_STATION = "bus_station"
    BUTCHER = "butcher"
    CAFE = "cafe"
    CAR_DEALER = "car_dealer"
    CAR_RENTAL = "car_rental"
    CAR_REPAIR = "car_repair"
    CAR_WASH = "car_wash"
    CASINO = "casino"
    CHARGING_STATION = "charging_station"
    CINEMA = "cinema"
    CLINIC = "clinic"
    CLOTHING_STORE = "clothing_store"
    COFFEE_SHOP = "coffee_shop"
    COMMUNITY_CENTRE = "community_centre"
    COMPUTER_STORE = "computer_store"
    CONFECTIONERY = "confectionery"
    CONVENIENCE_STORE = "convenience_store"
    COSMETICS_STORE = "cosmetics_store"
    DENTIST = "dentist"
    DEPARTMENT_STORE = "department_store"
    DOCTORS = "doctors"
    DRY_CLEANING = "dry_cleaning"
    ELECTRONICS_STORE = "electronics_store"
    FAST_FOOD = "fast_food"
    FIRE_STATION = "fire_station"
    FITNESS_CENTRE = "fitness_centre"
    FLOWER_SHOP = "flower_shop"
    FOOD_COURT = "food_court"
    FUEL = "fuel"
    FURNITURE_STORE = "furniture_store"
    GALLERY = "gallery"
    GARDEN_CENTRE = "garden_centre"
    GIFT_SHOP = "gift_shop"
    GREENGROCER = "greengrocer"
    GUEST_HOUSE = "guest_house"
    HAIRDRESSER = "hairdresser"
    HARDWARE_STORE = "hardware_store"
    HOSPITAL = "hospital"
    HOSTEL = "hostel"
    HOTEL = "hotel"
    ICE_CREAM = "ice_cream"
    JEWELRY_STORE = "jewelry_store"
    KINDERGARTEN = "kindergarten"
    LAUNDRY = "laundry"
    LIBRARY = "library"
    MALL = "mall"
    MARKETPLACE = "marketplace"
    MOBILE_PHONE_STORE = "mobile_phone_store"
    MUSEUM = "museum"
    MUSIC_VENUE = "music_venue"
    NIGHTCLUB = "nightclub"
    OPTICIAN = "optician"
    PARK = "park"
    PARKING = "parking"
    PET_STORE = "pet_store"
    PHARMACY = "pharmacy"
    PIZZERIA = "pizzeria"
    PLACE_OF_WORSHIP = "place_of_worship"
    PLAYGROUND = "playground"
    POLICE = "police"
    POST_OFFICE = "post_office"
    PUB = "pub"
    RESTAURANT = "restaurant"
    SCHOOL = "school"
    SHOE_STORE = "shoe_store"
    SPORTS_CENTRE = "sports_centre"
    SPORTS_STORE = "sports_store"
    STATIONERY_STORE = "stationery_store"
    SUPERMARKET = "supermarket"
    SWIMMING_POOL = "swimming_pool"
    TAXI = "taxi"
    THEATRE = "theatre"
    TOY_STORE = "toy_store"
    TRAIN_STATION = "train_station"
    TRAVEL_AGENCY = "travel_agency"
    UNIVERSITY = "university"
    VETERINARY = "veterinary"
    VIEWPOINT = "viewpoint"
    ZOO = "zoo"


class PlacesSearchInput(BaseModel):
    """Filled in by the MODEL. This class *is* the tool's input schema."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "mode": {"const": "area"},
                        },
                        "required": ["mode"],
                    },
                    "then": {
                        "required": ["query", "area"],
                        "properties": {
                            # Inactive fields may be omitted or explicitly null.
                            "near": {"type": "null"},
                            "organisations": {"maxItems": 0},
                        },
                    },
                },
                {
                    "if": {
                        "properties": {
                            "mode": {"const": "near"},
                        },
                        "required": ["mode"],
                    },
                    "then": {
                        "required": ["query", "near"],
                        "properties": {
                            "organisations": {"maxItems": 0},
                        },
                        "oneOf": [
                            {
                                "properties": {
                                    "near": {"pattern": PLACE_REF_PATTERN},
                                    "area": {"type": "null"},
                                },
                            },
                            {
                                "required": ["area"],
                                "properties": {
                                    "near": {"not": {"pattern": PLACE_REF_PATTERN}},
                                },
                            },
                        ],
                    },
                },
                {
                    "if": {
                        "properties": {"mode": {"const": "resolve"}},
                        "required": ["mode"],
                    },
                    "then": {
                        "required": ["organisations", "area"],
                        "properties": {
                            "query": {"type": "null"},
                            "category": {"type": "null"},
                            "near": {"type": "null"},
                            "organisations": {
                                "minItems": 1,
                                "maxItems": MAX_RESOLVE_ORGANISATIONS,
                            },
                        },
                    },
                },
            ],
        },
    )

    def __init__(self, **data: Any) -> None:
        """Validate dynamic tool arguments, including temporary legacy aliases."""

        super().__init__(**data)

    mode: SearchMode = Field(
        description=(
            "area discovers one or more matching places across a city; near discovers one or "
            "more places within a true circular radius around an anchor; resolve identifies one "
            "geographic record for each already-known organisation. A named organisation used "
            "only as a nearby-search anchor belongs directly in mode=near, not in a preliminary "
            "mode=resolve call"
        ),
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Already processed search target, not the user's full sentence. Keep only the "
            "requested place type or proper name in the user's original language and script; never "
            "translate or transliterate it. Remove command words, city, near-anchor text, radius, "
            "opening-hours wording, and other constraints because they belong in their dedicated "
            "fields. For category discovery use a generic target, "
            "for example query='coffee shops' with category='coffee_shop'. For a named place, "
            "organisation, or chain use only its name, for example query='Starbucks', and omit "
            "category"
        ),
    )
    category: PlaceCategory | None = Field(
        default=None,
        description=(
            "Canonical category for discovery requests. Set it only when the user asks for "
            "any places of a type, for example supermarkets, pharmacies, or hotels. Omit it "
            "when query names a particular place, organisation, or chain, even if its type "
            "is obvious; for example query='Starbucks' must not set coffee_shop. "
            "Choose only an allowed enum value; never pass raw OpenStreetMap tags. "
            "Use coffee_shop for coffee-focused venues and cafe for general cafes; "
            "use pizzeria for pizza-focused venues. For a generic restaurant request such as "
            "query='restaurants', set restaurant; otherwise providers interpret query as an "
            "organisation name. "
            "If no value is a good semantic match, omit it"
        ),
    )
    area: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Locality scope as either its name or a reusable plc_ ref from "
            "PlacesSearchOutput.area.ref. Required in area and resolve modes, and for a textual "
            "near anchor. Omit it when near is an existing plc_ ref because the backend loads "
            "that anchor's locality. Keep textual values in the user's original language and "
            "script; never translate or transliterate them"
        ),
    )
    near: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Place to search around, as either a reusable plc_ ref or a textual address, "
            "geographic toponym, named organisation, or POI. Copy the complete textual anchor "
            "phrase while "
            "removing only surrounding command words: never reduce it to a bare proper name. "
            "Keep its original language and script; never translate or transliterate it. "
            "Preserve qualifiers such as territory, metro/station, park, pier, railway station, "
            "shopping centre, entrance, building, terminal, or address—for example "
            "'Manhattan district', 'Oxford Circus station', 'Hyde Park', or "
            "'Westminster Pier'. "
            "These words select the intended provider object type. With a textual value, also "
            "pass area; with an existing plc_ ref, omit area. Do not call mode=resolve first only "
            "to obtain a ref for mode=near: textual anchors are resolved internally"
        ),
    )
    organisations: list[OrganisationCandidate] = Field(
        default_factory=list,
        max_length=MAX_RESOLVE_ORGANISATIONS,
        description=(
            "Concrete organisations to identify individually in mode=resolve. Use this for a "
            "specific establishment or branch whose map facts are requested, or for candidates "
            "already obtained from conversation or web evidence. Do not use it to list branches. "
            "Keep arbitrary facts such as price, menu, rooftop seating, and their src_ refs "
            "outside this field; client_id correlates the result with that evidence"
        ),
    )
    radius_m: int = Field(
        default=1000,
        ge=100,
        le=10_000,
        description=(
            "True circular search radius in metres; used only in mode=near and not for "
            "city-wide area searches"
        ),
    )
    open_24h: bool = Field(
        default=False,
        description=(
            "Return only places explicitly confirmed as open 24 hours every day. "
            "This does not mean open now or open on a particular weekday; use open_now "
            "for the current opening status"
        ),
    )
    open_now: bool = Field(
        default=False,
        description=(
            "Set true only when the user asks for places open now or at the current moment. "
            "The tool checks each place's published schedule in its local time zone at tool "
            "execution time and excludes unknown current status. Keep phrases such as 'open "
            "now' out of query because this field applies the filter. This is distinct from "
            "open_24h, which requires a place to operate continuously every day"
        ),
    )
    min_rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description=(
            "Inclusive minimum organisation rating. Set it only when the user requests a rating "
            "threshold, for example min_rating=4.6 for places rated 4.6 or higher. This filter "
            "requires 2GIS coverage for the search locality; if places_search returns "
            "unsupported_filter, use web_search for the same rated-place request instead of "
            "retrying this tool with another provider. Places with no published rating are "
            "excluded."
        ),
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description=(
            "Requested number of places. For a generic category discovery with no requested "
            "count, use 10. A user's explicit count always wins. Named searches (category "
            "omitted) return at most 5 distinct matches; category discovery may use the full "
            "requested limit"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_scope_fields(cls, value: object) -> object:
        """Translate the former split fields during the API migration window."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_area = [data.pop(key) for key in ("city", "area_ref") if data.get(key) is not None]
        legacy_near = data.pop("near_query", None)
        if "city" in data:
            data.pop("city")
        if "area_ref" in data:
            data.pop("area_ref")
        if legacy_area:
            if data.get("area") is not None or len(legacy_area) > 1:
                raise ValueError("pass one unified area value")
            data["area"] = legacy_area[0]
        if legacy_near is not None:
            if data.get("near") is not None:
                raise ValueError("pass one unified near value")
            data["near"] = legacy_near
        return data

    @field_validator("query", "area", "near")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Collapse whitespace so equivalent calls produce the same tool_hash.
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("area", "near")
    @classmethod
    def _reject_coordinates(cls, value: str | None) -> str | None:
        if value is not None and _COORDINATE_QUERY_PATTERN.fullmatch(value):
            raise ValueError("coordinates are not allowed; use a place name, address, or ref")
        return value

    @property
    def result_limit(self) -> int:
        """Apply the shared cap for a request intended to identify one named place."""

        if self.category is None:
            return min(self.limit, MAX_NAMED_PLACE_RESULTS)
        return self.limit

    @model_validator(mode="after")
    def _check_mode(self) -> PlacesSearchInput:
        normalized_category = normalized_category_value(
            self.query,
            self.category.value if self.category is not None else None,
        )
        self.category = (
            PlaceCategory(normalized_category) if normalized_category is not None else None
        )
        if self.mode is SearchMode.RESOLVE:
            if self.area is None:
                raise PydanticCustomError(
                    "places_search_resolve_area_required",
                    "mode=resolve requires area",
                )
            if not self.organisations:
                raise PydanticCustomError(
                    "places_search_resolve_organisations_required",
                    "mode=resolve requires at least one organisation",
                )
            client_ids = [item.client_id for item in self.organisations]
            if len(client_ids) != len(set(client_ids)):
                raise PydanticCustomError(
                    "places_search_resolve_client_ids_unique",
                    "mode=resolve requires unique organisation client_id values",
                )
            if self.query is not None or self.category is not None:
                raise PydanticCustomError(
                    "places_search_resolve_discovery_fields_forbidden",
                    "query and category are not used in mode=resolve",
                )
            if self.near is not None:
                raise PydanticCustomError(
                    "places_search_resolve_scope_fields_forbidden",
                    "near is not used in mode=resolve",
                )
            return self

        if self.query is None:
            raise PydanticCustomError(
                "places_search_query_required",
                "query is required in mode=area and mode=near",
            )
        if self.organisations:
            raise PydanticCustomError(
                "places_search_organisations_forbidden",
                "organisations is used only in mode=resolve",
            )

        if self.mode is SearchMode.AREA:
            if self.area is None:
                raise PydanticCustomError(
                    "places_search_area_required",
                    "mode=area requires area",
                )

            if self.near is not None:
                raise PydanticCustomError(
                    "places_search_area_near_forbidden",
                    "near is not used in mode=area",
                )

            return self

        if self.near is None:
            raise PydanticCustomError(
                "places_search_near_anchor_required",
                "mode=near requires near",
            )

        if self.near_ref is not None and self.area is not None:
            raise PydanticCustomError(
                "places_search_near_ref_scope_forbidden",
                "mode=near with an existing near ref does not use area",
            )

        if self.near_query is not None and self.area is None:
            raise PydanticCustomError(
                "places_search_near_area_required",
                "mode=near with a textual near requires area",
            )

        return self

    @property
    def city(self) -> str | None:
        """Return a textual area for provider compatibility."""

        return self.area if self.area is not None and not _is_place_ref(self.area) else None

    @property
    def area_ref(self) -> PlaceRef | None:
        """Return a resolved area ref for provider compatibility."""

        if self.area is not None and _is_place_ref(self.area):
            return self.area
        return None

    @property
    def near_ref(self) -> PlaceRef | None:
        """Return a resolved anchor ref, if the unified value is one."""

        if self.near is not None and _is_place_ref(self.near):
            return self.near
        return None

    @property
    def near_query(self) -> str | None:
        """Return an unresolved textual anchor, if present."""

        return self.near if self.near is not None and not _is_place_ref(self.near) else None


class Place(BaseModel):
    """One organisation, as the MODEL sees it.

    ``ref`` is the handle to pass on to ``routing_tool``. ``id`` is the provider's
    own identifier, kept so an answer can be traced back to the exact record.
    Coordinates are deliberately absent — they are in Redis under ``ref``.
    """

    ref: PlaceRef = Field(
        description=(
            "Stable internal place ref for later places_search or routing_tool calls; "
            "do not substitute the provider id"
        )
    )
    id: str = Field(description="Provider record id; not a plc_ place ref")
    name: str
    address: str
    categories: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description=(
            "Provider-supplied rating when available; None means the selected provider did not "
            "supply it and must not be interpreted as zero"
        ),
    )
    review_count: int | None = Field(
        default=None,
        ge=0,
        description="Provider-supplied number of reviews associated with rating, when available",
    )
    #: Provider-derived hours, compacted when the source returns dated ranges.
    hours_text: str | None = None
    #: True only when the source confirms complete 24-hour coverage; None = unknown.
    open_24h: bool | None = None
    is_open_now: bool | None = Field(
        default=None,
        description=(
            "Whether published hours confirmed the place was open at tool execution time. "
            "True is guaranteed for results when open_now was requested; None means the current "
            "status was not evaluated or could not be verified"
        ),
    )
    #: Features[] that are present, e.g. ["wheelchair_access", "ramp"].
    accessibility: list[str] = Field(default_factory=list)
    #: Only in `near` mode — computed by us, never returned by the API.
    distance_m: int | None = None


class OrganisationResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class OrganisationResolution(BaseModel):
    """Independent resolution outcome for one mode=resolve input candidate."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1, max_length=64)
    input_name: str = Field(min_length=1, max_length=200)
    status: OrganisationResolutionStatus
    place: Place | None = Field(
        default=None,
        description="Unique resolved organisation when status=resolved",
    )
    options: list[Place] = Field(
        default_factory=list,
        max_length=MAX_RESOLVE_OPTIONS,
        description=(
            "Up to three geographically valid alternatives when status=ambiguous; preserve "
            "their refs and ask the user only when the unresolved candidate is needed"
        ),
    )
    error: str | None = Field(
        default=None,
        description="Safe per-candidate provider error when status=error",
    )

    @model_validator(mode="after")
    def _check_status_payload(self) -> OrganisationResolution:
        if self.status is OrganisationResolutionStatus.RESOLVED:
            if self.place is None or self.options or self.error is not None:
                raise ValueError("resolved outcome requires only place")
        elif self.status is OrganisationResolutionStatus.AMBIGUOUS:
            if self.place is not None or not self.options or self.error is not None:
                raise ValueError("ambiguous outcome requires only options")
        elif self.status is OrganisationResolutionStatus.ERROR:
            if self.place is not None or self.options or self.error is None:
                raise ValueError("error outcome requires only error")
        elif self.place is not None or self.options or self.error is not None:
            raise ValueError("not_found outcome cannot carry place, options, or error")
        return self


class ResolvedSearchArea(BaseModel):
    """A locality resolved for area or nearby search and safe to reuse by ref."""

    ref: PlaceRef
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)


class PlacesSearchOutput(BaseModel):
    places: list[Place] = Field(
        default_factory=list,
        description="Matching organisations; each ref can be reused by geographic tools",
    )
    #: Resolved locality for mode=area or a scoped textual near. Reuse its ref as area.
    area: ResolvedSearchArea | None = None
    #: True when the provider response omitted candidates, or result_limit omitted
    #: eligible matches from the fetched response. Unfetched provider candidates
    #: are not guaranteed to satisfy our local filters.
    truncated: bool = False
    #: Echo of the anchor ref, so the answer can say what it measured from.
    anchor: PlaceRef | None = None
    resolved: list[OrganisationResolution] = Field(
        default_factory=list,
        description=(
            "Per-candidate outcomes for mode=resolve in the same order as the input. "
            "client_id links each geographic result back to external web evidence"
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]
        description="Exact number of matching organisations included in places",
        return_type=int,
    )
    @property
    def returned_count(self) -> int:
        """Return ordinary places or uniquely resolved batch candidates."""

        if self.resolved:
            return sum(
                item.status is OrganisationResolutionStatus.RESOLVED for item in self.resolved
            )
        return len(self.places)


# The model needs the calling convention, not the Pydantic implementation of
# every cross-field constraint. PlacesSearchInput remains the authoritative
# validator after the model returns its function arguments.
PLACES_SEARCH_LLM_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {
            "type": "string",
            "enum": [mode.value for mode in SearchMode],
            "description": (
                "area: query plus area; near: query plus near (and area for a textual anchor); "
                "resolve: area and organisations only. Choose by the expected result, not merely "
                "by whether query contains a proper name. Use `area` to discover "
                "a set of places or branches in a city (for example 'find Starbucks stores in "
                "Chicago' is mode=area, query='Starbucks', without category), `near` to discover a "
                "set around one anchor, and `resolve` to identify one record for each concrete "
                "named establishment (for example 'phone of Starbucks on Oxford Street' uses "
                "mode=resolve, name='Starbucks', address_hint='Oxford Street'). Resolve is not "
                "branch discovery or anchor preparation. A named organisation used only as the "
                "anchor for nearby discovery goes directly in mode=near as textual `near`; never "
                "call mode=resolve first to obtain its ref. A named chain alone does not imply "
                "resolve: choose by whether the user wants a result set or one identified entity."
            ),
        },
        "query": {
            "type": "string",
            "description": (
                "Required for area and near modes, including category discovery. Place type or "
                "name only. Preserve that place type or name in the source language and script; "
                "never translate, transliterate, anglicize, or localize it. Only `category` gets "
                "the canonical English enum. Omit city, anchor, radius, filters, ownership, "
                "provenance, and subjective or stylistic qualifiers. Those qualifiers are not "
                "map fields and must not be simulated by reformulating query across repeated "
                "calls."
            ),
        },
        "category": {
            "type": "string",
            "enum": [category.value for category in PlaceCategory],
            "description": (
                "Required whenever query is a generic place type: for example "
                "query='restaurants' requires category='restaurant'. Omit only for a specific "
                "organisation or chain name; without category, query is interpreted as a specific "
                "name search. Never choose the nearest enum merely to approximate an unsupported "
                "type: for example, `artisan workshop` is neither `gift_shop` nor `arts_centre`. "
                "When the user requests multiple distinct supported categories, make one "
                "places_search call per category. Each call must contain exactly one category and "
                "its matching generic query; never merge multiple categories into one query."
            ),
        },
        "area": {
            "type": "string",
            "description": (
                "Locality name or its reusable plc_ ref. Required for area/resolve and for a "
                "textual near anchor. For another area search or a new textual near in an "
                "established locality, reuse its valid ref. Pass locality text only for the first "
                "search there, when no ref is available, or after an unknown_ref error. On "
                "select_area, reissue the original call with the selected value as area. Keep "
                "the locality in its source language and script; never translate, transliterate, "
                "anglicize, or localize it. If the user wrote 'München', pass area='München', not "
                "area='Munich'. Omit area with an already resolved near plc_ ref."
            ),
        },
        "near": {
            "type": "string",
            "description": (
                "Anchor as either an existing plc_ ref or a complete name/address. For a local "
                "request use current_location_ref. Keep textual qualifiers such as metro, park, "
                "station, terminal, or address, and pass area with textual anchors. Keep text in "
                "its source language and script; never translate, transliterate, anglicize, or "
                "localize it. If a textual anchor is ambiguous, present the select_anchor "
                "clarification instead of choosing a candidate. A named organisation that is only "
                "the nearby-search anchor belongs directly in this field; do not call mode=resolve "
                "first because the tool resolves textual anchors internally."
            ),
        },
        "organisations": {
            "type": "array",
            "description": (
                "Resolve-mode candidates. Use the shared city (including one established in the "
                "conversation) and exact names from the user, conversation, or evidence. Preserve "
                "each name's source spelling, language, and script; never translate or "
                "transliterate it. Give every candidate a unique `client_id`. Whenever the "
                "evidence contains an address, street, district, "
                "metro or station, branch label, or any other location qualifier for a candidate, "
                "copy that qualifier into `address_hint` in its original language and script. "
                "Never resolve a bare organisation name when such a qualifier is available; omit "
                "`address_hint` only when the evidence provides none. For candidates supplied by "
                "one web_search criterion, choose at most ten and include all chosen candidates in "
                "one resolve call. After that call succeeds, never call resolve again for the same "
                "criterion with additional or different candidates; return the verified subset "
                "instead. Keep price, menu, "
                "availability, and `src_...` facts outside this tool. Reuse a completed resolved, "
                "not_found outcome; repeat it only after new user information or a retryable "
                "temporary error. Resolution selects the first provider-ranked candidate with "
                "an address and does not semantically rerank or clarify provider results; anchor "
                "ambiguity belongs to near mode and requires clarification. After select_area, "
                "retry resolve with the same organisations, "
                "filters, and limits and the selected area; never convert it to area discovery."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "client_id": {"type": "string"},
                    "name": {
                        "type": "string",
                        "description": (
                            "Exact organisation name copied from the user, conversation, or "
                            "evidence in its source spelling, language, and script. Never "
                            "translate or transliterate it."
                        ),
                    },
                    "address_hint": {
                        "type": "string",
                        "description": (
                            "Location qualifier copied from evidence in its original language and "
                            "script. Never translate or transliterate it. Required whenever "
                            "evidence provides an address, street, district, metro or station, "
                            "branch label, or another location hint; omit only when no such "
                            "qualifier is available."
                        ),
                    },
                },
                "required": ["client_id", "name"],
            },
        },
        "radius_m": {"type": "integer", "description": "Near-mode radius in metres; default 1000."},
        "open_24h": {
            "type": "boolean",
            "description": "Only places open 24 hours every day; `open_24h` means every day.",
        },
        "open_now": {
            "type": "boolean",
            "description": (
                "Only places confirmed open now; require that the user asks about the current "
                "moment."
            ),
        },
        "min_rating": {
            "type": "number",
            "description": (
                "Optional inclusive rating threshold. Set it only when the user requests a "
                "rating threshold. Supported only where the locality has 2GIS coverage."
            ),
        },
        "limit": {
            "type": "integer",
            "description": (
                "Requested result count. For generic category discovery, default 10 unless "
                "the user asked for a different number."
            ),
        },
    },
    "required": ["mode"],
    # Keep this one cross-field rule in the compact schema: without it, a
    # function-calling model can legally emit just mode + category, although
    # both discovery modes require a textual search target.
    "allOf": [
        {
            "if": {
                "properties": {"mode": {"enum": ["area", "near"]}},
                "required": ["mode"],
            },
            "then": {"required": ["query"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "near"}},
                "required": ["mode"],
            },
            "then": {"required": ["near"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "area"}},
                "required": ["mode"],
            },
            "then": {"required": ["area"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "resolve"}},
                "required": ["mode"],
            },
            "then": {"required": ["area", "organisations"]},
        },
    ],
}


PLACES_SEARCH_SPEC = ToolSpec[PlacesSearchInput, PlacesSearchOutput](
    name="places_search",
    description=(
        "Find organisations and their map fields for ordinary place discovery. The advertised JSON "
        "schema is the authoritative contract for modes, categories, filters, and limits. "
        "LANGUAGE INVARIANT: `category` is the only field normalized to an English enum. Preserve "
        "every free-text value (`query`, `area`, `near`, `organisations[].name`, and "
        "`address_hint`) in its original language and script. Never translate, transliterate, "
        "anglicize, localize, or otherwise normalize these values. Before calling the tool, "
        "compare every free-text argument with its source text and correct any language or script "
        "change. "
        "This tool cannot filter by average check, price or budget, menu items, availability, "
        "or arbitrary amenities and special features. Do not put unsupported criteria into "
        "`query`; use web_search for them. Its only supported attribute filters are opening hours "
        "(`open_now` and `open_24h`) and minimum rating (`min_rating`); rating filtering requires "
        "2GIS coverage for the locality. "
        "When web_search supplies candidates for an unsupported criterion, use only "
        "`mode=resolve` on those candidates for that part of the request. Select at most ten and "
        "resolve them in exactly one batch. After a successful resolve, never call places_search "
        "again for the same web criterion with additional or different candidates; return the "
        "verified subset even when it contains fewer results than desired. Never use `mode=area` "
        "or `mode=near` before or after resolve to replace, expand, or pad that candidate set. "
        "Before including a resolved candidate in the answer, verify that its place type or "
        "category matches the entity type requested; resolution alone does not prove type fit. "
        "Do not use any places_search mode solely to obtain endpoint refs for a route, even for "
        "named organisations or chains with multiple branches; routing_tool resolves textual "
        "route endpoints and uses the provider-ranked first address-bearing POI card without "
        "semantic reranking or clarification. "
        "Before every places_search call, classify the target. For a generic place type requested "
        "as a result set, such as pharmacies or restaurants, use "
        "mode=area or mode=near and always set the matching `category` enum. For a specific "
        "organisation, establishment, or chain name, such as Starbucks, Cofix, or Cafe Molot, "
        "omit `category`. Classify the named organisation by its role: use mode=resolve "
        "when that organisation itself is the requested result, use mode=area or mode=near without "
        "`category` when discovering its branches by name, and put it directly in textual `near` "
        "when it is only the anchor for discovering other places. Never make a separate "
        "mode=resolve call solely to obtain a ref for a subsequent mode=near call; the tool "
        "resolves textual near anchors internally. Examples: "
        "'pharmacies nearby' means query='pharmacies' and category='pharmacy'; 'restaurants in "
        "Moscow' means query='restaurants' and category='restaurant'; 'Starbucks nearby' "
        "means query='Starbucks' with no category; 'give me the address and opening hours of Cafe "
        "Molot in Gorodets' means mode=resolve with organisations and no category; 'find "
        "pharmacies within 3 km of Cafe Molot in Gorodets' means mode=near, near='Cafe Molot', "
        "area='Gorodets', query='pharmacies', category='pharmacy', and radius_m=3000. In area "
        "and near modes, `query` is required even "
        "when `category` is set. For multiple distinct supported categories, make one "
        "places_search call per category, with exactly one category and its matching generic "
        "query in each call; never merge multiple categories into one query. "
        "For a non-English generic category request, keep the original category wording in "
        "`query`; put its normalized English meaning only in `category`. Reuse `area.ref` and "
        "`place.ref`; a provider id is not a ref, and internal refs normally are not displayed. "
        "When the tool returns one or more results, use them for the part of the request they "
        "answer. Do not call places_search again with the same search intent and materially "
        "equivalent parameters; another call is appropriate only for a distinct unresolved part "
        "that requires materially different search parameters."
    ),
    input_model=PlacesSearchInput,
    llm_parameters=PLACES_SEARCH_LLM_PARAMETERS,
    output_model=PlacesSearchOutput,
    eval_metrics=[
        "input_validation_accuracy",
        "constraint_preservation_rate",
        "category_correctness",
        "search_mode_accuracy",
        "area_resolution_accuracy",
        "anchor_resolution_accuracy",
        "top1_accuracy",
        "hitrate_at_k",
        "recall_at_k",
        "precision_at_k",
        "ndcg_at_k",
        "radius_correctness",
        "constraint_correctness",
        "duplicate_rate",
    ],
    answer_fields=("places", "resolved.place", "resolved.options"),
)
