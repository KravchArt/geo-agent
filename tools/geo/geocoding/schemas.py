"""Pydantic schemas for internal place geocoding.

Turns a text description of a place into a ``ref`` (see :mod:`tools.refs`).
The model gets ``ref`` + ``name`` + ``address``. It does **not** get coordinates:
those are stored separately under ``place:<ref>`` for adapters and the map UI.

``city`` is folded into the query text by the service because the backing API
has no separate city parameter.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.refs import PlaceRef

#: Cap on how many candidates we hand back for disambiguation.
MAX_MATCHES = 5


class ToponymKind(StrEnum):
    """``GeocoderMetaData.kind`` — the documented set of values."""

    HOUSE = "house"
    STREET = "street"
    METRO = "metro"
    DISTRICT = "district"
    LOCALITY = "locality"


class GeocodePlaceInput(BaseModel):
    """Input for internal place geocoding."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Place name or address, for example 'Red Square' or 'Tverskaya Street 1'",
    )
    city: str | None = Field(
        default=None,
        max_length=120,
        description="Optional city used to disambiguate the query, for example 'Moscow'",
    )
    limit: int = Field(
        default=MAX_MATCHES,
        ge=1,
        le=MAX_MATCHES,
        description=(
            "Number of candidates to return. Use 1 only when the place is definitely unambiguous"
        ),
    )
    locality_only: bool = Field(
        default=False,
        description=(
            "Internal resolver flag: restrict a city-scope lookup to municipality geographies. "
            "It is never supplied by the model-facing tools."
        ),
    )

    @field_validator("query", "city")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Collapse whitespace so equivalent calls produce the same tool_hash.
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class PlaceMatch(BaseModel):
    """One resolved candidate, as the model sees it.

    ``lat`` and ``lon`` are intentionally absent. They are stored separately
    under ``ref``; other tools receive that handle rather than coordinates.
    """

    ref: PlaceRef
    name: str
    address: str
    kind: ToponymKind | None = None
    precision: str | None = None
    municipality: str | None = None
    country_secondary_subdivision: str | None = None
    country_subdivision: str | None = None
    country_code: str | None = None
    provider_entity_type: str | None = Field(
        default=None,
        description=(
            "Provider-specific geographic entity type retained for internal locality "
            "resolution, for example TomTom's Municipality."
        ),
    )
    relevance_score: float | None = Field(
        default=None,
        description=(
            "Provider ranking relevance. It is not a calibrated probability and may be absent."
        ),
    )
    match_confidence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Provider confidence in the textual match to the query. Unlike ranking relevance, "
            "this value is normalized to the [0, 1] range when the provider supplies it."
        ),
    )


class GeocodePlaceOutput(BaseModel):
    """Candidates returned by a geocoding request."""

    best: PlaceMatch | None = None
    matches: list[PlaceMatch] = Field(default_factory=list)
    ambiguous: bool = False


class ReverseGeocodeOutput(BaseModel):
    """Structured administrative and address data resolved from coordinates."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    municipality: str | None = None
    municipality_subdivision: str | None = None
    neighbourhood: str | None = None
    country_subdivision: str | None = None
    country_secondary_subdivision: str | None = None
    postal_code: str | None = None
    country: str | None = None
    country_code: str | None = None
    country_code_iso3: str | None = None
