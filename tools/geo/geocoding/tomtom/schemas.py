"""Validated subset of TomTom Geocoding API responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TomTomGeocodePosition(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TomTomGeocodeBounds(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    top_left: TomTomGeocodePosition = Field(validation_alias="topLeftPoint")
    bottom_right: TomTomGeocodePosition = Field(validation_alias="btmRightPoint")


class TomTomMatchConfidence(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    score: float = Field(ge=0, le=1)


class TomTomGeocodeAddress(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    street_number: str | None = Field(default=None, validation_alias="streetNumber")
    street_name: str | None = Field(default=None, validation_alias="streetName")
    municipality_subdivision: str | None = Field(
        default=None,
        validation_alias="municipalitySubdivision",
    )
    municipality: str | None = None
    neighbourhood: str | None = None
    country_subdivision: str | None = Field(
        default=None,
        validation_alias="countrySubdivision",
    )
    country_secondary_subdivision: str | None = Field(
        default=None,
        validation_alias="countrySecondarySubdivision",
    )
    postal_code: str | None = Field(default=None, validation_alias="postalCode")
    country_code: str | None = Field(default=None, validation_alias="countryCode")
    country_code_iso3: str | None = Field(default=None, validation_alias="countryCodeISO3")
    country: str | None = None
    freeform_address: str | None = Field(default=None, validation_alias="freeformAddress")
    local_name: str | None = Field(default=None, validation_alias="localName")


class TomTomGeocodeResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str = Field(min_length=1)
    id: str = Field(min_length=1)
    score: float | None = None
    entity_type: str | None = Field(default=None, validation_alias="entityType")
    match_confidence: TomTomMatchConfidence | None = Field(
        default=None,
        validation_alias="matchConfidence",
    )
    address: TomTomGeocodeAddress
    position: TomTomGeocodePosition
    bounding_box: TomTomGeocodeBounds | None = Field(
        default=None,
        validation_alias="boundingBox",
    )


class TomTomGeocodeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    results: list[TomTomGeocodeResult] = Field(default_factory=list)


class TomTomReverseGeocodeResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    address: TomTomGeocodeAddress
    id: str | None = None
    entity_type: str | None = Field(default=None, validation_alias="entityType")


class TomTomReverseGeocodeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    addresses: list[TomTomReverseGeocodeResult] = Field(default_factory=list)
