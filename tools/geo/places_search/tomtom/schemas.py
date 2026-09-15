"""Validated subset of TomTom Fuzzy Search v2 responses."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TomTomPosition(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TomTomEntryPoint(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str = Field(min_length=1)
    position: TomTomPosition


class TomTomAddress(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    freeform_address: str | None = Field(
        default=None,
        validation_alias="freeformAddress",
    )
    municipality: str | None = None


class TomTomClassificationName(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)


class TomTomClassification(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    code: str = Field(min_length=1)
    names: list[TomTomClassificationName] = Field(default_factory=list)


class TomTomCategoryId(BaseModel):
    """One most-specific TomTom POI category returned in categorySet."""

    model_config = ConfigDict(extra="ignore", strict=True)

    id: int = Field(gt=0)


class TomTomTimePoint(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    date: Date
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)

    @field_validator("date", mode="before")
    @classmethod
    def _parse_iso_date(cls, value: object) -> Date:
        # Strict response models still need an explicit conversion because JSON
        # represents an ISO date as a string.
        if isinstance(value, Date):
            return value
        if isinstance(value, str):
            return Date.fromisoformat(value)
        raise ValueError("date must be an ISO-8601 calendar date")

    @property
    def value(self) -> datetime:
        return datetime.combine(self.date, time(self.hour, self.minute))


class TomTomTimeRange(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    start_time: TomTomTimePoint = Field(validation_alias="startTime")
    end_time: TomTomTimePoint = Field(validation_alias="endTime")


class TomTomOpeningHours(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    mode: str | None = None
    time_ranges: list[TomTomTimeRange] = Field(
        default_factory=list,
        validation_alias="timeRanges",
    )


class TomTomTimeZone(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    iana_id: str = Field(min_length=1, validation_alias="ianaId")


class TomTomPoi(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    phone: str | None = None
    categories: list[str] = Field(default_factory=list)
    category_set: list[TomTomCategoryId] = Field(
        default_factory=list,
        validation_alias="categorySet",
    )
    classifications: list[TomTomClassification] = Field(default_factory=list)
    opening_hours: TomTomOpeningHours | None = Field(
        default=None,
        validation_alias="openingHours",
    )
    time_zone: TomTomTimeZone | None = Field(
        default=None,
        validation_alias="timeZone",
    )

    @property
    def category_names(self) -> list[str]:
        """Prefer classifications because the legacy categories field is deprecated."""

        values = [
            name.name for classification in self.classifications for name in classification.names
        ]
        if not values:
            values = self.categories

        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(value.split())
            normalized = cleaned.casefold()
            if cleaned and normalized not in seen:
                seen.add(normalized)
                result.append(cleaned)
        return result

    @property
    def classification_codes(self) -> frozenset[str]:
        return frozenset(item.code for item in self.classifications)

    @property
    def category_ids(self) -> frozenset[int]:
        return frozenset(item.id for item in self.category_set)


class TomTomSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: Literal["POI"]
    id: str = Field(min_length=1)
    score: float = 0.0
    poi: TomTomPoi
    address: TomTomAddress
    position: TomTomPosition
    entry_points: list[TomTomEntryPoint] = Field(
        default_factory=list,
        validation_alias="entryPoints",
    )

    @property
    def routing_position(self) -> TomTomPosition:
        """Prefer the main entrance over the POI's visual centre."""

        for entry_point in self.entry_points:
            if entry_point.type.casefold() == "main":
                return entry_point.position
        return self.position


class TomTomSearchSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    total_results: int = Field(default=0, ge=0, validation_alias="totalResults")


class TomTomSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    summary: TomTomSearchSummary
    results: list[TomTomSearchResult] = Field(default_factory=list)
