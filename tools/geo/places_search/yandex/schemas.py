"""Pydantic models for Yandex Organization Search API responses."""

from __future__ import annotations

from datetime import time
from typing import Any

from pydantic import AliasPath, BaseModel, ConfigDict, Field, field_validator

_ALL_DAYS = frozenset(range(7))
_WEEKDAYS = frozenset(range(5))
_WEEKEND = frozenset({5, 6})


class YandexOrganisationPoint(BaseModel):
    """GeoJSON point. Yandex sends coordinates as ``[lon, lat]``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    coordinates: tuple[float, float]

    @field_validator("coordinates", mode="before")
    @classmethod
    def _parse_coordinates(cls, value: object) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("geometry.coordinates must contain longitude and latitude")

        lon, lat = value

        if isinstance(lon, bool) or isinstance(lat, bool):
            raise ValueError("coordinates must be numbers")

        try:
            parsed_lon = float(lon)
            parsed_lat = float(lat)
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinates must be numbers") from exc

        if not -180 <= parsed_lon <= 180:
            raise ValueError("longitude must be between -180 and 180")
        if not -90 <= parsed_lat <= 90:
            raise ValueError("latitude must be between -90 and 90")

        return parsed_lon, parsed_lat

    @property
    def lon(self) -> float:
        return self.coordinates[0]

    @property
    def lat(self) -> float:
        return self.coordinates[1]


class YandexCompanyAddress(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    formatted: str = Field(min_length=1)


class YandexCompanyCategory(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)


class YandexCompanyPhone(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    formatted: str = Field(min_length=1)


class YandexCompanyInterval(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    from_time: time = Field(validation_alias="from")
    to_time: time = Field(validation_alias="to")

    @field_validator("from_time", "to_time", mode="before")
    @classmethod
    def _parse_time(cls, value: object) -> time:
        if not isinstance(value, str):
            raise ValueError("working-hours interval must be a time string")
        try:
            parsed = time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("working-hours interval must use ISO time") from exc
        if parsed.tzinfo is not None:
            raise ValueError("working-hours interval must not include a time zone")
        return parsed


class YandexCompanyAvailability(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    everyday: bool = Field(default=False, validation_alias="Everyday")
    weekdays: bool = Field(default=False, validation_alias="Weekdays")
    weekend: bool = Field(default=False, validation_alias="Weekend")
    monday: bool = Field(default=False, validation_alias="Monday")
    tuesday: bool = Field(default=False, validation_alias="Tuesday")
    wednesday: bool = Field(default=False, validation_alias="Wednesday")
    thursday: bool = Field(default=False, validation_alias="Thursday")
    friday: bool = Field(default=False, validation_alias="Friday")
    saturday: bool = Field(default=False, validation_alias="Saturday")
    sunday: bool = Field(default=False, validation_alias="Sunday")
    twenty_four_hours: bool = Field(
        default=False,
        validation_alias="TwentyFourHours",
    )
    intervals: list[YandexCompanyInterval] = Field(
        default_factory=list,
        validation_alias="Intervals",
    )

    @property
    def days(self) -> frozenset[int]:
        """Return the weekdays covered by this availability; Monday is 0."""

        if self.everyday:
            return _ALL_DAYS

        days: set[int] = set()
        if self.weekdays:
            days.update(_WEEKDAYS)
        if self.weekend:
            days.update(_WEEKEND)

        individual_days = (
            self.monday,
            self.tuesday,
            self.wednesday,
            self.thursday,
            self.friday,
            self.saturday,
            self.sunday,
        )
        days.update(index for index, enabled in enumerate(individual_days) if enabled)
        return frozenset(days)


class YandexCompanyHours(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    text: str | None = None
    availabilities: list[YandexCompanyAvailability] = Field(
        default_factory=list,
        validation_alias="Availabilities",
    )


class YandexCompanyAccessibilityFeature(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    value: Any = None


class YandexCompanyMetaData(BaseModel):
    """Documented ``CompanyMetaData`` fields only."""

    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: YandexCompanyAddress = Field(validation_alias="Address")
    url: str | None = None
    categories: list[YandexCompanyCategory] = Field(
        default_factory=list,
        validation_alias="Categories",
    )
    phones: list[YandexCompanyPhone] = Field(
        default_factory=list,
        validation_alias="Phones",
    )
    hours: YandexCompanyHours | None = Field(
        default=None,
        validation_alias="Hours",
    )
    features: list[YandexCompanyAccessibilityFeature] = Field(
        default_factory=list,
        validation_alias="Features",
    )

    @property
    def open_24h(self) -> bool | None:
        if self.hours is None or not self.hours.availabilities:
            return None

        covered_days: set[int] = set()
        has_day_schedule = False

        for availability in self.hours.availabilities:
            days = availability.days
            if not days:
                continue

            has_day_schedule = True
            if availability.twenty_four_hours:
                covered_days.update(days)

        if not has_day_schedule:
            return None

        return covered_days == _ALL_DAYS


class YandexOrganisation(BaseModel):
    """One GeoJSON feature representing an organisation."""

    model_config = ConfigDict(extra="ignore", strict=True)

    company: YandexCompanyMetaData = Field(
        validation_alias=AliasPath("properties", "CompanyMetaData"),
    )
    point: YandexOrganisationPoint = Field(
        validation_alias=AliasPath("geometry"),
    )


class YandexOrganisationSearchResponse(BaseModel):
    """Top-level Yandex response for ``type=biz``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    found: int = Field(
        ge=0,
        validation_alias=AliasPath(
            "properties",
            "ResponseMetaData",
            "SearchResponse",
            "found",
        ),
    )
    organisations: list[YandexOrganisation] = Field(
        default_factory=list,
        validation_alias="features",
    )
