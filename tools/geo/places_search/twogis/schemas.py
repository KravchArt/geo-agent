"""Validated subsets of 2GIS Regions 2.0 and Places 3.0 responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TwoGisMeta(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    code: int
    api_version: str | None = None


class TwoGisPoint(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TwoGisNameEx(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    primary: str | None = None
    short_name: str | None = None
    extension: str | None = None
    legal_name: str | None = None


class TwoGisAdmDiv(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    is_default: bool = False


class TwoGisRubric(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    name: str = Field(min_length=1)
    alias: str | None = None
    kind: str | None = None


class TwoGisEntityRef(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    name: str | None = None


class TwoGisStructuredAddress(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    building_id: str | None = None


class TwoGisWorkingPeriod(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    from_time: str = Field(min_length=1, validation_alias="from")
    to_time: str = Field(min_length=1, validation_alias="to")


class TwoGisScheduleDay(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    working_hours: list[TwoGisWorkingPeriod] = Field(default_factory=list)


class TwoGisSchedule(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    mon: TwoGisScheduleDay | None = Field(default=None, validation_alias="Mon")
    tue: TwoGisScheduleDay | None = Field(default=None, validation_alias="Tue")
    wed: TwoGisScheduleDay | None = Field(default=None, validation_alias="Wed")
    thu: TwoGisScheduleDay | None = Field(default=None, validation_alias="Thu")
    fri: TwoGisScheduleDay | None = Field(default=None, validation_alias="Fri")
    sat: TwoGisScheduleDay | None = Field(default=None, validation_alias="Sat")
    sun: TwoGisScheduleDay | None = Field(default=None, validation_alias="Sun")
    is_24x7: bool = False
    comment: str | None = None
    description: str | None = None

    @property
    def days(self) -> tuple[TwoGisScheduleDay | None, ...]:
        return (self.mon, self.tue, self.wed, self.thu, self.fri, self.sat, self.sun)


class TwoGisReviews(BaseModel):
    """Provider review statistics requested through ``items.reviews``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    general_rating: float | None = Field(default=None, ge=0, le=5)
    general_review_count: int | None = Field(default=None, ge=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int | None = Field(default=None, ge=0)

    @field_validator("general_rating", "rating", mode="before")
    @classmethod
    def _parse_numeric_rating(cls, value: object) -> object:
        # The public 2GIS schema documents ratings as strings (for example,
        # "4.73"), while some responses return JSON numbers.
        return float(value) if isinstance(value, str) else value

    @property
    def display_rating(self) -> float | None:
        """Prefer the aggregate 2GIS rating over a source-specific rating."""

        return self.general_rating if self.general_rating is not None else self.rating

    @property
    def display_review_count(self) -> int | None:
        """Return the count belonging to the preferred aggregate rating."""

        if self.general_review_count is not None:
            return self.general_review_count
        return self.review_count


class TwoGisItem(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    # 2GIS catalog project id.  It is distinct from ``adm_div`` parent ids
    # and lets a city lookup verify that a text-search candidate belongs to
    # the coverage region selected through Regions API.
    region_id: str | None = None
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    subtype: str | None = None
    route_type: str | None = None
    address_name: str | None = None
    full_address_name: str | None = None
    full_name: str | None = None
    building_name: str | None = None
    structured_address: TwoGisStructuredAddress | None = Field(
        default=None,
        validation_alias="address",
    )
    point: TwoGisPoint | None = None
    name_ex: TwoGisNameEx | None = None
    adm_div: list[TwoGisAdmDiv] = Field(default_factory=list)
    rubrics: list[TwoGisRubric] = Field(default_factory=list)
    org: TwoGisEntityRef | None = None
    brand: TwoGisEntityRef | None = None
    schedule: TwoGisSchedule | None = None
    reviews: TwoGisReviews | None = None

    @property
    def aliases(self) -> tuple[str, ...]:
        # Building cards often put the user-facing landmark name in
        # ``building_name`` while ``name`` contains a type extension such as
        # "Алые паруса, жилой комплекс".  Treat both as searchable aliases.
        values: list[str | None] = [self.name, self.building_name]
        if self.name_ex is not None:
            values.extend((self.name_ex.primary, self.name_ex.short_name))

        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value is None:
                continue
            cleaned = " ".join(value.split())
            normalized = cleaned.casefold()
            if cleaned and normalized not in seen:
                seen.add(normalized)
                result.append(cleaned)
        return tuple(result)

    @property
    def locality_names(self) -> tuple[str, ...]:
        locality_types = {"city", "settlement", "village", "locality"}
        return tuple(item.name for item in self.adm_div if item.type in locality_types)

    @property
    def locality(self) -> str | None:
        localities = [
            item
            for item in self.adm_div
            if item.type in {"city", "settlement", "village", "locality"}
        ]
        if not localities:
            return None
        default = next((item for item in localities if item.is_default), None)
        return (default or localities[0]).name

    @property
    def address(self) -> str:
        return (
            self.full_address_name
            or self.address_name
            or self.full_name
            or self.locality
            or self.name
        )

    @property
    def category_names(self) -> list[str]:
        return [rubric.name for rubric in self.rubrics]


class TwoGisItemsResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    items: list[TwoGisItem] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class TwoGisItemsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    meta: TwoGisMeta
    result: TwoGisItemsResult


class TwoGisRubricSearchItem(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    alias: str | None = None


class TwoGisRubricSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    items: list[TwoGisRubricSearchItem] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class TwoGisRubricSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    meta: TwoGisMeta
    result: TwoGisRubricSearchResult


class TwoGisSatellite(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)


class TwoGisRegion(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    country_code: str | None = None
    settlements: list[str] = Field(default_factory=list)
    satellites: list[TwoGisSatellite] = Field(default_factory=list)


class TwoGisRegionsResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    items: list[TwoGisRegion] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class TwoGisRegionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    meta: TwoGisMeta
    result: TwoGisRegionsResult
