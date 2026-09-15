"""Pydantic models for Yandex Maps Geocoder API responses."""

from __future__ import annotations

from pydantic import (
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


def _parse_coordinate_pair(value: object, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    parts = value.split()
    if len(parts) != 2:
        raise ValueError(f"{field_name} must contain longitude and latitude")

    lon = float(parts[0])
    lat = float(parts[1])

    if not -180 <= lon <= 180:
        raise ValueError("longitude must be between -180 and 180")
    if not -90 <= lat <= 90:
        raise ValueError("latitude must be between -90 and 90")

    return lon, lat


class YandexPoint(BaseModel):
    """Coordinates from Yandex ``Point.pos`` in ``longitude latitude`` order."""

    model_config = ConfigDict(extra="ignore")

    pos: tuple[float, float]

    @field_validator("pos", mode="before")
    @classmethod
    def _parse_pos(cls, value: object) -> tuple[float, float]:
        return _parse_coordinate_pair(value, field_name="Point.pos")

    @property
    def lon(self) -> float:
        return self.pos[0]

    @property
    def lat(self) -> float:
        return self.pos[1]


class YandexEnvelope(BaseModel):
    """Yandex ``boundedBy.Envelope`` in longitude/latitude order."""

    model_config = ConfigDict(extra="ignore", strict=True)

    lower_corner: tuple[float, float] = Field(validation_alias="lowerCorner")
    upper_corner: tuple[float, float] = Field(validation_alias="upperCorner")

    @field_validator("lower_corner", "upper_corner", mode="before")
    @classmethod
    def _parse_corner(cls, value: object, info: ValidationInfo) -> tuple[float, float]:
        return _parse_coordinate_pair(value, field_name=info.field_name or "corner")

    @model_validator(mode="after")
    def _check_corner_order(self) -> YandexEnvelope:
        if self.lower_corner[0] >= self.upper_corner[0]:
            raise ValueError("lowerCorner longitude must be less than upperCorner longitude")
        if self.lower_corner[1] >= self.upper_corner[1]:
            raise ValueError("lowerCorner latitude must be less than upperCorner latitude")
        return self

    @property
    def west(self) -> float:
        return self.lower_corner[0]

    @property
    def south(self) -> float:
        return self.lower_corner[1]

    @property
    def east(self) -> float:
        return self.upper_corner[0]

    @property
    def north(self) -> float:
        return self.upper_corner[1]


class YandexAddressComponent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    kind: str = Field(min_length=1)
    name: str = Field(min_length=1)


class YandexAddress(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    components: list[YandexAddressComponent] = Field(
        default_factory=list,
        validation_alias="Components",
    )

    def component_name(self, kind: str) -> str | None:
        return next(
            (component.name for component in self.components if component.kind == kind),
            None,
        )


class YandexGeocoderMetaData(BaseModel):
    """Address and match metadata for one Yandex geocoding candidate."""

    model_config = ConfigDict(extra="ignore", strict=True)

    kind: str = Field(min_length=1)
    text: str = Field(min_length=1)
    precision: str | None = None
    address: YandexAddress | None = Field(default=None, validation_alias="Address")

    @property
    def locality(self) -> str | None:
        if self.address is None:
            return None
        return self.address.component_name("locality")


class YandexGeoObject(BaseModel):
    """One geocoding candidate in Yandex ``featureMember``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    uri: str | None = None
    geocoder_metadata: YandexGeocoderMetaData = Field(
        validation_alias=AliasPath("metaDataProperty", "GeocoderMetaData"),
    )
    point: YandexPoint = Field(
        validation_alias="Point",
    )
    bounds: YandexEnvelope | None = Field(
        default=None,
        validation_alias=AliasPath("boundedBy", "Envelope"),
    )


class YandexGeocoderResponse(BaseModel):
    """Top-level response returned by Yandex Geocoder API."""

    model_config = ConfigDict(extra="ignore", strict=True)

    candidates: list[YandexGeoObject] = Field(
        default_factory=list,
        validation_alias=AliasPath(
            "response",
            "GeoObjectCollection",
            "featureMember",
        ),
    )

    @field_validator("candidates", mode="before")
    @classmethod
    def _unwrap_feature_members(cls, value: object) -> list[object]:
        if not isinstance(value, list):
            raise ValueError("featureMember must be a list")

        candidates: list[object] = []

        for member in value:
            if not isinstance(member, dict):
                raise ValueError("featureMember item must be an object")

            geo_object = member.get("GeoObject")
            if geo_object is None:
                raise ValueError("featureMember item must contain GeoObject")

            candidates.append(geo_object)

        return candidates
