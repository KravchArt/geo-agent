"""Pydantic models for the Overpass JSON subset used by place search."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OsmCenter(BaseModel):
    """Representative coordinate returned by ``out center``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class OsmElement(BaseModel):
    """One OSM element or the pseudo-element emitted by ``out count``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    type: Literal["node", "way", "relation", "area", "count"]
    id: int = Field(ge=0)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    center: OsmCenter | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coordinate_pair_is_complete(self) -> OsmElement:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be returned together")
        return self

    @property
    def coordinates(self) -> tuple[float, float] | None:
        """Return ``(lon, lat)`` for nodes, ways, and relations."""

        if self.lat is not None and self.lon is not None:
            return self.lon, self.lat
        if self.center is not None:
            return self.center.lon, self.center.lat
        return None


class OsmResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    timestamp_osm_base: str | None = None
    copyright: str | None = None


class OsmOverpassResponse(BaseModel):
    """Validated Overpass JSON response."""

    model_config = ConfigDict(extra="ignore", strict=True)

    version: float
    generator: str
    osm3s: OsmResponseMetadata
    elements: list[OsmElement] = Field(default_factory=list)

    @property
    def total_found(self) -> int | None:
        """Return the total emitted by ``out count``."""

        for element in reversed(self.elements):
            if element.type != "count":
                continue
            value = element.tags.get("total")
            if value is None:
                return None
            try:
                total = int(value)
            except ValueError:
                return None
            return total if total >= 0 else None
        return None
