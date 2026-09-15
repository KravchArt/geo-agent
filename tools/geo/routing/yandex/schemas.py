"""Validated subsets of Yandex routing API responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.geo.routing.schemas import TrafficType, TransportMode


class _YandexRoutingModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class YandexRouteStep(_YandexRoutingModel):
    length: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(ge=0, allow_inf_nan=False)
    mode: TransportMode = Field(strict=False)


class YandexRouteLeg(_YandexRoutingModel):
    status: Literal["OK", "FAIL"]
    steps: list[YandexRouteStep] = Field(default_factory=list)


class YandexRouteFlags(_YandexRoutingModel):
    has_tolls: bool = Field(default=False, validation_alias="hasTolls")


class YandexRoute(_YandexRoutingModel):
    legs: list[YandexRouteLeg]
    flags: YandexRouteFlags = Field(default_factory=YandexRouteFlags)


class YandexRouteOptimization(_YandexRoutingModel):
    waypoints_order: list[int]


class YandexRouteResponse(_YandexRoutingModel):
    traffic_type: TrafficType | None = Field(default=None, strict=False)
    route: YandexRoute
    optimization: YandexRouteOptimization | None = None


class YandexMatrixValue(_YandexRoutingModel):
    value: int = Field(ge=0)


class YandexMatrixElement(_YandexRoutingModel):
    status: Literal["OK", "FAIL"]
    distance: YandexMatrixValue | None = None
    duration: YandexMatrixValue | None = None

    @model_validator(mode="after")
    def _successful_element_has_costs(self) -> YandexMatrixElement:
        if self.status == "OK" and (self.distance is None or self.duration is None):
            raise ValueError("matrix element with status=OK requires distance and duration")
        return self


class YandexMatrixRow(_YandexRoutingModel):
    elements: list[YandexMatrixElement]


class YandexDistanceMatrixResponse(_YandexRoutingModel):
    rows: list[YandexMatrixRow]
