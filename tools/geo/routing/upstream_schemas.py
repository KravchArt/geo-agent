"""Shared primitives for strict routing API response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegativeNumber = Annotated[
    int | float,
    Field(ge=0, allow_inf_nan=False),
]


class RoutingUpstreamModel(BaseModel):
    """Strict response model that tolerates provider fields we do not consume."""

    model_config = ConfigDict(extra="ignore", strict=True)
