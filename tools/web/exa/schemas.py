"""Validated subset of the current Exa Search API response."""

from pydantic import BaseModel, ConfigDict, Field


class ExaSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    url: str = Field(min_length=1)
    published_date: str | None = Field(default=None, alias="publishedDate")
    highlights: list[str] = Field(default_factory=list)


class ExaSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[ExaSearchResult]
    request_id: str = Field(alias="requestId", min_length=1)
