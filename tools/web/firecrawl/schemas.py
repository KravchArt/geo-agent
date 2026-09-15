"""Validated subset of the Firecrawl v2 Search API response."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FirecrawlWebResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    description: str = ""
    url: str = Field(min_length=1)


class FirecrawlNewsResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    snippet: str = ""
    url: str = Field(min_length=1)
    date: str | None = None


class FirecrawlSearchData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    web: list[FirecrawlWebResult] = Field(default_factory=list)
    news: list[FirecrawlNewsResult] = Field(default_factory=list)


class FirecrawlSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: Literal[True]
    data: FirecrawlSearchData
    id: str = Field(min_length=1)
