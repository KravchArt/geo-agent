from pydantic import BaseModel, ConfigDict, Field


class TavilySearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = None
    published_date: str | None = None


class TavilySearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    results: list[TavilySearchResult] = Field(default_factory=list)
