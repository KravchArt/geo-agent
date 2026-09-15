"""Run 100 live checks of the production 2GIS category conversion.

The matrix contains 50 canonical categories, each tested with an English and
a Russian input query. For every case the output shows:

    model category + input query -> production rubric lookup -> exact 2GIS rubric

By default the checks use the Moscow catalog because it supports the Russian
rubric directory exercised by the production conversion. This script calls
only the 2GIS Categories API; it does not geocode cities or search for places.

Run from the repository root:

    uv run python scripts/inspect_twogis_categories.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from backend.app.config import get_settings
from tools.base import ToolExecutionError
from tools.geo.places_search.category_normalization import (
    english_category_query,
    russian_category_query,
)
from tools.geo.places_search.schemas import PlaceCategory
from tools.geo.places_search.twogis.categories import twogis_rubric_lookup
from tools.geo.places_search.twogis.client import TwoGisSearchClient
from tools.observability import ToolExecutionContext

REGION_ID = "32"
COUNTRY_CODE = "ru"

# Exactly 50 categories x two input languages = 100 live Categories API calls.
CATEGORIES = tuple(PlaceCategory)[:50]


async def _inspect_one(
    *,
    http_client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    timeout_s: float,
    category: PlaceCategory,
    input_language: str,
) -> dict[str, Any]:
    input_query = (
        english_category_query(category.value)
        if input_language == "en"
        else russian_category_query(category.value)
    )
    lookup = twogis_rubric_lookup(
        category=category,
        query=input_query,
        country_code=COUNTRY_CODE,
    )

    # A fresh client deliberately avoids its production rubric cache: this is
    # a request matrix, so every row must perform one real Categories API call.
    client = TwoGisSearchClient(
        api_key=api_key,
        http_client=http_client,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    try:
        rubrics = await client.find_rubrics(
            query=lookup.query,
            region_id=REGION_ID,
            locale=lookup.locale,
            context=ToolExecutionContext(),
            # Unsupported mappings still make their diagnostic request, but
            # use strict exact matching rather than becoming production-ready.
            allowed_aliases=lookup.allowed_aliases or (),
        )
    except ToolExecutionError as exc:
        result: dict[str, Any] = {
            "status": "error",
            "error_code": exc.error_code.value,
            "provider_code": exc.provider_code,
            "message": str(exc),
        }
    else:
        result = (
            {
                "status": "matched",
                "rubrics": [
                    {"id": rubric.id, "name": rubric.name, "alias": rubric.alias}
                    for rubric in rubrics
                ],
            }
            if rubrics
            else {"status": "no_exact_match"}
        )

    return {
        "canonical_category": category.value,
        "input_language": input_language,
        "input_query": input_query,
        "lookup_query": lookup.query,
        "lookup_locale": lookup.locale,
        "mapping_status": ("supported" if lookup.allowed_aliases is not None else "unsupported"),
        "allowed_aliases": list(lookup.allowed_aliases or ()),
        "result": result,
    }


async def main() -> None:
    if len(CATEGORIES) != 50:
        raise RuntimeError("category matrix must contain exactly 50 categories")

    settings = get_settings()
    if settings.dgis_api_key is None:
        raise SystemExit("DGIS_API_KEY is required")

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
    ) as http_client:
        for category in CATEGORIES:
            for input_language in ("en", "ru"):
                row = await _inspect_one(
                    http_client=http_client,
                    api_key=settings.dgis_api_key,
                    base_url=settings.dgis_catalog_base_url,
                    timeout_s=float(settings.dgis_catalog_timeout),
                    category=category,
                    input_language=input_language,
                )
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    statuses = [str(row["result"]["status"]) for row in results]
    supported = [row for row in results if row["mapping_status"] == "supported"]
    print(
        json.dumps(
            {
                "summary": {
                    "requests": len(results),
                    "supported_requests": len(supported),
                    "unsupported_requests": len(results) - len(supported),
                    "matched": statuses.count("matched"),
                    "no_exact_match": statuses.count("no_exact_match"),
                    "errors": statuses.count("error"),
                }
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
