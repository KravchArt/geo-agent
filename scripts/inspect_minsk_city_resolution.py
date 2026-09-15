"""Run the configured places-search stack for English ``Minsk`` without an LLM.

The call enters through ``ToolExecutor`` and the normal ``places_search`` tool,
so it uses the project's configured provider clients and its
``PlacesSearchCoordinator`` resolution/fallback policy.

Run from the repository root:

    uv run python scripts/inspect_minsk_city_resolution.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from backend.app.config import get_settings
from backend.app.tools import build_runtime_tools
from tools import ToolExecutor
from tools.geo.place_store import InMemoryPlaceStore
from tools.web import InMemorySourceStore


async def main() -> None:
    settings = get_settings()
    if not settings.places_search_providers:
        raise SystemExit(
            "PLACES_SEARCH_PROVIDERS is empty; configure at least one provider before running this."
        )

    place_store = InMemoryPlaceStore()
    geocoder_candidates: list[dict[str, Any]] = []

    async def capture_geocoder_response(response: httpx.Response) -> None:
        """Keep the ranking fields that ToolExecutor intentionally hides."""

        if "/geocode/" not in response.request.url.path:
            return
        await response.aread()
        try:
            payload = response.json()
            results = payload.get("results", []) if isinstance(payload, dict) else []
        except ValueError:
            return

        for candidate in results:
            if not isinstance(candidate, dict):
                continue
            address = candidate.get("address")
            confidence = candidate.get("matchConfidence")
            geocoder_candidates.append(
                {
                    "id": candidate.get("id"),
                    "type": candidate.get("type"),
                    "entity_type": candidate.get("entityType"),
                    # This is the value copied to PlaceMatch.relevance_score.
                    "score": candidate.get("score"),
                    "match_confidence_score": (
                        confidence.get("score") if isinstance(confidence, dict) else None
                    ),
                    "address": address,
                }
            )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
        event_hooks={"response": [capture_geocoder_response]},
    ) as http_client:
        tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
            source_store=InMemorySourceStore(),
        )
        result = await ToolExecutor(tools).run(
            "places_search",
            {
                "mode": "area",
                "city": "Minsk",
                "query": "coffee shop",
                "limit": 1,
            },
        )

    report: dict[str, Any] = {
        "configured_providers": settings.places_search_providers,
        "tomtom_geocoder_candidates": geocoder_candidates,
        "tool_result": result.model_dump(mode="json"),
    }
    if result.ok and result.data is not None and (area := result.data.get("area")) is not None:
        area_record = await place_store.get(area["ref"])
        report["resolved_area_record"] = (
            area_record.model_dump(mode="json") if area_record is not None else None
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
