"""Inspect one named-place search without routing or an LLM.

The request enters through the production ``places_search`` tool and its
configured ``PlacesSearchCoordinator``. Raw 2GIS candidates are retained for
comparison with the normalized result. The first output fields summarize the
objects that the production tool ultimately returned. ``PLACES_ARGUMENTS`` may
use either an area discovery or a one-entity resolve call.

Run from the repository root:

    uv run python scripts/inspect_place_resolution.py
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
from tools.geo.places_search.twogis.matching import build_named_candidates
from tools.geo.places_search.twogis.schemas import TwoGisItemsResponse
from tools.web import InMemorySourceStore

PLACES_ARGUMENTS = {
    "mode": "area",
    "query": "restaurants",
    "category": "restaurant",
    "area": "Moscow",
}

# Set true only when raw provider cards and the complete ToolResult are needed.
INCLUDE_DIAGNOSTICS = False


def _requested_name_and_city() -> tuple[str, str]:
    mode = PLACES_ARGUMENTS.get("mode")
    if mode == "resolve":
        organisations = PLACES_ARGUMENTS.get("organisations")
        if not isinstance(organisations, list) or not organisations:
            raise ValueError("resolve requires a non-empty organisations list")
        first = organisations[0]
        if not isinstance(first, dict) or not isinstance(first.get("name"), str):
            raise ValueError("the first organisation must have a string name")
        requested_name = first["name"]
    else:
        query = PLACES_ARGUMENTS.get("query")
        if not isinstance(query, str):
            raise ValueError("area inspection requires a string query")
        requested_name = query

    area = PLACES_ARGUMENTS.get("area", PLACES_ARGUMENTS.get("city"))
    if not isinstance(area, str) or area.startswith("plc_"):
        raise ValueError("inspection requires a textual area/city")
    return requested_name, area


def _capture_twogis_candidates(response: httpx.Response) -> dict[str, Any] | None:
    query = response.request.url.params.get("q")
    if query is None or "/3.0/items" not in response.request.url.path:
        return None
    try:
        payload = response.json()
    except ValueError:
        return {"query": query, "status_code": response.status_code, "candidates": []}

    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    items = result.get("items", []) if isinstance(result, dict) else []
    parsed = TwoGisItemsResponse.model_validate(payload)
    requested_name, requested_city = _requested_name_and_city()
    normalized_candidates = build_named_candidates(
        parsed.result.items,
        query=requested_name,
        city=requested_city,
    )
    return {
        "query": query,
        "status_code": response.status_code,
        "total": result.get("total") if isinstance(result, dict) else None,
        "candidates": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "full_address_name": item.get("full_address_name"),
                "type": item.get("type"),
                "subtype": item.get("subtype"),
                "route_type": item.get("route_type"),
                "rubrics": item.get("rubrics", []),
                "org": item.get("org"),
                "brand": item.get("brand"),
                "name_ex": item.get("name_ex"),
                "point": item.get("point"),
                "address": item.get("address"),
            }
            for item in items
            if isinstance(item, dict)
        ],
        "after_filtering_and_deduplication": [
            {
                "provider_rank": candidate.provider_rank,
                "id": candidate.item.id,
                "name": candidate.item.name,
                "semantic_kind": candidate.semantic_kind,
                "building_id": (
                    candidate.item.structured_address.building_id
                    if candidate.item.structured_address is not None
                    else None
                ),
                "point": (
                    candidate.item.point.model_dump(mode="json")
                    if candidate.item.point is not None
                    else None
                ),
            }
            for candidate in normalized_candidates
        ],
    }


def _result_refs(result_payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    result_data = result_payload.get("data")
    if not isinstance(result_data, dict):
        result_data = {}
    for place in result_data.get("places", []):
        if isinstance(place, dict) and isinstance(place.get("ref"), str):
            refs.append(place["ref"])
    for outcome in result_data.get("resolved", []):
        if not isinstance(outcome, dict):
            continue
        place = outcome.get("place")
        if isinstance(place, dict) and isinstance(place.get("ref"), str):
            refs.append(place["ref"])
        for option in outcome.get("options", []):
            if isinstance(option, dict) and isinstance(option.get("ref"), str):
                refs.append(option["ref"])
    clarification = result_payload.get("clarification")
    if isinstance(clarification, dict):
        for option in clarification.get("options", []):
            if (
                isinstance(option, dict)
                and isinstance(option.get("value"), str)
                and option["value"].startswith("plc_")
            ):
                refs.append(option["value"])
    return list(dict.fromkeys(refs))


def _final_outcome(result_payload: dict[str, Any]) -> str:
    if not result_payload.get("ok"):
        return "clarification" if result_payload.get("clarification") else "error"
    data = result_payload.get("data")
    if not isinstance(data, dict):
        return "empty"
    if data.get("places"):
        return "places"
    resolved = data.get("resolved")
    if isinstance(resolved, list) and resolved:
        statuses = {item.get("status") for item in resolved if isinstance(item, dict)}
        if statuses == {"not_found"}:
            return "not_found"
        if "ambiguous" in statuses:
            return "clarification"
        return "resolved"
    return "not_found"


def _final_objects(
    result_payload: dict[str, Any],
    records_by_ref: dict[str, dict[str, Any]],
    semantic_kinds_by_provider_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Return only objects exposed by the final production tool outcome."""

    objects: list[dict[str, Any]] = []

    def add(
        place: dict[str, Any],
        *,
        disposition: str,
        client_id: str | None = None,
        input_name: str | None = None,
    ) -> None:
        place_ref = place.get("ref")
        if not isinstance(place_ref, str):
            return
        record = records_by_ref.get(place_ref, {})
        objects.append(
            {
                "disposition": disposition,
                "client_id": client_id,
                "input_name": input_name,
                "ref": place_ref,
                "provider": record.get("provider"),
                "provider_id": record.get("provider_id", place.get("id")),
                "name": place.get("name", record.get("name")),
                "address": place.get("address", record.get("address")),
                "semantic_kind": record.get("kind")
                or semantic_kinds_by_provider_id.get(str(record.get("provider_id", ""))),
                "categories": place.get("categories", []),
                "rating": place.get("rating"),
                "review_count": place.get("review_count"),
                "hours_text": place.get("hours_text"),
                "coordinates": (
                    {"lat": record.get("lat"), "lon": record.get("lon")}
                    if record.get("lat") is not None and record.get("lon") is not None
                    else None
                ),
            }
        )

    data = result_payload.get("data")
    if isinstance(data, dict):
        for place in data.get("places", []):
            if isinstance(place, dict):
                add(place, disposition="returned")
        for resolution in data.get("resolved", []):
            if not isinstance(resolution, dict):
                continue
            common = {
                "client_id": resolution.get("client_id"),
                "input_name": resolution.get("input_name"),
            }
            place = resolution.get("place")
            if isinstance(place, dict):
                add(place, disposition="resolved", **common)
            for option in resolution.get("options", []):
                if isinstance(option, dict):
                    add(option, disposition="clarification_option", **common)

    clarification = result_payload.get("clarification")
    if isinstance(clarification, dict):
        for option in clarification.get("options", []):
            if not isinstance(option, dict):
                continue
            value = option.get("value")
            if not isinstance(value, str) or not value.startswith("plc_"):
                continue
            record = records_by_ref.get(value, {})
            add(
                {
                    "ref": value,
                    "name": option.get("label", record.get("name")),
                    "address": option.get("description", record.get("address")),
                },
                disposition="clarification_option",
            )
    return objects


async def main() -> None:
    settings = get_settings()
    if not settings.places_search_providers:
        raise SystemExit("PLACES_SEARCH_PROVIDERS is empty")

    raw_twogis_results: list[dict[str, Any]] = []

    async def capture_response(response: httpx.Response) -> None:
        await response.aread()
        captured = _capture_twogis_candidates(response)
        if captured is not None:
            raw_twogis_results.append(captured)

    place_store = InMemoryPlaceStore()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
        event_hooks={"response": [capture_response]},
    ) as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
            source_store=InMemorySourceStore(),
        )
        if "places_search" not in runtime_tools:
            raise SystemExit("places_search was not built from the current configuration")
        result = await ToolExecutor(runtime_tools).run("places_search", PLACES_ARGUMENTS)

    result_payload = result.model_dump(mode="json")
    semantic_kinds_by_provider_id = {
        str(candidate["id"]): str(candidate["semantic_kind"])
        for response in raw_twogis_results
        for candidate in response.get("after_filtering_and_deduplication", [])
        if isinstance(candidate, dict)
        and candidate.get("id") is not None
        and candidate.get("semantic_kind") is not None
    }
    records_by_ref: dict[str, dict[str, Any]] = {}
    for place_ref in _result_refs(result_payload):
        record = await place_store.get(place_ref)
        if record is not None:
            records_by_ref[place_ref] = record.model_dump(mode="json")

    output: dict[str, Any] = {
        "places_arguments": PLACES_ARGUMENTS,
        "final_outcome": _final_outcome(result_payload),
        "final_objects": _final_objects(
            result_payload,
            records_by_ref,
            semantic_kinds_by_provider_id,
        ),
        "final_area": (
            result_payload.get("data", {}).get("area")
            if isinstance(result_payload.get("data"), dict)
            else None
        ),
    }
    if INCLUDE_DIAGNOSTICS:
        output["diagnostics"] = {
            "configured_providers": settings.places_search_providers,
            "raw_twogis_results": raw_twogis_results,
            "stored_records": list(records_by_ref.values()),
            "tool_result": result_payload,
        }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
