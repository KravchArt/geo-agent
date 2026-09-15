"""Run the configured routing coordinator from Alye Parusa to KLPK in Kirov.

There is no LLM or agent loop in this script. ``build_runtime_tools`` assembles
the production ``RoutingCoordinator`` with its configured text-place resolver
and routing-provider fallback chain; ``ToolExecutor`` only applies the normal
tool validation and result envelope around that coordinator.

Run from the repository root:

    uv run python scripts/inspect_kirov_route.py
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

ROUTING_ARGUMENTS = {
    "mode": "route",
    "waypoints": [
        {"query": "Алые Паруса", "city": "Киров, Кировская область"},
        {"query": "КЛПК", "city": "Киров, Кировская область"},
    ],
}


def _twogis_resolution_candidates(response: httpx.Response) -> dict[str, Any] | None:
    """Keep the identity/type fields from one 2GIS named-place response."""

    query = response.request.url.params.get("q")
    if query is None or "/3.0/items" not in response.request.url.path:
        return None
    try:
        payload = response.json()
    except ValueError:
        return {"query": query, "status_code": response.status_code, "candidates": []}

    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    items = result.get("items", []) if isinstance(result, dict) else []
    candidates: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rubrics = item.get("rubrics", [])
        candidates.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "full_address_name": item.get("full_address_name"),
                "type": item.get("type"),
                "subtype": item.get("subtype"),
                "route_type": item.get("route_type"),
                "rubrics": [
                    {
                        "name": rubric.get("name"),
                        "alias": rubric.get("alias"),
                        "kind": rubric.get("kind"),
                    }
                    for rubric in rubrics
                    if isinstance(rubric, dict)
                ],
                "org": item.get("org"),
                "brand": item.get("brand"),
                "name_ex": item.get("name_ex"),
                "point": item.get("point"),
            }
        )
    return {
        "query": query,
        "status_code": response.status_code,
        "total": result.get("total") if isinstance(result, dict) else None,
        "candidates": candidates,
    }


def _resolved_refs(result_data: dict[str, Any] | None, clarification: Any) -> list[str]:
    """Collect selected route refs or ambiguity-option refs for store inspection."""

    refs: list[str] = []
    if result_data is not None:
        route = result_data.get("route")
        if isinstance(route, dict):
            for waypoint in route.get("waypoints", []):
                if isinstance(waypoint, dict) and isinstance(waypoint.get("ref"), str):
                    refs.append(waypoint["ref"])
    if clarification is not None:
        for option in clarification.options:
            if isinstance(option.value, str) and option.value.startswith("plc_"):
                refs.append(option.value)
    return list(dict.fromkeys(refs))


async def main() -> None:
    settings = get_settings()
    if not settings.routing_providers:
        raise SystemExit("ROUTING_PROVIDERS is empty")

    place_store = InMemoryPlaceStore()
    resolution_results: list[dict[str, Any]] = []

    async def capture_resolution_response(response: httpx.Response) -> None:
        await response.aread()
        captured = _twogis_resolution_candidates(response)
        if captured is not None:
            resolution_results.append(captured)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
        event_hooks={"response": [capture_resolution_response]},
    ) as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
            source_store=InMemorySourceStore(),
        )
        if "routing_tool" not in runtime_tools:
            raise SystemExit("routing_tool was not built from the current configuration")

        result = await ToolExecutor(runtime_tools).run(
            "routing_tool",
            ROUTING_ARGUMENTS,
        )

    selected_records = []
    for place_ref in _resolved_refs(result.data, result.clarification):
        record = await place_store.get(place_ref)
        if record is not None:
            selected_records.append(record.model_dump(mode="json"))

    report = {
        "routing_arguments": ROUTING_ARGUMENTS,
        "configured_providers": {
            "place_resolution": settings.text_place_resolution_providers,
            "routing": settings.routing_providers,
        },
        "resolution_results": resolution_results,
        "selected_place_records": selected_records,
        "result": result.model_dump(mode="json"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
