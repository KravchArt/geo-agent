"""Inspect raw provider responses for resolving Moscow's Red Square without an LLM.

The script sends one ``places_search(mode="resolve")`` request through the
configured runtime stack and records both upstream API payloads and the final
normalised tool result. API credentials are redacted from printed URLs.

Run from the repository root:

    uv run python scripts/inspect_red_square_resolution.py
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

TOOL_ARGUMENTS = {
    "mode": "resolve",
    "city": "Москва",
    "organisations": [
        {
            "client_id": "red_square",
            "name": "Красная площадь",
        }
    ],
}

_SECRET_QUERY_KEYS = {"key", "apikey", "api_key", "token", "access_token"}


def _safe_request(response: httpx.Response) -> dict[str, Any]:
    params = {
        key: "<redacted>" if key.lower() in _SECRET_QUERY_KEYS else value
        for key, value in response.request.url.params.multi_items()
    }
    return {
        "method": response.request.method,
        "origin": str(response.request.url.copy_with(query=None)),
        "params": params,
    }


async def main() -> None:
    settings = get_settings()
    if not settings.places_search_providers:
        raise SystemExit("PLACES_SEARCH_PROVIDERS is empty")

    upstream_responses: list[dict[str, Any]] = []

    async def capture_response(response: httpx.Response) -> None:
        await response.aread()
        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"non_json_body": response.text[:2_000]}
        upstream_responses.append(
            {
                "request": _safe_request(response),
                "status_code": response.status_code,
                "response": payload,
            }
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
        event_hooks={"response": [capture_response]},
    ) as http_client:
        runtime_tools = build_runtime_tools(
            settings=settings,
            http_client=http_client,
            place_store=InMemoryPlaceStore(),
            source_store=InMemorySourceStore(),
        )
        tool_result = await ToolExecutor(runtime_tools).run("places_search", TOOL_ARGUMENTS)

    report = {
        "tool_arguments": TOOL_ARGUMENTS,
        "configured_providers": settings.places_search_providers,
        "upstream_responses": upstream_responses,
        "tool_result": tool_result.model_dump(mode="json"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
