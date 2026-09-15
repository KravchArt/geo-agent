"""Live orchestrator check for a deliberately large TomTom places search.

This test uses only production project components: ``Orchestrator``,
``ToolExecutor``, ``build_runtime_tools``, the shared Yandex-backed place
resolver, the configured TomTom provider and the real model-facing
``places_search`` wrapper. No upstream response is mocked.

Run from the repository root with live provider keys configured in ``.env``::

    RUN_LIVE_TOMTOM_HEAVY=1 uv run pytest -q \
        tools/tests/test_tomtom_heavy_orchestrator_live.py -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from backend.app.config import Settings
from backend.app.llm.base import LLMClient
from backend.app.services.orchestrator import Orchestrator, OrchestratorRun
from backend.app.tools import build_runtime_tools
from common.models import (
    LLMRequest,
    LLMResponse,
    ReActStep,
    ReActStepType,
    ReActTrace,
)
from tools import ToolExecutor
from tools.geo import PlacesSearchOutput
from tools.geo.place_store import InMemoryPlaceStore
from tools.registry import get_tool_spec
from tools.web import InMemorySourceStore

HEAVY_TOMTOM_QUERY: dict[str, Any] = {
    "mode": "near",
    "query": "аптеки",
    "category": "pharmacy",
    "near_query": "Парк Горького",
    "city": "Москва",
    "radius_m": 10_000,
    "open_now": True,
    "limit": 20,
}


class _HeavyTomTomScenarioLLM(LLMClient):
    """Deterministically choose the heavy call; providers remain completely live."""

    mode = "mock"
    model = "tomtom-heavy-live-scenario"

    @staticmethod
    def _response(trace: ReActTrace) -> LLMResponse:
        return LLMResponse(
            model=_HeavyTomTomScenarioLLM.model,
            mode="mock",
            content=trace.final_answer or "",
            trace=trace,
        )

    @classmethod
    def _action(cls, arguments: dict[str, Any], *, call_id: str) -> LLMResponse:
        return cls._response(
            ReActTrace(
                steps=[
                    ReActStep(
                        type=ReActStepType.ACTION,
                        content="Call places_search with the requested live TomTom workload.",
                        tool_name="places_search",
                        tool_input=dict(arguments),
                        tool_call_id=call_id,
                    )
                ]
            )
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        observations = [message for message in request.messages if message.role == "tool"]
        if not observations:
            return self._action(
                HEAVY_TOMTOM_QUERY,
                call_id="live_heavy_tomtom_search",
            )

        latest_observation = observations[-1].content
        try:
            observation = json.loads(latest_observation)
        except json.JSONDecodeError:
            answer = f"places_search failed: {latest_observation}"
        else:
            count = observation.get("returned_count", 0)
            answer = f"TomTom вернул {count} открытых сейчас аптек около Парка Горького."

        return self._response(
            ReActTrace(
                steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=answer)],
                final_answer=answer,
            )
        )

    async def health(self) -> bool:
        return True


async def _run_live_search(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: InMemoryPlaceStore,
) -> OrchestratorRun:
    runtime_tools = build_runtime_tools(
        settings=settings,
        http_client=http_client,
        place_store=place_store,
        source_store=InMemorySourceStore(),
    )
    assert "places_search" in runtime_tools

    orchestrator = Orchestrator(
        llm=_HeavyTomTomScenarioLLM(),
        tool_executor=ToolExecutor(runtime_tools),
        tool_specs=[get_tool_spec(name) for name in runtime_tools],
        max_steps=2,
    )
    return await orchestrator.run(
        "Найди в Москве до 20 аптек, которые открыты прямо сейчас, "
        "в радиусе 10 километров от Парка Горького."
    )


def _provider_calls(run: OrchestratorRun) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for sequence, upstream_call in enumerate(
        (
            upstream_call
            for tool_call in run.tool_calls
            if tool_call.result.metrics is not None
            for upstream_call in tool_call.result.metrics.upstream_calls
        ),
        start=1,
    ):
        calls.append(
            {
                "sequence": sequence,
                "provider": upstream_call.provider,
                "operation": upstream_call.operation,
                "outcome": upstream_call.outcome.value,
                "status_code": upstream_call.status_code,
                "latency_ms": upstream_call.latency_ms,
                "error_code": upstream_call.error_code,
                "failure_kind": upstream_call.failure_kind,
                "retryable": upstream_call.retryable,
            }
        )
    return calls


@pytest.mark.live_places
async def test_heavy_tomtom_query_runs_through_the_existing_orchestrator() -> None:
    if os.environ.get("RUN_LIVE_TOMTOM_HEAVY") != "1":
        pytest.skip("set RUN_LIVE_TOMTOM_HEAVY=1 to call the real TomTom API")

    settings = Settings(app_env="dev")
    if "tomtom" not in settings.places_search_providers:
        pytest.skip("configure TomTom in PLACES_SEARCH_PROVIDERS")

    place_store = InMemoryPlaceStore()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
    ) as http_client:
        run = await _run_live_search(
            settings=settings,
            http_client=http_client,
            place_store=place_store,
        )

    assert len(run.tool_calls) == 1
    tool_call = run.tool_calls[0]
    assert tool_call.tool_name == "places_search"
    assert tool_call.arguments == HEAVY_TOMTOM_QUERY
    assert tool_call.result.ok, tool_call.result.model_dump_json(indent=2)
    assert tool_call.result.data is not None

    output = PlacesSearchOutput.model_validate(tool_call.result.data)
    assert output.anchor is not None
    anchor_record = await place_store.get(output.anchor)
    assert anchor_record is not None
    assert anchor_record.provider == "tomtom"
    assert len(output.places) <= HEAVY_TOMTOM_QUERY["limit"]
    assert all(place.is_open_now is True for place in output.places)

    for place in output.places:
        record = await place_store.get(place.ref)
        assert record is not None
        assert record.provider == "tomtom"

    assert tool_call.result.metrics is not None
    upstream_calls = tool_call.result.metrics.upstream_calls
    assert any(call.provider == "yandex_geocoder" for call in upstream_calls)
    assert sum(call.provider == "tomtom_search" for call in upstream_calls) == 2
    assert any(
        call.provider == "tomtom_search" and call.status_code == 200 for call in upstream_calls
    )
    assert not any(call.provider == "osm_overpass" for call in upstream_calls)

    print(
        json.dumps(
            {
                "request": HEAVY_TOMTOM_QUERY,
                "configured_places_providers": settings.places_search_providers,
                "provider_calls": _provider_calls(run),
                "tool_attempts": [
                    {
                        "arguments": call.arguments,
                        "result": call.result.model_dump(mode="json"),
                    }
                    for call in run.tool_calls
                ],
                "answer": run.llm_response.content,
                "places_search": tool_call.result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
