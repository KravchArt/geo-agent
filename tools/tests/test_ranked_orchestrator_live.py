"""Live end-to-end check for ``routing_tool(mode="rank")``.

The test deliberately drives the production assembly path instead of wiring a
routing provider by hand:

* :func:`backend.app.tools.build_runtime_tools` builds the configured place
  search providers, routing providers, shared geocoder and coordinators;
* :class:`backend.app.services.orchestrator.Orchestrator` executes the same
  model-facing tool loop as the application;
* ``places_search`` mints the candidate ``plc_`` refs that are passed to
  ``routing_tool`` on the next model turn;
* the routing tool geocodes the textual origin and ranks those candidates with
  the configured matrix provider.

The tiny LLM below is only a deterministic scenario driver. It does not replace
a geocoder, place-search provider, routing provider, coordinator, tool wrapper
or store. Every upstream response comes from the live providers configured in
``.env``.

Run from the repository root with provider keys configured in ``.env``::

    RUN_LIVE_RANKED=1 uv run pytest -q \
        tools/tests/test_ranked_orchestrator_live.py -s
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
from tools.geo import PlacesSearchOutput, RoutingOutput
from tools.geo.place_store import InMemoryPlaceStore
from tools.registry import get_tool_spec
from tools.web import InMemorySourceStore


class _RankedScenarioLLM(LLMClient):
    """Choose the two required tool calls while preserving their real data flow."""

    mode = "mock"
    model = "ranked-live-scenario"

    def __init__(self) -> None:
        self.candidate_refs: list[str] = []
        self.anchor_ref: str | None = None

    @staticmethod
    def _action(tool_name: str, arguments: dict[str, Any], call_id: str) -> LLMResponse:
        return LLMResponse(
            model=_RankedScenarioLLM.model,
            mode="mock",
            content="",
            trace=ReActTrace(
                steps=[
                    ReActStep(
                        type=ReActStepType.ACTION,
                        content=f"Call {tool_name}.",
                        tool_name=tool_name,
                        tool_input=arguments,
                        tool_call_id=call_id,
                    )
                ]
            ),
        )

    @staticmethod
    def _final(content: str) -> LLMResponse:
        return LLMResponse(
            model=_RankedScenarioLLM.model,
            mode="mock",
            content=content,
            trace=ReActTrace(
                steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=content)],
                final_answer=content,
            ),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        observations = [message for message in request.messages if message.role == "tool"]

        if not observations:
            return self._action(
                "places_search",
                {
                    "mode": "near",
                    "query": "рестораны",
                    "category": "restaurant",
                    "near_query": "площадь Минина и Пожарского",
                    "city": "Нижний Новгород",
                    "radius_m": 3_000,
                    "limit": 5,
                },
                "live_places_search",
            )

        if len(observations) == 1:
            try:
                place_observation = json.loads(observations[0].content)
            except json.JSONDecodeError:
                return self._final(f"places_search failed: {observations[0].content}")

            anchor_ref = place_observation.get("anchor")
            if not isinstance(anchor_ref, str) or not anchor_ref.startswith("plc_"):
                return self._final("places_search returned no anchor ref for ranking")
            self.anchor_ref = anchor_ref

            places = place_observation.get("places", [])
            self.candidate_refs = [
                place["ref"]
                for place in places
                if isinstance(place, dict)
                and isinstance(place.get("ref"), str)
                and place["ref"].startswith("plc_")
            ]
            if not self.candidate_refs:
                return self._final("places_search returned no candidates to rank")

            return self._action(
                "routing_tool",
                {
                    "mode": "rank",
                    "transport": "walking",
                    "origins": [self.anchor_ref],
                    "candidates": self.candidate_refs,
                    "optimize_by": "distance",
                    "aggregate": "min",
                    "limit": min(3, len(self.candidate_refs)),
                },
                "live_rank_candidates",
            )

        return self._final(
            "Рестораны ранжированы по расстоянию пешком от площади Минина и Пожарского."
        )

    async def health(self) -> bool:
        return True


async def _run_ranked_scenario(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> tuple[OrchestratorRun, _RankedScenarioLLM, InMemoryPlaceStore]:
    place_store = InMemoryPlaceStore()
    scenario_llm = _RankedScenarioLLM()
    runtime_tools = build_runtime_tools(
        settings=settings,
        http_client=http_client,
        place_store=place_store,
        source_store=InMemorySourceStore(),
    )
    assert {"places_search", "routing_tool"} <= set(runtime_tools)

    orchestrator = Orchestrator(
        llm=scenario_llm,
        tool_executor=ToolExecutor(runtime_tools),
        tool_specs=[get_tool_spec(name) for name in runtime_tools],
        max_steps=3,
    )
    run = await orchestrator.run(
        "Найди рестораны в Нижнем Новгороде и ранжируй их по расстоянию пешком "
        "от площади Минина и Пожарского."
    )
    return run, scenario_llm, place_store


async def _assert_ranked_scenario(
    run: OrchestratorRun,
    scenario_llm: _RankedScenarioLLM,
    place_store: InMemoryPlaceStore,
) -> tuple[PlacesSearchOutput, RoutingOutput]:
    assert run.tool_calls, "orchestrator made no tool calls"
    failed_calls = [
        call.result.model_dump(mode="json") for call in run.tool_calls if not call.result.ok
    ]
    assert not failed_calls, json.dumps(failed_calls, ensure_ascii=False, indent=2)
    assert [call.tool_name for call in run.tool_calls] == [
        "places_search",
        "routing_tool",
    ]

    places_call, routing_call = run.tool_calls
    assert places_call.result.data is not None
    places_output = PlacesSearchOutput.model_validate(places_call.result.data)
    assert places_output.places, "places_search returned no restaurants to rank"
    assert places_output.anchor is not None
    assert scenario_llm.anchor_ref == places_output.anchor

    returned_place_refs = {place.ref for place in places_output.places}
    assert scenario_llm.candidate_refs
    assert set(scenario_llm.candidate_refs) <= returned_place_refs
    assert routing_call.arguments["mode"] == "rank"
    assert routing_call.arguments["origins"] == [places_output.anchor]
    assert routing_call.arguments["candidates"] == scenario_llm.candidate_refs

    for place_ref in returned_place_refs:
        record = await place_store.get(place_ref)
        assert record is not None
        assert record.provider == "tomtom"

    assert routing_call.result.data is not None
    routing_output = RoutingOutput.model_validate(routing_call.result.data)
    assert routing_output.ranked, "routing provider returned no ranked candidates"
    assert [origin.ref for origin in routing_output.origins] == [places_output.anchor]
    assert [candidate.rank for candidate in routing_output.ranked] == list(
        range(1, len(routing_output.ranked) + 1)
    )
    assert {candidate.point.ref for candidate in routing_output.ranked} <= returned_place_refs

    reachable_scores = [
        candidate.score for candidate in routing_output.ranked if candidate.score is not None
    ]
    assert reachable_scores == sorted(reachable_scores)

    assert places_call.result.metrics is not None
    places_upstream = places_call.result.metrics.upstream_calls
    assert "yandex_geocoder" in {call.provider for call in places_upstream}
    assert any(
        call.provider == "tomtom_search" and call.status_code == 200 for call in places_upstream
    )

    assert routing_call.result.metrics is not None
    routing_upstream = routing_call.result.metrics.upstream_calls
    assert "yandex_geocoder" not in {call.provider for call in routing_upstream}
    assert any(call.operation == "distance_matrix" for call in routing_upstream)

    return places_output, routing_output


def _provider_calls(run: OrchestratorRun) -> list[dict[str, object]]:
    """Return the upstream call sequence without provider URLs or API keys."""

    calls: list[dict[str, object]] = []
    sequence = 0
    for tool_call in run.tool_calls:
        metrics = tool_call.result.metrics
        if metrics is None:
            continue
        for upstream_call in metrics.upstream_calls:
            sequence += 1
            calls.append(
                {
                    "sequence": sequence,
                    "tool": tool_call.tool_name,
                    "provider": upstream_call.provider,
                    "operation": upstream_call.operation,
                    "outcome": upstream_call.outcome.value,
                    "status_code": upstream_call.status_code,
                    "latency_ms": upstream_call.latency_ms,
                    "parallel_group": upstream_call.parallel_group,
                    "error_code": upstream_call.error_code,
                    "failure_kind": upstream_call.failure_kind,
                    "provider_code": upstream_call.provider_code,
                    "retryable": upstream_call.retryable,
                }
            )
    return calls


def _print_run(
    run: OrchestratorRun,
    *,
    places_providers: list[str],
    routing_providers: list[str],
) -> None:
    places_call = next(
        (call for call in run.tool_calls if call.tool_name == "places_search"),
        None,
    )
    routing_call = next(
        (call for call in run.tool_calls if call.tool_name == "routing_tool"),
        None,
    )
    print(
        json.dumps(
            {
                "configured_providers": {
                    "places_search": places_providers,
                    "routing": routing_providers,
                },
                "provider_calls": _provider_calls(run),
                "answer": run.llm_response.content,
                "places_search": (
                    places_call.result.model_dump(mode="json") if places_call is not None else None
                ),
                "routing_rank": (
                    routing_call.result.model_dump(mode="json")
                    if routing_call is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@pytest.mark.live_places
@pytest.mark.live_routing
async def test_places_search_results_are_ranked_through_the_orchestrator() -> None:
    if os.environ.get("RUN_LIVE_RANKED") != "1":
        pytest.skip("set RUN_LIVE_RANKED=1 to call real place-search and routing APIs")

    settings = Settings(app_env="dev")
    if "tomtom" not in settings.places_search_providers:
        pytest.skip("configure TomTom in PLACES_SEARCH_PROVIDERS")
    if not settings.routing_providers:
        pytest.skip("configure at least one ROUTING_PROVIDERS entry")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.tools_http_timeout)),
        proxy=settings.tools_http_proxy,
        trust_env=False,
    ) as http_client:
        run, scenario_llm, place_store = await _run_ranked_scenario(
            settings=settings,
            http_client=http_client,
        )

    await _assert_ranked_scenario(run, scenario_llm, place_store)
    _print_run(
        run,
        places_providers=list(settings.places_search_providers),
        routing_providers=list(settings.routing_providers),
    )
