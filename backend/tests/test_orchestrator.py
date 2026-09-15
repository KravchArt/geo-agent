"""Orchestrator ReAct loop — unit tests (no network, no real providers)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from backend.app.llm.base import LLMClient
from backend.app.services.orchestrator import (
    Orchestrator,
    _observation_text,
    to_openai_tools,
)
from common.models import LLMRequest, LLMResponse, ReActStep, ReActStepType, ReActTrace
from tools.base import PydanticTool, ToolErrorCode, ToolResult, ToolSpec
from tools.executor import ToolExecutor
from tools.geo.places_search.schemas import (
    PLACES_SEARCH_SPEC,
    PlacesSearchInput,
    PlacesSearchOutput,
)
from tools.observability import ToolExecutionContext
from tools.registry import list_tool_specs
from tools.web.search import WebSearchInput


class _EchoIn(BaseModel):
    text: str = "hi"


class _EchoOut(BaseModel):
    echoed: str


class _WebOut(BaseModel):
    results: list[str]


_ECHO_SPEC = ToolSpec[_EchoIn, _EchoOut](
    name="echo_tool",
    description="Echo back the text.",
    input_model=_EchoIn,
    output_model=_EchoOut,
)


async def _echo(params: _EchoIn, _ctx: ToolExecutionContext) -> _EchoOut:
    return _EchoOut(echoed=params.text)


async def _web_result(_params: _EchoIn, _ctx: ToolExecutionContext) -> _WebOut:
    return _WebOut(results=["candidate"])


async def _empty_web_result(_params: _EchoIn, _ctx: ToolExecutionContext) -> _WebOut:
    return _WebOut(results=[])


def _executor(*, warning: str | None = None) -> ToolExecutor:
    async def handler(
        params: _EchoIn,
        context: ToolExecutionContext,
    ) -> _EchoOut:
        if warning is not None:
            context.add_warning(warning)
        return await _echo(params, context)

    tool = PydanticTool(spec=_ECHO_SPEC, handler=handler)
    return ToolExecutor({tool.name: tool})


class _ScriptedLLM(LLMClient):
    """Returns pre-baked responses in order; records the requests it received."""

    mode = "mock"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def health(self) -> bool:
        return True


class _ChunkedScriptedLLM(_ScriptedLLM):
    """Scripted client that splits each response as a native provider would."""

    def __init__(self, responses: list[LLMResponse], chunks: list[list[str]]) -> None:
        super().__init__(responses)
        self._chunks = chunks

    async def generate_stream(self, request: LLMRequest, on_text_chunk) -> LLMResponse:
        self.requests.append(request)
        for chunk in self._chunks.pop(0):
            on_text_chunk(chunk)
        return self._responses.pop(0)


def _tool_turn(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        model="m",
        mode="mock",
        content="",
        trace=ReActTrace(
            steps=[
                ReActStep(type=ReActStepType.THOUGHT, content="use a tool"),
                ReActStep(
                    type=ReActStepType.ACTION, content="call", tool_name=name, tool_input=args
                ),
            ]
        ),
    )


def _final_turn(answer: str) -> LLMResponse:
    return LLMResponse(
        model="m",
        mode="mock",
        content=answer,
        trace=ReActTrace(
            steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=answer)],
            final_answer=answer,
        ),
    )


def _final_turn_with_places(answer: str, place_refs: list[str]) -> LLMResponse:
    markers = " ".join(f"[[{ref}]]" for ref in place_refs)
    return _final_turn(f"{answer} {markers}".rstrip())


async def test_rejected_answer_is_regenerated_once() -> None:
    llm = _ScriptedLLM([_final_turn("Bad [[plc_deadbeef00]]"), _final_turn("Clean answer")])
    seen: list[str] = []

    async def validator(answer: str, _executed) -> str | None:
        seen.append(answer)
        return "Use only valid refs." if "deadbeef" in answer else None

    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        answer_validator=validator,
    )

    run = await orch.run("q")

    assert run.regenerations == 1
    assert run.llm_response.content == "Clean answer"
    assert seen == ["Bad [[plc_deadbeef00]]", "Clean answer"]
    assert any("Use only valid refs" in message.content for message in llm.requests[1].messages)


async def test_exposed_planning_is_regenerated_before_becoming_final_answer() -> None:
    leaked = (
        "I have gathered the places.\n"
        "I will structure the itinerary chronologically.\n"
        "I will check the refs."
    )
    llm = _ScriptedLLM([_final_turn(leaked), _final_turn("Here is the itinerary.")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
    )

    run = await orch.run("q")

    assert run.regenerations == 1
    assert run.llm_response.content == "Here is the itinerary."
    assert "exposed internal planning" in llm.requests[1].messages[-1].content


async def test_one_first_person_phrase_is_not_mistaken_for_exposed_planning() -> None:
    answer = "I will start with the quietest waterfront and leave the museum for the afternoon."
    llm = _ScriptedLLM([_final_turn(answer)])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
    )

    run = await orch.run("q")

    assert run.regenerations == 0
    assert run.llm_response.content == answer


async def test_answer_validation_regeneration_is_bounded() -> None:
    llm = _ScriptedLLM([_final_turn("Bad one"), _final_turn("Bad two")])

    async def always_bad(_answer: str, _executed) -> str | None:
        return "Still invalid."

    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        answer_validator=always_bad,
    )

    run = await orch.run("q")

    assert run.regenerations == 1
    assert run.llm_response.content == "Bad two"


async def test_broken_answer_validator_does_not_cost_the_answer() -> None:
    llm = _ScriptedLLM([_final_turn("Answer")])

    async def broken_validator(_answer: str, _executed) -> str | None:
        raise RuntimeError("validator unavailable")

    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        answer_validator=broken_validator,
    )

    run = await orch.run("q")

    assert run.regenerations == 0
    assert run.llm_response.content == "Answer"


async def test_loop_executes_tool_then_answers():
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "prague"}), _final_turn("done")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("plan a trip")

    # Two model turns; exactly one tool executed.
    assert run.llm_turns == 2
    assert len(run.tool_calls) == 1
    call = run.tool_calls[0]
    assert call.tool_name == "echo_tool"
    assert call.result.ok is True
    assert call.result.data == {"echoed": "prague"}

    # Trace threads action -> observation -> final answer.
    kinds = [s.type for s in run.llm_response.trace.steps]
    assert ReActStepType.ACTION in kinds
    assert ReActStepType.OBSERVATION in kinds
    assert kinds[-1] is ReActStepType.FINAL_ANSWER
    assert run.llm_response.content == "done"


async def test_final_answer_keeps_internal_selected_place_markers() -> None:
    selected = "plc_a1b2c3d4e5"
    llm = _ScriptedLLM([_final_turn_with_places("Try the museum.", [selected])])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("q")

    assert run.llm_response.content == f"Try the museum. [[{selected}]]"
    assert run.llm_response.trace.final_answer == f"Try the museum. [[{selected}]]"


async def test_sampling_parameters_are_used_for_every_agent_turn():
    llm = _ScriptedLLM([_tool_turn("echo_tool", {}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        temperature=0.2,
        top_p=0.9,
    )

    await orch.run("plan a trip")

    assert len(llm.requests) == 2
    assert all(request.temperature == 0.2 for request in llm.requests)
    assert all(request.top_p == 0.9 for request in llm.requests)


async def test_progress_callback_reports_model_and_tool_lifecycle():
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "prague"}), _final_turn("done")])
    events: list[tuple[str, dict[str, Any]]] = []
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        progress_callback=lambda kind, details: events.append((kind, details)),
    )

    await orch.run("plan a trip")

    assert [kind for kind, _details in events] == [
        "model_started",
        "tool_started",
        "tool_finished",
        "model_started",
        "answer_ready",
    ]
    assert events[1][1] == {
        "turn_index": 1,
        "tool_name": "echo_tool",
        "tool_input": {"text": "prague"},
    }
    assert events[2][1]["ok"] is True
    assert events[-1][1]["turn_index"] == 2
    assert llm.requests[0].max_tokens is None
    assert llm.requests[1].max_tokens is None


async def test_broken_progress_callback_does_not_break_the_agent():
    def broken_callback(_kind: str, _details: dict[str, Any]) -> None:
        raise RuntimeError("UI disconnected")

    llm = _ScriptedLLM([_final_turn("still done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        progress_callback=broken_callback,
    )

    run = await orch.run("q")

    assert run.llm_response.content == "still done"


async def test_answer_chunks_are_forwarded() -> None:
    chunks: list[str] = []
    orch = Orchestrator(
        llm=_ScriptedLLM([_final_turn("stream me")]),
        tool_executor=_executor(),
        answer_chunk_callback=chunks.append,
    )

    await orch.run("q")

    # The base LLM fallback emits one chunk; native providers may emit many.
    assert chunks == ["stream me"]


async def test_answer_is_streamed_without_internal_place_refs() -> None:
    answer = "Часть 1\nЧасть 2"
    place_ref = "plc_a1b2c3d4e5"
    internal_answer = f"{answer} [[{place_ref}]]"
    response = LLMResponse(
        model="m",
        mode="mock",
        content=internal_answer,
        trace=ReActTrace(
            steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=internal_answer)],
            final_answer=internal_answer,
        ),
    )
    chunks: list[str] = []
    llm = _ChunkedScriptedLLM(
        [response],
        [["Часть 1\nЧасть 2 [[plc_a1", "b2c3d4e5]]"]],
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        answer_chunk_callback=chunks.append,
    )

    run = await orch.run("q")

    assert "".join(chunks) == answer
    assert all("plc_" not in chunk for chunk in chunks)
    assert run.llm_response.content == internal_answer


def test_places_observation_keeps_user_facts_but_drops_provider_noise() -> None:
    result = ToolResult(
        tool_name="places_search",
        ok=True,
        data={
            "places": [
                {
                    "ref": "plc_a1b2c3d4e5",
                    "id": "2gis-provider-record-id",
                    "name": "The Бык",
                    "address": "Москва, Ветошный переулок, 13",
                    "categories": ["демократичный мясной ресторан"],
                    "phones": ["+7 495 000-00-00"],
                    "rating": 4.9,
                    "review_count": 1234,
                    "hours_text": "ежедневно 12:00–23:00",
                    "is_open_now": True,
                    "accessibility": [],
                }
            ],
            "area": {
                "ref": "plc_f0e1d2c3b4",
                "name": "Москва",
                "address": "Москва, Россия",
            },
        },
    )

    observation = json.loads(_observation_text(result))

    assert observation == {
        "places": [
            {
                "ref": "plc_a1b2c3d4e5",
                "name": "The Бык",
                "address": "Москва, Ветошный переулок, 13",
                "categories": ["демократичный мясной ресторан"],
                "phones": ["+7 495 000-00-00"],
                "rating": 4.9,
                "review_count": 1234,
                "hours": "ежедневно 12:00–23:00",
                "is_open_now": True,
            }
        ],
        "area": {
            "ref": "plc_f0e1d2c3b4",
            "name": "Москва",
            "address": "Москва, Россия",
        },
    }
    assert result.data is not None
    assert result.data["places"][0]["id"] == "2gis-provider-record-id"


async def test_tool_turn_text_is_reset_before_the_final_answer() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    chunks: list[str] = []
    tool_turn = _tool_turn("echo_tool", {"text": "x"}).model_copy(
        update={"content": "Let me check."}
    )
    llm = _ScriptedLLM([tool_turn, _final_turn("final")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(),
        tool_specs=[_ECHO_SPEC],
        progress_callback=lambda kind, details: events.append((kind, details)),
        answer_chunk_callback=chunks.append,
    )

    await orch.run("q")

    assert chunks == ["Let me check.", "final"]
    assert [kind for kind, _details in events].count("answer_reset") == 1


async def test_observation_is_replayed_in_the_openai_tool_protocol():
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "x"}), _final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run("q")

    # The SECOND model call must replay the turn the way tool-calling models are
    # trained on: an assistant message carrying tool_calls, then a `tool` message
    # whose tool_call_id matches it.
    second = llm.requests[1]
    assistant = next(m for m in second.messages if m.tool_calls)
    tool_msg = next(m for m in second.messages if m.role == "tool")

    assert assistant.tool_calls is not None
    call = assistant.tool_calls[0]
    assert call["function"]["name"] == "echo_tool"
    assert tool_msg.tool_call_id == call["id"]
    assert "echoed" in tool_msg.content and "x" in tool_msg.content


async def test_successful_web_search_adds_a_strict_follow_up_instruction() -> None:
    web_spec = ToolSpec[_EchoIn, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=_EchoIn,
        output_model=_WebOut,
    )
    web_tool = PydanticTool(spec=web_spec, handler=_web_result)
    llm = _ScriptedLLM([_tool_turn("web_search", {"text": "cafes"}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    await orch.run("find affordable cafes")

    second = llm.requests[1]
    tool_message = next(message for message in second.messages if message.role == "tool")
    observation = json.loads(tool_message.content)
    assert observation["results"] == ["candidate"]
    assert observation["message"] == (
        "The web search completed successfully and returned relevant results."
    )
    assert "Do not repeat or reformulate" in observation["agent_guidance"]["retry_policy"]
    assert "[[src_...]] ref" in observation["agent_guidance"]["next_action"]
    assert "places_search may only use mode=resolve" in observation["agent_guidance"]["next_action"]
    assert "never use mode=area or mode=near" in observation["agent_guidance"]["next_action"]
    assert "resolved place type or category matches" in observation["agent_guidance"]["next_action"]
    assert all(message.role != "system" for message in second.messages[1:])


async def test_equivalent_successful_web_search_is_blocked_before_execution() -> None:
    calls: list[str] = []

    async def handler(params: WebSearchInput, _context: ToolExecutionContext) -> _WebOut:
        calls.append(params.query)
        return _WebOut(results=["candidate"])

    web_spec = ToolSpec[WebSearchInput, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=WebSearchInput,
        output_model=_WebOut,
        answer_fields=("results",),
    )
    web_tool = PydanticTool(spec=web_spec, handler=handler)
    first_query = "рестораны Москва средний чек 2000 рублей 2026"
    reformulation = "список ресторанов в Москве со средним чеком 2000 рублей 2026"
    llm = _ScriptedLLM(
        [
            _tool_turn("web_search", {"query": first_query}),
            _tool_turn("web_search", {"query": reformulation}),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    run = await orch.run("find restaurants")

    assert calls == [first_query]
    assert len(run.tool_calls) == 2
    blocked = run.tool_calls[1].result
    assert blocked.ok is False
    assert blocked.error_code is ToolErrorCode.DUPLICATE_CALL
    assert blocked.retryable is False
    final_request_tool_messages = [
        message for message in llm.requests[2].messages if message.role == "tool"
    ]
    assert "ERROR[duplicate_call]" in final_request_tool_messages[-1].content
    assert "previously returned evidence" in final_request_tool_messages[-1].content


async def test_web_search_with_changed_numeric_constraint_is_not_blocked() -> None:
    calls: list[str] = []

    async def handler(params: WebSearchInput, _context: ToolExecutionContext) -> _WebOut:
        calls.append(params.query)
        return _WebOut(results=["candidate"])

    web_spec = ToolSpec[WebSearchInput, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=WebSearchInput,
        output_model=_WebOut,
        answer_fields=("results",),
    )
    web_tool = PydanticTool(spec=web_spec, handler=handler)
    llm = _ScriptedLLM(
        [
            _tool_turn(
                "web_search",
                {"query": "рестораны Москва средний чек 2000 рублей"},
            ),
            _tool_turn(
                "web_search",
                {"query": "рестораны Москва средний чек 3000 рублей"},
            ),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    run = await orch.run("find restaurants")

    assert calls == [
        "рестораны Москва средний чек 2000 рублей",
        "рестораны Москва средний чек 3000 рублей",
    ]
    assert all(call.result.ok for call in run.tool_calls)


async def test_web_search_with_changed_semantic_constraint_is_not_blocked() -> None:
    calls: list[str] = []

    async def handler(params: WebSearchInput, _context: ToolExecutionContext) -> _WebOut:
        calls.append(params.query)
        return _WebOut(results=["candidate"])

    web_spec = ToolSpec[WebSearchInput, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=WebSearchInput,
        output_model=_WebOut,
        answer_fields=("results",),
    )
    web_tool = PydanticTool(spec=web_spec, handler=handler)
    llm = _ScriptedLLM(
        [
            _tool_turn("web_search", {"query": "family-run pottery workshops Florence"}),
            _tool_turn("web_search", {"query": "family-run leather workshops Florence"}),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    run = await orch.run("find workshops")

    assert calls == [
        "family-run pottery workshops Florence",
        "family-run leather workshops Florence",
    ]
    assert all(call.result.ok for call in run.tool_calls)


async def test_equivalent_web_search_is_blocked_after_empty_result() -> None:
    calls: list[str] = []

    async def handler(params: WebSearchInput, _context: ToolExecutionContext) -> _WebOut:
        calls.append(params.query)
        results = [] if len(calls) == 1 else ["candidate"]
        return _WebOut(results=results)

    web_spec = ToolSpec[WebSearchInput, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=WebSearchInput,
        output_model=_WebOut,
        answer_fields=("results",),
    )
    web_tool = PydanticTool(spec=web_spec, handler=handler)
    llm = _ScriptedLLM(
        [
            _tool_turn("web_search", {"query": "кафе Москва открыто сейчас"}),
            _tool_turn("web_search", {"query": "список кафе в Москве открытых сейчас"}),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    run = await orch.run("find cafes")

    assert calls == ["кафе Москва открыто сейчас"]
    assert len(run.tool_calls) == 2
    assert run.tool_calls[1].result.error_code is ToolErrorCode.DUPLICATE_CALL
    assert "previous search returned no evidence" in (run.tool_calls[1].result.error or "").lower()


async def test_empty_web_search_allows_another_tool_or_a_not_found_answer() -> None:
    web_spec = ToolSpec[_EchoIn, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=_EchoIn,
        output_model=_WebOut,
    )
    web_tool = PydanticTool(spec=web_spec, handler=_empty_web_result)
    llm = _ScriptedLLM([_tool_turn("web_search", {"text": "cafes"}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"web_search": web_tool}),
        tool_specs=[web_spec],
    )

    await orch.run("find cafes")

    second = llm.requests[1]
    tool_message = next(message for message in second.messages if message.role == "tool")
    observation = json.loads(tool_message.content)
    assert observation["status"] == "NO_RESULTS"
    assert observation["results"] == []
    assert observation["message"] == ("The search completed successfully but returned no results.")
    guidance = observation["agent_guidance"]
    assert "Do not repeat an equivalent search using the same tool" in guidance["retry_policy"]
    assert "another appropriate available tool or strategy" in guidance["next_action"]
    assert "For example, use web_search" not in guidance["next_action"]


async def test_empty_non_web_search_mentions_web_as_an_optional_example() -> None:
    places_spec = ToolSpec[_EchoIn, _WebOut](
        name="places_search",
        description="Search for places.",
        input_model=_EchoIn,
        output_model=_WebOut,
    )
    web_spec = ToolSpec[_EchoIn, _WebOut](
        name="web_search",
        description="Search the web.",
        input_model=_EchoIn,
        output_model=_WebOut,
    )
    places_tool = PydanticTool(spec=places_spec, handler=_empty_web_result)
    web_tool = PydanticTool(spec=web_spec, handler=_empty_web_result)
    llm = _ScriptedLLM([_tool_turn("places_search", {"text": "cafes"}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor(
            {"places_search": places_tool, "web_search": web_tool},
        ),
        tool_specs=[places_spec, web_spec],
    )

    await orch.run("find cafes")

    second = llm.requests[1]
    tool_message = next(message for message in second.messages if message.role == "tool")
    observation = json.loads(tool_message.content)
    assert observation["status"] == "NO_RESULTS"
    assert "For example, use web_search" in observation["agent_guidance"]["next_action"]
    assert "Call web_search" not in observation["agent_guidance"]["next_action"]


async def test_successful_places_search_adds_a_strict_follow_up_instruction() -> None:
    places_spec = ToolSpec[_EchoIn, _EchoOut](
        name="places_search",
        description="Discover places.",
        input_model=_EchoIn,
        output_model=_EchoOut,
    )
    places_tool = PydanticTool(spec=places_spec, handler=_echo)
    llm = _ScriptedLLM([_tool_turn("places_search", {"text": "restaurants"}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"places_search": places_tool}),
        tool_specs=[places_spec],
    )

    await orch.run("find restaurants")

    second = llm.requests[1]
    tool_message = next(message for message in second.messages if message.role == "tool")
    observation = json.loads(tool_message.content)
    assert observation["places"] == []
    assert observation["message"] == (
        "The places search completed successfully and returned results."
    )
    guidance = observation["agent_guidance"]
    assert (
        "Do not call places_search again with the same search intent" in (guidance["retry_policy"])
    )
    assert "additional or different candidates" in guidance["retry_policy"]
    assert "Use the returned results for the part" in guidance["next_action"]
    assert (
        "do not follow resolve with places_search mode=area or mode=near"
        in (guidance["next_action"])
    )
    assert "place type or category matches the entity type" in guidance["next_action"]
    assert "do not make another resolve call with more candidates" in guidance["next_action"]
    assert "Return the verified subset even if fewer places qualify" in guidance["next_action"]
    assert "distinct part remains unresolved" in guidance["next_action"]
    assert "materially different search parameters" in guidance["next_action"]
    assert all(message.role != "system" for message in second.messages[1:])


async def test_successful_routing_adds_structured_follow_up_guidance() -> None:
    routing_spec = ToolSpec[_EchoIn, _EchoOut](
        name="routing_tool",
        description="Calculate a route.",
        input_model=_EchoIn,
        output_model=_EchoOut,
    )
    routing_tool = PydanticTool(spec=routing_spec, handler=_echo)
    llm = _ScriptedLLM([_tool_turn("routing_tool", {"text": "route"}), _final_turn("done")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"routing_tool": routing_tool}),
        tool_specs=[routing_spec],
    )

    await orch.run("build a route")

    second = llm.requests[1]
    tool_message = next(message for message in second.messages if message.role == "tool")
    observation = json.loads(tool_message.content)
    assert observation["echoed"] == "route"
    assert observation["message"] == (
        "The routing calculation completed successfully and returned results."
    )
    guidance = observation["agent_guidance"]
    assert "same routing intent" in guidance["retry_policy"]
    assert "distinct routing task remains unresolved" in guidance["next_action"]
    assert "materially different parameters" in guidance["next_action"]
    assert "every non-arrival step" in guidance["presentation_policy"]
    assert "authoritative length_m" in guidance["presentation_policy"]
    assert "street_name is absent" in guidance["presentation_policy"]
    assert "zero-length arrival step" in guidance["presentation_policy"]


async def test_observation_replays_non_fatal_tool_warnings() -> None:
    warning = "Route times do not include live traffic."
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "x"}), _final_turn("ok")])
    orch = Orchestrator(
        llm=llm,
        tool_executor=_executor(warning=warning),
        tool_specs=[_ECHO_SPEC],
    )

    await orch.run("q")

    tool_msg = next(m for m in llm.requests[1].messages if m.role == "tool")
    observation = json.loads(tool_msg.content)
    assert observation == {
        "echoed": "x",
        "warnings": [warning],
    }


async def test_provider_tool_call_id_is_preserved():
    # The id the model gave us must come back on the tool reply, not a made-up one.
    turn = _tool_turn("echo_tool", {"text": "y"})
    turn.trace.steps[-1].tool_call_id = "call_from_model"
    llm = _ScriptedLLM([turn, _final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run("q")

    tool_msg = next(m for m in llm.requests[1].messages if m.role == "tool")
    assert tool_msg.tool_call_id == "call_from_model"


async def test_all_tool_calls_in_one_turn_are_executed():
    # Models emit parallel calls; every one must run and get its own reply.
    parallel = LLMResponse(
        model="m",
        mode="mock",
        content="",
        trace=ReActTrace(
            steps=[
                ReActStep(
                    type=ReActStepType.ACTION,
                    content="call",
                    tool_name="echo_tool",
                    tool_input={"text": "a"},
                    tool_call_id="c1",
                ),
                ReActStep(
                    type=ReActStepType.ACTION,
                    content="call",
                    tool_name="echo_tool",
                    tool_input={"text": "b"},
                    tool_call_id="c2",
                ),
            ]
        ),
    )
    llm = _ScriptedLLM([parallel, _final_turn("both done")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("q")

    assert len(run.tool_calls) == 2
    assert [c.result.data for c in run.tool_calls] == [{"echoed": "a"}, {"echoed": "b"}]
    tool_ids = [m.tool_call_id for m in llm.requests[1].messages if m.role == "tool"]
    assert tool_ids == ["c1", "c2"]


async def test_tools_are_advertised_as_openai_functions():
    llm = _ScriptedLLM([_final_turn("hi")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run("q")

    tools = llm.requests[0].tools
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "echo_tool"
    assert "text" in tools[0]["function"]["parameters"]["properties"]


async def test_russian_generic_places_query_is_restored_from_category() -> None:
    calls: list[PlacesSearchInput] = []

    async def handler(
        params: PlacesSearchInput,
        _context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        calls.append(params)
        return PlacesSearchOutput()

    tool = PydanticTool(spec=PLACES_SEARCH_SPEC, handler=handler)
    llm = _ScriptedLLM(
        [
            _tool_turn(
                "places_search",
                {
                    "mode": "area",
                    "area": "Москва",
                    "query": "restaurants",
                    "category": "restaurant",
                },
            ),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"places_search": tool}),
        tool_specs=[PLACES_SEARCH_SPEC],
    )

    run = await orch.run("А теперь дай рестораны в Москве")

    assert calls[0].query == "рестораны"
    assert calls[0].category is not None and calls[0].category.value == "restaurant"
    assert run.tool_calls[0].arguments["query"] == "рестораны"


@pytest.mark.parametrize(
    ("user_message", "query"),
    [
        ("Find restaurants in Moscow", "restaurants"),
        ("Найди итальянские рестораны в Москве", "Italian restaurants"),
    ],
)
async def test_places_query_language_restoration_leaves_non_matching_queries_unchanged(
    user_message: str,
    query: str,
) -> None:
    calls: list[PlacesSearchInput] = []

    async def handler(
        params: PlacesSearchInput,
        _context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        calls.append(params)
        return PlacesSearchOutput()

    tool = PydanticTool(spec=PLACES_SEARCH_SPEC, handler=handler)
    llm = _ScriptedLLM(
        [
            _tool_turn(
                "places_search",
                {
                    "mode": "area",
                    "area": "Moscow",
                    "query": query,
                    "category": "restaurant",
                },
            ),
            _final_turn("done"),
        ]
    )
    orch = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({"places_search": tool}),
        tool_specs=[PLACES_SEARCH_SPEC],
    )

    await orch.run(user_message)

    assert calls[0].query == query


def test_places_search_advertises_its_compact_llm_contract():
    """The model gets a compact contract, including its one essential mode rule."""

    tool = to_openai_tools([PLACES_SEARCH_SPEC])[0]["function"]

    assert tool["parameters"] == PLACES_SEARCH_SPEC.llm_parameters
    assert tool["parameters"]["allOf"][0]["then"] == {"required": ["query"]}
    assert "$defs" not in tool["parameters"]
    assert tool["parameters"] != PlacesSearchInput.model_json_schema()


def test_model_facing_tool_contracts_contain_no_cyrillic_text() -> None:
    payload = json.dumps(to_openai_tools(list_tool_specs()), ensure_ascii=False)

    assert not any("\u0400" <= character <= "\u04ff" for character in payload)


async def test_final_answer_on_first_turn_skips_tools():
    llm = _ScriptedLLM([_final_turn("no tools needed")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("q")

    assert run.llm_turns == 1
    assert run.tool_calls == []
    assert run.llm_response.content == "no tools needed"


async def test_step_budget_is_enforced():
    # A model that never stops asking for tools must be cut off.
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "loop"}) for _ in range(10)])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC], max_steps=3)

    run = await orch.run("q")

    assert run.llm_turns == 3
    assert run.llm_response.trace.steps[-1].type is ReActStepType.FINAL_ANSWER
    assert run.llm_response.content  # a fallback answer, not empty
    # The final turn is answer-only, so it cannot execute a third tool call.
    assert len(run.tool_calls) == 2
    assert llm.requests[-1].tools == []
    budget_message = next(
        message
        for message in llm.requests[-1].messages
        if "produce the final answer now" in message.content
    )
    assert budget_message.role == "user"


async def test_tool_error_is_data_not_a_crash():
    # Unknown tool -> ToolResult(ok=False); the loop keeps going and still answers.
    llm = _ScriptedLLM([_tool_turn("does_not_exist", {}), _final_turn("recovered")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("q")

    assert run.tool_calls[0].result.ok is False
    assert run.llm_response.content == "recovered"


async def test_hallucinated_tool_is_told_the_closed_set():
    # A made-up name must not reach the executor, and the model must learn which
    # names are real so it can correct itself on the next turn.
    llm = _ScriptedLLM([_tool_turn("search_places", {}), _final_turn("fixed")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    run = await orch.run("q")

    error = run.tool_calls[0].result.error or ""
    assert "unknown tool: search_places" in error
    assert "echo_tool" in error  # the available set is named
    observation = next(m.content for m in llm.requests[1].messages if m.role == "tool")
    assert "echo_tool" in observation


class _RecordingEchoStore:
    def __init__(self) -> None:
        self.written: dict[str, str] = {}

    async def set_echo(self, tool_hash: str, value: str, ttl: int | None = None) -> None:
        self.written[tool_hash] = value


async def test_tool_results_are_echo_recorded_by_hash():
    # tool_hash -> payload is what later proves a cited result was really produced.
    echo = _RecordingEchoStore()
    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "z"}), _final_turn("ok")])
    orch = Orchestrator(
        llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC], echo_store=echo
    )

    run = await orch.run("q")

    tool_hash = run.tool_calls[0].result.tool_hash
    assert tool_hash
    assert tool_hash in echo.written
    assert "echoed" in echo.written[tool_hash]


async def test_echo_failure_never_breaks_the_request():
    class _Broken:
        async def set_echo(self, tool_hash: str, value: str, ttl: int | None = None) -> None:
            raise RuntimeError("redis down")

    llm = _ScriptedLLM([_tool_turn("echo_tool", {"text": "z"}), _final_turn("still ok")])
    orch = Orchestrator(
        llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC], echo_store=_Broken()
    )

    run = await orch.run("q")

    # Echo is observability: losing it must not cost the user their answer.
    assert run.llm_response.content == "still ok"
    assert run.tool_calls[0].result.ok is True


async def test_history_is_replayed_to_the_model():
    # Without it a follow-up arrives with nothing to resolve "рядом" against.
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "А что рядом?",
        history=[
            {"role": "user", "content": "Найди кофейни у Лувра"},
            {"role": "assistant", "content": "Вот три: plc_aaaaaaaaaa, ..."},
        ],
    )

    messages = llm.requests[0].messages
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "Найди кофейни у Лувра"
    # The current question stays last, so it is what the model answers.
    assert messages[-1].content == "А что рядом?"
    # Refs from the previous answer come along, so they can be reused instead of
    # resolving the same place again.
    assert "plc_aaaaaaaaaa" in messages[2].content


async def test_history_replays_hidden_map_refs_for_follow_up_tools() -> None:
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "Найди аптеки рядом с Frank by Баста",
        history=[
            {"role": "user", "content": "Найди рестораны в Москве"},
            {
                "role": "assistant",
                "content": "Вот Frank by Баста на Мясницкой улице, 24.",
                "_place_context": (
                    "- Frank by Баста, реберная — Москва, Мясницкая улица, 24 → plc_77a91385f1"
                ),
            },
        ],
    )

    assistant_history = llm.requests[0].messages[2].content
    assert "INTERNAL PLACE CONTEXT FOR FOLLOW-UP TOOLS ONLY" in assistant_history
    assert "plc_77a91385f1" in assistant_history
    assert "Use a matching plc_ ref directly" in assistant_history


async def test_history_tells_model_to_reuse_private_area_ref() -> None:
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "А теперь найди аптеки в этом же городе",
        history=[
            {"role": "user", "content": "Найди рестораны в Москве"},
            {
                "role": "assistant",
                "content": "Вот найденные рестораны.",
                "_place_context": ("- Search area: Москва — Москва, Россия → plc_77a91385f1"),
            },
        ],
    )

    assistant_history = llm.requests[0].messages[2].content
    assert "plc_77a91385f1" in assistant_history
    assert "pass its ref as `area`" in assistant_history
    assert "pass the locality name as `area` only when" in assistant_history


async def test_latest_search_area_is_repeated_next_to_current_follow_up() -> None:
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "Теперь давай найдем кафе рядом с метро Римская",
        history=[
            {"role": "user", "content": "Найди рестораны в Москве"},
            {
                "role": "assistant",
                "content": "Вот найденные рестораны.",
                "_place_context": (
                    "- Search area: Москва — Москва, Центральный федеральный округ, Россия "
                    "→ plc_77a91385f1"
                ),
            },
        ],
    )

    messages = llm.requests[0].messages
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    active_context = messages[-2].content
    assert "ACTIVE CONVERSATION GEO CONTEXT" in active_context
    assert "Most recent established locality: Москва" in active_context
    assert "Area ref: plc_77a91385f1" in active_context
    assert "`mode=near` call with a new textual `near`" in active_context
    assert "use the ref as `area`" in active_context
    assert messages[-1].content == "Теперь давай найдем кафе рядом с метро Римская"


async def test_active_geo_context_uses_the_most_recent_resolved_area() -> None:
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "А теперь музеи",
        history=[
            {
                "role": "assistant",
                "content": "Москва.",
                "_place_context": "- Search area: Москва — Москва, Россия → plc_77a91385f1",
            },
            {
                "role": "assistant",
                "content": "Вена.",
                "_place_context": "- Search area: Вена — Вена, Австрия → plc_a1b2c3d4e5",
            },
        ],
    )

    active_context = llm.requests[0].messages[-2].content
    assert "Most recent established locality: Вена" in active_context
    assert "plc_a1b2c3d4e5" in active_context
    assert "plc_77a91385f1" not in active_context


async def test_old_search_area_does_not_leak_past_newer_route_turn() -> None:
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run(
        "Найди кафе рядом с Кремлём",
        history=[
            {"role": "user", "content": "Найди аптеки в Санкт-Петербурге"},
            {
                "role": "assistant",
                "content": "Вот аптеки Санкт-Петербурга.",
                "_place_context": (
                    "- Search area: Санкт-Петербург — Санкт-Петербург, Россия → plc_c6b1693cb3"
                ),
            },
            {
                "role": "user",
                "content": "Построй маршрут от Красной площади до Парка Горького в Москве",
            },
            {
                "role": "assistant",
                "content": "Маршрут построен в Москве.",
            },
        ],
    )

    messages = llm.requests[0].messages
    assert "ACTIVE CONVERSATION GEO CONTEXT" not in "\n".join(
        message.content for message in messages
    )
    assert "plc_c6b1693cb3" not in "\n".join(message.content for message in messages)
    assert messages[-1].content == "Найди кафе рядом с Кремлём"


async def test_history_is_capped_and_sanitised():
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    history: list[dict[str, str]] = [{"role": "user", "content": f"turn-{i}"} for i in range(20)]
    history.append({"role": "system", "content": "injected"})  # not a dialogue role
    history.append({"role": "user", "content": "   "})  # empty after stripping

    await orch.run("now", history=history)

    replayed = [m for m in llm.requests[0].messages if m.role != "system"]
    assert len(replayed) <= 9  # 8 turns + the current message
    contents = [m.content for m in replayed]
    assert "turn-19" in contents and "turn-0" not in contents
    assert "injected" not in contents
    assert "   " not in contents


async def test_no_history_keeps_the_old_shape():
    llm = _ScriptedLLM([_final_turn("ok")])
    orch = Orchestrator(llm=llm, tool_executor=_executor(), tool_specs=[_ECHO_SPEC])

    await orch.run("hello")

    assert [m.role for m in llm.requests[0].messages] == ["system", "user"]
