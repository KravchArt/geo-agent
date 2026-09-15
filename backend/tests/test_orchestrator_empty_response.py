"""Focused coverage for recovery from an empty provider completion."""

from __future__ import annotations

from backend.app.llm.base import LLMClient
from backend.app.services.orchestrator import (
    LLMCallRecord,
    Orchestrator,
    OrchestratorExecutionError,
)
from backend.app.services.pipeline import _failed_orchestrator_run, _is_llm_timeout
from common.models import LLMRequest, LLMResponse, ReActStep, ReActStepType, ReActTrace
from tools.executor import ToolExecutor


class _ScriptedLLM(LLMClient):
    mode = "mock"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def health(self) -> bool:
        return True


def _response(content: str) -> LLMResponse:
    trace = (
        ReActTrace()
        if not content
        else ReActTrace(
            steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=content)],
            final_answer=content,
        )
    )
    return LLMResponse(model="m", mode="mock", content=content, trace=trace)


async def test_empty_completion_gets_one_recovery_turn() -> None:
    llm = _ScriptedLLM([_response(""), _response("Готовый ответ")])
    orchestrator = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({}),
        max_steps=2,
    )

    run = await orchestrator.run("Построй маршрут")

    assert run.llm_response.content == "Готовый ответ"
    assert run.llm_turns == 2
    assert run.regenerations == 1
    assert "previous generation was empty" in llm.requests[1].messages[-1].content
    assert "Stop reasoning now" in llm.requests[1].messages[-1].content


async def test_second_empty_completion_is_returned_for_fresh_pipeline_retry() -> None:
    llm = _ScriptedLLM([_response(""), _response("")])
    orchestrator = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({}),
        max_steps=2,
    )

    run = await orchestrator.run("Построй маршрут")

    assert run.llm_response.content == ""
    assert run.empty_response is True
    assert run.llm_turns == 2
    assert run.regenerations == 1


async def test_empty_recovery_keeps_existing_observations() -> None:
    """The cheap in-run retry must stay in the same ReAct conversation."""
    llm = _ScriptedLLM([_response(""), _response("Ответ")])
    orchestrator = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({}),
        max_steps=2,
    )

    await orchestrator.run(
        "Продолжи",
        history=[{"role": "assistant", "content": "Ранее найдено место"}],
    )

    retry_messages = llm.requests[1].messages
    assert any(message.content == "Ранее найдено место" for message in retry_messages)
    assert "previous generation was empty" in retry_messages[-1].content


async def test_empty_final_step_also_gets_a_recovery_turn() -> None:
    empty_final = LLMResponse(
        model="m",
        mode="mock",
        content="",
        trace=ReActTrace(
            steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content="")],
            final_answer="",
        ),
    )
    llm = _ScriptedLLM([empty_final, _response("Восстановленный ответ")])
    orchestrator = Orchestrator(
        llm=llm,
        tool_executor=ToolExecutor({}),
        max_steps=2,
    )

    run = await orchestrator.run("Построй маршрут")

    assert run.llm_response.content == "Восстановленный ответ"
    assert run.regenerations == 1


def test_only_llm_timeout_is_eligible_for_clean_pipeline_retry() -> None:
    timeout = OrchestratorExecutionError(
        TimeoutError(),
        llm_calls=[],
        tool_calls=[],
        turns=1,
        regenerations=0,
    )
    provider_error = OrchestratorExecutionError(
        RuntimeError("403"),
        llm_calls=[],
        tool_calls=[],
        turns=1,
        regenerations=0,
    )

    assert _is_llm_timeout(timeout) is True
    assert _is_llm_timeout(provider_error) is False


def test_timed_out_attempt_is_kept_for_success_observability() -> None:
    error = OrchestratorExecutionError(
        TimeoutError(),
        llm_calls=[
            LLMCallRecord(
                turn_index=1,
                model="m",
                mode="vllm",
                latency_ms=60_000,
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
                reasoning_tokens=2,
                success=False,
                error_type="TimeoutError",
            )
        ],
        tool_calls=[],
        turns=1,
        regenerations=0,
    )

    run = _failed_orchestrator_run(error)

    assert run.llm_turns == 1
    assert run.llm_response.usage.total_tokens == 13
    assert run.llm_response.usage.reasoning_tokens == 2
    assert run.llm_calls[0].success is False
