"""Mock LLM client.

Deterministic, no network — the default client and the ONLY one used in CI. It
produces the same :class:`LLMResponse` shape as the real vLLM client, and is
loop-aware so it can drive the orchestrator without a GPU:

* no tools advertised  -> answer in one turn (the lean default path).
* tools advertised, no observation yet -> call the first tool.
* tools advertised, an observation present -> give the final answer.
"""

from __future__ import annotations

from backend.app.llm.base import LLMClient
from common.models import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReActStep,
    ReActStepType,
    ReActTrace,
)

_FINAL_ANSWER = (
    "Here is a simple 1-day plan for your trip: start at the old town square, "
    "have lunch nearby, then visit the riverside park in the afternoon."
)


def _seen_tool_result(request: LLMRequest) -> bool:
    """True once the orchestrator has replayed a tool result into the conversation."""
    return any(m.role == "tool" for m in request.messages)


class MockLLMClient(LLMClient):
    """Deterministic mock — no external calls."""

    mode = "mock"

    def __init__(self, model: str = "geoagent-model") -> None:
        self.model = model

    def _response(self, request: LLMRequest, trace: ReActTrace) -> LLMResponse:
        return LLMResponse(
            model=request.model or self.model,
            mode="mock",
            content=trace.final_answer or "",
            trace=trace,
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Tool turn: advertise-and-call the first tool, once, before any result.
        if request.tools and not _seen_tool_result(request):
            name = request.tools[0]["function"]["name"]
            trace = ReActTrace(
                steps=[
                    ReActStep(
                        type=ReActStepType.THOUGHT,
                        content="I should use a tool to ground the answer.",
                    ),
                    ReActStep(
                        type=ReActStepType.ACTION,
                        content=f"Call {name}.",
                        tool_name=name,
                        # Deliberately minimal: exercises the loop; a real tool
                        # rejects bad args as data (INVALID_INPUT), not a crash.
                        tool_input={},
                        tool_call_id="mock_call_0",
                    ),
                ],
            )
            return self._response(request, trace)

        # Final turn.
        trace = ReActTrace(
            steps=[
                ReActStep(type=ReActStepType.THOUGHT, content="I have enough to answer."),
                ReActStep(
                    type=ReActStepType.FINAL_ANSWER,
                    content=_FINAL_ANSWER,
                ),
            ],
            final_answer=_FINAL_ANSWER,
        )
        return self._response(request, trace)

    async def health(self) -> bool:
        return True
