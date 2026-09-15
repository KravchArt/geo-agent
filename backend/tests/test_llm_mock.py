"""Mock LLM client tests (no external services, runs everywhere)."""

from __future__ import annotations

from backend.app.config import Settings
from backend.app.llm.base import get_llm_client
from backend.app.llm.mock import MockLLMClient
from common.models import LLMMessage, LLMRequest, LLMResponse, ReActStepType


async def test_mock_returns_react_trace():
    client = MockLLMClient()
    resp = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="plan a day trip")])
    )
    assert isinstance(resp, LLMResponse)
    assert resp.mode == "mock"
    assert resp.trace.steps, "mock must return a non-empty ReAct trace"
    assert resp.trace.final_answer
    assert resp.trace.steps[-1].type is ReActStepType.FINAL_ANSWER


async def test_mock_response_matches_shared_contract():
    # The whole point of Phase 0: mock output validates against the shared
    # contract, so a real vLLM response (same shape) is a drop-in.
    client = MockLLMClient()
    resp = await client.generate(LLMRequest(messages=[LLMMessage(role="user", content="hi")]))
    LLMResponse.model_validate(resp.model_dump())


def test_factory_returns_mock_for_mock_mode():
    client = get_llm_client(Settings(llm_mode="mock"))
    assert client.mode == "mock"
