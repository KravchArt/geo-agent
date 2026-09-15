"""LLM scope classification — verdict parsing and the regex fallback."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.app.llm.base import LLMClient
from backend.app.services.gates import (
    GateEvaluator,
    LLMScopeGate,
    parse_scope_verdict,
)
from common.models import (
    ConversationTurn,
    GateDecision,
    GateName,
    GateVerdict,
    LLMRequest,
    LLMResponse,
    ReActTrace,
)


class _StubLLM(LLMClient):
    """Returns a fixed completion, or raises to simulate the model being down."""

    mode = "mock"

    def __init__(self, content: str = "", *, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMResponse(
            model="scoper",
            mode="mock",
            content=self._content,
            trace=ReActTrace(final_answer=self._content),
        )

    async def health(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("yes", True),
        ("no", False),
        ("Yes", True),
        ("NO", False),
        ("yes.", True),
        ("  no\n", False),
        # Reasoning models narrate first and conclude last.
        ("There is no clear geo intent... no", False),
        ("The user asks about museums, so yes", True),
        # Nothing decidable.
        ("maybe", None),
        ("", None),
        ("normally people travel", None),  # "no" inside a word is not a verdict
        # A weak model answers the question instead of classifying it. Reading the
        # leading "Yes" as a verdict would make the gate pass everything, so a
        # reply that does not END in a verdict is not a verdict at all.
        ("Yes, I can answer that. The museums in Kazan are", None),
        ("No problem! Here is a sorting function in Python", None),
    ],
)
def test_verdict_parsing(content, expected):
    assert parse_scope_verdict(content) is expected


async def test_yes_allows_the_request():
    gate = LLMScopeGate(_StubLLM("yes"))

    decision = await gate.evaluate("Which museums to visit in Kazan?")

    assert decision.passed is True
    assert decision.verdict is GateVerdict.ALLOW
    assert decision.name is GateName.SCOPE
    assert decision.provider == "model"


async def test_no_rejects_the_request():
    gate = LLMScopeGate(_StubLLM("no"))

    decision = await gate.evaluate("Write a sorting function in Python")

    assert decision.passed is False
    assert decision.verdict is GateVerdict.REJECT


async def test_the_classifier_prompt_and_user_text_are_sent():
    llm = _StubLLM("yes")

    await gate_eval(llm, "Set a route around Moscow")

    request = llm.requests[0]
    assert request.messages[0].role == "system"
    assert "classifier" in request.messages[0].content
    assert "Set a route around Moscow" in request.messages[-1].content
    # A one-word verdict does not need a large budget or sampling.
    assert request.temperature == 0.0
    assert request.max_tokens is not None


async def gate_eval(llm: LLMClient, text: str) -> None:
    await LLMScopeGate(llm).evaluate(text)


async def test_unparseable_answer_falls_back_to_the_rules():
    # The regex gate matches this travel phrasing, so the fallback allows it.
    gate = LLMScopeGate(_StubLLM("I am not sure what you mean"))

    decision = await gate.evaluate("Plan one day in Prague")

    assert decision.provider == "rule_based"
    assert decision.passed is True


async def test_model_failure_falls_back_to_the_rules():
    # Never let a dead classifier decide by default — the deterministic gate does.
    gate = LLMScopeGate(_StubLLM(error=RuntimeError("connection refused")))

    decision = await gate.evaluate("Write a sorting function in Python")

    assert decision.provider == "rule_based"
    assert decision.passed is False


class _AlwaysAllow(GateEvaluator):
    name = GateName.SCOPE

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        return GateDecision(
            name=GateName.SCOPE,
            verdict=GateVerdict.ALLOW,
            passed=True,
            response="ok",
            reason="stub",
            provider="rule_based",
            confidence=1.0,
            latency_ms=0,
        )


def test_scope_client_defaults_to_the_main_backend():
    from backend.app.config import Settings
    from backend.app.llm.base import get_scope_llm_client

    # scope_base_url is set explicitly: Settings also reads the developer's .env,
    # and this must assert the default, not whatever that machine happens to run.
    client = get_scope_llm_client(Settings(llm_mode="mock", scope_base_url=None))

    assert client.mode == "mock"


def test_scope_client_uses_its_own_endpoint_when_configured():
    # A small CPU model answers yes/no without queueing behind the GPU model.
    from backend.app.config import Settings
    from backend.app.llm.base import get_scope_llm_client
    from backend.app.llm.vllm import VLLMClient

    settings = Settings(
        llm_mode="mock",
        scope_base_url="http://localhost:9001/v1",
        scope_model="qwen3-0.6b",
    )
    client = get_scope_llm_client(settings)

    # Not the mock: an explicit endpoint means a real HTTP client.
    assert isinstance(client, VLLMClient)
    assert client.mode == "vllm"
    assert client.model == "qwen3-0.6b"


async def test_fallback_gate_is_injectable():
    gate = LLMScopeGate(_StubLLM("garbage"), fallback=_AlwaysAllow())

    decision = await gate.evaluate("anything")

    assert decision.reason == "stub"


async def test_history_is_given_to_the_classifier_as_context():
    # A follow-up is in scope only because of what came before: judged alone,
    # "и что рядом?" looks like nothing at all.
    llm = _StubLLM("yes")
    gate = LLMScopeGate(llm)

    await gate.evaluate(
        "И что рядом?",
        history=[
            {"role": "user", "content": "Найди кофейни в Москве"},
            {"role": "assistant", "content": "Вот три кофейни в центре."},
        ],
    )

    sent = "".join(m.content for m in llm.requests[0].messages)
    assert "Найди кофейни в Москве" in sent
    assert "И что рядом?" in sent
    # Context must be labelled as context, or the model classifies the whole
    # dialogue and one in-scope opener would whitelist every later turn.
    assert "context only" in sent
    assert "Classify this request" in sent


async def test_only_the_recent_turns_are_sent():
    llm = _StubLLM("yes")
    gate = LLMScopeGate(llm)
    history = [{"role": "user", "content": f"turn-{i}"} for i in range(12)]

    await gate.evaluate("и дальше?", history=history)

    sent = "".join(m.content for m in llm.requests[0].messages)
    assert "turn-11" in sent  # newest kept
    assert "turn-0" not in sent  # oldest dropped


async def test_no_history_sends_only_the_message():
    llm = _StubLLM("yes")

    await LLMScopeGate(llm).evaluate("Find cafes in Prague")

    # system + the message itself; no empty context block.
    assert len(llm.requests[0].messages) == 2
    assert "context only" not in "".join(m.content for m in llm.requests[0].messages)


async def test_history_is_forwarded_to_the_fallback():
    # The regex gate ignores context, but a model-backed fallback might not.
    class _Recording(GateEvaluator):
        name = GateName.SCOPE

        def __init__(self) -> None:
            self.seen: Sequence[ConversationTurn] = ()

        async def evaluate(
            self, text: str, *, history: Sequence[ConversationTurn] = ()
        ) -> GateDecision:
            self.seen = history
            return GateDecision(
                name=GateName.SCOPE,
                verdict=GateVerdict.ALLOW,
                passed=True,
                response="ok",
                reason="stub",
                provider="rule_based",
                confidence=1.0,
                latency_ms=0,
            )

    fallback = _Recording()
    gate = LLMScopeGate(_StubLLM(error=RuntimeError("down")), fallback=fallback)

    await gate.evaluate("и что рядом?", history=[{"role": "user", "content": "кафе в Москве"}])

    assert fallback.seen and fallback.seen[0]["content"] == "кафе в Москве"
