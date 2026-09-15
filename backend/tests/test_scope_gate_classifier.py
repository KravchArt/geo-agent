"""ClassifierScopeGate: threshold decisions, the grey-zone cascade and failures."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.app.services.gates import ClassifierScopeGate, GateEvaluator
from backend.app.services.scope_classifier import ScopeClassifierError, ScopeScore
from common.models import ConversationTurn, GateDecision, GateName, GateVerdict


class _StubClient:
    """Returns a fixed score, or raises to simulate the sidecar being down."""

    def __init__(self, probability: float = 0.9, *, error: Exception | None = None) -> None:
        self.model = "scope-bert"
        self._probability = probability
        self._error = error
        self.calls: list[str] = []

    async def score(self, text: str) -> ScopeScore:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return ScopeScore(
            in_scope_probability=self._probability,
            raw_probability=self._probability,
        )


class _RecordingFallback(GateEvaluator):
    name = GateName.SCOPE

    def __init__(self, *, passed: bool = True) -> None:
        self.calls: list[str] = []
        self.seen_history: Sequence[ConversationTurn] = ()
        self._passed = passed

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        self.calls.append(text)
        self.seen_history = history
        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if self._passed else GateVerdict.REJECT,
            passed=self._passed,
            response="scoper",
            reason="scoper",
            provider="model",
            confidence=1.0,
            latency_ms=0,
        )


def _gate(
    client: _StubClient,
    fallback: _RecordingFallback,
    *,
    low: float = 0.4,
    high: float = 0.6,
) -> ClassifierScopeGate:
    return ClassifierScopeGate(client, fallback, grey_low=low, grey_high=high)


async def test_confident_in_scope_decides_without_the_fallback() -> None:
    client, fallback = _StubClient(0.97), _RecordingFallback()

    decision = await _gate(client, fallback).evaluate("маршрут по Казани")

    assert decision.passed is True
    assert decision.verdict is GateVerdict.ALLOW
    assert decision.provider == "model"
    assert decision.model == "scope-bert"
    assert decision.confidence == pytest.approx(0.97)
    assert fallback.calls == []


async def test_confident_out_of_scope_decides_without_the_fallback() -> None:
    client, fallback = _StubClient(0.02), _RecordingFallback()

    decision = await _gate(client, fallback).evaluate("напиши сортировку на питоне")

    assert decision.passed is False
    assert decision.verdict is GateVerdict.REJECT
    # Confidence is in the VERDICT, so a 0.02 in-scope score is a confident reject.
    assert decision.confidence == pytest.approx(0.98)
    assert fallback.calls == []


@pytest.mark.parametrize("probability", [0.41, 0.5, 0.59])
async def test_grey_zone_defers_to_the_fallback(probability: float) -> None:
    """This is why the cascade exists: the encoder cannot judge a follow-up."""
    client, fallback = _StubClient(probability), _RecordingFallback(passed=True)
    history = [{"role": "user", "content": "найди кофейни в Москве"}]

    decision = await _gate(client, fallback).evaluate("а что рядом?", history=history)

    assert decision.passed is True
    assert fallback.calls == ["а что рядом?"]
    # The fallback is the one that can use context, so it must receive it.
    assert fallback.seen_history == history
    # Both attempts are kept for the audit trail, classifier first.
    assert len(decision.attempts) == 2
    assert decision.attempts[0].confidence == pytest.approx(probability)
    assert decision.attempts[0].success is True


@pytest.mark.parametrize(("probability", "expected"), [(0.4, False), (0.6, True)])
async def test_grey_zone_bounds_are_inclusive_for_the_classifier(
    probability: float, expected: bool
) -> None:
    """Exactly on a bound the classifier still answers; only strictly inside defers."""
    client, fallback = _StubClient(probability), _RecordingFallback(passed=not expected)

    decision = await _gate(client, fallback).evaluate("test")

    assert decision.passed is expected
    assert fallback.calls == []


async def test_equal_bounds_disable_the_cascade() -> None:
    """low == high is a plain threshold — no score may fall through to the fallback."""
    fallback = _RecordingFallback()

    for probability in (0.0, 0.49, 0.5, 0.51, 1.0):
        client = _StubClient(probability)
        decision = await _gate(client, fallback, low=0.5, high=0.5).evaluate("test")
        assert decision.passed is (probability >= 0.5)

    assert fallback.calls == []


async def test_classifier_failure_falls_back() -> None:
    """A dead sidecar must degrade to the LLM scoper, not to allowing everything."""
    client = _StubClient(error=ScopeClassifierError("connection refused"))
    fallback = _RecordingFallback(passed=False)

    decision = await _gate(client, fallback).evaluate("напиши сортировку")

    assert decision.passed is False
    assert fallback.calls == ["напиши сортировку"]
    assert decision.attempts[0].success is False
    assert decision.attempts[0].error_type == "ScopeClassifierError"
