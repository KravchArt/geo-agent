"""Scope OOD check: index arithmetic, the extra deferral, shadow mode, failures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from backend.app.services.gates import ClassifierScopeGate, GateEvaluator
from backend.app.services.scope_classifier import ScopeScore
from backend.app.services.scope_ood import (
    OODIndex,
    ScopeOODDetector,
    ScopeOODError,
)
from common.models import ConversationTurn, GateDecision, GateName, GateVerdict


def _unit(*values: float) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _index(k: int = 1) -> OODIndex:
    """Three training rows clustered around one direction, plus a reference.

    The reference is what the quantile threshold is read off, so it is set
    explicitly rather than derived — the tests are about the decision, not about
    how the reference was computed offline.
    """
    embeddings = np.stack([_unit(1, 0), _unit(0.99, 0.14), _unit(0.98, 0.2)])
    reference = np.asarray([0.90, 0.95, 0.99], dtype=np.float32)
    return OODIndex(embeddings, reference, k=k, encoder="scope-classifier")


class _StubClient:
    def __init__(self, probability: float = 0.98) -> None:
        self.model = "scope-classifier"
        self._probability = probability

    async def score(self, text: str) -> ScopeScore:
        return ScopeScore(in_scope_probability=self._probability, raw_probability=self._probability)


class _StubEmbedder:
    def __init__(self, vector: np.ndarray, *, error: Exception | None = None) -> None:
        self.model = "scope-classifier"
        self._vector = vector
        self._error = error
        self.calls: list[str] = []

    async def embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._vector


class _RecordingFallback(GateEvaluator):
    name = GateName.SCOPE

    def __init__(self, *, passed: bool = False) -> None:
        self.calls: list[str] = []
        self._passed = passed

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        self.calls.append(text)
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


def _detector(vector: np.ndarray, *, quantile: float = 0.5, error: Exception | None = None):
    return ScopeOODDetector(_StubEmbedder(vector, error=error), _index(), quantile=quantile)


# --- index arithmetic ------------------------------------------------------


def test_near_request_is_not_flagged() -> None:
    verdict = _index().evaluate(_unit(1, 0.02), quantile=0.5)
    assert verdict.similarity > 0.99
    assert not verdict.far


def test_far_request_is_flagged() -> None:
    verdict = _index().evaluate(_unit(0, 1), quantile=0.5)
    assert verdict.similarity < 0.3
    assert verdict.far
    assert verdict.quantile == 0.0


def test_embedding_is_normalised_before_comparison() -> None:
    """A client returning unnormalised vectors must not change the distance."""
    small = _index().evaluate(_unit(1, 0.02) * 0.01, quantile=0.5)
    large = _index().evaluate(_unit(1, 0.02) * 100.0, quantile=0.5)
    assert small.similarity == pytest.approx(large.similarity, abs=1e-5)


def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ScopeOODError, match="dims"):
        _index().evaluate(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), quantile=0.5)


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ScopeOODError, match="zeros"):
        _index().evaluate(np.zeros(2, dtype=np.float32), quantile=0.5)


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ScopeOODError, match="reference"):
        OODIndex(np.stack([_unit(1, 0)]), np.asarray([], dtype=np.float32))


def test_load_reports_a_missing_file() -> None:
    with pytest.raises(ScopeOODError, match="not found"):
        OODIndex.load("/nonexistent/index.npz")


def test_roundtrip_through_a_file(tmp_path) -> None:
    path = tmp_path / "index.npz"
    source = _index()
    np.savez_compressed(
        path,
        embeddings=np.stack([_unit(1, 0), _unit(0.99, 0.14), _unit(0.98, 0.2)]),
        reference=np.asarray([0.90, 0.95, 0.99], dtype=np.float32),
        k=np.int64(1),
        encoder=np.str_("scope-classifier"),
    )
    loaded = OODIndex.load(path)
    assert loaded.rows == source.rows
    assert loaded.encoder == "scope-classifier"
    assert loaded.evaluate(_unit(0, 1), quantile=0.5).far


# --- gate behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_far_request_is_deferred_despite_a_confident_score() -> None:
    """The whole point: 0.98 is not evidence when nothing like it was trained on."""
    fallback = _RecordingFallback(passed=False)
    gate = ClassifierScopeGate(
        _StubClient(probability=0.98),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(0, 1)),
    )
    decision = await gate.evaluate("сколько зарабатывать, чтобы жить в Швейцарии")
    assert fallback.calls == ["сколько зарабатывать, чтобы жить в Швейцарии"]
    assert not decision.passed
    assert len(decision.attempts) == 2


@pytest.mark.asyncio
async def test_near_request_is_decided_by_the_classifier_alone() -> None:
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(
        _StubClient(probability=0.98),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(1, 0.02)),
    )
    decision = await gate.evaluate("найди кофейни в Праге")
    assert fallback.calls == []
    assert decision.passed
    assert "Distance to corpus" in decision.reason


@pytest.mark.asyncio
async def test_shadow_mode_records_the_distance_without_changing_the_verdict() -> None:
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(
        _StubClient(probability=0.98),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(0, 1)),
        ood_enforce=False,
    )
    decision = await gate.evaluate("сколько зарабатывать, чтобы жить в Швейцарии")
    assert fallback.calls == []
    assert decision.passed
    assert "shadow mode" in decision.reason


@pytest.mark.asyncio
async def test_a_dead_embedding_endpoint_leaves_the_gate_working() -> None:
    """Fail-open by design: without the distance the gate is what it was before."""
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(
        _StubClient(probability=0.98),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(0, 1), error=ScopeOODError("sidecar down")),
    )
    decision = await gate.evaluate("найди кофейни в Праге")
    assert fallback.calls == []
    assert decision.passed


@pytest.mark.asyncio
async def test_out_of_scope_request_is_still_rejected_outright_when_familiar() -> None:
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(
        _StubClient(probability=0.02),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(1, 0.02)),
    )
    decision = await gate.evaluate("как приготовить плов")
    assert fallback.calls == []
    assert not decision.passed
    assert decision.verdict is GateVerdict.REJECT


@pytest.mark.asyncio
async def test_grey_zone_still_defers_when_the_request_is_familiar() -> None:
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(
        _StubClient(probability=0.5),
        fallback,
        grey_low=0.4,
        grey_high=0.6,
        ood=_detector(_unit(1, 0.02)),
    )
    await gate.evaluate("а что рядом?")
    assert fallback.calls == ["а что рядом?"]


@pytest.mark.asyncio
async def test_gate_without_an_index_behaves_exactly_as_before() -> None:
    fallback = _RecordingFallback()
    gate = ClassifierScopeGate(_StubClient(probability=0.98), fallback, grey_low=0.4, grey_high=0.6)
    decision = await gate.evaluate("найди кофейни в Праге")
    assert fallback.calls == []
    assert decision.passed
    assert "Distance to corpus" not in decision.reason
