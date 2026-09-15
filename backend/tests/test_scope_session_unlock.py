"""Session-unlock policy: once a session is in scope, later turns skip the classifier."""

from __future__ import annotations

from typing import Any, cast

import pytest

from backend.app.config import Settings
from backend.app.redis.client import RedisClient
from backend.app.services.gates import ClassifierScopeGate, RuleBasedScopeGate
from backend.app.services.pipeline import (
    SCOPE_UNLOCKED_FIELD,
    _remember_scope_verdict,
    _scope_is_unlocked,
    _session_unlock_applies,
)
from common.models import GateDecision, GateName, GateVerdict


class _FakeRedis:
    """Minimal session-value store; can be made to fail like a dead Redis."""

    def __init__(self, *, broken: bool = False) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self._broken = broken

    async def get_session_value(self, session_id: str, field: str) -> str | None:
        if self._broken:
            raise ConnectionError("redis is down")
        return self.values.get((session_id, field))

    async def set_session_value(
        self, session_id: str, field: str, value: str, ttl: int | None = None
    ) -> None:
        if self._broken:
            raise ConnectionError("redis is down")
        self.values[(session_id, field)] = value


def _settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


def _classifier_gate() -> ClassifierScopeGate:
    return ClassifierScopeGate(
        object(),  # type: ignore[arg-type]
        RuleBasedScopeGate(),
        grey_low=0.4,
        grey_high=0.6,
    )


def _decision(*, passed: bool) -> GateDecision:
    return GateDecision(
        name=GateName.SCOPE,
        verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
        passed=passed,
        response="x",
        reason="x",
        provider="model",
        confidence=1.0,
        latency_ms=0,
    )


def test_unlock_applies_only_to_the_classifier_gate() -> None:
    """Other gates read history already, so skipping them would only lose filtering."""
    settings = _settings(scope_session_unlock=True)

    assert _session_unlock_applies(settings, _classifier_gate()) is True
    assert _session_unlock_applies(settings, RuleBasedScopeGate()) is False
    assert _session_unlock_applies(settings, None) is False


def test_unlock_can_be_switched_off() -> None:
    settings = _settings(scope_session_unlock=False)

    assert _session_unlock_applies(settings, _classifier_gate()) is False


async def test_first_in_scope_verdict_unlocks_the_session() -> None:
    redis, settings, gate = _FakeRedis(), _settings(), _classifier_gate()

    assert (
        await _scope_is_unlocked(
            redis=cast(RedisClient, redis), settings=settings, session_id="s1", scope_gate=gate
        )
        is False
    )

    await _remember_scope_verdict(
        redis=cast(RedisClient, redis),
        settings=settings,
        session_id="s1",
        scope=_decision(passed=True),
        scope_gate=gate,
    )

    assert redis.values[("s1", SCOPE_UNLOCKED_FIELD)] == "1"
    assert (
        await _scope_is_unlocked(
            redis=cast(RedisClient, redis), settings=settings, session_id="s1", scope_gate=gate
        )
        is True
    )


async def test_rejected_verdict_leaves_the_session_locked() -> None:
    """Out-of-scope turns keep being classified until one passes."""
    redis, settings, gate = _FakeRedis(), _settings(), _classifier_gate()

    for _ in range(3):
        await _remember_scope_verdict(
            redis=cast(RedisClient, redis),
            settings=settings,
            session_id="s1",
            scope=_decision(passed=False),
            scope_gate=gate,
        )

    assert redis.values == {}
    assert (
        await _scope_is_unlocked(
            redis=cast(RedisClient, redis), settings=settings, session_id="s1", scope_gate=gate
        )
        is False
    )


async def test_unlock_is_per_session() -> None:
    redis, settings, gate = _FakeRedis(), _settings(), _classifier_gate()

    await _remember_scope_verdict(
        redis=cast(RedisClient, redis),
        settings=settings,
        session_id="s1",
        scope=_decision(passed=True),
        scope_gate=gate,
    )

    assert (
        await _scope_is_unlocked(
            redis=cast(RedisClient, redis), settings=settings, session_id="other", scope_gate=gate
        )
        is False
    )


async def test_redis_failure_fails_closed() -> None:
    """A dead Redis must classify, not skip the gate."""
    redis, settings, gate = _FakeRedis(broken=True), _settings(), _classifier_gate()

    assert (
        await _scope_is_unlocked(
            redis=cast(RedisClient, redis), settings=settings, session_id="s1", scope_gate=gate
        )
        is False
    )

    # Writing the flag must not propagate either — it only costs one extra call.
    await _remember_scope_verdict(
        redis=cast(RedisClient, redis),
        settings=settings,
        session_id="s1",
        scope=_decision(passed=True),
        scope_gate=gate,
    )


def test_unlocked_gate_records_why_it_passed() -> None:
    """The skip must be auditable, not an invisible allow."""
    import asyncio

    from backend.app.services.gates import SessionUnlockedScopeGate

    decision = asyncio.run(SessionUnlockedScopeGate().evaluate("напиши сортировку"))

    assert decision.passed is True
    assert decision.provider == "session_unlocked"
    assert "already produced an in-scope verdict" in decision.reason


@pytest.mark.parametrize("provider", ["rule_based", "llm", "classifier", "no_scoper"])
def test_scope_provider_accepts_the_new_value(provider: str) -> None:
    extra: dict[str, Any] = {"scope_provider": provider}
    if provider == "classifier":
        extra |= {
            "scope_classifier_base_url": "http://classifier:8000",
            "scope_classifier_model": "scope-bert",
        }
    assert Settings(**extra).scope_provider == provider


def test_classifier_provider_requires_an_endpoint() -> None:
    with pytest.raises(ValueError, match="SCOPE_CLASSIFIER_BASE_URL"):
        Settings(scope_provider="classifier", scope_classifier_model="scope-bert")


def test_inverted_grey_zone_is_rejected() -> None:
    with pytest.raises(ValueError, match="GREY_LOW"):
        Settings(scope_classifier_grey_low=0.7, scope_classifier_grey_high=0.3)
