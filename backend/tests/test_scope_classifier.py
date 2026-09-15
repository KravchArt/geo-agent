"""Scope classifier client: response parsing and temperature calibration."""

from __future__ import annotations

import math

import httpx
import pytest

from backend.app.services.scope_classifier import (
    ScopeClassifierClient,
    ScopeClassifierError,
    calibrate,
)

#: The value shipped in artifacts/scope_bert_meta.json.
META_TEMPERATURE = 2.332827091217041


def _reference_score(logits: tuple[float, float], temperature: float) -> float:
    """What the training script computed: softmax(logits / T)[1]."""
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    return exps[1] / sum(exps)


def _softmax(logits: tuple[float, float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


@pytest.mark.parametrize(
    "logits",
    [(2.0, -1.0), (-3.5, 4.25), (0.1, -0.1), (8.0, -8.0), (-0.75, 0.75), (0.0, 0.0)],
)
def test_calibration_reproduces_softmax_over_scaled_logits(logits: tuple[float, float]) -> None:
    """vLLM softmaxes for us, so calibration must work backwards from probabilities.

    For two classes that is lossless: the probability fixes the logit difference,
    which is the only thing the temperature acts on.
    """
    served_probability = _softmax(logits)[1]

    calibrated = calibrate(served_probability, META_TEMPERATURE)

    assert calibrated == pytest.approx(_reference_score(logits, META_TEMPERATURE), abs=1e-9)


def test_calibration_is_identity_at_temperature_one() -> None:
    assert calibrate(0.73, 1.0) == pytest.approx(0.73, abs=1e-12)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_calibration_survives_saturated_probabilities(probability: float) -> None:
    """float32 softmax rounds confident outputs to exactly 0.0/1.0; logit() must not blow up."""
    result = calibrate(probability, META_TEMPERATURE)

    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(probability, abs=1e-2)


def test_calibration_pulls_toward_the_middle_for_t_above_one() -> None:
    """T > 1 is a confidence *reduction*; getting the direction wrong is silent."""
    assert 0.5 < calibrate(0.95, META_TEMPERATURE) < 0.95
    assert 0.05 < calibrate(0.05, META_TEMPERATURE) < 0.5


def _client(handler: object, **kwargs: object) -> ScopeClassifierClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return ScopeClassifierClient(
        base_url="http://classifier:8000",
        model="scope-bert",
        http_client=httpx.AsyncClient(transport=transport),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_score_reads_the_positive_class_and_calibrates() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "probs": [0.1, 0.9], "num_classes": 2}]},
        )

    client = _client(handler, temperature=META_TEMPERATURE)
    score = await client.score("где в тбилиси поесть хинкали")

    assert seen["url"] == "http://classifier:8000/classify"
    assert score.raw_probability == pytest.approx(0.9)
    assert score.in_scope_probability == pytest.approx(calibrate(0.9, META_TEMPERATURE))


async def test_score_applies_the_e5_prefix() -> None:
    """e5 was fine-tuned with 'query: '; omitting it degrades quality silently."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"data": [{"probs": [0.2, 0.8]}]})

    await _client(handler, prefix="query: ").score("хочу к морю")

    assert '"input": "query: \\u0445\\u043e\\u0447\\u0443' in seen["body"] or (
        "query: хочу к морю" in seen["body"]
    )


async def test_http_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(ScopeClassifierError):
        await _client(handler).score("test")


@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [{"probs": []}]},
        {"data": [{"probs": [0.5, 0.3, 0.2]}]},
        {"data": [{"probs": ["a", "b"]}]},
        {},
    ],
)
async def test_unusable_payloads_raise(payload: dict[str, object]) -> None:
    """A three-class head or an empty body means the wrong model is served."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ScopeClassifierError):
        await _client(handler).score("test")
