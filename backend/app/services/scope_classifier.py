"""Client for the fine-tuned scope classifier served by vLLM.

The model is ``intfloat/multilingual-e5-base`` fine-tuned for binary scope
classification (see ``artifacts/scope_bert_meta.json`` next to the weights). It is
served by vLLM's pooling runner, so this talks to ``POST /classify`` — not to the
chat API. Nothing about it is generative: one forward pass, two probabilities.

Two details are easy to get wrong and both fail *quietly*, degrading quality
without raising anything:

* the ``query: `` prefix e5 was trained with,
* the calibration temperature, which applies to the LOGIT, not to the probability.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


class ScopeClassifierError(RuntimeError):
    """The classifier could not produce a usable score."""


@dataclass(frozen=True, slots=True)
class ScopeScore:
    """Calibrated probability that the request is in scope."""

    in_scope_probability: float
    raw_probability: float


class ScopeClassifierScorer(Protocol):
    """Minimal classifier contract required by the scope gate."""

    @property
    def model(self) -> str: ...

    async def score(self, text: str) -> ScopeScore: ...


def calibrate(in_scope_probability: float, temperature: float) -> float:
    """Apply the calibration temperature to a softmax probability.

    vLLM's classify endpoint applies softmax itself, so the raw logits are gone by
    the time we see the response. For two classes nothing is actually lost: the
    softmax probability determines the logit difference exactly
    (``logit(p1) = z1 - z0``), so dividing that by the temperature and squashing it
    back through the sigmoid reproduces ``softmax(logits / T)[1]`` to the last bit.

    Only exact 0.0 and 1.0 are special-cased, and they are already the limit of the
    calibrated value. Clamping anything else would corrupt genuinely small
    probabilities — a confident logit pair like (8, -8) legitimately yields 1.1e-7.
    """
    if in_scope_probability <= 0.0:
        return 0.0
    if in_scope_probability >= 1.0:
        return 1.0
    # log1p keeps the p -> 1 tail accurate, where 1 - p loses most of its digits.
    logit = math.log(in_scope_probability) - math.log1p(-in_scope_probability)
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _in_scope_probability(payload: dict[str, Any]) -> float:
    """Pull the positive-class probability out of a /classify response.

    Index 1 is the in-scope class — the training script scored
    ``softmax(logits / T)[:, 1]``, and the checkpoint carries no ``id2label`` to
    read a name from, so the position is the contract.
    """
    try:
        data = payload["data"]
        probs = data[0]["probs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ScopeClassifierError(f"unexpected /classify response shape: {payload!r}") from exc

    if not isinstance(probs, list) or len(probs) != 2:
        raise ScopeClassifierError(
            f"expected 2 class probabilities, got {probs!r} — is this the right model?"
        )
    try:
        return float(probs[1])
    except (TypeError, ValueError) as exc:
        raise ScopeClassifierError(f"non-numeric probability in {probs!r}") from exc


class ScopeClassifierClient:
    """Posts one request to vLLM's ``/classify`` and returns a calibrated score."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        prefix: str = "query: ",
        temperature: float = 1.0,
        timeout: int = 5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/classify"
        self._model = model
        self._api_key = api_key
        self._prefix = prefix
        self._temperature = temperature
        self._timeout = timeout
        self._client = http_client

    @property
    def model(self) -> str:
        return self._model

    async def _post(self, text: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {"model": self._model, "input": f"{self._prefix}{text}"}
        if self._client is not None:
            response = await self._client.post(self._url, json=body, headers=headers)
        else:
            # trust_env=False for the same reason as the LLM client: the model
            # server is addressed explicitly and a shell proxy must not reroute it.
            async with httpx.AsyncClient(timeout=float(self._timeout), trust_env=False) as client:
                response = await client.post(self._url, json=body, headers=headers)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    async def score(self, text: str) -> ScopeScore:
        """Return the calibrated in-scope probability for one request."""
        try:
            payload = await self._post(text)
        except httpx.HTTPError as exc:
            raise ScopeClassifierError(f"classify request failed: {exc}") from exc

        raw = _in_scope_probability(payload)
        return ScopeScore(
            in_scope_probability=calibrate(raw, self._temperature),
            raw_probability=raw,
        )

    async def health(self) -> bool:
        try:
            await self.score("health check")
        except Exception:
            return False
        return True
