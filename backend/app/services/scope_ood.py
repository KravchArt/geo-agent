"""Out-of-distribution check for the scope classifier: "have I seen this before?"

The classifier's probability answers *which side of the boundary* a request falls
on, and it answers honestly — measured on production-shaped logs it is well
calibrated (ECE 0.007, promising 0.984 where the true rate is 0.989). What it
cannot answer is whether there is any evidence behind that number. A request
about salaries in Switzerland sits far from anything in the training corpus, on
the wrong side of the boundary, and the classifier reports 0.984 for it.

That is epistemic uncertainty, and softmax has no way to express it. Temperature
does not help either: it is a single global scalar that divides every logit
alike. So the grey zone cannot catch these — the errors are not near the
threshold, they are confidently wrong at 0.89-0.98.

What does catch them is distance. The training corpus was generated from 203
templates; a request unlike all of them is one the classifier has no grounds to
judge. We embed the request with the same encoder, take the mean cosine
similarity to its ``k`` nearest training rows, and defer anything too far to the
LLM scoper — which reads the request rather than pattern-matching it.

Measured on 196 labelled production-shaped requests:

    mean similarity, correct verdicts : 0.963
    mean similarity, wrong verdicts   : 0.802
    defer the farthest 2.6%  -> 1 of 4 false positives caught
    defer the farthest 11.7% -> 3 of 4 false positives caught

Three properties of this signal that matter operationally:

* **It is not a probability.** 0.78 does not mean "22% chance of error", and it
  cannot be calibrated into one. Only its quantile against the training
  distribution is meaningful, which is why the threshold is expressed that way.
* **It is tied to one encoder and one corpus.** Retrain the classifier or extend
  the templates and the index must be rebuilt, or the distances drift silently.
  ``encoder`` in the index file is checked against the configured model for
  exactly this reason.
* **The reference distribution excludes same-template neighbours.** Every training
  row has ~50 paraphrase siblings; counting them makes the reference far too
  tight and every live request look anomalous. Building the index without that
  exclusion moved the 1% threshold from 0.83 to 0.95 and flagged 15% of traffic
  instead of 2.6%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class ScopeOODError(RuntimeError):
    """The OOD check could not produce a usable distance."""


@dataclass(frozen=True, slots=True)
class OODVerdict:
    """How far a request sits from the training corpus."""

    similarity: float
    #: Fraction of training rows that are *farther* from their neighbours than
    #: this request is from its own. 0.01 means "more distant than 99% of them".
    quantile: float
    far: bool


class ScopeEmbedder(Protocol):
    """Minimal embedding contract required by the OOD detector."""

    @property
    def model(self) -> str: ...

    async def embed(self, text: str) -> np.ndarray: ...


class OODIndex:
    """Training-corpus embeddings plus the reference distribution of distances.

    Built offline by ``tools/build_scope_ood_index.py`` in the classifier repo and
    shipped as a single ``.npz``. Loading is cheap; the whole index for the 11 142
    row corpus is 768 floats per row, about 33 MB in memory.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        reference: np.ndarray,
        *,
        k: int = 10,
        encoder: str | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ScopeOODError(f"expected a 2-D embedding matrix, got shape {embeddings.shape}")
        if reference.ndim != 1 or reference.size == 0:
            raise ScopeOODError("the reference distribution is empty — rebuild the index")
        self._emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._reference = np.sort(np.asarray(reference, dtype=np.float32))
        self._k = min(k, len(self._emb))
        self.encoder = encoder

    @property
    def rows(self) -> int:
        return int(self._emb.shape[0])

    @property
    def dim(self) -> int:
        return int(self._emb.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> OODIndex:
        path = Path(path)
        if not path.exists():
            raise ScopeOODError(f"OOD index not found at {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = {"embeddings", "reference"} - set(data.files)
            if missing:
                raise ScopeOODError(f"{path} is missing {sorted(missing)} — rebuild the index")
            encoder = str(data["encoder"]) if "encoder" in data.files else None
            k = int(data["k"]) if "k" in data.files else 10
            return cls(data["embeddings"], data["reference"], k=k, encoder=encoder)

    def threshold(self, quantile: float) -> float:
        """Similarity below which a request counts as far from the corpus."""
        return float(np.quantile(self._reference, quantile))

    def evaluate(self, embedding: np.ndarray, *, quantile: float) -> OODVerdict:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size != self.dim:
            raise ScopeOODError(f"embedding has {vec.size} dims, index has {self.dim}")
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ScopeOODError("embedding is all zeros")
        sims = self._emb @ (vec / norm)
        # Mean over k neighbours rather than the single nearest: one lucky match
        # in a corpus of paraphrases says much less than a consistent neighbourhood.
        similarity = float(np.sort(sims)[-self._k :].mean())
        rank = int(np.searchsorted(self._reference, similarity))
        return OODVerdict(
            similarity=similarity,
            quantile=rank / len(self._reference),
            far=similarity <= self.threshold(quantile),
        )


class ScopeEmbeddingClient:
    """Fetches one embedding from an OpenAI-compatible ``/v1/embeddings`` endpoint.

    This must serve the **same checkpoint** the classifier uses, run with
    ``--runner pooling --task embed``. The fine-tuned encoder's space is not the
    base model's space, and the index was built in the former; pointing this at
    stock ``multilingual-e5-base`` would produce distances that look plausible and
    mean nothing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        prefix: str = "query: ",
        timeout: int = 5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._api_key = api_key
        self._prefix = prefix
        self._timeout = timeout
        self._client = http_client

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {"model": self._model, "input": f"{self._prefix}{text}"}
        try:
            if self._client is not None:
                response = await self._client.post(self._url, json=body, headers=headers)
            else:
                # trust_env=False for the same reason as the classifier client: the
                # model server is addressed explicitly and a shell proxy must not
                # reroute it.
                async with httpx.AsyncClient(
                    timeout=float(self._timeout), trust_env=False
                ) as client:
                    response = await client.post(self._url, json=body, headers=headers)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except httpx.HTTPError as exc:
            raise ScopeOODError(f"embedding request failed: {exc}") from exc

        try:
            vector = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ScopeOODError(f"unexpected /embeddings response shape: {payload!r}") from exc
        if not isinstance(vector, list) or not vector:
            raise ScopeOODError(f"empty embedding in {payload!r}")
        return np.asarray(vector, dtype=np.float32)


class ScopeOODDetector:
    """Ties the embedding endpoint to the index. One call, one verdict."""

    def __init__(
        self,
        client: ScopeEmbedder,
        index: OODIndex,
        *,
        quantile: float,
    ) -> None:
        self._client = client
        self._index = index
        self._quantile = quantile
        if index.encoder and index.encoder != client.model:
            # A warning, not an error: the served model name is a deployment
            # detail and may legitimately differ from the name used at build time.
            # Getting it wrong is silent, so it must at least be loud in the log.
            logger.warning(
                "scope_ood_encoder_mismatch index_built_with=%r serving=%r "
                "— distances are only meaningful for the checkpoint the index was built from",
                index.encoder,
                client.model,
            )

    @property
    def quantile(self) -> float:
        return self._quantile

    @property
    def threshold(self) -> float:
        return self._index.threshold(self._quantile)

    async def evaluate(self, text: str) -> OODVerdict:
        embedding = await self._client.embed(text)
        return self._index.evaluate(embedding, quantile=self._quantile)
