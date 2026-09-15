"""LLM client interface + factory.

The factory returns a ``mock`` or ``vllm`` client based on ``Settings.llm_mode``.
Both implementations honour the same :class:`LLMClient` contract and return the
same response shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from backend.app.config import Settings, get_settings
from common.models import LLMRequest, LLMResponse


class LLMClient(ABC):
    """Abstract LLM client. Implementations MUST return identical structures."""

    #: "mock" or "vllm" — echoed into LLMResponse.mode.
    mode: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one generation and return a structured response."""
        raise NotImplementedError

    async def generate_stream(
        self,
        request: LLMRequest,
        on_text_chunk: Callable[[str], None],
    ) -> LLMResponse:
        """Generate while reporting assistant text chunks.

        Providers without a native streaming implementation retain the same
        contract and emit their completed text as one chunk.
        """
        response = await self.generate(request)
        if response.content:
            on_text_chunk(response.content)
        return response

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable/usable."""
        raise NotImplementedError


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the configured LLM client. Imports concrete clients lazily."""
    settings = settings or get_settings()
    if settings.llm_mode == "mock":
        from backend.app.llm.mock import MockLLMClient

        return MockLLMClient(model=settings.llm_model)

    from backend.app.llm.vllm import VLLMClient

    return VLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
        proxy=settings.llm_http_proxy,
        reasoning_effort=settings.llm_reasoning_effort,
        reasoning_max_tokens=settings.llm_reasoning_max_tokens,
    )


def get_scope_llm_client(settings: Settings | None = None) -> LLMClient:
    """Client for the scope classifier.

    With ``SCOPE_BASE_URL`` set it talks to a dedicated endpoint — a small model
    on CPU answers yes/no in tens of milliseconds, so a one-word classification
    on the critical path does not queue behind the main model on the GPU.
    Unset, it reuses the main backend.
    """
    settings = settings or get_settings()
    if not settings.scope_base_url:
        return get_llm_client(settings)

    from backend.app.llm.vllm import VLLMClient

    return VLLMClient(
        base_url=settings.scope_base_url,
        api_key=settings.scope_api_key,
        model=settings.scope_model or settings.llm_model,
        timeout=settings.llm_timeout,
    )


def get_censorship_llm_client(settings: Settings | None = None) -> LLMClient:
    """Client for the censorship classifier.

    Same reasoning as :func:`get_scope_llm_client`: with ``CENSORSHIP_BASE_URL``
    set the verdict comes from a dedicated endpoint instead of queueing behind the
    main model. Unset, it reuses the main backend.
    """
    settings = settings or get_settings()
    if not settings.censorship_base_url:
        return get_llm_client(settings)

    from backend.app.llm.vllm import VLLMClient

    return VLLMClient(
        base_url=settings.censorship_base_url,
        api_key=settings.censorship_api_key,
        model=settings.censorship_model or settings.llm_model,
        timeout=settings.llm_timeout,
    )
