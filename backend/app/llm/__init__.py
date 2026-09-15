"""LLM client layer.

Exposes a single :class:`LLMClient` interface with two implementations —
``mock`` and ``vllm`` — selected by config (``LLM_MODE``). Both return the
identical :class:`common.models.LLMResponse` structure, so swapping backends is
a config change, never a code change.
"""

from backend.app.llm.base import LLMClient, get_llm_client

__all__ = ["LLMClient", "get_llm_client"]
