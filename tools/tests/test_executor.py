from __future__ import annotations

from types import MappingProxyType
from typing import Any

from tools.base import Tool, ToolErrorCode
from tools.executor import ToolExecutor
from tools.registry import build_runtime_tool_registry
from tools.stubs import places_search_stub, routing_tool_stub, web_search_stub


def build_stub_tools() -> MappingProxyType[str, Tool[Any, Any]]:
    return build_runtime_tool_registry(
        places_search_handler=places_search_stub,
        routing_handler=routing_tool_stub,
        web_search_handler=web_search_stub,
    )


async def test_executor_runs_registered_tool():
    """Verify that executor runs registered tool."""

    executor = ToolExecutor(build_stub_tools())

    result = await executor.run("web_search", {"query": "  museum   renovation "})

    assert result.ok is True
    assert result.tool_name == "web_search"
    assert result.data == {
        "query": "museum renovation",
        "results": [],
    }
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.has_non_empty_answer is False


async def test_executor_maps_unknown_tool_to_invalid_input():
    """Verify that executor maps unknown tool to invalid input."""

    executor = ToolExecutor(build_stub_tools())

    result = await executor.run("geocode_place", {"query": "Красная площадь"})

    assert result.ok is False
    assert result.tool_name == "geocode_place"
    assert result.error == "unknown tool: geocode_place"
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.data is None
    assert result.tool_hash is None
    assert result.metrics is not None
    assert result.metrics.success is False
    assert result.metrics.has_non_empty_answer is False
    assert result.metrics.response_bytes > 0
    assert result.metrics.model_tokens_estimate > 0
    assert result.retryable is False


async def test_executor_treats_missing_args_as_empty_dict():
    """Verify that executor treats missing args as empty dict."""

    executor = ToolExecutor(build_stub_tools())

    result = await executor.run("web_search", None)

    assert result.ok is False
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.tool_name == "web_search"


async def test_executor_uses_the_injected_registry() -> None:
    """Verify that executor uses the injected registry."""

    executor = ToolExecutor({})

    result = await executor.run("web_search", {"query": "museum renovation"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.error == "unknown tool: web_search"
