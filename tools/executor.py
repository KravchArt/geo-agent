"""Tool execution service.

The executor is the boundary between an agent/orchestrator and the tool
registry. It resolves a tool by name, executes it with raw model-provided args,
and always returns a ToolResult.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from tools.base import (
    Tool,
    ToolErrorCode,
    ToolMetrics,
    ToolResult,
    render_tool_error_observation,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Execute model-facing tools from an application-provided registry."""

    def __init__(self, tools: Mapping[str, Tool[Any, Any]]) -> None:
        self._tools = tools

    async def run(self, tool_name: str, raw_args: dict[str, Any] | None) -> ToolResult:
        """Execute a registered tool by name.

        Unknown tools are returned as INVALID_INPUT instead of raising KeyError,
        because an invented tool name is a model/orchestrator input error.
        """

        start = perf_counter()

        try:
            tool = self._tools[tool_name]
        except KeyError:
            error = f"unknown tool: {tool_name}"
            observation = render_tool_error_observation(ToolErrorCode.INVALID_INPUT, error)
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=error,
                error_code=ToolErrorCode.INVALID_INPUT,
                retryable=False,
                metrics=ToolMetrics(
                    latency_ms=int((perf_counter() - start) * 1000),
                    response_bytes=len(observation.encode("utf-8")),
                    model_tokens_estimate=max(1, len(observation) // 4),
                    success=False,
                    has_non_empty_answer=False,
                ),
            )

        result = await tool.run(raw_args or {})
        logger.info(
            "tool_executor_complete tool_name=%s ok=%s latency_ms=%s"
            "upstream_calls=%s upstream_latency_ms=%s",
            tool_name,
            result.ok,
            result.metrics.latency_ms if result.metrics else int((perf_counter() - start) * 1000),
            len(result.metrics.upstream_calls) if result.metrics else 0,
            result.metrics.upstream_latency_ms if result.metrics else None,
        )
        return result
