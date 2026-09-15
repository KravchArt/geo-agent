"""Tool schemas and interfaces.

Phase 0: contracts only — no concrete tools, no external API clients, no logic.
This package depends only on ``common``.

The central rule that shapes every schema here lives in :mod:`tools.refs`:
resolved data (coordinates, URLs) flows OUT of tools into Redis and the UI, and
the model only ever passes around short refs (``plc_...``, ``src_...``). Data the
model cannot see is data it cannot mistype.
"""

from tools.base import (
    PydanticTool,
    Tool,
    ToolClarification,
    ToolClarificationOption,
    ToolErrorCode,
    ToolExecutionError,
    ToolFailureKind,
    ToolMetrics,
    ToolResult,
    ToolSpec,
    ToolValidationIssue,
)
from tools.coordination import ProviderExecutionStrategy
from tools.executor import ToolExecutor
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, SourceRecord, SourceRef

__all__ = [
    "PlaceRecord",
    "PlaceRef",
    "ProviderExecutionStrategy",
    "PydanticTool",
    "RecordOrigin",
    "SourceRecord",
    "SourceRef",
    "Tool",
    "ToolClarification",
    "ToolClarificationOption",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolFailureKind",
    "ToolMetrics",
    "ToolResult",
    "ToolSpec",
    "ToolValidationIssue",
]
