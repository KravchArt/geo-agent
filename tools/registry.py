"""Registry of model-facing tool contracts.

This module lists tools the model may call. Internal services such as the
geocoder are intentionally excluded.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from tools.base import PydanticTool, Tool, ToolHandler, ToolSpec
from tools.geo import (
    PLACES_SEARCH_SPEC,
    ROUTING_TOOL_SPEC,
    PlacesSearchInput,
    PlacesSearchOutput,
    RoutingInput,
    RoutingOutput,
)
from tools.web import (
    WEB_SEARCH_SPEC,
    WebSearchInput,
    WebSearchOutput,
)


def build_tool_specs(*specs: ToolSpec[Any, Any]) -> MappingProxyType[str, ToolSpec[Any, Any]]:
    registry: dict[str, ToolSpec[Any, Any]] = {}

    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"duplicate tool spec name: {spec.name}")
        registry[spec.name] = spec

    return MappingProxyType(registry)


def build_tools(
    specs: MappingProxyType[str, ToolSpec[Any, Any]],
    tools: dict[str, Tool[Any, Any]],
) -> MappingProxyType[str, Tool[Any, Any]]:
    spec_names = set(specs)
    tool_names = set(tools)

    if tool_names != spec_names:
        missing_tools = sorted(spec_names - tool_names)
        unknown_tools = sorted(tool_names - spec_names)
        raise ValueError(
            f"tool registry mismatch: missing_tools={missing_tools}, unknown_tools={unknown_tools}"
        )

    for name, tool in tools.items():
        spec = specs[name]

        if tool.name != name:
            raise ValueError(f"tool registry mismatch for {name}: tool.name={tool.name!r}")

        if tool.input_model is not spec.input_model:
            raise ValueError(f"tool registry mismatch for {name}: input_model does not match spec")

        if tool.output_model is not spec.output_model:
            raise ValueError(f"tool registry mismatch for {name}: output_model does not match spec")

    return MappingProxyType(dict(tools))


def build_runtime_tool_registry(
    *,
    places_search_handler: ToolHandler[PlacesSearchInput, PlacesSearchOutput] | None = None,
    routing_handler: ToolHandler[RoutingInput, RoutingOutput] | None = None,
    web_search_handler: ToolHandler[WebSearchInput, WebSearchOutput] | None = None,
    execution_timeout_s: float | None = None,
) -> MappingProxyType[str, Tool[Any, Any]]:
    """Build a registry containing only tools available in this runtime."""

    active_specs: list[ToolSpec[Any, Any]] = []
    active_tools: dict[str, Tool[Any, Any]] = {}

    if places_search_handler is not None:
        active_specs.append(PLACES_SEARCH_SPEC)
        active_tools[PLACES_SEARCH_SPEC.name] = PydanticTool(
            spec=PLACES_SEARCH_SPEC,
            handler=places_search_handler,
            timeout_s=execution_timeout_s,
        )

    if routing_handler is not None:
        active_specs.append(ROUTING_TOOL_SPEC)
        active_tools[ROUTING_TOOL_SPEC.name] = PydanticTool(
            spec=ROUTING_TOOL_SPEC,
            handler=routing_handler,
            timeout_s=execution_timeout_s,
        )

    if web_search_handler is not None:
        active_specs.append(WEB_SEARCH_SPEC)
        active_tools[WEB_SEARCH_SPEC.name] = PydanticTool(
            spec=WEB_SEARCH_SPEC,
            handler=web_search_handler,
            timeout_s=execution_timeout_s,
        )

    return build_tools(
        build_tool_specs(*active_specs),
        active_tools,
    )


TOOL_SPECS = build_tool_specs(
    PLACES_SEARCH_SPEC,
    ROUTING_TOOL_SPEC,
    WEB_SEARCH_SPEC,
)


def get_tool_spec(name: str) -> ToolSpec[Any, Any]:
    """Return a model-facing tool spec by name."""

    return TOOL_SPECS[name]


def list_tool_specs() -> list[ToolSpec[Any, Any]]:
    """Return all model-facing tool specs in deterministic order."""

    return [TOOL_SPECS[name] for name in sorted(TOOL_SPECS)]
