from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from pydantic import BaseModel

from tools.base import PydanticTool, Tool, ToolSpec
from tools.observability import ToolExecutionContext
from tools.registry import (
    TOOL_SPECS,
    build_runtime_tool_registry,
    build_tool_specs,
    build_tools,
    get_tool_spec,
    list_tool_specs,
)
from tools.stubs import places_search_stub, routing_tool_stub, web_search_stub


def build_stub_tools() -> MappingProxyType[str, Tool[Any, Any]]:
    return build_runtime_tool_registry(
        places_search_handler=places_search_stub,
        routing_handler=routing_tool_stub,
        web_search_handler=web_search_stub,
    )


def test_registry_lists_only_model_facing_tools():
    """Verify that registry lists only model facing tools."""

    assert set(TOOL_SPECS) == {
        "places_search",
        "routing_tool",
        "web_search",
    }
    assert "geocode_place" not in TOOL_SPECS
    assert "geocoder" not in TOOL_SPECS


def test_get_tool_spec_returns_registered_spec():
    """Verify that get tool spec returns registered spec."""

    spec = get_tool_spec("places_search")

    assert spec.name == "places_search"
    assert spec.input_model.model_json_schema()


def test_get_tool_spec_rejects_unknown_tool():
    """Verify that get tool spec rejects unknown tool."""

    with pytest.raises(KeyError):
        get_tool_spec("geocode_place")


def test_list_tool_specs_is_deterministic():
    """Verify that list tool specs is deterministic."""

    specs = list_tool_specs()

    assert [spec.name for spec in specs] == [
        "places_search",
        "routing_tool",
        "web_search",
    ]


def test_tool_specs_mapping_is_read_only():
    """Verify that tool specs mapping is read only."""

    with pytest.raises(TypeError):
        TOOL_SPECS["other"] = get_tool_spec("places_search")  # type: ignore[index]


class DummyInput(BaseModel):
    value: str


class DummyOutput(BaseModel):
    value: str


class OtherInput(BaseModel):
    value: str


class OtherOutput(BaseModel):
    value: str


async def dummy_handler(args: DummyInput, _context: ToolExecutionContext) -> DummyOutput:
    return DummyOutput(value=args.value)


def test_build_tool_specs_rejects_duplicate_names():
    """Verify that build tool specs rejects duplicate names."""

    first = ToolSpec[DummyInput, DummyOutput](
        name="duplicate",
        description="First dummy tool.",
        input_model=DummyInput,
        output_model=DummyOutput,
    )
    second = ToolSpec[DummyInput, DummyOutput](
        name="duplicate",
        description="Second dummy tool.",
        input_model=DummyInput,
        output_model=DummyOutput,
    )

    with pytest.raises(ValueError, match="duplicate tool spec name: duplicate"):
        build_tool_specs(first, second)


def test_build_tools_rejects_missing_executable_tool():
    """Verify that build tools rejects missing executable tool."""

    specs = build_tool_specs(
        ToolSpec[DummyInput, DummyOutput](
            name="dummy",
            description="Dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"tool registry mismatch: missing_tools=\['dummy'\], unknown_tools=\[\]",
    ):
        build_tools(specs, {})


def test_build_tools_rejects_unknown_executable_tool():
    """Verify that build tools rejects unknown executable tool."""

    specs = build_tool_specs(
        ToolSpec[DummyInput, DummyOutput](
            name="dummy",
            description="Dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        )
    )

    unknown_tool = build_stub_tools()["places_search"]

    with pytest.raises(
        ValueError,
        match=r"tool registry mismatch: missing_tools=\['dummy'\], unknown_tools=\['other'\]",
    ):
        build_tools(specs, {"other": unknown_tool})


def test_build_tools_rejects_tool_name_mismatch():
    """Verify that build tools rejects tool name mismatch."""

    specs = build_tool_specs(
        ToolSpec[DummyInput, DummyOutput](
            name="dummy",
            description="Dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        )
    )
    tool = PydanticTool(
        spec=ToolSpec[DummyInput, DummyOutput](
            name="other",
            description="Other dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        ),
        handler=dummy_handler,
    )

    with pytest.raises(
        ValueError,
        match=r"tool registry mismatch for dummy: tool.name='other'",
    ):
        build_tools(specs, {"dummy": tool})


def test_build_tools_rejects_tool_input_model_mismatch():
    """Verify that build tools rejects tool input model mismatch."""

    specs = build_tool_specs(
        ToolSpec[DummyInput, DummyOutput](
            name="dummy",
            description="Dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        )
    )

    async def other_input_handler(
        args: OtherInput,
        _context: ToolExecutionContext,
    ) -> DummyOutput:
        return DummyOutput(value=args.value)

    tool = PydanticTool(
        spec=ToolSpec[OtherInput, DummyOutput](
            name="dummy",
            description="Dummy tool with wrong input.",
            input_model=OtherInput,
            output_model=DummyOutput,
        ),
        handler=other_input_handler,
    )

    with pytest.raises(
        ValueError,
        match="tool registry mismatch for dummy: input_model does not match spec",
    ):
        build_tools(specs, {"dummy": tool})


def test_build_tools_rejects_tool_output_model_mismatch():
    """Verify that build tools rejects tool output model mismatch."""

    specs = build_tool_specs(
        ToolSpec[DummyInput, DummyOutput](
            name="dummy",
            description="Dummy tool.",
            input_model=DummyInput,
            output_model=DummyOutput,
        )
    )

    async def other_output_handler(
        args: DummyInput,
        _context: ToolExecutionContext,
    ) -> OtherOutput:
        return OtherOutput(value=args.value)

    tool = PydanticTool(
        spec=ToolSpec[DummyInput, OtherOutput](
            name="dummy",
            description="Dummy tool with wrong output.",
            input_model=DummyInput,
            output_model=OtherOutput,
        ),
        handler=other_output_handler,
    )

    with pytest.raises(
        ValueError,
        match="tool registry mismatch for dummy: output_model does not match spec",
    ):
        build_tools(specs, {"dummy": tool})


def test_runtime_registry_lists_executable_model_facing_tools():
    """Verify that runtime registry lists executable model facing tools."""

    tools = build_stub_tools()

    assert set(tools) == {
        "places_search",
        "routing_tool",
        "web_search",
    }
    assert "geocode_place" not in tools
    assert "geocoder" not in tools


def test_runtime_registry_returns_executable_tool():
    """Verify that runtime registry returns executable tool."""

    tool = build_stub_tools()["places_search"]

    assert tool.name == "places_search"
    assert tool.input_model is get_tool_spec("places_search").input_model
    assert tool.output_model is get_tool_spec("places_search").output_model


def test_runtime_registry_rejects_unknown_tool():
    """Verify that runtime registry rejects unknown tool."""

    with pytest.raises(KeyError):
        build_stub_tools()["geocode_place"]


def test_runtime_registry_is_deterministic():
    """Verify that runtime registry is deterministic."""

    tools = build_stub_tools()

    assert [tools[name].name for name in sorted(tools)] == [
        "places_search",
        "routing_tool",
        "web_search",
    ]


def test_runtime_tools_mapping_is_read_only():
    """Verify that runtime tools mapping is read only."""

    tools = build_stub_tools()

    with pytest.raises(TypeError):
        tools["other"] = tools["places_search"]  # type: ignore[index]


async def test_registered_stub_tool_runs_through_wrapper():
    """Verify that registered stub tool runs through wrapper."""

    tool = build_stub_tools()["web_search"]

    result = await tool.run({"query": "  museum   renovation "})

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


async def test_registered_places_stub_returns_empty_successful_result():
    """Verify that registered places stub returns empty successful result."""

    tool = build_stub_tools()["places_search"]

    result = await tool.run({"mode": "area", "query": "coffee", "city": "Moscow"})

    assert result.ok is True
    assert result.tool_name == "places_search"
    assert result.data == {
        "places": [],
        "area": None,
        "truncated": False,
        "anchor": None,
        "resolved": [],
        "returned_count": 0,
    }
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.has_non_empty_answer is False


async def test_registered_routing_stub_returns_empty_successful_result():
    """Verify that registered routing stub returns empty successful result."""

    tool = build_stub_tools()["routing_tool"]

    result = await tool.run(
        {
            "mode": "rank",
            "origins": ["plc_a1b2c3d4e5"],
            "candidates": ["plc_b2c3d4e5f6"],
        }
    )

    assert result.ok is True
    assert result.tool_name == "routing_tool"
    assert result.data is not None
    assert result.data["mode"] == "rank"
    assert result.data["transport"] == "driving"
    assert result.data["ranked"] == []
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.has_non_empty_answer is False


async def test_registered_places_stub_preserves_near_anchor():
    """Verify that registered places stub preserves near anchor."""

    tool = build_stub_tools()["places_search"]

    result = await tool.run(
        {
            "mode": "near",
            "query": "coffee",
            "near": "plc_a1b2c3d4e5",
        }
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["anchor"] == "plc_a1b2c3d4e5"
    assert result.metrics is not None
    assert result.metrics.has_non_empty_answer is False


async def test_registered_routing_stub_returns_route_for_route_mode():
    """Verify that registered routing stub returns route for route mode."""

    tool = build_stub_tools()["routing_tool"]

    result = await tool.run(
        {
            "mode": "route",
            "waypoints": ["plc_a1b2c3d4e5", "plc_b2c3d4e5f6"],
        }
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["mode"] == "route"
    assert result.data["route"] is not None
    assert result.data["route"]["length_m"] == 0
    assert result.data["route"]["duration_s"] == 0
    assert result.data["route"]["waypoint_order"] == [0, 1]
    assert set(result.data) == {"mode", "transport", "route"}
    assert result.metrics is not None
    assert result.metrics.has_non_empty_answer is True


def test_runtime_registry_may_contain_only_active_tools() -> None:
    """Verify that runtime registry may contain only active tools."""

    tools = build_runtime_tool_registry(
        web_search_handler=web_search_stub,
    )

    assert set(tools) == {"web_search"}


def test_runtime_registry_may_be_empty() -> None:
    """Verify that runtime registry may be empty."""

    tools = build_runtime_tool_registry()

    assert not tools
