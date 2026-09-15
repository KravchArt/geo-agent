from __future__ import annotations

import asyncio
import json
import logging

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.base import (
    PydanticTool,
    ToolClarification,
    ToolClarificationOption,
    ToolErrorCode,
    ToolExecutionError,
    ToolFailureKind,
    ToolSpec,
    has_non_empty_answer,
    render_tool_observation,
)
from tools.geo.places_search import PLACES_SEARCH_SPEC
from tools.observability import ToolExecutionContext, UpstreamCallMetrics
from tools.stubs import places_search_stub


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return " ".join(value.split())


class EchoOutput(BaseModel):
    text: str


async def echo_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
    return EchoOutput(text=args.text)


async def test_pydantic_tool_wraps_successful_handler_result():
    """Verify that Pydantic tool wraps successful handler result."""

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=echo_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.tool_name == "echo"
    assert result.data == {"text": "hello"}
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.upstream_latency_ms is None
    assert result.metrics.has_non_empty_answer is True
    assert result.metrics.response_bytes > 0
    assert result.metrics.model_tokens_estimate > 0


async def test_pydantic_tool_rejects_invalid_input_before_handler():
    """Verify that Pydantic tool rejects invalid input before handler."""

    handler_was_called = False

    async def handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        nonlocal handler_was_called
        handler_was_called = True
        return EchoOutput(text=args.text)

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=handler,
    )

    result = await tool.run({"text": "hello", "apikey": "super-secret"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.error is not None
    assert result.error.startswith("Invalid tool input; correct the listed fields and retry")
    assert "VALIDATION_ERRORS:" in result.error
    assert result.data is None
    assert result.tool_hash is None
    assert result.metrics is not None
    assert result.metrics.success is False
    assert result.metrics.has_non_empty_answer is False
    assert result.metrics.response_bytes > 0
    assert result.metrics.model_tokens_estimate > 0
    assert handler_was_called is False

    assert len(result.validation_errors) == 1
    issue = result.validation_errors[0]
    assert issue.path == "apikey"
    assert issue.code == "extra_forbidden"
    assert issue.message == "Extra inputs are not permitted"
    assert "super-secret" not in result.model_dump_json()


async def test_pydantic_tool_exposes_stable_model_validation_code():
    """Verify that Pydantic tool exposes stable model validation code."""

    tool = PydanticTool(
        spec=PLACES_SEARCH_SPEC,
        handler=places_search_stub,
    )

    result = await tool.run(
        {
            "mode": "near",
            "query": "кофейни",
        }
    )

    assert result.ok is False
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert len(result.validation_errors) == 1

    issue = result.validation_errors[0]
    assert issue.path == "$"
    assert issue.code == "places_search_near_anchor_required"


async def test_places_resolve_rejects_duplicate_client_ids() -> None:
    tool = PydanticTool(spec=PLACES_SEARCH_SPEC, handler=places_search_stub)

    result = await tool.run(
        {
            "mode": "resolve",
            "city": "Berlin",
            "organisations": [
                {"client_id": "candidate", "name": "Alpha"},
                {"client_id": "candidate", "name": "Beta"},
            ],
        }
    )

    assert result.ok is False
    assert result.validation_errors[0].code == "places_search_resolve_client_ids_unique"


async def test_places_resolve_rejects_discovery_query() -> None:
    tool = PydanticTool(spec=PLACES_SEARCH_SPEC, handler=places_search_stub)

    result = await tool.run(
        {
            "mode": "resolve",
            "query": "restaurants",
            "city": "Berlin",
            "organisations": [{"client_id": "candidate", "name": "Alpha"}],
        }
    )

    assert result.ok is False
    assert result.validation_errors[0].code == ("places_search_resolve_discovery_fields_forbidden")


async def test_pydantic_tool_rejects_invalid_handler_output(
    caplog: pytest.LogCaptureFixture,
):
    """Verify that Pydantic tool rejects invalid handler output."""

    async def bad_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        return {"url": "https://provider.example/?apikey=super-secret"}  # type: ignore[return-value]

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=bad_handler,
    )

    with caplog.at_level(logging.ERROR, logger="tools.base"):
        result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert result.error == "Incorrect answer from tool"
    assert "super-secret" not in result.error
    assert result.data is None
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is False
    assert result.metrics.has_non_empty_answer is False
    assert result.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert result.retryable is False
    assert result.metrics.response_bytes > 0
    assert result.metrics.model_tokens_estimate > 0

    log_output = "\n".join(caplog.messages)
    assert "tool_output_validation_failed" in log_output
    assert "tool_name=echo" in log_output
    assert "validation_error_count=1" in log_output
    assert "super-secret" not in log_output


async def test_pydantic_tool_maps_timeout_error():
    """Verify that Pydantic tool maps timeout error."""

    async def timeout_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        raise TimeoutError("provider timed out with api key super-secret")

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=timeout_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.TIMEOUT
    assert result.error == "tool timed out"
    assert "super-secret" not in result.error
    assert result.data is None
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is False
    assert result.metrics.has_non_empty_answer is False
    assert result.failure_kind is ToolFailureKind.TIMEOUT
    assert result.retryable is True
    assert result.metrics.response_bytes > 0
    assert result.metrics.model_tokens_estimate > 0


async def test_pydantic_tool_enforces_execution_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that the wrapper cancels a slow handler and preserves collected metrics."""

    handler_was_cancelled = False

    async def slow_handler(
        args: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        nonlocal handler_was_cancelled
        context.record_upstream_call(
            provider="slow_provider",
            operation="wait",
            latency_ms=1,
        )

        try:
            await asyncio.Event().wait()
        finally:
            handler_was_cancelled = True

        return EchoOutput(text=args.text)

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=slow_handler,
        timeout_s=0.01,
    )

    with caplog.at_level(logging.ERROR, logger="tools.base"):
        result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.TIMEOUT
    assert result.error == "tool timed out"
    assert handler_was_cancelled is True
    assert result.metrics is not None
    assert result.metrics.upstream_calls == (
        UpstreamCallMetrics(
            provider="slow_provider",
            operation="wait",
            latency_ms=1,
        ),
    )
    assert "tool_execution_timed_out" in caplog.text
    assert "timeout_s=0.01" in caplog.text
    assert result.failure_kind is ToolFailureKind.TIMEOUT
    assert result.retryable is True


def test_pydantic_tool_rejects_non_positive_execution_timeout() -> None:
    """Verify that a direct tool construction rejects a non-positive deadline."""

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        PydanticTool(
            spec=ToolSpec[EchoInput, EchoOutput](
                name="echo",
                description="Echo test tool.",
                input_model=EchoInput,
                output_model=EchoOutput,
            ),
            handler=echo_handler,
            timeout_s=0,
        )


async def test_pydantic_tool_maps_unexpected_handler_error(
    caplog: pytest.LogCaptureFixture,
):
    """Verify that Pydantic tool maps unexpected handler error."""

    async def failing_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        raise RuntimeError("https://provider.example/?apikey=super-secret")

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=failing_handler,
    )

    with caplog.at_level(logging.ERROR, logger="tools.base"):
        result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert result.error == "Inner failure from tool"
    assert "super-secret" not in result.error
    assert result.data is None
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is False
    assert result.metrics.has_non_empty_answer is False
    assert result.failure_kind is ToolFailureKind.INTERNAL_CONTRACT
    assert result.retryable is False

    log_output = "\n".join(caplog.messages)
    assert "tool_handler_failed" in log_output
    assert "tool_name=echo" in log_output
    assert "exception_type=RuntimeError" in log_output
    assert "traceback=" in log_output
    assert "failing_handler" in log_output
    assert "super-secret" not in log_output


async def test_pydantic_tool_exposes_only_explicit_public_tool_errors(
    caplog: pytest.LogCaptureFixture,
):
    """Verify that Pydantic tool exposes only explicit public tool errors."""

    async def failing_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        raise ToolExecutionError(ToolErrorCode.UNKNOWN_REF, "place ref is expired")

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=failing_handler,
    )

    with caplog.at_level(logging.INFO, logger="tools.base"):
        result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.UNKNOWN_REF
    assert result.error == "place ref is expired"

    record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("tool_execution_failed")
    )
    assert record.levelno == logging.INFO
    assert "provider=internal" in record.getMessage()
    assert "error_code=unknown_ref" in record.getMessage()
    assert "failure_kind=unspecified" in record.getMessage()


async def test_pydantic_tool_preserves_safe_clarification_choices() -> None:
    clarification = ToolClarification(
        kind="select_anchor",
        question="Which VDNH do you mean?",
        options=[
            ToolClarificationOption(
                value="plc_a1b2c3d4e5",
                label="ВДНХ",
                description="metro station",
            ),
            ToolClarificationOption(
                value="plc_b2c3d4e5f6",
                label="ВДНХ",
                description="exhibition centre",
            ),
        ],
    )

    async def ambiguous_handler(
        args: EchoInput,
        _context: ToolExecutionContext,
    ) -> EchoOutput:
        raise ToolExecutionError(
            ToolErrorCode.INVALID_INPUT,
            f"ambiguous: {args.text}",
            clarification=clarification,
        )

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=ambiguous_handler,
    )

    result = await tool.run({"text": "ВДНХ"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.clarification == clarification
    assert len(result.clarification.options) == 2
    assert result.error is not None
    assert "CLARIFICATION:" in result.error
    assert "plc_a1b2c3d4e5" in result.error
    assert "metro station" in result.error


async def test_pydantic_tool_logs_provider_failure_at_error_level(
    caplog: pytest.LogCaptureFixture,
):
    """Verify that Pydantic tool logs provider failure at error level."""

    async def failing_handler(args: EchoInput, _context: ToolExecutionContext) -> EchoOutput:
        raise ToolExecutionError(
            ToolErrorCode.UPSTREAM_ERROR,
            "provider is temporarily unavailable",
            status_code=503,
            provider="yandex_organisation_search",
            failure_kind=ToolFailureKind.HTTP_STATUS,
            retryable=True,
        )

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=failing_handler,
    )

    with caplog.at_level(logging.ERROR, logger="tools.base"):
        result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert result.error == "provider is temporarily unavailable"
    assert result.provider == "yandex_organisation_search"
    assert result.status_code == 503
    assert result.failure_kind is ToolFailureKind.HTTP_STATUS
    assert result.retryable is True

    record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("tool_execution_failed")
    )
    assert record.levelno == logging.ERROR
    log_output = record.getMessage()
    assert "provider=yandex_organisation_search" in log_output
    assert "error_code=upstream_error" in log_output
    assert "failure_kind=http_status" in log_output
    assert "status_code=503" in log_output
    assert "retryable=True" in log_output


async def test_pydantic_tool_hash_uses_normalized_input():
    """Verify that Pydantic tool hash uses normalized input."""

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=echo_handler,
    )

    first = await tool.run({"text": "  hello   world "})
    second = await tool.run({"text": "hello world"})

    assert first.ok is True
    assert second.ok is True
    assert first.tool_hash == second.tool_hash
    assert first.data == {"text": "hello world"}
    assert second.data == {"text": "hello world"}


class EmptyOutput(BaseModel):
    items: list[str] = Field(default_factory=list)


async def test_pydantic_tool_allows_empty_successful_output():
    """Verify that Pydantic tool allows empty successful output."""

    async def empty_handler(args: EchoInput, _context: ToolExecutionContext) -> EmptyOutput:
        return EmptyOutput()

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EmptyOutput](
            name="empty_echo",
            description="Empty output test tool.",
            input_model=EchoInput,
            output_model=EmptyOutput,
        ),
        handler=empty_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.data == {"items": []}
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.has_non_empty_answer is False


async def test_pydantic_tool_uses_spec_answer_fields_for_non_empty_metric():
    """Verify that Pydantic tool uses spec answer fields for non empty metric."""

    class SearchOutput(BaseModel):
        query: str
        results: list[str] = Field(default_factory=list)

    async def empty_search_handler(
        args: EchoInput,
        _context: ToolExecutionContext,
    ) -> SearchOutput:
        return SearchOutput(query=args.text, results=[])

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, SearchOutput](
            name="search",
            description="Search test tool.",
            input_model=EchoInput,
            output_model=SearchOutput,
            answer_fields=("results",),
        ),
        handler=empty_search_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.data == {"query": "hello", "results": []}
    assert result.metrics is not None
    assert result.metrics.success is True
    assert result.metrics.has_non_empty_answer is False


def test_non_empty_answer_paths_traverse_resolution_items() -> None:
    fields = ("places", "resolved.place", "resolved.options")

    assert not has_non_empty_answer(
        {"places": [], "resolved": [{"status": "not_found", "place": None, "options": []}]},
        answer_fields=fields,
    )
    assert not has_non_empty_answer(
        {
            "places": [],
            "resolved": [{"status": "error", "place": None, "options": [], "error": "timeout"}],
        },
        answer_fields=fields,
    )
    assert has_non_empty_answer(
        {
            "places": [],
            "resolved": [{"status": "resolved", "place": {"ref": "plc_a1b2c3d4e5"}, "options": []}],
        },
        answer_fields=fields,
    )
    assert has_non_empty_answer(
        {
            "places": [],
            "resolved": [
                {
                    "status": "ambiguous",
                    "place": None,
                    "options": [{"ref": "plc_a1b2c3d4e5"}],
                }
            ],
        },
        answer_fields=fields,
    )


async def test_pydantic_tool_exposes_context_warnings() -> None:
    async def warning_handler(
        args: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        context.add_warning("Route times do not include live traffic.")
        return EchoOutput(text=args.text)

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="warning_echo",
            description="Warning propagation test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=warning_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.warnings == ["Route times do not include live traffic."]


def test_large_tool_observation_is_complete_and_valid_json() -> None:
    rendered = render_tool_observation(
        {
            "route": {
                "length_m": 1_000,
                "steps": [
                    {
                        "instruction": f"Continue on street number {index}",
                        "length_m": 10,
                    }
                    for index in range(100)
                ],
            }
        },
        ["Route times do not include live traffic."],
    )

    parsed = json.loads(rendered)
    assert parsed["warnings"] == ["Route times do not include live traffic."]
    assert parsed["route"]["length_m"] == 1_000
    assert len(parsed["route"]["steps"]) == 100
    assert parsed["route"]["steps"][50] == {
        "instruction": "Continue on street number 50",
        "length_m": 10,
    }


async def test_pydantic_tool_treats_handler_validation_error_as_upstream_error():
    """Verify that Pydantic tool treats handler validation error as upstream error."""

    async def validation_error_handler(
        args: EchoInput,
        _context: ToolExecutionContext,
    ) -> EchoOutput:
        EchoInput.model_validate({"text": "hello", "apikey": "super-secret"})
        return EchoOutput(text=args.text)

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=validation_error_handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert result.error == "Inner failure from tool"
    assert "super-secret" not in result.error
    assert result.tool_hash
    assert result.metrics is not None
    assert result.metrics.success is False


async def test_pydantic_tool_propagates_upstream_latency_to_metrics() -> None:
    """Verify that Pydantic tool propagates upstream latency to metrics."""

    async def handler(
        args: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        context.record_upstream_call(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=120,
        )
        context.record_upstream_call(
            provider="yandex_organisation_search",
            operation="search",
            latency_ms=280,
        )
        return EchoOutput(text=args.text)

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is True
    assert result.metrics is not None
    assert result.metrics.upstream_calls == (
        UpstreamCallMetrics(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=120,
        ),
        UpstreamCallMetrics(
            provider="yandex_organisation_search",
            operation="search",
            latency_ms=280,
        ),
    )
    assert result.metrics.upstream_latency_ms == 400

    serialized_metrics = result.metrics.model_dump(mode="json")
    assert serialized_metrics["upstream_calls"] == [
        {
            "provider": "yandex_geocoder",
            "operation": "search",
            "latency_ms": 120,
            "parallel_group": None,
            "outcome": "unknown",
            "status_code": None,
            "error_code": None,
            "failure_kind": None,
            "provider_code": None,
            "retryable": None,
        },
        {
            "provider": "yandex_organisation_search",
            "operation": "search",
            "latency_ms": 280,
            "parallel_group": None,
            "outcome": "unknown",
            "status_code": None,
            "error_code": None,
            "failure_kind": None,
            "provider_code": None,
            "retryable": None,
        },
    ]
    assert serialized_metrics["upstream_latency_ms"] == 400


async def test_pydantic_tool_keeps_upstream_latency_on_failure() -> None:
    """Verify that Pydantic tool keeps upstream latency on failure."""

    async def handler(
        _args: EchoInput,
        context: ToolExecutionContext,
    ) -> EchoOutput:
        context.record_upstream_call(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=150,
        )
        raise ToolExecutionError(
            ToolErrorCode.TIMEOUT,
            "provider timed out",
            provider="yandex_geocoder",
            failure_kind=ToolFailureKind.TIMEOUT,
            retryable=True,
        )

    tool = PydanticTool(
        spec=ToolSpec[EchoInput, EchoOutput](
            name="echo",
            description="Echo test tool.",
            input_model=EchoInput,
            output_model=EchoOutput,
        ),
        handler=handler,
    )

    result = await tool.run({"text": "hello"})

    assert result.ok is False
    assert result.error_code is ToolErrorCode.TIMEOUT
    assert result.metrics is not None
    assert result.metrics.upstream_calls == (
        UpstreamCallMetrics(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=150,
        ),
    )
    assert result.metrics.upstream_latency_ms == 150
