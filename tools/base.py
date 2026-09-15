"""Tool contract — schemas and interface only. No implementations.

A tool's *validation contract* is two Pydantic models:

* **input**  — what the **LLM** is allowed to decide and what validates
  ``ReActStep.tool_input``.
* **output** — what the tool returns. It is serialized into
  :attr:`ToolResult.data` (JSON-safe → lands in ``tool_call.result`` JSONB).

``ToolSpec.llm_parameters`` may provide a compact function-calling schema for
the model. The input model remains the authoritative backend validator.

Anything a tool needs but the model must NOT choose (API keys, ``lang``,
pagination, request signing) belongs to config/adapters — never to the schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from time import perf_counter
from traceback import extract_tb
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from tools.observability import (
    ToolExecutionContext,
    UpstreamCallMetrics,
    aggregate_upstream_latency_ms,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ERROR_CHARS = 4_000


class ToolErrorCode(StrEnum):
    """Typed failure taxonomy shared by all tools."""

    INVALID_INPUT = "invalid_input"
    DUPLICATE_CALL = "duplicate_call"
    NOT_FOUND = "not_found"
    #: A `plc_`/`src_` ref the cache does not know: expired, or invented by the
    #: model. NEVER guess what it meant — make the model resolve it again.
    UNKNOWN_REF = "unknown_ref"
    UNSUPPORTED_FILTER = "unsupported_filter"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"


class ToolFailureKind(StrEnum):
    """Internal failure stage used for diagnostics, never as a model-facing code."""

    HTTP_STATUS = "http_status"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER_RESPONSE = "provider_response"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    INTERNAL_CONTRACT = "internal_contract"
    COVERAGE_MISS = "coverage_miss"


class ToolClarificationOption(BaseModel):
    """One safe model-facing choice for resolving an ambiguous tool input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(
        min_length=1,
        description="Opaque value to reuse only after the user selects this option.",
    )
    label: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=500)


class ToolClarification(BaseModel):
    """A bounded choice the model must present instead of guessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["select_anchor", "select_area", "select_area_query"]
    question: str = Field(min_length=1, max_length=500)
    options: list[ToolClarificationOption] = Field(min_length=2, max_length=3)


class ToolExecutionError(Exception):
    """Expected tool failure with separate public and operational metadata.

    ``error_code`` and the exception message are safe to return to the model.
    The remaining fields are internal diagnostics for logging and retry policy.
    ``retryable`` means the same call can be repeated with the same arguments.
    """

    def __init__(
        self,
        error_code: ToolErrorCode,
        public_message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        provider_code: str | None = None,
        failure_kind: ToolFailureKind | None = None,
        retryable: bool = False,
        clarification: ToolClarification | None = None,
    ) -> None:
        super().__init__(public_message)
        self.error_code = error_code
        self.status_code = status_code
        self.provider = provider
        self.provider_code = provider_code
        self.failure_kind = failure_kind
        self.retryable = retryable
        self.clarification = clarification


class ToolMetrics(BaseModel):
    """Runtime metrics for one tool call.

    These are per-call raw measurements. Aggregates like p95/p99 are computed
    later by the eval/observability layer from many ToolMetrics records.
    """

    latency_ms: int = Field(
        ge=0,
        description="Total wall-clock tool latency, including validation and wrapping.",
    )
    response_bytes: int = Field(
        default=0,
        ge=0,
        description="Serialized JSON size of the data sent back to the model.",
    )
    model_tokens_estimate: int = Field(
        default=0,
        ge=0,
        description="Approximate token count of the data sent back to the model.",
    )
    success: bool = Field(
        description="True when the tool call completed without a tool-level error.",
    )
    has_non_empty_answer: bool = Field(
        description="True when the tool returned at least one useful non-empty value.",
    )
    upstream_calls: tuple[UpstreamCallMetrics, ...] = Field(
        default_factory=tuple,
        description="External API calls performed during this tool execution.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upstream_latency_ms(self) -> int | None:
        """Estimated upstream critical-path latency for this tool execution."""

        return aggregate_upstream_latency_ms(self.upstream_calls)


class ToolValidationIssue(BaseModel):
    """One safe, model-facing input-validation problem."""

    path: str = Field(
        description="Dot-separated path to the invalid argument; $ means the whole input.",
    )
    code: str = Field(
        description="Stable machine-readable validation error code.",
    )
    message: str = Field(
        description="Short explanation of how the argument is invalid.",
    )


class ToolResult(BaseModel):
    """Uniform envelope every tool returns.

    ``ok=False`` means the call failed. An empty result set is **not** a
    failure — that is ``ok=True`` with an empty list inside ``data``.
    """

    tool_name: str
    ok: bool
    #: The output model, dumped with ``mode="json"``.
    data: dict[str, Any] | None = None
    error: str | None = None
    error_code: ToolErrorCode | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    provider: str | None = None
    provider_code: str | None = None
    failure_kind: ToolFailureKind | None = None
    retryable: bool | None = None
    validation_errors: list[ToolValidationIssue] = Field(
        default_factory=list,
        description=(
            "Safe field-level validation errors. Populated only when error_code=invalid_input."
        ),
    )
    clarification: ToolClarification | None = Field(
        default=None,
        description=("A bounded user choice required before retrying an ambiguous tool request."),
    )
    #: sha256 over (tool_name + canonical args) — the Redis `echo:<hash>` key.
    tool_hash: str | None = None
    #: Non-fatal notes for the model (e.g. "this filter is not supported").
    warnings: list[str] = Field(default_factory=list)
    metrics: ToolMetrics | None = None


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolSpec(BaseModel, Generic[InputT, OutputT]):
    """Machine-readable tool contract for registry, model schemas, policy and eval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        min_length=1,
        max_length=128,
        description="Stable snake_case tool name.",
    )
    description: str = Field(
        min_length=1,
        description="Short model-facing description of when to use the tool.",
    )
    input_model: type[InputT] = Field(
        description="Pydantic model used to validate model-provided tool arguments.",
    )
    llm_parameters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional compact JSON Schema sent to the model for function calling. "
            "When absent, input_model.model_json_schema() is used."
        ),
    )
    output_model: type[OutputT] = Field(
        description="Pydantic model used to validate tool output before serialization.",
    )
    eval_metrics: list[str] = Field(
        default_factory=list,
        description="Metric names expected for this tool in the eval harness.",
    )
    answer_fields: tuple[str, ...] = Field(
        default=(),
        description=(
            "Output fields or dot-separated nested paths that indicate a useful non-empty "
            "answer; paths traverse list items automatically. "
            "If empty, the wrapper falls back to generic non-empty detection."
        ),
    )
    output_exclude_none: bool = Field(
        default=False,
        description="Omit null output fields from the model-facing serialized data.",
    )


ToolHandler = Callable[
    [InputT, ToolExecutionContext],
    Awaitable[OutputT],
]


def canonical_tool_hash(tool_name: str, args: BaseModel) -> str:
    """Return stable sha256 over tool name and normalized Pydantic args."""

    payload = {
        "tool_name": tool_name,
        "args": args.model_dump(mode="json", exclude_none=False),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def estimate_model_tokens(payload: dict[str, Any] | None) -> int:
    """Approximate token count for JSON data sent to the model."""

    if payload is None:
        return 0

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return max(1, len(encoded) // 4)


def tool_observation_payload(
    data: dict[str, Any],
    warnings: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build the exact successful payload shown to the model."""

    if not warnings:
        return data
    return {
        **data,
        "warnings": list(warnings),
    }


def render_tool_observation(
    data: dict[str, Any],
    warnings: tuple[str, ...] | list[str],
) -> str:
    """Serialize the complete successful tool observation for the model."""
    payload = tool_observation_payload(data, warnings)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_tool_error_observation(
    error_code: ToolErrorCode,
    error: str,
    *,
    max_chars: int = DEFAULT_MAX_TOOL_ERROR_CHARS,
) -> str:
    """Render the exact size-bounded error observation shown to the model."""

    if max_chars < 1:
        raise ValueError("tool error observation limit must be positive")
    return f"ERROR[{error_code.value}]: {error}"[:max_chars]


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return len(value) > 0
    if isinstance(value, bool | int | float):
        return True
    return True


def _answer_path_values(data: Any, path: tuple[str, ...]) -> list[Any]:
    """Collect values from a mapping, traversing lists between path segments."""

    if not path:
        return [data]
    if isinstance(data, dict):
        key, *remaining = path
        if key not in data:
            return []
        return _answer_path_values(data[key], tuple(remaining))
    if isinstance(data, list | tuple):
        return [value for item in data for value in _answer_path_values(item, path)]
    return []


def has_non_empty_answer(
    data: dict[str, Any] | None,
    *,
    answer_fields: tuple[str, ...] = (),
) -> bool:
    """Return True when tool data contains a useful non-empty answer."""

    if not data:
        return False

    if answer_fields:
        return any(
            _is_non_empty_value(value)
            for field in answer_fields
            for value in _answer_path_values(data, tuple(field.split(".")))
        )

    return any(_is_non_empty_value(value) for value in data.values())


def public_validation_issues(
    exc: ValidationError,
    *,
    max_validation_issues: int = 10,
    max_validation_message_chars: int = 200,
) -> list[ToolValidationIssue]:
    """Convert Pydantic errors without exposing rejected input values."""

    issues: list[ToolValidationIssue] = []

    errors = exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )

    for error in errors[:max_validation_issues]:
        path = ".".join(str(part) for part in error["loc"]) or "$"

        message = " ".join(error["msg"].split())
        if len(message) > max_validation_message_chars:
            message = message[: max_validation_message_chars - 1] + "…"

        issues.append(
            ToolValidationIssue(
                path=path,
                code=error["type"],
                message=message,
            )
        )

    return issues


def _model_error_text(
    message: str,
    *,
    validation_errors: list[ToolValidationIssue] | None = None,
    clarification: ToolClarification | None = None,
) -> str:
    """Attach only explicitly public recovery data to the model-visible error."""

    parts = [message]
    if validation_errors:
        parts.append(
            "VALIDATION_ERRORS: "
            + json.dumps(
                [issue.model_dump(mode="json") for issue in validation_errors],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if clarification is not None:
        parts.append(f"CLARIFICATION: {clarification.model_dump_json()}")
    return "\n".join(parts)


def _safe_traceback(exc: Exception) -> str:
    """Render traceback locations without exception text or source-code secrets."""

    return " <- ".join(
        f"{frame.filename}:{frame.lineno} in {frame.name}"
        for frame in extract_tb(exc.__traceback__)
    )


@runtime_checkable
class Tool(Protocol[InputT, OutputT]):
    """Minimal tool interface. Implementations come in a later phase."""

    #: Stable snake_case id. Stored in `tool_call.tool_name` (<=128 chars).
    name: str
    #: Read by the MODEL: what it does, when to use it, when NOT to.
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]

    async def run(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate ``raw_args`` against ``input_model``, execute, wrap the result.

        TODO(phase-1): implement.
        """
        ...


class PydanticTool(Generic[InputT, OutputT]):
    """Wrap one async tool handler with Pydantic validation and ToolResult output."""

    def __init__(
        self,
        *,
        spec: ToolSpec[InputT, OutputT],
        handler: ToolHandler[InputT, OutputT],
        timeout_s: float | None = None,
    ) -> None:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self.input_model = spec.input_model
        self.output_model = spec.output_model
        self._handler = handler
        self._timeout_s = timeout_s

    @staticmethod
    def _error_metrics(
        start: float,
        context: ToolExecutionContext | None = None,
        *,
        error_code: ToolErrorCode | None = None,
        error: str | None = None,
    ) -> ToolMetrics:
        observation = (
            render_tool_error_observation(error_code, error)
            if error_code is not None and error is not None
            else ""
        )
        return ToolMetrics(
            latency_ms=int((perf_counter() - start) * 1000),
            response_bytes=len(observation.encode("utf-8")),
            model_tokens_estimate=(max(1, len(observation) // 4) if observation else 0),
            upstream_calls=(context.upstream_calls if context is not None else ()),
            success=False,
            has_non_empty_answer=False,
        )

    @staticmethod
    def _success_metrics(
        start: float,
        data: dict[str, Any],
        answer_fields: tuple[str, ...],
        context: ToolExecutionContext,
    ) -> ToolMetrics:
        observation = render_tool_observation(data, context.warnings)
        response_bytes = len(observation.encode("utf-8"))
        model_tokens_estimate = max(
            1,
            len(observation) // 4,
        )

        return ToolMetrics(
            latency_ms=int((perf_counter() - start) * 1000),
            upstream_calls=context.upstream_calls,
            response_bytes=response_bytes,
            model_tokens_estimate=model_tokens_estimate,
            success=True,
            has_non_empty_answer=has_non_empty_answer(
                data,
                answer_fields=answer_fields,
            ),
        )

    async def run(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate model-provided args before the handler sees them."""

        start = perf_counter()

        try:
            args = self.input_model.model_validate(raw_args)
        except ValidationError as exc:
            validation_errors = public_validation_issues(exc)
            error = _model_error_text(
                "Invalid tool input; correct the listed fields and retry",
                validation_errors=validation_errors,
            )
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=error,
                error_code=ToolErrorCode.INVALID_INPUT,
                validation_errors=validation_errors,
                retryable=False,
                metrics=self._error_metrics(
                    start,
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error=error,
                ),
            )

        tool_hash = canonical_tool_hash(self.name, args)
        context = ToolExecutionContext()

        try:
            # One tool may perform several individually timed HTTP requests.
            # This outer deadline prevents their combined work from running on.
            async with asyncio.timeout(self._timeout_s):
                raw_output = await self._handler(args, context)
        except ToolExecutionError as exc:
            error = _model_error_text(
                str(exc),
                clarification=exc.clarification,
            )
            metrics = self._error_metrics(
                start,
                context,
                error_code=exc.error_code,
                error=error,
            )

            expected_outcome_codes = {
                ToolErrorCode.INVALID_INPUT,
                ToolErrorCode.NOT_FOUND,
                ToolErrorCode.UNKNOWN_REF,
                ToolErrorCode.UNSUPPORTED_FILTER,
            }

            log_level = logging.INFO if exc.error_code in expected_outcome_codes else logging.ERROR

            logger.log(
                log_level,
                "tool_execution_failed "
                "tool_name=%s tool_hash=%s provider=%s "
                "error_code=%s failure_kind=%s status_code=%s provider_code=%s "
                "retryable=%s latency_ms=%s",
                self.name,
                tool_hash,
                exc.provider or "internal",
                exc.error_code.value,
                exc.failure_kind.value if exc.failure_kind is not None else "unspecified",
                exc.status_code,
                exc.provider_code,
                exc.retryable,
                metrics.latency_ms,
            )

            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=error,
                error_code=exc.error_code,
                status_code=exc.status_code,
                provider=exc.provider,
                provider_code=exc.provider_code,
                failure_kind=exc.failure_kind,
                retryable=exc.retryable,
                clarification=exc.clarification,
                tool_hash=tool_hash,
                metrics=metrics,
            )
        except TimeoutError:
            error = "tool timed out"
            metrics = self._error_metrics(
                start,
                context,
                error_code=ToolErrorCode.TIMEOUT,
                error=error,
            )

            logger.error(
                "tool_execution_timed_out tool_name=%s tool_hash=%s timeout_s=%s latency_ms=%s",
                self.name,
                tool_hash,
                self._timeout_s,
                metrics.latency_ms,
            )

            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=error,
                error_code=ToolErrorCode.TIMEOUT,
                failure_kind=ToolFailureKind.TIMEOUT,
                retryable=True,
                tool_hash=tool_hash,
                metrics=metrics,
            )
        except Exception as exc:
            error = "Inner failure from tool"
            logger.error(
                "tool_handler_failed tool_name=%s tool_hash=%s exception_type=%s traceback=%s",
                self.name,
                tool_hash,
                type(exc).__name__,
                _safe_traceback(exc),
            )
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=error,
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
                retryable=False,
                tool_hash=tool_hash,
                metrics=self._error_metrics(
                    start,
                    context,
                    error_code=ToolErrorCode.UPSTREAM_ERROR,
                    error=error,
                ),
            )

        try:
            output = self.output_model.model_validate(raw_output)
        except ValidationError as exc:
            error = "Incorrect answer from tool"
            logger.error(
                "tool_output_validation_failed tool_name=%s tool_hash=%s validation_error_count=%s",
                self.name,
                tool_hash,
                exc.error_count(),
            )

            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=error,
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
                tool_hash=tool_hash,
                metrics=self._error_metrics(
                    start,
                    context,
                    error_code=ToolErrorCode.UPSTREAM_ERROR,
                    error=error,
                ),
            )

        data = output.model_dump(
            mode="json",
            exclude_none=self.spec.output_exclude_none,
        )

        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            tool_hash=tool_hash,
            warnings=list(context.warnings),
            metrics=self._success_metrics(
                start,
                data,
                self.spec.answer_fields,
                context,
            ),
        )
