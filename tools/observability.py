"""Per-tool execution context for collecting operational metrics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import count

# Group identifiers must remain unique when a coordinator merges metrics from
# several provider attempts. Otherwise sequential fallback attempts could be
# mistaken for one parallel batch and their latency would be under-reported.
_PARALLEL_GROUP_IDS = count(1)


class UpstreamCallOutcome(StrEnum):
    """Terminal outcome of one external API call."""

    SUCCESS = "success"
    FAILURE = "failure"
    #: Existing non-routing clients do not classify call outcomes yet.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class UpstreamCallMetrics:
    """Metrics of one external API call made by a tool."""

    provider: str
    operation: str
    latency_ms: int
    parallel_group: int | None = None
    outcome: UpstreamCallOutcome = UpstreamCallOutcome.UNKNOWN
    status_code: int | None = None
    error_code: str | None = None
    failure_kind: str | None = None
    provider_code: str | None = None
    retryable: bool | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider cannot be empty")

        if not self.operation:
            raise ValueError("operation cannot be empty")

        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")

        if self.parallel_group is not None and self.parallel_group < 1:
            raise ValueError("parallel_group must be positive")

        if self.status_code is not None and not 100 <= self.status_code <= 599:
            raise ValueError("status_code must be a valid HTTP status")

        if self.outcome is UpstreamCallOutcome.SUCCESS and (
            self.error_code is not None or self.failure_kind is not None
        ):
            raise ValueError("successful upstream call cannot contain error metadata")


def aggregate_upstream_latency_ms(
    calls: Iterable[UpstreamCallMetrics],
) -> int | None:
    """Estimate upstream critical-path latency from individual call metrics.

    Independent calls are sequential unless they share a parallel group.
    Calls within the same group contribute only the slowest call; separate
    groups are sequential and therefore remain additive.
    """

    total_ms = 0
    call_count = 0
    parallel_maxima: dict[int, int] = {}

    for call in calls:
        call_count += 1
        if call.parallel_group is None:
            total_ms += call.latency_ms
            continue

        parallel_maxima[call.parallel_group] = max(
            parallel_maxima.get(call.parallel_group, 0),
            call.latency_ms,
        )

    if call_count == 0:
        return None

    return total_ms + sum(parallel_maxima.values())


@dataclass(slots=True)
class ToolExecutionContext:
    """Mutable metrics and warning collector scoped to one logical tool call."""

    _upstream_calls: list[UpstreamCallMetrics] = field(
        default_factory=list,
        repr=False,
    )
    _warnings: list[str] = field(
        default_factory=list,
        repr=False,
    )
    _parallel_group: ContextVar[int | None] = field(
        default_factory=lambda: ContextVar("upstream_parallel_group", default=None),
        repr=False,
    )

    @property
    def upstream_calls(self) -> tuple[UpstreamCallMetrics, ...]:
        """Return an immutable snapshot of recorded upstream calls."""

        return tuple(self._upstream_calls)

    @property
    def upstream_latency_ms(self) -> int | None:
        """Return critical-path API latency, or None if no API was called."""

        return aggregate_upstream_latency_ms(self._upstream_calls)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return de-duplicated non-fatal notes safe to expose to the model."""

        return tuple(self._warnings)

    def record_upstream_call(
        self,
        *,
        provider: str,
        operation: str,
        latency_ms: int,
        outcome: UpstreamCallOutcome = UpstreamCallOutcome.UNKNOWN,
        status_code: int | None = None,
        error_code: str | None = None,
        failure_kind: str | None = None,
        provider_code: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Record one completed, failed or timed-out external API call."""

        self._upstream_calls.append(
            UpstreamCallMetrics(
                provider=provider,
                operation=operation,
                latency_ms=latency_ms,
                parallel_group=self._parallel_group.get(),
                outcome=outcome,
                status_code=status_code,
                error_code=error_code,
                failure_kind=failure_kind,
                provider_code=provider_code,
                retryable=retryable,
            )
        )

    def record_cancelled_upstream_call(
        self,
        *,
        provider: str,
        operation: str,
        latency_ms: int,
        status_code: int | None = None,
    ) -> None:
        """Record an in-flight request cancelled by the enclosing tool deadline."""

        self.record_upstream_call(
            provider=provider,
            operation=operation,
            latency_ms=latency_ms,
            outcome=UpstreamCallOutcome.FAILURE,
            status_code=status_code,
            error_code="timeout",
            failure_kind="timeout",
            retryable=True,
        )

    @contextmanager
    def parallel_upstream_calls(self) -> Iterator[None]:
        """Mark calls started in this scope as one concurrent batch.

        ``ContextVar`` values are copied into tasks created by
        ``asyncio.gather`` while keeping unrelated tool executions isolated.
        """

        token = self._parallel_group.set(next(_PARALLEL_GROUP_IDS))
        try:
            yield
        finally:
            self._parallel_group.reset(token)

    def extend_upstream_calls(self, calls: tuple[UpstreamCallMetrics, ...]) -> None:
        """Preserve complete call records while merging provider attempts."""

        self._upstream_calls.extend(calls)

    def add_warning(self, warning: str) -> None:
        """Record one non-empty warning once, preserving insertion order."""

        normalized = " ".join(warning.split())
        if not normalized:
            raise ValueError("warning cannot be empty")
        if normalized not in self._warnings:
            self._warnings.append(normalized)
