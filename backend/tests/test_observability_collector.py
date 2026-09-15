"""Lightweight request-stage timing and structured logging helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypedDict


class MetricsSnapshot(TypedDict):
    stage_latency_ms: dict[str, int]
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    """One completed pipeline-stage observation, ready for DB persistence."""

    name: str
    occurrence: int
    latency_ms: int
    status: str
    fields: dict[str, object]
    error_type: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class RequestMetricsCollector:
    """Collect named stage latencies and individual stage executions."""

    request_id: str
    session_id: str
    logger: logging.Logger
    stages_ms: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    measurements: list[StageMeasurement] = field(default_factory=list)
    _occurrences: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, **fields: object) -> Iterator[None]:
        started = perf_counter()
        occurrence = self._occurrences.get(name, 0) + 1
        self._occurrences[name] = occurrence
        self.logger.info(
            "stage_started request_id=%s session_id=%s stage=%s occurrence=%s fields=%s",
            self.request_id,
            self.session_id,
            name,
            occurrence,
            fields or None,
        )
        try:
            yield
        except Exception as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            self.stages_ms[name] = self.stages_ms.get(name, 0) + latency_ms
            self.measurements.append(
                StageMeasurement(
                    name=name,
                    occurrence=occurrence,
                    latency_ms=latency_ms,
                    status="failed",
                    fields=dict(fields),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                )
            )
            self.logger.exception(
                "stage_failed request_id=%s session_id=%s stage=%s occurrence=%s latency_ms=%s",
                self.request_id,
                self.session_id,
                name,
                occurrence,
                latency_ms,
            )
            raise
        else:
            latency_ms = int((perf_counter() - started) * 1000)
            self.stages_ms[name] = self.stages_ms.get(name, 0) + latency_ms
            self.measurements.append(
                StageMeasurement(
                    name=name,
                    occurrence=occurrence,
                    latency_ms=latency_ms,
                    status="completed",
                    fields=dict(fields),
                )
            )
            self.logger.info(
                "stage_complete request_id=%s session_id=%s stage=%s occurrence=%s latency_ms=%s",
                self.request_id,
                self.session_id,
                name,
                occurrence,
                latency_ms,
            )

    def record_stage(
        self,
        name: str,
        latency_ms: int,
        *,
        status: str = "completed",
        error_type: str | None = None,
        error_message: str | None = None,
        **fields: object,
    ) -> None:
        """Record a stage timed by a component itself (for parallel/nested work)."""
        occurrence = self._occurrences.get(name, 0) + 1
        self._occurrences[name] = occurrence
        self.stages_ms[name] = self.stages_ms.get(name, 0) + latency_ms
        self.measurements.append(
            StageMeasurement(
                name=name,
                occurrence=occurrence,
                latency_ms=latency_ms,
                status=status,
                fields=dict(fields),
                error_type=error_type,
                error_message=(error_message[:2000] if error_message else None),
            )
        )
        self.logger.info(
            "stage_recorded request_id=%s session_id=%s stage=%s occurrence=%s"
            "latency_ms=%s status=%s fields=%s",
            self.request_id,
            self.session_id,
            name,
            occurrence,
            latency_ms,
            status,
            fields or None,
        )

    def set_count(self, name: str, value: int) -> None:
        self.counts[name] = value

    def increment(self, name: str, value: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + value

    def snapshot(self) -> MetricsSnapshot:
        return {"stage_latency_ms": dict(self.stages_ms), "counts": dict(self.counts)}
