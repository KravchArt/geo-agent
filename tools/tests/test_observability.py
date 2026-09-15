import pytest

from tools.observability import (
    ToolExecutionContext,
    UpstreamCallMetrics,
    UpstreamCallOutcome,
)


def test_empty_context_has_no_upstream_latency() -> None:
    """Verify that empty context has no upstream latency."""

    context = ToolExecutionContext()

    assert context.upstream_calls == ()
    assert context.upstream_latency_ms is None
    assert context.warnings == ()


def test_context_accumulates_upstream_calls() -> None:
    """Verify that context accumulates upstream calls."""

    context = ToolExecutionContext()

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

    assert context.upstream_latency_ms == 400
    assert context.upstream_calls == (
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


def test_parallel_upstream_calls_contribute_only_the_slowest_latency() -> None:
    context = ToolExecutionContext()

    context.record_upstream_call(
        provider="scope_resolver",
        operation="search",
        latency_ms=50,
    )
    with context.parallel_upstream_calls():
        context.record_upstream_call(
            provider="geocoder",
            operation="geocode",
            latency_ms=120,
        )
        context.record_upstream_call(
            provider="geocoder",
            operation="geocode",
            latency_ms=280,
        )
        context.record_upstream_call(
            provider="geocoder",
            operation="geocode",
            latency_ms=200,
        )

    group_ids = {call.parallel_group for call in context.upstream_calls[1:]}
    assert len(group_ids) == 1
    assert None not in group_ids
    assert context.upstream_latency_ms == 330


def test_separate_parallel_groups_remain_additive_after_context_merge() -> None:
    first_attempt = ToolExecutionContext()
    second_attempt = ToolExecutionContext()
    merged = ToolExecutionContext()

    with first_attempt.parallel_upstream_calls():
        first_attempt.record_upstream_call(
            provider="graphhopper_routing",
            operation="geocode",
            latency_ms=100,
        )
        first_attempt.record_upstream_call(
            provider="graphhopper_routing",
            operation="geocode",
            latency_ms=180,
        )
    with second_attempt.parallel_upstream_calls():
        second_attempt.record_upstream_call(
            provider="osrm_routing",
            operation="geocode",
            latency_ms=90,
        )
        second_attempt.record_upstream_call(
            provider="osrm_routing",
            operation="geocode",
            latency_ms=140,
        )

    merged.extend_upstream_calls(first_attempt.upstream_calls)
    merged.extend_upstream_calls(second_attempt.upstream_calls)

    assert first_attempt.upstream_calls[0].parallel_group != (
        second_attempt.upstream_calls[0].parallel_group
    )
    assert merged.upstream_latency_ms == 320


def test_contexts_do_not_share_metrics() -> None:
    """Verify that contexts do not share metrics."""

    first = ToolExecutionContext()
    second = ToolExecutionContext()

    first.record_upstream_call(
        provider="yandex_geocoder",
        operation="search",
        latency_ms=100,
    )

    assert first.upstream_latency_ms == 100
    assert second.upstream_latency_ms is None


def test_context_preserves_failure_metadata_when_merging_calls() -> None:
    source = ToolExecutionContext()
    target = ToolExecutionContext()
    source.record_upstream_call(
        provider="graphhopper_routing",
        operation="build_route",
        latency_ms=2_001,
        outcome=UpstreamCallOutcome.FAILURE,
        status_code=504,
        error_code="timeout",
        failure_kind="timeout",
        retryable=True,
    )

    target.extend_upstream_calls(source.upstream_calls)

    assert target.upstream_calls == source.upstream_calls
    assert target.upstream_calls[0].outcome is UpstreamCallOutcome.FAILURE
    assert target.upstream_calls[0].status_code == 504
    assert target.upstream_calls[0].error_code == "timeout"


def test_upstream_call_rejects_negative_latency() -> None:
    """Verify that upstream call rejects negative latency."""

    with pytest.raises(ValueError, match="latency_ms cannot be negative"):
        UpstreamCallMetrics(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=-1,
        )


def test_upstream_call_rejects_invalid_parallel_group() -> None:
    with pytest.raises(ValueError, match="parallel_group must be positive"):
        UpstreamCallMetrics(
            provider="yandex_geocoder",
            operation="search",
            latency_ms=1,
            parallel_group=0,
        )


def test_context_deduplicates_and_normalizes_warnings() -> None:
    context = ToolExecutionContext()

    context.add_warning("  Route times   exclude live traffic. ")
    context.add_warning("Route times exclude live traffic.")

    assert context.warnings == ("Route times exclude live traffic.",)


def test_context_rejects_empty_warning() -> None:
    with pytest.raises(ValueError, match="warning cannot be empty"):
        ToolExecutionContext().add_warning("  ")
