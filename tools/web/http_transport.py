"""Shared HTTP/JSON transport behavior for web-search API clients."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Set
from time import perf_counter
from typing import Any

import httpx

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext, UpstreamCallOutcome

DEFAULT_INVALID_INPUT_STATUSES = frozenset({400, 422})
DEFAULT_TIMEOUT_STATUSES: frozenset[int] = frozenset()


async def post_search_json(
    *,
    http_client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    provider: str,
    context: ToolExecutionContext,
    invalid_input_statuses: Set[int] = DEFAULT_INVALID_INPUT_STATUSES,
    timeout_statuses: Set[int] = DEFAULT_TIMEOUT_STATUSES,
) -> dict[str, Any]:
    """POST a provider payload and map shared failures into tool errors."""

    started = perf_counter()
    response: httpx.Response | None = None
    try:
        try:
            response = await http_client.post(
                url,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(
                error_code=ToolErrorCode.TIMEOUT,
                public_message="web search timed out",
                provider=provider,
                failure_kind=ToolFailureKind.TIMEOUT,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                public_message="web search provider is unavailable",
                provider=provider,
                failure_kind=ToolFailureKind.NETWORK,
                retryable=True,
            ) from exc

        if response.status_code in timeout_statuses:
            raise ToolExecutionError(
                error_code=ToolErrorCode.TIMEOUT,
                public_message="web search timed out",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.TIMEOUT,
                retryable=True,
            )

        if response.status_code in invalid_input_statuses:
            raise ToolExecutionError(
                error_code=ToolErrorCode.INVALID_INPUT,
                public_message="web search request was rejected",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=False,
            )

        if response.status_code == 429:
            raise ToolExecutionError(
                error_code=ToolErrorCode.RATE_LIMITED,
                public_message="web search provider rate limit exceeded",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=True,
            )

        if response.status_code >= 400:
            raise ToolExecutionError(
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                public_message="web search provider returned an error",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.HTTP_STATUS,
                retryable=response.status_code >= 500,
            )

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                public_message="web search provider returned invalid data",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.INVALID_JSON,
                retryable=False,
            ) from exc

        if not isinstance(data, dict):
            raise ToolExecutionError(
                error_code=ToolErrorCode.UPSTREAM_ERROR,
                public_message="web search provider returned invalid data",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            )
    except ToolExecutionError as exc:
        context.record_upstream_call(
            provider=provider,
            operation="search",
            latency_ms=int((perf_counter() - started) * 1000),
            outcome=UpstreamCallOutcome.FAILURE,
            status_code=exc.status_code or (response.status_code if response is not None else None),
            error_code=exc.error_code.value,
            failure_kind=exc.failure_kind.value if exc.failure_kind is not None else None,
            provider_code=exc.provider_code,
            retryable=exc.retryable,
        )
        raise
    except asyncio.CancelledError:
        context.record_cancelled_upstream_call(
            provider=provider,
            operation="search",
            latency_ms=int((perf_counter() - started) * 1000),
            status_code=response.status_code if response is not None else None,
        )
        raise

    context.record_upstream_call(
        provider=provider,
        operation="search",
        latency_ms=int((perf_counter() - started) * 1000),
        outcome=UpstreamCallOutcome.SUCCESS,
        status_code=response.status_code,
    )
    return data
