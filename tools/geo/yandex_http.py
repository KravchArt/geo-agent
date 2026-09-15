"""Shared HTTP/JSON transport behavior for Yandex geo API clients."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any, cast

import httpx

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext, UpstreamCallOutcome


async def get_yandex_json(
    *,
    http_client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, str],
    provider: str,
    operation: str,
    context: ToolExecutionContext,
) -> dict[str, Any]:
    """Perform one Yandex GET request and return a validated JSON object."""

    started = perf_counter()
    response: httpx.Response | None = None

    try:
        try:
            response = await http_client.get(url, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(
                ToolErrorCode.TIMEOUT,
                "Yandex request timed out",
                provider=provider,
                failure_kind=ToolFailureKind.TIMEOUT,
                retryable=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _http_status_error(
                status_code=exc.response.status_code,
                provider=provider,
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex service is unavailable",
                provider=provider,
                failure_kind=ToolFailureKind.NETWORK,
                retryable=True,
            ) from exc

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex returned invalid JSON",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.INVALID_JSON,
                retryable=False,
            ) from exc

        if not isinstance(payload, dict):
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex returned an unexpected response format",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            )
    except asyncio.CancelledError:
        context.record_cancelled_upstream_call(
            provider=provider,
            operation=operation,
            latency_ms=int((perf_counter() - started) * 1000),
            status_code=response.status_code if response is not None else None,
        )
        raise
    except ToolExecutionError as exc:
        context.record_upstream_call(
            provider=provider,
            operation=operation,
            latency_ms=int((perf_counter() - started) * 1000),
            outcome=UpstreamCallOutcome.FAILURE,
            status_code=exc.status_code or (response.status_code if response is not None else None),
            error_code=exc.error_code.value,
            failure_kind=exc.failure_kind.value if exc.failure_kind is not None else None,
            provider_code=exc.provider_code,
            retryable=exc.retryable,
        )
        raise

    context.record_upstream_call(
        provider=provider,
        operation=operation,
        latency_ms=int((perf_counter() - started) * 1000),
        outcome=UpstreamCallOutcome.SUCCESS,
        status_code=response.status_code,
    )
    return cast(dict[str, Any], payload)


def _http_status_error(
    *,
    status_code: int,
    provider: str,
) -> ToolExecutionError:
    """Translate an unsuccessful Yandex HTTP status into one tool error."""

    if status_code in {408, 504}:
        error_code = ToolErrorCode.TIMEOUT
        public_message = "Yandex request timed out"
        retryable = True
    elif status_code == 429:
        error_code = ToolErrorCode.RATE_LIMITED
        public_message = "Yandex rate limit exceeded"
        retryable = True
    else:
        # Model input has already passed the tool schema. In particular, an
        # upstream 400 now signals an adapter/upstream problem, not bad LLM input.
        error_code = ToolErrorCode.UPSTREAM_ERROR
        public_message = "Yandex returned an HTTP error"
        retryable = status_code >= 500

    return ToolExecutionError(
        error_code,
        public_message,
        status_code=status_code,
        provider=provider,
        failure_kind=ToolFailureKind.HTTP_STATUS,
        retryable=retryable,
    )
