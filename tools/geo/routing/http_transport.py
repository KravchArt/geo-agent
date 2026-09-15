"""Shared HTTP transport for routing API clients."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Literal, TypeVar, cast

import httpx

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import (
    ToolExecutionContext,
    UpstreamCallOutcome,
)

ResponseT = TypeVar("ResponseT")
RoutingResponseParser = Callable[[int, Any], ResponseT]


async def request_routing_json(
    *,
    http_client: httpx.AsyncClient,
    method: Literal["GET", "POST"],
    url: str,
    params: httpx.QueryParams | Mapping[str, str],
    operation: str,
    provider: str,
    service_name: str,
    context: ToolExecutionContext,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    http_timeout_s: float | None = None,
    response_parser: RoutingResponseParser[ResponseT],
    response_body: Literal["object", "array", "object_or_array"] = "object",
) -> ResponseT:
    """Send, decode, and classify one routing response before recording it."""

    if http_timeout_s is not None and http_timeout_s <= 0:
        raise ValueError("routing HTTP timeout must be positive")

    started = perf_counter()
    response: httpx.Response | None = None
    request_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        **(headers or {}),
    }

    try:
        try:
            pending_response = http_client.request(
                method,
                url,
                params=params,
                json=payload,
                headers=request_headers,
            )
            if http_timeout_s is None:
                response = await pending_response
            else:
                # Limit only the network exchange. Provider orchestration, place
                # resolution, response validation, and normalization run outside
                # this deadline.
                async with asyncio.timeout(http_timeout_s):
                    response = await pending_response
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ToolExecutionError(
                ToolErrorCode.TIMEOUT,
                f"{service_name} request timed out",
                provider=provider,
                failure_kind=ToolFailureKind.TIMEOUT,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                f"{service_name} service is unavailable",
                provider=provider,
                failure_kind=ToolFailureKind.NETWORK,
                retryable=True,
            ) from exc

        try:
            body: Any = response.json()
        except ValueError as exc:
            if response.status_code == 204:
                body = {} if response_body == "object" else []
            else:
                common_error = common_routing_http_error(
                    status_code=response.status_code,
                    provider=provider,
                    service_name=service_name,
                )
                if common_error is not None:
                    raise common_error from exc
                raise ToolExecutionError(
                    ToolErrorCode.UPSTREAM_ERROR,
                    f"{service_name} returned invalid JSON",
                    status_code=response.status_code,
                    provider=provider,
                    failure_kind=ToolFailureKind.INVALID_JSON,
                    retryable=False,
                ) from exc

        body_matches = (
            isinstance(body, Mapping)
            if response_body == "object"
            else isinstance(body, list)
            if response_body == "array"
            else isinstance(body, (Mapping, list))
        )
        if not body_matches:
            common_error = common_routing_http_error(
                status_code=response.status_code,
                provider=provider,
                service_name=service_name,
            )
            if common_error is not None:
                raise common_error
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                f"{service_name} returned an unexpected response format",
                status_code=response.status_code,
                provider=provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            )

        result = response_parser(
            response.status_code,
            cast(Any, dict(body) if isinstance(body, Mapping) else body),
        )
    except asyncio.CancelledError:
        context.record_cancelled_upstream_call(
            provider=provider,
            operation=operation,
            latency_ms=int((perf_counter() - started) * 1_000),
            status_code=response.status_code if response is not None else None,
        )
        raise
    except ToolExecutionError as exc:
        context.record_upstream_call(
            provider=provider,
            operation=operation,
            latency_ms=int((perf_counter() - started) * 1_000),
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
        latency_ms=int((perf_counter() - started) * 1_000),
        outcome=UpstreamCallOutcome.SUCCESS,
        status_code=response.status_code,
    )
    return result


def common_routing_http_error(
    *,
    status_code: int,
    provider: str,
    service_name: str,
    provider_code: str | None = None,
) -> ToolExecutionError | None:
    """Classify provider-independent statuses after the JSON body is available."""

    if status_code in {401, 403}:
        error_code = ToolErrorCode.UPSTREAM_ERROR
        public_message = f"{service_name} authentication failed"
        failure_kind = ToolFailureKind.AUTHENTICATION
        retryable = False
    elif status_code in {408, 504}:
        error_code = ToolErrorCode.TIMEOUT
        public_message = f"{service_name} request timed out"
        failure_kind = ToolFailureKind.TIMEOUT
        retryable = True
    elif status_code == 429:
        error_code = ToolErrorCode.RATE_LIMITED
        public_message = f"{service_name} rate limit exceeded"
        failure_kind = ToolFailureKind.HTTP_STATUS
        retryable = True
    elif status_code >= 500:
        error_code = ToolErrorCode.UPSTREAM_ERROR
        public_message = f"{service_name} returned an HTTP error"
        failure_kind = ToolFailureKind.HTTP_STATUS
        retryable = True
    else:
        return None

    return ToolExecutionError(
        error_code,
        public_message,
        status_code=status_code,
        provider=provider,
        provider_code=provider_code,
        failure_kind=failure_kind,
        retryable=retryable,
    )
