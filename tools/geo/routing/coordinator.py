"""Provider coordination for ``routing_tool``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.coordination import ProviderExecutionStrategy
from tools.geo.routing.provider import RoutingProvider
from tools.geo.routing.schemas import RoutingInput, RoutingOutput
from tools.observability import ToolExecutionContext

logger = logging.getLogger(__name__)

ROUTING_PROVIDER_PRIORITY = ("twogis", "graphhopper", "osrm", "yandex")
_PROVIDER_LABELS = {
    "twogis": "2GIS",
    "graphhopper": "GraphHopper",
    "osrm": "OSRM",
    "yandex": "Yandex",
}


class RoutingInputResolver(Protocol):
    """Replace textual routing points before provider attempts."""

    async def resolve_to_refs(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingInput:
        """Replace textual routing points with reusable place refs."""
        ...


class _FallbackAction(StrEnum):
    STOP = "stop"
    TRY_NEXT = "try_next"
    TRY_NEXT_AND_ALERT = "try_next_and_alert"


class RoutingCoordinator:
    """Execute routing providers in priority order with safe fallback."""

    def __init__(
        self,
        *,
        providers: Sequence[RoutingProvider],
        input_resolver: RoutingInputResolver | None = None,
        strategy: ProviderExecutionStrategy = ProviderExecutionStrategy.FALLBACK,
    ) -> None:
        if not providers:
            raise ValueError("routing coordinator requires at least one provider")

        provider_names = [provider.provider for provider in providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError(f"duplicate routing providers: {provider_names}")

        if strategy is ProviderExecutionStrategy.PARALLEL:
            raise NotImplementedError("parallel routing strategy is not implemented")
        if input_resolver is None:
            raise ValueError("routing coordinator requires an input resolver")

        priority = {
            provider_name: index for index, provider_name in enumerate(ROUTING_PROVIDER_PRIORITY)
        }
        # Unknown/custom providers retain their relative configured order after
        # the built-in priority list.
        self._providers = tuple(
            sorted(
                providers,
                key=lambda provider: priority.get(provider.provider, len(priority)),
            )
        )
        self._input_resolver = input_resolver
        self._strategy = strategy

    async def route(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        resolved_args = await self._input_resolver.resolve_to_refs(args, context)
        if self._strategy is ProviderExecutionStrategy.FIRST:
            return await self._providers[0].route(resolved_args, context)
        if self._strategy is ProviderExecutionStrategy.FALLBACK:
            return await self._route_with_fallback(resolved_args, context)

        raise AssertionError(f"unsupported strategy: {self._strategy}")

    async def _route_with_fallback(
        self,
        args: RoutingInput,
        context: ToolExecutionContext,
    ) -> RoutingOutput:
        for index, provider in enumerate(self._providers):
            attempt_context = ToolExecutionContext()
            try:
                result = await provider.route(args, attempt_context)
            except ToolExecutionError as exc:
                self._merge_context(context, attempt_context, include_warnings=False)
                is_last = index == len(self._providers) - 1
                action = self._fallback_action(exc, provider)
                if is_last or action is _FallbackAction.STOP:
                    raise

                next_provider = self._providers[index + 1]
                self._log_fallback(
                    error=exc,
                    provider=provider,
                    next_provider=next_provider,
                    action=action,
                )
                reason = "timed out" if exc.error_code is ToolErrorCode.TIMEOUT else "failed"
                context.add_warning(
                    f"{self._provider_label(provider)} routing {reason}; "
                    f"retrying with {self._provider_label(next_provider)}."
                )
                continue

            self._merge_context(context, attempt_context, include_warnings=True)
            return result

        raise AssertionError("routing fallback completed without a result or error")

    @staticmethod
    def _fallback_action(
        error: ToolExecutionError,
        provider: RoutingProvider,
    ) -> _FallbackAction:
        routing_provider_names = {
            provider.provider,
            f"{provider.provider}_routing",
        }
        # Ref-loading failures and errors from another component must not be
        # hidden by retrying the same prepared input through another routing engine.
        if error.provider not in routing_provider_names:
            return _FallbackAction.STOP

        # Authentication and internal contract failures require an operator or
        # code change. A successful secondary provider must not mask them.
        if error.status_code in {401, 403} or error.failure_kind in {
            ToolFailureKind.AUTHENTICATION,
            ToolFailureKind.INTERNAL_CONTRACT,
        }:
            return _FallbackAction.STOP

        # A different road graph may snap or connect the same points.
        if error.error_code is ToolErrorCode.NOT_FOUND:
            return _FallbackAction.TRY_NEXT

        # Capability gaps are provider-local. For example, 2GIS cannot disable
        # traffic for a fastest driving route, while GraphHopper can serve the
        # same prepared refs with its static graph.
        if error.error_code is ToolErrorCode.UNSUPPORTED_FILTER:
            return _FallbackAction.TRY_NEXT

        # A malformed provider response is recoverable for the user, but it is
        # still an integration incident and must remain visible to operations.
        if error.failure_kind in {
            ToolFailureKind.INVALID_JSON,
            ToolFailureKind.INVALID_SCHEMA,
        }:
            return _FallbackAction.TRY_NEXT_AND_ALERT

        transient_failure = (
            error.error_code in {ToolErrorCode.RATE_LIMITED, ToolErrorCode.TIMEOUT}
            or error.failure_kind
            in {
                ToolFailureKind.NETWORK,
                ToolFailureKind.TIMEOUT,
                ToolFailureKind.PROVIDER_RESPONSE,
            }
            or (
                error.failure_kind is ToolFailureKind.HTTP_STATUS
                and error.status_code is not None
                and (error.status_code == 429 or error.status_code >= 500)
            )
        )
        if error.retryable and transient_failure:
            return _FallbackAction.TRY_NEXT
        return _FallbackAction.STOP

    @staticmethod
    def _log_fallback(
        *,
        error: ToolExecutionError,
        provider: RoutingProvider,
        next_provider: RoutingProvider,
        action: _FallbackAction,
    ) -> None:
        if action is _FallbackAction.TRY_NEXT_AND_ALERT:
            log_level = logging.ERROR
        elif error.error_code is ToolErrorCode.NOT_FOUND:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
        logger.log(
            log_level,
            "routing_provider_fallback provider=%s next_provider=%s "
            "error_code=%s failure_kind=%s status_code=%s provider_code=%s retryable=%s",
            provider.provider,
            next_provider.provider,
            error.error_code.value,
            error.failure_kind.value if error.failure_kind is not None else "unspecified",
            error.status_code,
            error.provider_code,
            error.retryable,
        )

    @staticmethod
    def _merge_context(
        target: ToolExecutionContext,
        source: ToolExecutionContext,
        *,
        include_warnings: bool,
    ) -> None:
        target.extend_upstream_calls(source.upstream_calls)
        if include_warnings:
            for warning in source.warnings:
                target.add_warning(warning)

    @staticmethod
    def _provider_label(provider: RoutingProvider) -> str:
        return _PROVIDER_LABELS.get(provider.provider, provider.provider)
