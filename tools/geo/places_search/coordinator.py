from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.coordination import ProviderExecutionStrategy
from tools.geo.places_search.provider import (
    FirstAddressPlacesSearchProvider,
    PlacesSearchProvider,
)
from tools.geo.places_search.schemas import (
    MAX_NAMED_PLACE_RESULTS,
    OrganisationCandidate,
    OrganisationResolution,
    OrganisationResolutionStatus,
    Place,
    PlacesSearchInput,
    PlacesSearchOutput,
    SearchMode,
)
from tools.geo.places_search.tomtom.matching import normalized_address_tokens
from tools.observability import ToolExecutionContext

logger = logging.getLogger(__name__)

# 2GIS is coverage-aware: the shared geocoder first resolves a city into an
# opaque Redis-backed point ref, then 2GIS checks Regions API by that point.
# If no 2GIS project covers it, the normal fallback advances to TomTom.
PLACES_SEARCH_PROVIDER_PRIORITY = ("twogis", "tomtom", "yandex")
_PROVIDER_LABELS = {
    "tomtom": "TomTom",
    "twogis": "2GIS",
    "yandex": "Yandex",
}


class SearchScopeResolver(Protocol):
    """Prepare one provider-independent scope for all search attempts."""

    async def resolve_scope(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchInput:
        """Replace textual area/anchor input with a reusable place ref."""
        ...


class _FallbackAction(StrEnum):
    STOP = "stop"
    TRY_NEXT = "try_next"
    TRY_NEXT_AND_ALERT = "try_next_and_alert"


class PlacesSearchCoordinator:
    """Search providers in priority order without mixing their result sets."""

    def __init__(
        self,
        *,
        providers: Sequence[PlacesSearchProvider],
        scope_resolver: SearchScopeResolver | None = None,
        strategy: ProviderExecutionStrategy = ProviderExecutionStrategy.FALLBACK,
    ) -> None:
        if not providers:
            raise ValueError("places search coordinator requires at least one provider")

        provider_names = [provider.provider for provider in providers]

        if len(provider_names) != len(set(provider_names)):
            raise ValueError(f"duplicate places search providers: {provider_names}")

        if strategy is ProviderExecutionStrategy.PARALLEL:
            raise NotImplementedError("parallel places search strategy is not implemented")
        if scope_resolver is None:
            raise ValueError("places search coordinator requires a scope resolver")

        priority = {
            provider_name: index
            for index, provider_name in enumerate(PLACES_SEARCH_PROVIDER_PRIORITY)
        }
        # Built-ins have a stable product-level priority. Custom/test providers
        # keep their configured relative order after the built-ins.
        self._providers = tuple(
            sorted(
                providers,
                key=lambda provider: priority.get(provider.provider, len(priority)),
            )
        )
        self._scope_resolver = scope_resolver
        self._strategy = strategy

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        if args.mode is SearchMode.RESOLVE:
            return await self._resolve_organisations(args, context)

        providers = self._providers_for(args)
        if self._strategy is ProviderExecutionStrategy.FIRST:
            provider = providers[0]
            prepared_args = await self._prepare_args_for_provider(args, provider, context)
            return await provider.search(prepared_args, context)
        if self._strategy is ProviderExecutionStrategy.FALLBACK:
            return await self._search_with_fallback(args, context, providers)

        raise AssertionError(f"unsupported strategy: {self._strategy}")

    async def _resolve_organisations(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Resolve known names independently with lazy shared-scope fallback."""

        assert args.area is not None  # Enforced by PlacesSearchInput.
        assert args.organisations  # Enforced by PlacesSearchInput.

        providers = self._providers_for(args)
        shared_area_ref = args.area_ref
        if shared_area_ref is None and not getattr(providers[0], "resolves_area_natively", False):
            assert args.city is not None
            scope_seed = PlacesSearchInput(
                mode=SearchMode.AREA,
                query=args.organisations[0].name,
                area=args.city,
            )
            prepared_scope = await self._scope_resolver.resolve_scope(scope_seed, context)
            if prepared_scope.area_ref is None:
                raise AssertionError("resolve scope did not produce an area_ref")
            shared_area_ref = prepared_scope.area_ref

        outcomes: list[OrganisationResolution] = []
        for candidate in args.organisations:
            item_args = PlacesSearchInput(
                mode=SearchMode.AREA,
                query=candidate.name,
                area=shared_area_ref if shared_area_ref is not None else args.city,
                open_24h=args.open_24h,
                open_now=args.open_now,
                min_rating=args.min_rating,
                limit=MAX_NAMED_PLACE_RESULTS,
            )
            try:
                result = await self._search_with_fallback(
                    item_args,
                    context,
                    providers,
                    first_with_address=True,
                )
            except ToolExecutionError as exc:
                # A shared city-resolution failure affects the whole batch and
                # must preserve its top-level clarification. Provider failures
                # for one organisation remain independent item outcomes.
                if exc.provider is None or exc.provider.endswith("_geocoder"):
                    raise
                outcomes.append(
                    OrganisationResolution(
                        client_id=candidate.client_id,
                        input_name=candidate.name,
                        status=OrganisationResolutionStatus.ERROR,
                        error=str(exc),
                    )
                )
                continue
            if shared_area_ref is None and result.area is not None:
                shared_area_ref = result.area.ref
            outcomes.append(self._select_resolution(candidate, result.places))

        return PlacesSearchOutput(resolved=outcomes)

    @staticmethod
    def _select_resolution(
        candidate: OrganisationCandidate,
        places: list[Place],
    ) -> OrganisationResolution:
        """Select the first provider result for one known entity.

        ``resolve`` is an identity lookup, not branch discovery. Built-in
        providers return their first address-bearing card without semantic
        reranking. Textual nearby anchors follow a separate path and still
        raise a clarification when more than one anchor survives normalization.
        """

        unique_places = list({place.ref: place for place in places}.values())
        if candidate.address_hint is not None:
            hint_tokens = normalized_address_tokens(candidate.address_hint)
            address_matches = [
                place
                for place in unique_places
                if hint_tokens and hint_tokens.issubset(normalized_address_tokens(place.address))
            ]
            # Address hints often come from an informal user phrase (for
            # example, "Покровка") rather than the provider's official street
            # name. Prefer an exact address match when present, but do not turn
            # otherwise valid provider results into a false ``not_found``.
            if address_matches:
                unique_places = address_matches

        if not unique_places:
            return OrganisationResolution(
                client_id=candidate.client_id,
                input_name=candidate.name,
                status=OrganisationResolutionStatus.NOT_FOUND,
            )

        return OrganisationResolution(
            client_id=candidate.client_id,
            input_name=candidate.name,
            status=OrganisationResolutionStatus.RESOLVED,
            place=unique_places[0],
        )

    async def _search_with_fallback(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
        providers: Sequence[PlacesSearchProvider],
        *,
        first_with_address: bool = False,
    ) -> PlacesSearchOutput:
        shared_resolved_args: PlacesSearchInput | None = None
        for index, provider in enumerate(providers):
            attempt_context = ToolExecutionContext()
            try:
                if self._needs_shared_scope(args, provider):
                    if shared_resolved_args is None:
                        shared_resolved_args = await self._scope_resolver.resolve_scope(
                            args,
                            context,
                        )
                    provider_args = shared_resolved_args
                else:
                    provider_args = args
                if first_with_address and isinstance(
                    provider,
                    FirstAddressPlacesSearchProvider,
                ):
                    result = await provider.search_first_address(
                        provider_args,
                        attempt_context,
                    )
                else:
                    result = await provider.search(provider_args, attempt_context)
            except ToolExecutionError as exc:
                self._merge_context(context, attempt_context, include_warnings=False)
                is_last = index == len(providers) - 1
                action = self._fallback_action(exc, provider)
                if is_last or action is _FallbackAction.STOP:
                    raise

                next_provider = providers[index + 1]
                self._log_fallback(
                    error=exc,
                    provider=provider,
                    next_provider=next_provider,
                    action=action,
                )
                if exc.failure_kind is ToolFailureKind.COVERAGE_MISS:
                    context.add_warning(
                        f"{self._provider_label(provider)} does not cover the requested locality; "
                        f"using {self._provider_label(next_provider)}."
                    )
                    continue
                if exc.error_code is ToolErrorCode.NOT_FOUND:
                    reason = "returned no matches"
                elif exc.error_code is ToolErrorCode.TIMEOUT:
                    reason = "timed out"
                else:
                    reason = "failed"
                context.add_warning(
                    f"{self._provider_label(provider)} place search {reason}; "
                    f"retrying with {self._provider_label(next_provider)}."
                )
                continue

            self._merge_context(context, attempt_context, include_warnings=True)
            is_last = index == len(providers) - 1
            if result.places or is_last:
                return result

            next_provider = providers[index + 1]
            logger.info(
                "places_search_provider_fallback provider=%s next_provider=%s reason=empty_result",
                provider.provider,
                next_provider.provider,
            )
            context.add_warning(
                f"{self._provider_label(provider)} place search returned no matches; "
                f"retrying with {self._provider_label(next_provider)}."
            )

        raise AssertionError("places search fallback completed without a result or error")

    async def _prepare_args_for_provider(
        self,
        args: PlacesSearchInput,
        provider: PlacesSearchProvider,
        context: ToolExecutionContext,
    ) -> PlacesSearchInput:
        if not self._needs_shared_scope(args, provider):
            return args
        return await self._scope_resolver.resolve_scope(args, context)

    @staticmethod
    def _needs_shared_scope(
        args: PlacesSearchInput,
        provider: PlacesSearchProvider,
    ) -> bool:
        """Return whether this attempt needs the shared ref preparation.

        Nearby text always becomes one portable point ref before any provider
        call. Existing area refs are already prepared. A raw city may instead
        be handed directly to providers such as 2GIS that have a native region
        lookup; the geocoder is then paid only if fallback reaches a provider
        that actually needs a bounded locality record.
        """

        return not (
            args.mode is SearchMode.AREA
            and args.city is not None
            and getattr(provider, "resolves_area_natively", False)
        )

    def _providers_for(
        self,
        args: PlacesSearchInput,
    ) -> tuple[PlacesSearchProvider, ...]:
        providers = self._providers
        if args.min_rating is not None:
            providers = tuple(
                provider
                for provider in providers
                if getattr(provider, "supports_min_rating", False)
            )
            if not providers:
                raise ToolExecutionError(
                    ToolErrorCode.UNSUPPORTED_FILTER,
                    "Minimum-rating filtering requires a configured 2GIS provider. "
                    "Use web_search for the same rated-place request.",
                    retryable=False,
                )

        if not args.open_now:
            return providers

        providers = tuple(provider for provider in providers if provider.supports_open_now)
        if providers:
            return providers

        raise ToolExecutionError(
            ToolErrorCode.UNSUPPORTED_FILTER,
            "Current opening status is unavailable from the configured place providers.",
            retryable=False,
        )

    @staticmethod
    def _fallback_action(
        error: ToolExecutionError,
        provider: PlacesSearchProvider,
    ) -> _FallbackAction:
        if error.provider is None or error.provider.endswith("_geocoder"):
            return _FallbackAction.STOP
        if error.provider != provider.provider and not error.provider.startswith(
            f"{provider.provider}_"
        ):
            return _FallbackAction.STOP

        if (
            provider.provider == "twogis"
            and error.status_code == 403
            and error.provider_code is not None
            and error.provider_code.casefold() == "backendexception"
        ):
            return _FallbackAction.TRY_NEXT_AND_ALERT

        if error.status_code in {401, 403} or error.failure_kind in {
            ToolFailureKind.AUTHENTICATION,
            ToolFailureKind.INTERNAL_CONTRACT,
        }:
            return _FallbackAction.STOP

        if error.error_code is ToolErrorCode.NOT_FOUND:
            return _FallbackAction.TRY_NEXT
        if error.failure_kind in {
            ToolFailureKind.INVALID_JSON,
            ToolFailureKind.INVALID_SCHEMA,
        }:
            return _FallbackAction.TRY_NEXT_AND_ALERT

        transient_failure = (
            error.error_code in {ToolErrorCode.RATE_LIMITED, ToolErrorCode.TIMEOUT}
            or error.failure_kind in {ToolFailureKind.NETWORK, ToolFailureKind.TIMEOUT}
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
        provider: PlacesSearchProvider,
        next_provider: PlacesSearchProvider,
        action: _FallbackAction,
    ) -> None:
        log_level = (
            logging.ERROR if action is _FallbackAction.TRY_NEXT_AND_ALERT else logging.WARNING
        )
        logger.log(
            log_level,
            "places_search_provider_fallback provider=%s next_provider=%s "
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
    def _provider_label(provider: PlacesSearchProvider) -> str:
        return _PROVIDER_LABELS.get(provider.provider, provider.provider)
