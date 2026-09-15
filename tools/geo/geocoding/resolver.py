"""Geocode address/toponym queries into shared place refs."""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol, runtime_checkable

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.matching import select_unique_place_match, uses_added_house_suffix
from tools.geo.geocoding.schemas import MAX_MATCHES, GeocodePlaceInput, PlaceMatch, ToponymKind
from tools.geo.geocoding.service import GeocoderService
from tools.geo.place_store import PlaceStore
from tools.geo.text_place_query import same_locality_anchor, uses_cyrillic
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRef, ResolvedPlace


@runtime_checkable
class _LocalityReverseGeocoder(Protocol):
    async def reverse_geocode_city(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
    ) -> str | None: ...


class PlaceResolutionContractError(ToolExecutionError):
    """The geocoder returned a ref without persisting its hidden record."""

    def __init__(
        self,
        *,
        provider: str,
        message: str = "Internal geocoder returned a place ref without persisting its record",
    ) -> None:
        super().__init__(
            ToolErrorCode.UPSTREAM_ERROR,
            message,
            provider=provider,
            failure_kind=ToolFailureKind.INTERNAL_CONTRACT,
            retryable=False,
        )


class GeocodedPlaceResolver:
    """Geocode addresses, toponyms, and bounded localities into hidden records."""

    def __init__(
        self,
        *,
        geocoder: GeocoderService,
        place_store: PlaceStore,
        allowed_country_codes: Collection[str] | None = None,
    ) -> None:
        """Bind address/toponym geocoding to reusable place storage."""

        self._geocoder = geocoder
        self._place_store = place_store
        self._allowed_country_codes = (
            frozenset(code.strip().casefold() for code in allowed_country_codes if code.strip())
            if allowed_country_codes is not None
            else None
        )

    @property
    def provider(self) -> str:
        """Return the primary geocoder provider used by this resolver."""

        return self._geocoder.provider

    async def geocode_place(
        self,
        *,
        query: str,
        city: str | None,
        context: ToolExecutionContext,
    ) -> ResolvedPlace | None:
        """Resolve one address or geographic toponym through the geocoder."""

        result = await self._geocoder.geocode(
            GeocodePlaceInput(
                query=query,
                city=city,
                limit=MAX_MATCHES,
            ),
            context,
        )
        match = select_unique_place_match(
            result,
            query=query,
            city=city,
        )
        if match is None:
            return None

        if uses_added_house_suffix(match, query=query):
            context.add_warning(
                f"Geocoder normalized the requested address {query!r} to "
                f"{match.name!r}; verify the building suffix if exact entrance "
                "accuracy matters."
            )

        record = await self._place_store.get(match.ref)
        if record is None:
            raise PlaceResolutionContractError(provider=self._geocoder.provider)

        return ResolvedPlace(ref=match.ref, record=record)

    async def geocode_bounded_locality(
        self,
        query: str,
        context: ToolExecutionContext,
    ) -> ResolvedPlace | None:
        """Resolve one locality whose stored record includes provider bounds."""

        result = await self._geocoder.geocode(
            GeocodePlaceInput(query=query, limit=MAX_MATCHES, locality_only=True),
            context,
        )

        localities: list[tuple[PlaceMatch, ResolvedPlace]] = []
        for match in result.matches:
            if match.kind is not ToponymKind.LOCALITY:
                continue

            record = await self._place_store.get(match.ref)
            if record is None:
                raise PlaceResolutionContractError(
                    provider=self.provider,
                    message=(
                        "Internal geocoder returned a locality ref without persisting its record"
                    ),
                )
            localities.append(
                (
                    match,
                    ResolvedPlace(ref=match.ref, record=record),
                )
            )

        if not localities:
            return None

        localities = self._filter_allowed_countries(localities)
        if not localities:
            return None

        localities = self._prefer_municipalities(localities)
        localities = self._deduplicate_administrative_localities(localities)

        exact_matches = [
            item
            for item in localities
            if same_locality_anchor(item[0].name, query)
            or (
                item[1].record.locality is not None
                and same_locality_anchor(item[1].record.locality, query)
            )
        ]
        # Keep TomTom's top-ranked locality when its localized label differs
        # from the user's exonym (Rome -> Roma, Munich -> München). Literal
        # namesakes may narrow the list only when the provider's own first
        # result is also an exact match; otherwise they would incorrectly
        # discard the most relevant city before score comparison.
        candidates = exact_matches if localities[0] in exact_matches else localities
        # Locality resolution is scope preparation, not anchor selection. Once
        # provider results have passed country filtering, municipality
        # preference, administrative deduplication, and exact-name narrowing,
        # trust their remaining order and select the first candidate. Textual
        # POI/anchor resolution retains its separate ambiguity policy.
        _, resolved = candidates[0]
        if resolved.record.kind != ToponymKind.LOCALITY.value:
            raise PlaceResolutionContractError(
                provider=self.provider,
                message="Internal geocoder returned inconsistent locality metadata",
            )
        if resolved.record.bounds is None:
            return None
        return resolved

    async def localize_bounded_locality(
        self,
        resolved: ResolvedPlace,
        *,
        language: str,
        context: ToolExecutionContext,
    ) -> ResolvedPlace:
        """Cache a municipality label in the language used by place search."""

        cached = resolved.record.localized_localities.get(language)
        if cached is not None:
            return resolved

        # A locality originally geocoded in this script needs no additional
        # request; store its existing structured municipality under the same
        # language key used by the places provider.
        if language == "ru-RU":
            existing_label = next(
                (
                    label
                    for label in (resolved.record.locality, resolved.record.name)
                    if label is not None and uses_cyrillic(label)
                ),
                None,
            )
            if existing_label is not None:
                return await self._save_localized_locality(
                    resolved,
                    language=language,
                    localized=existing_label,
                )

        if not isinstance(self._geocoder, _LocalityReverseGeocoder):
            return resolved

        try:
            localized = await self._geocoder.reverse_geocode_city(
                lat=resolved.record.lat,
                lon=resolved.record.lon,
                language=language,
                context=context,
            )
        except ToolExecutionError:
            # Localization improves cross-language matching but must not turn a
            # previously valid strict search into a hard failure. Without an
            # alias, providers retain the original locality comparison.
            context.add_warning(
                f"Could not localize the search-area municipality to {language}; "
                "using its original name."
            )
            return resolved
        if localized is None or not (localized := " ".join(localized.split())):
            return resolved

        return await self._save_localized_locality(
            resolved,
            language=language,
            localized=localized,
        )

    async def localize_bounded_locality_ref(
        self,
        ref: PlaceRef,
        *,
        language: str,
        context: ToolExecutionContext,
    ) -> ResolvedPlace | None:
        """Localize a reusable area ref when a follow-up changes language."""

        record = await self._place_store.get(ref)
        if record is None or record.kind != ToponymKind.LOCALITY.value or record.bounds is None:
            return None
        return await self.localize_bounded_locality(
            ResolvedPlace(ref=record.ref, record=record),
            language=language,
            context=context,
        )

    async def load_bounded_locality_ref(self, ref: PlaceRef) -> ResolvedPlace | None:
        """Load a reusable locality ref without issuing another geocoder request."""

        record = await self._place_store.get(ref)
        if record is None or record.kind != ToponymKind.LOCALITY.value or record.bounds is None:
            return None
        return ResolvedPlace(ref=record.ref, record=record)

    async def _save_localized_locality(
        self,
        resolved: ResolvedPlace,
        *,
        language: str,
        localized: str,
    ) -> ResolvedPlace:
        """Persist one normalized language-specific municipality label."""

        aliases = dict(resolved.record.localized_localities)
        aliases[language] = localized
        record = resolved.record.model_copy(update={"localized_localities": aliases})
        await self._place_store.save(record)
        return ResolvedPlace(ref=resolved.ref, record=record)

    def _filter_allowed_countries(
        self,
        localities: list[tuple[PlaceMatch, ResolvedPlace]],
    ) -> list[tuple[PlaceMatch, ResolvedPlace]]:
        """Discard locality candidates outside the configured product geography.

        A missing country code is not sufficient evidence that a candidate is
        inside a closed country scope.  Filtering happens before score-gap
        ambiguity detection, so an unrelated high-ranked namesake cannot force
        a clarification or win the selection.
        """

        if self._allowed_country_codes is None:
            return localities
        return [
            item
            for item in localities
            if item[0].country_code is not None
            and item[0].country_code.casefold() in self._allowed_country_codes
        ]

    @staticmethod
    def _prefer_municipalities(
        localities: list[tuple[PlaceMatch, ResolvedPlace]],
    ) -> list[tuple[PlaceMatch, ResolvedPlace]]:
        """Use administrative geographies only when TomTom found no municipality.

        A localized municipality name need not textually match the user's
        exonym. Keeping its raw provider entity type prevents a broader
        CountrySecondarySubdivision from replacing that city in such cases.
        Other geocoders leave ``provider_entity_type`` unset, so their
        existing behavior remains unchanged.
        """

        municipalities = [
            item for item in localities if item[0].provider_entity_type == "Municipality"
        ]
        return municipalities or localities

    @classmethod
    def _deduplicate_administrative_localities(
        cls,
        candidates: list[tuple[PlaceMatch, ResolvedPlace]],
    ) -> list[tuple[PlaceMatch, ResolvedPlace]]:
        """Collapse provider variants of the same municipality and region."""

        groups: list[
            tuple[
                int,
                tuple[str, str],
                set[str],
                tuple[PlaceMatch, ResolvedPlace],
            ]
        ] = []
        ungrouped: list[tuple[int, tuple[PlaceMatch, ResolvedPlace]]] = []
        for candidate_index, candidate in enumerate(candidates):
            identity = cls._administrative_locality_identity(candidate[0])
            if identity is None:
                ungrouped.append((candidate_index, candidate))
                continue

            context, aliases = identity
            matching_indexes = [
                index
                for index, (_, group_context, group_aliases, _) in enumerate(groups)
                if group_context == context and not aliases.isdisjoint(group_aliases)
            ]
            if not matching_indexes:
                groups.append((candidate_index, context, aliases, candidate))
                continue

            primary_index = matching_indexes[0]
            first_index, group_context, group_aliases, selected = groups[primary_index]
            group_aliases.update(aliases)
            if cls._administrative_locality_preference(candidate) > (
                cls._administrative_locality_preference(selected)
            ):
                selected = candidate
            groups[primary_index] = (first_index, group_context, group_aliases, selected)

            for duplicate_index in reversed(matching_indexes[1:]):
                (
                    duplicate_first_index,
                    _,
                    duplicate_aliases,
                    duplicate_candidate,
                ) = groups.pop(duplicate_index)
                first_index = min(first_index, duplicate_first_index)
                group_aliases.update(duplicate_aliases)
                if cls._administrative_locality_preference(duplicate_candidate) > (
                    cls._administrative_locality_preference(selected)
                ):
                    selected = duplicate_candidate
                groups[primary_index] = (first_index, group_context, group_aliases, selected)

        positioned = [(first_index, candidate) for first_index, _, _, candidate in groups]
        positioned.extend(ungrouped)
        positioned.sort(key=lambda item: item[0])
        return [candidate for _, candidate in positioned]

    @staticmethod
    def _administrative_locality_identity(
        match: PlaceMatch,
    ) -> tuple[tuple[str, str], set[str]] | None:
        """Return region/country context plus municipality-level aliases."""

        if (
            match.municipality is None
            or match.country_subdivision is None
            or match.country_code is None
        ):
            return None
        aliases = {
            " ".join(value.casefold().split())
            for value in (
                match.municipality,
                match.country_secondary_subdivision,
            )
            if value is not None and value.strip()
        }
        return (
            (
                " ".join(match.country_subdivision.casefold().split()),
                match.country_code.casefold(),
            ),
            aliases,
        )

    @staticmethod
    def _administrative_locality_preference(
        candidate: tuple[PlaceMatch, ResolvedPlace],
    ) -> tuple[bool, float, float]:
        """Prefer the canonical municipality, then broader bounds and score."""

        match, resolved = candidate
        municipality = match.municipality or ""
        canonical_name = " ".join(match.name.casefold().split()) == " ".join(
            municipality.casefold().split()
        )
        bounds = resolved.record.bounds
        bounds_area = (
            abs((bounds.east - bounds.west) * (bounds.north - bounds.south))
            if bounds is not None
            else 0.0
        )
        return (canonical_name, bounds_area, match.relevance_score or 0.0)
