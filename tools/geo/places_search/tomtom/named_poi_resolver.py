"""TomTom fallback for resolving one named POI as a nearby-search anchor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.base import ToolClarification, ToolClarificationOption
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.locality import same_locality
from tools.geo.places_search.tomtom.client import TomTomSearchEndpoint
from tools.geo.places_search.tomtom.matching import (
    clean_optional_text,
    is_auxiliary,
    name_match_score,
    named_result_identity,
    normalized_tokens,
)
from tools.geo.places_search.tomtom.schemas import (
    TomTomSearchResponse,
    TomTomSearchResult,
)
from tools.geo.text_place_query import (
    compose_scoped_place_query,
    tomtom_response_language,
    uses_cyrillic,
)
from tools.geo.text_place_resolution import PlaceResolutionArea
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, ResolvedPlace, mint_place_ref

_CANDIDATE_LIMIT = 20
_ANCHOR_EQUIVALENCE_DISTANCE_M = 50
_MAX_CLARIFICATION_OPTIONS = 3
_AMBIGUITY_SCORE_GAP_RATIO = 0.03
_SHOPPING_CENTER_CLASSIFICATIONS = frozenset({"SHOPPING_CENTER"})
_RAILWAY_STATION_CLASSIFICATIONS = frozenset({"RAILWAY_STATION"})
_NAMED_PLACE_TYPE_HINTS = (
    (("торгово", "развлекательный", "центр"), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("торговый", "центр"), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("shopping", "center"), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("shopping", "centre"), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("трц",), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("трк",), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("тц",), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("mall",), _SHOPPING_CENTER_CLASSIFICATIONS),
    (("railway", "station"), _RAILWAY_STATION_CLASSIFICATIONS),
    (("train", "station"), _RAILWAY_STATION_CLASSIFICATIONS),
    (("железнодорожный", "вокзал"), _RAILWAY_STATION_CLASSIFICATIONS),
    (("жд", "вокзал"), _RAILWAY_STATION_CLASSIFICATIONS),
    (("вокзал",), _RAILWAY_STATION_CLASSIFICATIONS),
    (("station",), _RAILWAY_STATION_CLASSIFICATIONS),
)


@dataclass(frozen=True, slots=True)
class _NamedCandidate:
    """An address-bearing TomTom POI plus optional semantic rank metadata."""

    result: TomTomSearchResult
    address: str
    name_score: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _NamedPoiTypeHint:
    name_query: str
    classification_codes: frozenset[str]


class _NamedSearchClient(Protocol):
    async def search(
        self,
        *,
        endpoint: TomTomSearchEndpoint,
        query: str | None,
        limit: int,
        context: ToolExecutionContext,
        language: str,
        bbox: GeoBounds | None,
    ) -> TomTomSearchResponse:
        """Execute the small Fuzzy request required by this resolver."""
        ...


class TomTomNamedPoiResolver:
    """Resolve one defensible POI or clarify distinct anchor locations."""

    provider = "tomtom"
    requires_city_bounds = True

    def __init__(
        self,
        *,
        client: _NamedSearchClient,
        place_store: PlaceStore,
    ) -> None:
        self._client = client
        self._place_store = place_store

    async def resolve_named_poi(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Resolve one city-scoped POI and preserve provider-ranked ambiguity."""

        del area

        response = await self._search(
            query=query, city=city, city_bounds=city_bounds, context=context
        )
        candidates = _build_candidates(
            response,
            query=query,
            city=city,
            city_bounds=city_bounds,
        )
        groups = _candidate_groups(candidates)
        if not groups:
            return None
        if len(groups) > 1 and _scores_require_clarification(
            groups[0][0].result.score,
            groups[1][0].result.score,
        ):
            representatives = [group[0] for group in groups[:_MAX_CLARIFICATION_OPTIONS]]
            records = [_place_record(candidate) for candidate in representatives]
            await self._place_store.save_many(records)
            raise AmbiguousPlaceError(
                query,
                clarification=ToolClarification(
                    kind="select_anchor",
                    question=_clarification_question(query),
                    options=[
                        ToolClarificationOption(
                            value=record.ref,
                            label=_clarification_label(candidate),
                            description=record.address[:500],
                        )
                        for candidate, record in zip(representatives, records, strict=True)
                    ],
                ),
            )

        return await self._persist(groups[0][0])

    async def resolve_first_address(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Return the first provider card with an address for routing."""

        del area

        response = await self._search(
            query=query, city=city, city_bounds=city_bounds, context=context
        )
        candidate = _first_address_candidate(response)
        return await self._persist(candidate) if candidate is not None else None

    async def _search(
        self,
        *,
        query: str,
        city: str,
        city_bounds: GeoBounds | None,
        context: ToolExecutionContext,
    ) -> TomTomSearchResponse:
        return await self._client.search(
            endpoint=TomTomSearchEndpoint.FUZZY,
            query=compose_scoped_place_query(
                query=query,
                city=city,
                city_first=False,
            ),
            limit=_CANDIDATE_LIMIT,
            language=tomtom_response_language(query, city),
            bbox=city_bounds,
            context=context,
        )

    async def _persist(self, candidate: _NamedCandidate) -> ResolvedPlace:
        """Store the chosen routable point behind an opaque shared ref."""

        record = _place_record(candidate)
        await self._place_store.save(record)
        return ResolvedPlace(ref=record.ref, record=record)


def _build_candidates(
    response: TomTomSearchResponse,
    *,
    query: str,
    city: str,
    city_bounds: GeoBounds | None = None,
) -> list[_NamedCandidate]:
    """Keep only defensible city-scoped cards in provider-ranked order."""

    type_hint = _extract_type_hint(query)
    candidates: list[_NamedCandidate] = []
    for result in response.results:
        address = clean_optional_text(result.address.freeform_address)
        locality = clean_optional_text(result.address.municipality)
        if address is None or locality is None:
            continue
        if not same_locality(city, locality) and not _is_inside_bounds(
            result,
            city_bounds,
        ):
            continue
        name_score = name_match_score(result.poi.name, query, locality=city)
        if (
            type_hint is not None
            and name_score != (0, 0)
            and not result.poi.classification_codes & type_hint.classification_codes
        ):
            continue
        if name_score is None and type_hint is not None:
            name_score = name_match_score(
                result.poi.name,
                type_hint.name_query,
                locality=city,
            )
        if name_score is None:
            continue
        candidates.append(
            _NamedCandidate(
                result=result,
                address=address,
                name_score=name_score,
            )
        )
    return candidates


def _first_address_candidate(response: TomTomSearchResponse) -> _NamedCandidate | None:
    for result in response.results:
        address = clean_optional_text(result.address.freeform_address)
        if address is not None:
            return _NamedCandidate(result=result, address=address, name_score=(0, 0))
    return None


def _scores_require_clarification(first_score: float, second_score: float) -> bool:
    """Return whether two leading distinct places are within 3% of the top score."""

    if first_score <= 0:
        return first_score == second_score
    return (first_score - second_score) / first_score < _AMBIGUITY_SCORE_GAP_RATIO


def _is_inside_bounds(
    result: TomTomSearchResult,
    bounds: GeoBounds | None,
) -> bool:
    """Use provider-independent city bounds when locality names are translated."""

    if bounds is None:
        return False
    point = result.position
    return bounds.west <= point.lon <= bounds.east and bounds.south <= point.lat <= bounds.north


def _candidate_groups(
    candidates: list[_NamedCandidate],
) -> list[list[_NamedCandidate]]:
    """Return distinct anchor locations in TomTom provider order."""

    if not candidates:
        return []
    candidates = _deduplicate(candidates)
    candidates = _exclude_auxiliary_when_primary_exists(candidates)
    groups: list[list[_NamedCandidate]] = []
    for candidate in candidates:
        group = next(
            (
                existing
                for existing in groups
                if any(_same_anchor_location(candidate, member) for member in existing)
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    return groups


def _extract_type_hint(query: str) -> _NamedPoiTypeHint | None:
    """Extract only explicit type words already present in the user's text."""

    query_tokens = normalized_tokens(query)
    for hint_tokens, classification_codes in _NAMED_PLACE_TYPE_HINTS:
        if len(query_tokens) <= len(hint_tokens):
            continue
        if query_tokens[: len(hint_tokens)] == hint_tokens:
            name_tokens = query_tokens[len(hint_tokens) :]
        elif query_tokens[-len(hint_tokens) :] == hint_tokens:
            name_tokens = query_tokens[: -len(hint_tokens)]
        else:
            continue
        return _NamedPoiTypeHint(
            name_query=" ".join(name_tokens),
            classification_codes=classification_codes,
        )
    return None


def _same_anchor_location(left: _NamedCandidate, right: _NamedCandidate) -> bool:
    left_point = left.result.routing_position
    right_point = right.result.routing_position
    return (
        calculate_distance_m(
            from_lat=left_point.lat,
            from_lon=left_point.lon,
            to_lat=right_point.lat,
            to_lon=right_point.lon,
        )
        <= _ANCHOR_EQUIVALENCE_DISTANCE_M
    )


def _place_record(candidate: _NamedCandidate) -> PlaceRecord:
    result = candidate.result
    point = result.routing_position
    return PlaceRecord(
        ref=mint_place_ref(f"tomtom:poi:{result.id}"),
        name=result.poi.name,
        address=candidate.address,
        lat=point.lat,
        lon=point.lon,
        locality=result.address.municipality,
        provider="tomtom",
        provider_id=result.id,
        origin=RecordOrigin.PLACES_SEARCH,
    )


def _clarification_label(candidate: _NamedCandidate) -> str:
    result = candidate.result
    category_names = result.poi.category_names
    label = f"{result.poi.name} ({category_names[0]})" if category_names else result.poi.name
    return label[:200]


def _clarification_question(query: str) -> str:
    if uses_cyrillic(query):
        return f"Какой объект по запросу {query!r} вы имели в виду?"
    return f"Which place matching {query!r} do you mean?"


def _deduplicate(candidates: list[_NamedCandidate]) -> list[_NamedCandidate]:
    result: list[_NamedCandidate] = []
    seen: set[tuple[tuple[str, ...], frozenset[str]]] = set()
    for candidate in candidates:
        identity = named_result_identity(candidate.result, candidate.address)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(candidate)
    return result


def _exclude_auxiliary_when_primary_exists(
    candidates: list[_NamedCandidate],
) -> list[_NamedCandidate]:
    has_exact_primary = any(
        candidate.name_score == (0, 0) and not is_auxiliary(candidate.result)
        for candidate in candidates
    )
    if not has_exact_primary:
        return candidates
    return [candidate for candidate in candidates if not is_auxiliary(candidate.result)]
