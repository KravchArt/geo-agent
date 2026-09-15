"""Conservative filtering and order-preserving deduplication for 2GIS objects."""

from __future__ import annotations

from dataclasses import dataclass

from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.places_search.locality import compatible_locality
from tools.geo.places_search.tomtom.matching import normalized_tokens
from tools.geo.places_search.twogis.schemas import TwoGisItem

_DUPLICATE_DISTANCE_M = 50
_EXTENSION_STOP_WORDS = frozenset({"в", "во", "на", "по", "для", "и", "the", "of"})
_TECHNICAL_ADDRESS_LABELS = frozenset({"object", "объект"})


@dataclass(frozen=True, slots=True)
class TwoGisCandidate:
    item: TwoGisItem
    semantic_kind: str
    provider_rank: int


@dataclass(frozen=True, slots=True)
class _TypeHint:
    tokens: tuple[str, ...]
    allowed_kinds: frozenset[str]


def has_explicit_address(item: TwoGisItem) -> bool:
    """Return whether 2GIS supplied a model-facing address for the card."""

    return bool(item.full_address_name or item.address_name)


_TYPE_HINTS = (
    _TypeHint(("офис",), frozenset({"branch", "building"})),
    _TypeHint(("office",), frozenset({"branch", "building"})),
    _TypeHint(("железнодорожный", "вокзал"), frozenset({"rail_station"})),
    _TypeHint(("станция", "метро"), frozenset({"metro_station"})),
    _TypeHint(
        ("выставочный", "комплекс"),
        frozenset({"exhibition_center", "park", "branch"}),
    ),
    _TypeHint(
        ("торговый", "центр"),
        frozenset({"shopping_center", "branch"}),
    ),
    _TypeHint(("shopping", "center"), frozenset({"shopping_center", "branch"})),
    _TypeHint(("shopping", "centre"), frozenset({"shopping_center", "branch"})),
    _TypeHint(("метро",), frozenset({"metro_station"})),
    _TypeHint(("остановка",), frozenset({"transit_stop"})),
    _TypeHint(("парк",), frozenset({"park"})),
    _TypeHint(("трц",), frozenset({"shopping_center", "branch"})),
    _TypeHint(("трк",), frozenset({"shopping_center", "branch"})),
    _TypeHint(("тц",), frozenset({"shopping_center", "branch"})),
    _TypeHint(("mall",), frozenset({"shopping_center", "branch"})),
    _TypeHint(("стадион",), frozenset({"stadium"})),
    _TypeHint(("аэропорт",), frozenset({"airport"})),
    _TypeHint(("вокзал",), frozenset({"rail_station"})),
    _TypeHint(("музей",), frozenset({"museum"})),
    _TypeHint(("театр",), frozenset({"theatre"})),
    _TypeHint(("кинотеатр",), frozenset({"cinema"})),
    _TypeHint(("гостиница",), frozenset({"hotel"})),
    _TypeHint(("отель",), frozenset({"hotel"})),
)


def build_named_candidates(
    items: list[TwoGisItem],
    *,
    query: str,
    city: str | None,
) -> list[TwoGisCandidate]:
    """Keep defensible city-scoped matches in provider response order."""

    type_hint, core_query = _extract_type_hint(query)
    candidates: list[TwoGisCandidate] = []
    scoped_candidates: list[TwoGisCandidate] = []

    for provider_rank, item in enumerate(items):
        if item.point is None or (city is not None and not _matches_locality(item, city)):
            continue
        kind = semantic_kind(item)
        if type_hint is not None and kind not in type_hint.allowed_kinds:
            continue
        candidate = TwoGisCandidate(
            item=item,
            semantic_kind=kind,
            provider_rank=provider_rank,
        )
        scoped_candidates.append(candidate)
        if not _matches_name(
            item,
            query=query,
            core_query=core_query,
            has_type_hint=type_hint is not None,
        ):
            continue
        candidates.append(candidate)

    if not candidates and type_hint is not None and city is not None:
        strict_locality_candidates = [
            candidate
            for candidate in scoped_candidates
            if _explicitly_matches_locality(candidate.item, city)
        ]
        unique_scoped_candidates = deduplicate_candidates(
            strict_locality_candidates,
            merge_nearby_anchors=True,
        )
        if len(unique_scoped_candidates) == 1:
            candidates = unique_scoped_candidates

    candidates = _prefer_explicit_extension(candidates, query=query)
    candidates = _prefer_exact_non_transit_match(candidates, query=query)
    candidates = _prefer_addressable_landmark(candidates)
    candidates = _prefer_exact_type_hint_name(
        candidates,
        core_query=core_query,
        has_type_hint=type_hint is not None,
    )
    return deduplicate_candidates(candidates, merge_nearby_anchors=True)


def _prefer_exact_type_hint_name(
    candidates: list[TwoGisCandidate],
    *,
    core_query: str,
    has_type_hint: bool,
) -> list[TwoGisCandidate]:
    """Prefer exact aliases after removing an explicit object-type hint.

    A query such as ``ТЦ Небо`` intentionally identifies the shopping centre
    whose short/primary alias is exactly ``Небо``. The broader core-token
    matcher also admits ``Седьмое небо``; retain that fallback only when no
    exact alias exists. Multiple exact aliases remain candidates so genuinely
    distinct branches still produce a clarification.
    """

    if not has_type_hint:
        return candidates
    core_tokens = normalized_tokens(core_query)
    if not core_tokens:
        return candidates
    exact = [
        candidate
        for candidate in candidates
        if any(normalized_tokens(alias) == core_tokens for alias in candidate.item.aliases)
    ]
    return exact or candidates


def _prefer_explicit_extension(
    candidates: list[TwoGisCandidate],
    *,
    query: str,
) -> list[TwoGisCandidate]:
    """Apply extension as a preference without making it a hard filter.

    An explicitly requested qualifier wins. Otherwise prefer ordinary cards
    without an extension, but retain qualified cards when that is all 2GIS
    returned for the requested name.
    """

    explicitly_requested = [
        candidate for candidate in candidates if _query_requests_extension(candidate.item, query)
    ]
    if explicitly_requested:
        return explicitly_requested

    ordinary = [candidate for candidate in candidates if not _extension_tokens(candidate.item)]
    return ordinary or candidates


def _prefer_exact_non_transit_match(
    candidates: list[TwoGisCandidate],
    *,
    query: str,
) -> list[TwoGisCandidate]:
    """Drop a same-name bus stop when an exact non-transit place exists.

    2GIS can return a landmark and a nearby surface-transport stop with the
    same name (for example, Red Square).  In an unqualified named-place search
    the stop is not an alternative identity for the landmark.  Explicit stop
    queries have already been narrowed to ``transit_stop`` by the type-hint
    filter above, so they are intentionally unaffected.
    """

    query_tokens = normalized_tokens(query)
    has_exact_non_transit_match = any(
        candidate.semantic_kind != "transit_stop"
        and any(normalized_tokens(alias) == query_tokens for alias in candidate.item.aliases)
        for candidate in candidates
    )
    if not has_exact_non_transit_match:
        return candidates
    return [candidate for candidate in candidates if candidate.semantic_kind != "transit_stop"]


def _prefer_addressable_landmark(
    candidates: list[TwoGisCandidate],
) -> list[TwoGisCandidate]:
    """Drop nearby-search artefacts when the top match has a real address.

    A 2GIS name lookup may append a street stop or a small named square near an
    addressable building. They remain useful for explicit stop/park queries,
    where the type filter makes them the first result, but they are not useful
    alternatives to an already top-ranked addressable landmark.
    """

    if not candidates or not _has_address_identity(candidates[0].item):
        return candidates
    return [
        candidate
        for candidate in candidates
        if candidate.semantic_kind not in {"transit_stop", "park"}
        or _has_address_identity(candidate.item)
    ]


def _has_address_identity(item: TwoGisItem) -> bool:
    if item.full_address_name or item.address_name:
        return True

    has_building_id = (
        item.structured_address is not None and item.structured_address.building_id is not None
    )
    return has_building_id and not _has_technical_placeholder_address(item)


def _has_technical_placeholder_address(item: TwoGisItem) -> bool:
    """Detect provider-only labels such as ``Нижний Новгород, Объект``."""

    if item.full_name is None:
        return False
    address_tail = item.full_name.rsplit(",", maxsplit=1)[-1].strip().casefold()
    return address_tail in _TECHNICAL_ADDRESS_LABELS


def deduplicate_candidates(
    candidates: list[TwoGisCandidate],
    *,
    merge_nearby_anchors: bool = False,
) -> list[TwoGisCandidate]:
    """Remove later aliases without replacing the first provider record."""

    result: list[TwoGisCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(result)
                if _same_named_object(
                    existing,
                    candidate,
                    merge_nearby_anchors=merge_nearby_anchors,
                )
            ),
            None,
        )
        if duplicate_index is None:
            result.append(candidate)
    return result


def semantic_kind(item: TwoGisItem) -> str:
    rubric_tokens = _rubric_tokens(item)
    route_type = (item.route_type or "").casefold()
    subtype = (item.subtype or "").casefold()

    if item.type == "station":
        if route_type == "metro" or subtype == "metro":
            return "metro_station"
        if route_type in {"train", "railway"} or subtype in {"railway", "train"}:
            return "rail_station"
        return "transit_stop"
    if item.type == "adm_div.place" or (item.type == "adm_div" and subtype == "place"):
        return "park"
    if "parki" in rubric_tokens or "парки" in rubric_tokens:
        return "park"
    if rubric_tokens & {"torgovye", "торговые", "mall", "моллы"}:
        return "shopping_center"
    if rubric_tokens & {"stadiony", "стадионы", "stadium"}:
        return "stadium"
    if rubric_tokens & {"aeroporty", "аэропорты", "airport"}:
        return "airport"
    if rubric_tokens & {"muzei", "музеи", "museum"}:
        return "museum"
    if rubric_tokens & {"teatry", "театры", "theatre"}:
        return "theatre"
    if rubric_tokens & {"kinoteatry", "кинотеатры", "cinema"}:
        return "cinema"
    if rubric_tokens & {"gostinicy", "гостиницы", "hotel"}:
        return "hotel"
    if rubric_tokens & {"vystavochnye", "выставочные", "exhibition"}:
        return "exhibition_center"
    if rubric_tokens & {"zheleznodorozhnye", "железнодорожные", "railway"}:
        return "rail_station"
    if item.type == "attraction":
        return "attraction"
    if item.type == "building":
        return "building"
    return item.type


def matches_explicit_type_hint(item: TwoGisItem, query: str) -> bool:
    """Reject only objects contradicting an explicit type word in the query."""

    type_hint, _ = _extract_type_hint(query)
    return type_hint is None or semantic_kind(item) in type_hint.allowed_kinds


def _extension_tokens(item: TwoGisItem) -> tuple[str, ...]:
    """Return meaningful normalized words from a 2GIS name extension."""

    extension = item.name_ex.extension if item.name_ex is not None else None
    if extension is None:
        return ()
    return tuple(
        token for token in normalized_tokens(extension) if token not in _EXTENSION_STOP_WORDS
    )


def _query_requests_extension(item: TwoGisItem, query: str) -> bool:
    """Return whether the user explicitly named this card's qualifier."""

    extension_tokens = _extension_tokens(item)
    if not extension_tokens:
        return False
    query_tokens = normalized_tokens(query)
    return any(
        _same_extension_lexeme(extension_token, query_token)
        for extension_token in extension_tokens
        for query_token in query_tokens
    )


def _same_extension_lexeme(left: str, right: str) -> bool:
    if left == right:
        return True
    return min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left))


def _rubric_tokens(item: TwoGisItem) -> set[str]:
    tokens: set[str] = set()
    for rubric in item.rubrics:
        for value in (rubric.alias, rubric.name):
            if value is None:
                continue
            # 2GIS rubric aliases use underscores (for example
            # ``torgovye_centry``), while ``normalized_tokens`` deliberately
            # treats them as word characters. Split them here so aliases and
            # localized display names follow the same semantic rules.
            tokens.update(normalized_tokens(value.replace("_", " ")))
    return tokens


def _extract_type_hint(query: str) -> tuple[_TypeHint | None, str]:
    query_tokens = normalized_tokens(query)
    for hint in _TYPE_HINTS:
        hint_length = len(hint.tokens)
        for start in range(len(query_tokens) - hint_length + 1):
            if query_tokens[start : start + hint_length] != hint.tokens:
                continue
            remaining = query_tokens[:start] + query_tokens[start + hint_length :]
            if remaining:
                return hint, " ".join(remaining)
    return None, query


def _matches_name(
    item: TwoGisItem,
    *,
    query: str,
    core_query: str,
    has_type_hint: bool,
) -> bool:
    query_tokens = normalized_tokens(query)
    core_tokens = normalized_tokens(core_query)

    for alias in item.aliases:
        alias_tokens = normalized_tokens(alias)
        if alias_tokens == query_tokens:
            return True
        if alias_tokens and all(
            any(_same_name_lexeme(alias_token, query_token) for query_token in query_tokens)
            for alias_token in alias_tokens
        ):
            return True
        if has_type_hint and core_tokens and set(core_tokens).issubset(alias_tokens):
            return True
        if set(query_tokens).issubset(alias_tokens):
            # 2GIS often appends an object kind to the requested landmark,
            # e.g. "Алые паруса, жилой комплекс".  It is still a valid named
            # anchor and must reach spatial deduplication.
            return True

    return False


def _same_name_lexeme(left: str, right: str) -> bool:
    if left == right:
        return True
    return min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left))


def _matches_locality(item: TwoGisItem, city: str) -> bool:
    # Some non-address objects contain only a region and no city. Keep them if
    # 2GIS did not contradict the requested locality; reject explicit mismatch.
    return not item.locality_names or any(
        compatible_locality(city, locality) for locality in item.locality_names
    )


def _explicitly_matches_locality(item: TwoGisItem, city: str) -> bool:
    """Require positive locality evidence for the unique scoped fallback."""

    return bool(item.locality_names) and any(
        compatible_locality(city, locality) for locality in item.locality_names
    )


def _same_named_object(
    left: TwoGisCandidate,
    right: TwoGisCandidate,
    *,
    merge_nearby_anchors: bool,
) -> bool:
    if left.item.id == right.item.id:
        return True

    # The catalog commonly returns both the physical building and an
    # organisation/landmark card for that same building.  Their semantic kinds
    # legitimately differ, but for point resolution they are one anchor.  Keep
    # the first provider-ranked card instead of presenting both as alternatives.
    left_building_id = (
        left.item.structured_address.building_id
        if left.item.structured_address is not None
        else None
    )
    right_building_id = (
        right.item.structured_address.building_id
        if right.item.structured_address is not None
        else None
    )
    if (
        merge_nearby_anchors
        and left_building_id is not None
        and left_building_id == right_building_id
    ):
        return True

    if left.item.point is None or right.item.point is None:
        return False
    distance = calculate_distance_m(
        from_lat=left.item.point.lat,
        from_lon=left.item.point.lon,
        to_lat=right.item.point.lat,
        to_lon=right.item.point.lon,
    )
    if distance > _DUPLICATE_DISTANCE_M:
        return False
    if merge_nearby_anchors:
        # Point resolution needs one coordinate, not separate catalogue
        # identities. Nearby cards describe the same textual anchor.
        return True

    # Category discovery still needs separate nearby businesses. Only collapse
    # aliases of the same semantic object in that mode.
    if left.semantic_kind != right.semantic_kind:
        return False
    left_tokens = frozenset(normalized_tokens(left.item.name))
    right_tokens = frozenset(normalized_tokens(right.item.name))
    return bool(left_tokens and right_tokens) and (
        left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)
    )
