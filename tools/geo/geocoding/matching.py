"""Deterministic selection of one candidate from internal geocoder results."""

from __future__ import annotations

import re

from tools.base import ToolClarification, ToolClarificationOption
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.geocoding.schemas import GeocodePlaceOutput, PlaceMatch, ToponymKind
from tools.geo.text_place_query import uses_cyrillic

_THOROUGHFARE_TYPE_TOKENS = frozenset(
    {
        "бульвар",
        "набережная",
        "переулок",
        "площадь",
        "проспект",
        "улица",
        "шоссе",
        "бул",
        "наб",
        "пер",
        "пл",
        "просп",
        "ул",
    }
)
_HOUSE_NUMBER_WITH_LETTER = re.compile(r"^(\d+)[^\W\d_]$", flags=re.UNICODE)
_HOUSE_NUMBER_TOKEN = re.compile(r"^\d+[^\W_]*$", flags=re.UNICODE)
_CYRILLIC_LOOKUP_FOLDS = str.maketrans({"ё": "е", "і": "и"})


def select_unique_place_match(
    result: GeocodePlaceOutput,
    *,
    query: str,
    city: str | None,
) -> PlaceMatch | None:
    """Return one defensible candidate, ``None``, or raise on ambiguity."""

    candidates = list(result.matches)
    if not candidates:
        return None

    if city is not None:
        city_tokens = _tokens(city)
        candidates = [
            match for match in candidates if city_tokens and city_tokens <= _tokens(match.address)
        ]
        if not candidates:
            return None

    normalized_query = _normalize(query)
    query_tokens = _tokens(query)

    # A city constraint proves only that a candidate is inside that city. It
    # does not prove that a locality fallback is the requested street, house,
    # or POI. Keep candidates tied to the actual query before accepting even a
    # single result.
    candidates = [
        match
        for match in candidates
        if _matches_query(
            match,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
        )
    ]
    if not candidates:
        return None

    exact_name_matches = [
        match for match in candidates if _normalize(match.name) == normalized_query
    ]
    if exact_name_matches:
        candidates = exact_name_matches
    if len(candidates) == 1:
        return candidates[0]

    exact_precision_matches = [match for match in candidates if match.precision == "exact"]
    if exact_precision_matches:
        candidates = exact_precision_matches
    if len(candidates) == 1:
        return candidates[0]

    raise AmbiguousPlaceError(
        query,
        clarification=ToolClarification(
            kind="select_anchor",
            question=(
                f"Какой объект по запросу {query!r} вы имели в виду?"
                if uses_cyrillic(query)
                else f"Which {query!r} location do you mean?"
            ),
            options=[
                ToolClarificationOption(
                    value=match.ref,
                    label=match.name[:200],
                    description=match.address[:500] or None,
                )
                for match in candidates[:3]
            ],
        ),
    )


def uses_added_house_suffix(match: PlaceMatch, *, query: str) -> bool:
    """Return whether the provider added a letter suffix to a requested house."""

    if match.kind is not ToponymKind.HOUSE:
        return False

    query_tokens = _tokens(query)
    candidate_tokens = _tokens(match.name) | _tokens(match.address)
    return (
        not query_tokens <= candidate_tokens
        and query_tokens <= candidate_tokens | _bare_house_number_aliases(candidate_tokens)
    )


def _matches_query(
    match: PlaceMatch,
    *,
    normalized_query: str,
    query_tokens: set[str],
) -> bool:
    """Reject broad geocoder fallbacks unrelated to the requested point."""

    if _normalize(match.name) == normalized_query:
        return True

    candidate_tokens = _tokens(match.name) | _tokens(match.address)
    if query_tokens and query_tokens <= candidate_tokens:
        return True

    # A house result may omit the generic thoroughfare type from its rendered
    # name ("Большая Покровская, 2" vs "Большая Покровская улица, 2").
    # Keep every proper-name and house-number token mandatory.
    significant_query_tokens = query_tokens - _THOROUGHFARE_TYPE_TOKENS
    comparable_house_tokens = candidate_tokens | _bare_house_number_aliases(candidate_tokens)
    if (
        match.kind is ToponymKind.HOUSE
        and significant_query_tokens
        and significant_query_tokens <= comparable_house_tokens
    ):
        return True

    # Trust an explicit exact house match even when the response language or
    # spelling normalization prevents useful token comparison.
    return (
        match.kind is ToponymKind.HOUSE
        and match.precision == "exact"
        and _house_numbers_are_compatible(query_tokens, candidate_tokens)
    )


def _bare_house_number_aliases(tokens: set[str]) -> set[str]:
    """Expose ``2`` as an alias for provider-normalized ``2А``, never vice versa."""

    aliases: set[str] = set()
    for token in tokens:
        match = _HOUSE_NUMBER_WITH_LETTER.fullmatch(token)
        if match is not None:
            aliases.add(match.group(1))
    return aliases


def _house_numbers_are_compatible(
    query_tokens: set[str],
    candidate_tokens: set[str],
) -> bool:
    query_numbers = {token for token in query_tokens if _HOUSE_NUMBER_TOKEN.fullmatch(token)}
    if not query_numbers:
        return True
    comparable_candidate_numbers = {
        token for token in candidate_tokens if _HOUSE_NUMBER_TOKEN.fullmatch(token)
    }
    comparable_candidate_numbers.update(_bare_house_number_aliases(comparable_candidate_numbers))
    return query_numbers <= comparable_candidate_numbers


def _tokens(value: str) -> set[str]:
    """Return comparable Unicode word/number tokens without punctuation."""

    return set(re.findall(r"[^\W_]+", _fold_lookup_spelling(value), flags=re.UNICODE))


def _normalize(value: str) -> str:
    """Normalize case and whitespace without weakening name equality."""

    return " ".join(_fold_lookup_spelling(value).split())


def _fold_lookup_spelling(value: str) -> str:
    """Fold only established Russian/Ukrainian lookup variants.

    Users often enter a Russian rendering of a Ukrainian street name, while
    the provider correctly returns the native spelling: ``Незалежности`` vs
    ``Незалежності``.  This narrow fold preserves the otherwise strict token
    matcher and does not transliterate between unrelated scripts.
    """

    return value.casefold().translate(_CYRILLIC_LOOKUP_FOLDS)
