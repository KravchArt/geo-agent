"""Compare locality labels returned by places-search providers."""

from __future__ import annotations

import re

_LOCALITY_QUALIFIERS = frozenset(
    {
        "city",
        "district",
        "municipality",
        "of",
        "urban",
        "г",
        "город",
        "городской",
        "муниципальный",
        "округ",
        "район",
    }
)


def _significant_tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(
        r"[^\W_]+",
        value.casefold().replace("ё", "е"),
        flags=re.UNICODE,
    )
    return tuple(token for token in tokens if token not in _LOCALITY_QUALIFIERS)


def same_locality(first: str, second: str) -> bool:
    """Compare provider locality labels while ignoring administrative wrappers."""

    first_tokens = frozenset(_significant_tokens(first))
    second_tokens = frozenset(_significant_tokens(second))
    return bool(first_tokens) and first_tokens == second_tokens


def compatible_locality(first: str, second: str) -> bool:
    """Accept an exact locality or a defensible shortened provider label.

    Search providers sometimes shorten an official name (``Frankfurt am Main``
    to ``Frankfurt``). A one-token abbreviation is accepted only when it is the
    leading identity token, so a generic suffix such as ``Main`` cannot match.
    Geographic bounds must still be checked separately by the caller.
    """

    first_tokens = _significant_tokens(first)
    second_tokens = _significant_tokens(second)
    first_set = frozenset(first_tokens)
    second_set = frozenset(second_tokens)
    if not first_set or not second_set:
        return False
    if first_set == second_set:
        return True

    shorter_tokens, shorter_set, longer_tokens, longer_set = (
        (first_tokens, first_set, second_tokens, second_set)
        if len(first_set) < len(second_set)
        else (second_tokens, second_set, first_tokens, first_set)
    )
    if not shorter_set < longer_set:
        return False
    return len(shorter_set) >= 2 or shorter_tokens[0] == longer_tokens[0]
