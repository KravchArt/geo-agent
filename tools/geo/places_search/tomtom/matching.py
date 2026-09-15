"""Shared matching rules for TomTom named-POI results."""

from __future__ import annotations

import re

from tools.geo.places_search.tomtom.schemas import TomTomSearchResult

_NON_WORDS = re.compile(r"[^\w]+", re.UNICODE)
_AUXILIARY_CLASSIFICATIONS = frozenset(
    {
        "OPEN_PARKING_AREA",
        "PARKING_GARAGE",
        "PARKING_LOT",
    }
)
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "ә": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "ғ": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "қ": "q",
        "л": "l",
        "м": "m",
        "н": "n",
        "ң": "ng",
        "о": "o",
        "ө": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ұ": "u",
        "ү": "u",
        "ф": "f",
        "х": "kh",
        "һ": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "є": "ie",
        "і": "i",
        "ї": "i",
    }
)
_ADDRESS_TYPE_TOKENS = frozenset(
    {
        "avenue",
        "ave",
        "boulevard",
        "blvd",
        "drive",
        "dr",
        "embankment",
        "highway",
        "hwy",
        "lane",
        "ln",
        "pereulok",
        "pere",
        "prospekt",
        "road",
        "rd",
        "street",
        "st",
        "ulitsa",
        "ul",
    }
)


def clean_optional_text(value: str | None) -> str | None:
    """Collapse whitespace and convert blank provider text to ``None``."""

    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def normalized_tokens(value: str) -> tuple[str, ...]:
    """Return stable Unicode tokens for conservative name/address comparison."""

    normalized = value.casefold().replace("ё", "е")
    return tuple(token for token in _NON_WORDS.sub(" ", normalized).split() if token)


def normalized_address_tokens(value: str) -> frozenset[str]:
    """Return strict comparable address tokens across Cyrillic and Latin text.

    This is deliberately narrower than fuzzy address matching: it only
    transliterates letters and removes generic thoroughfare-type words. Every
    remaining proper-name token and building number must still match exactly.
    """

    transliterated = value.casefold().translate(_CYRILLIC_TO_LATIN)
    return frozenset(
        token for token in normalized_tokens(transliterated) if token not in _ADDRESS_TYPE_TOKENS
    )


def name_match_score(
    name: str,
    query: str,
    *,
    locality: str | None = None,
) -> tuple[int, int] | None:
    """Accept exact names and compact qualifiers, not incidental mentions."""

    name_tokens = normalized_tokens(name)
    query_tokens = normalized_tokens(query)
    if not name_tokens or not query_tokens:
        return None
    if name_tokens == query_tokens:
        return (0, 0)
    if set(query_tokens).issubset(name_tokens):
        extra_words = len(name_tokens) - len(query_tokens)
        if extra_words <= 1:
            return (1, extra_words)
        # A locality adjective such as "Санкт-Петербургский" can legitimately
        # qualify a POI name. Arbitrary multi-word prefixes must not broaden a
        # named-POI match.
        if extra_words == 2 and locality is not None:
            locality_tokens = normalized_tokens(locality)
            for index in range(len(name_tokens) - len(query_tokens) + 1):
                if name_tokens[index : index + len(query_tokens)] != query_tokens:
                    continue
                prefix = name_tokens[:index]
                suffix = name_tokens[index + len(query_tokens) :]
                if prefix and suffix:
                    continue
                qualifier = prefix or suffix
                if _matches_locality_qualifier(qualifier, locality_tokens):
                    return (1, extra_words)

    # Providers sometimes localize one proper name with a one-letter spelling
    # variant (for example, "Медеу" versus "Медео каток"). Accept that only
    # for a single substantial query token and at most one descriptive word;
    # broader fuzzy matching would turn incidental organisations into anchors.
    if len(query_tokens) == 1 and len(query_tokens[0]) >= 4 and len(name_tokens) <= 2:
        query_token = query_tokens[0]
        if any(_is_single_edit_variant(query_token, token) for token in name_tokens):
            return (2, len(name_tokens))
    return None


def is_auxiliary(result: TomTomSearchResult) -> bool:
    """Identify parking records that TomTom can name after a landmark."""

    return bool(result.poi.classification_codes & _AUXILIARY_CLASSIFICATIONS)


def named_result_identity(
    result: TomTomSearchResult,
    address: str,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Build a conservative identity for duplicate named-POI cards."""

    return (
        normalized_tokens(result.poi.name),
        frozenset(normalized_tokens(address)),
    )


def _matches_locality_qualifier(
    qualifier: tuple[str, ...],
    locality: tuple[str, ...],
) -> bool:
    return len(qualifier) == len(locality) and all(
        candidate == city_word or candidate.startswith(city_word)
        for candidate, city_word in zip(qualifier, locality, strict=True)
    )


def _is_single_edit_variant(left: str, right: str) -> bool:
    """Return whether two substantial tokens differ by exactly one edit."""

    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    edits = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        edits += 1
        long_index += 1
        if edits > 1:
            return False
    return True
