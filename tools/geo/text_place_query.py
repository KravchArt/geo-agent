"""Normalize textual place scopes before sending them to geo providers."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LOCALITY_QUALIFIERS = frozenset(
    {
        "city",
        "of",
        "г",
        "город",
    }
)
_TWOGIS_LOCALES_BY_COUNTRY = {
    # Supported 2GIS locales, indexed by ISO 3166-1 alpha-2 country code and
    # response language. Prefer the caller's language; when it is unavailable,
    # retain the covered country's catalog rather than silently switching to
    # the Russian catalog.
    "ae": {"ar": "ar_AE", "en": "en_AE"},
    "am": {"en": "en_AM", "hy": "hy_AM", "ru": "ru_AM"},
    "az": {"az": "az_AZ", "ru": "ru_AZ"},
    "bh": {"ar": "ar_BH", "en": "en_BH"},
    "by": {"ru": "ru_BY"},
    "cl": {"es": "es_CL"},
    "cn": {"en": "en_CN", "ru": "ru_CN", "zh": "zh_CN"},
    "cy": {"en": "en_CY"},
    "cz": {"cs": "cs_CZ"},
    "eg": {"ar": "ar_EG", "en": "en_EG"},
    "ge": {"ka": "ka_GE", "ru": "ru_GE"},
    "iq": {"ar": "ar_IQ", "en": "en_IQ"},
    "it": {"it": "it_IT"},
    "kg": {"ky": "ky_KG", "ru": "ru_KG"},
    "kz": {"kk": "kk_KZ", "ru": "ru_KZ"},
    "kw": {"ar": "ar_KW", "en": "en_KW"},
    "ma": {"ar": "ar_MA", "en": "en_MA"},
    "mn": {"en": "en_MN", "mn": "mn_MN"},
    "om": {"ar": "ar_OM", "en": "en_OM"},
    "qa": {"ar": "ar_QA", "en": "en_QA"},
    "ru": {
        "ar": "ar_RU",
        "cs": "cs_RU",
        "en": "en_RU",
        "es": "es_RU",
        "it": "it_RU",
        "ru": "ru_RU",
    },
    "sa": {"ar": "ar_SA", "en": "en_SA"},
    "tj": {"ru": "ru_TJ", "tg": "tg_TJ"},
    "uz": {"ru": "ru_UZ", "uz": "uz_UZ"},
}


def same_locality_anchor(query: str, city: str) -> bool:
    """Return whether a nearby anchor is the locality supplied as its context.

    A comma-delimited suffix may qualify either value with a country or region,
    so only the primary locality label participates in this comparison. Named
    objects remain distinct: ``Hauptbahnhof, Frankfurt am Main`` is not the
    locality ``Frankfurt am Main``.
    """

    query_tokens = _significant_tokens(_primary_label(query))
    city_tokens = _significant_tokens(_primary_label(city))
    return bool(query_tokens) and query_tokens == city_tokens


def compose_scoped_place_query(
    *,
    query: str,
    city: str | None,
    city_first: bool,
) -> str:
    """Add a city context exactly once to one provider search string."""

    cleaned_query = " ".join(query.split())
    if city is None:
        return cleaned_query

    cleaned_city = " ".join(city.split())
    if same_locality_anchor(cleaned_query, cleaned_city) or any(
        same_locality_anchor(segment, cleaned_city) for segment in cleaned_query.split(",")[1:]
    ):
        return cleaned_query

    if city_first:
        return f"{cleaned_city}, {cleaned_query}"
    return f"{cleaned_query}, {cleaned_city}"


def yandex_response_locale(query: str) -> str:
    """Choose one supported Yandex response locale matching the query script."""

    return "ru_RU" if _CYRILLIC_RE.search(query) is not None else "en_US"


def tomtom_response_language(*values: str) -> str:
    """Choose Russian or English TomTom output for a Russian or English query."""

    return "ru-RU" if _CYRILLIC_RE.search(" ".join(values)) is not None else "en-US"


def uses_cyrillic(value: str) -> bool:
    """Return whether text contains at least one Cyrillic character."""

    return _CYRILLIC_RE.search(value) is not None


def twogis_response_locale(
    *values: str,
    country_code: str | None = None,
) -> str:
    """Choose a supported 2GIS response locale for the query and covered country."""

    language = "ru" if _CYRILLIC_RE.search(" ".join(values)) is not None else "en"
    if country_code is not None:
        locales = _TWOGIS_LOCALES_BY_COUNTRY.get(country_code.strip().casefold())
        if locales is not None:
            if locale := locales.get(language):
                return locale
            # Keep the resolved country catalogue if 2GIS has no locale for
            # the query script. Russian is preferred for Russian-language
            # users; otherwise use English, then the provider's first choice.
            return locales.get("ru") or locales.get("en") or next(iter(locales.values()))

    return "ru_RU" if language == "ru" else "en_RU"


def _primary_label(value: str) -> str:
    return value.split(",", maxsplit=1)[0]


def _significant_tokens(value: str) -> frozenset[str]:
    tokens = _WORD_RE.findall(value.casefold().replace("ё", "е"))
    return frozenset(token for token in tokens if token not in _LOCALITY_QUALIFIERS)
