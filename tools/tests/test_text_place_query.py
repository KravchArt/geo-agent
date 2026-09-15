"""Tests for provider-facing textual place query normalization."""

from __future__ import annotations

import pytest

from tools.geo.text_place_query import (
    compose_scoped_place_query,
    same_locality_anchor,
    tomtom_response_language,
    twogis_response_locale,
    yandex_response_locale,
)


@pytest.mark.parametrize(
    ("query", "city"),
    [
        ("Frankfurt am Main", "Frankfurt am Main"),
        ("Frankfurt am Main, Germany", "Frankfurt am Main"),
        ("город Frankfurt am Main", "Frankfurt am Main, Germany"),
    ],
)
def test_same_locality_anchor_accepts_optional_qualifiers(query: str, city: str) -> None:
    assert same_locality_anchor(query, city)


def test_same_locality_anchor_keeps_named_object_distinct() -> None:
    assert not same_locality_anchor("Hauptbahnhof, Frankfurt am Main", "Frankfurt am Main")


@pytest.mark.parametrize(
    ("query", "city", "city_first", "expected"),
    [
        ("Frankfurt am Main", "Frankfurt am Main", True, "Frankfurt am Main"),
        (
            "Frankfurt am Main, Germany",
            "Frankfurt am Main",
            False,
            "Frankfurt am Main, Germany",
        ),
        (
            "Hauptbahnhof, Frankfurt am Main",
            "Frankfurt am Main",
            False,
            "Hauptbahnhof, Frankfurt am Main",
        ),
        (
            "Hauptbahnhof",
            "Frankfurt am Main",
            False,
            "Hauptbahnhof, Frankfurt am Main",
        ),
        (
            "Hauptbahnhof",
            "Frankfurt am Main",
            True,
            "Frankfurt am Main, Hauptbahnhof",
        ),
    ],
)
def test_compose_scoped_place_query_adds_city_exactly_once(
    query: str,
    city: str,
    city_first: bool,
    expected: str,
) -> None:
    assert (
        compose_scoped_place_query(
            query=query,
            city=city,
            city_first=city_first,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Москва, Красная площадь", "ru_RU"),
        ("Frankfurt am Main, Germany", "en_US"),
    ],
)
def test_yandex_response_locale_follows_query_script(query: str, expected: str) -> None:
    assert yandex_response_locale(query) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("кофейни", "Франкфурт-на-Майне"), "ru-RU"),
        (("coffee shops", "Frankfurt am Main"), "en-US"),
    ],
)
def test_tomtom_response_language_follows_query_script(
    values: tuple[str, ...],
    expected: str,
) -> None:
    assert tomtom_response_language(*values) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("рестораны", "Москва"), "ru_RU"),
        (("restaurants", "Moscow"), "en_RU"),
        (("restaurants", "Москва"), "ru_RU"),
    ],
)
def test_twogis_response_locale_follows_query_script(
    values: tuple[str, ...],
    expected: str,
) -> None:
    assert twogis_response_locale(*values) == expected


@pytest.mark.parametrize(
    ("country_code", "expected"),
    [
        ("am", "ru_AM"),
        ("AZ", "ru_AZ"),
        ("by", "ru_BY"),
        ("cn", "ru_CN"),
        ("ge", "ru_GE"),
        ("kg", "ru_KG"),
        ("kz", "ru_KZ"),
        ("ru", "ru_RU"),
        ("tj", "ru_TJ"),
        (" UZ ", "ru_UZ"),
    ],
)
def test_twogis_response_locale_uses_russian_catalog_for_covered_country(
    country_code: str,
    expected: str,
) -> None:
    assert twogis_response_locale("рестораны", country_code=country_code) == expected


@pytest.mark.parametrize(
    ("query", "country_code", "expected"),
    [
        ("restaurants", "ru", "en_RU"),
        ("рестораны", "ru", "ru_RU"),
        ("restaurants", "cn", "en_CN"),
        ("рестораны", "cn", "ru_CN"),
        # 2GIS has no English locale for Uzbekistan or Kazakhstan. Keep the
        # covered country catalogue with its Russian response locale.
        ("restaurants", "uz", "ru_UZ"),
        ("restaurants", "kz", "ru_KZ"),
        # The UAE has no Russian locale, so a Cyrillic query uses its English
        # catalogue rather than incorrectly switching to ru_RU.
        ("рестораны", "ae", "en_AE"),
    ],
)
def test_twogis_response_locale_prefers_query_language_within_country_catalogue(
    query: str,
    country_code: str,
    expected: str,
) -> None:
    assert twogis_response_locale(query, country_code=country_code) == expected
