"""Tests for provider locality-label comparison."""

from __future__ import annotations

import pytest

from tools.geo.places_search.locality import compatible_locality, same_locality


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Frankfurt am Main", "Frankfurt"),
        ("New York City", "New York"),
        ("город Москва", "Москва"),
    ],
)
def test_compatible_locality_accepts_exact_and_shortened_labels(
    first: str,
    second: str,
) -> None:
    assert compatible_locality(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Frankfurt am Main", "Eschborn"),
        ("Frankfurt am Main", "Offenbach am Main"),
        ("Frankfurt am Main", "Main"),
    ],
)
def test_compatible_locality_rejects_neighbours_and_generic_suffixes(
    first: str,
    second: str,
) -> None:
    assert not compatible_locality(first, second)


def test_same_locality_remains_strict_for_other_providers() -> None:
    assert not same_locality("Frankfurt am Main", "Frankfurt")
