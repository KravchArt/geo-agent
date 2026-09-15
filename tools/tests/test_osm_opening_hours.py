"""OSM opening-hours evaluation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tools.geo.opening_hours import is_open_at

_MOSCOW = {"lat": 55.75, "lon": 37.62}


@pytest.mark.parametrize(
    ("hours_text", "expected"),
    [
        ("24/7", True),
        ("Sa 14:00-16:00", True),
        ("Sa 16:00-18:00", False),
        ("unknown", None),
        ("not a valid schedule", None),
        (None, None),
    ],
)
def test_evaluates_schedule_in_timezone_inferred_from_coordinates(
    hours_text: str | None,
    expected: bool | None,
) -> None:
    # 12:00 UTC is 15:00 in Moscow on this date.
    assert (
        is_open_at(
            hours_text,
            **_MOSCOW,
            now=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        )
        is expected
    )


def test_evaluates_interval_crossing_midnight() -> None:
    # Friday 22:30 UTC is Saturday 01:30 in Moscow.
    assert (
        is_open_at(
            "Fr 22:00-03:00",
            **_MOSCOW,
            now=datetime(2026, 7, 31, 22, 30, tzinfo=UTC),
        )
        is True
    )


def test_rejects_naive_reference_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        is_open_at(
            "24/7",
            **_MOSCOW,
            now=datetime(2026, 8, 1, 12, 0),
        )
