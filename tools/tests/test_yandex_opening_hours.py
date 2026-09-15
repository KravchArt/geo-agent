"""Yandex structured opening-hours evaluation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tools.geo.places_search.yandex.opening_hours import current_opening_status
from tools.geo.places_search.yandex.schemas import YandexCompanyHours

_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_MOSCOW = {"lat": 55.75, "lon": 37.62}


def _hours(*availabilities: dict[str, object]) -> YandexCompanyHours:
    return YandexCompanyHours.model_validate({"Availabilities": list(availabilities)})


@pytest.mark.parametrize(
    ("availabilities", "expected"),
    [
        (
            ({"Everyday": True, "Intervals": [{"from": "14:00:00", "to": "16:00:00"}]},),
            True,
        ),
        (
            ({"Everyday": True, "Intervals": [{"from": "16:00:00", "to": "18:00:00"}]},),
            False,
        ),
        (
            ({"Weekend": True, "TwentyFourHours": True},),
            True,
        ),
        (
            ({"Weekdays": True, "TwentyFourHours": True},),
            False,
        ),
        (
            (
                {
                    "Saturday": True,
                    "Intervals": [
                        {"from": "09:00:00", "to": "12:00:00"},
                        {"from": "14:00:00", "to": "18:00:00"},
                    ],
                },
            ),
            True,
        ),
    ],
)
def test_evaluates_documented_availabilities_in_local_time(
    availabilities: tuple[dict[str, object], ...],
    expected: bool,
) -> None:
    assert (
        current_opening_status(
            _hours(*availabilities),
            **_MOSCOW,
            now=_NOW,
        )
        is expected
    )


def test_evaluates_interval_crossing_midnight() -> None:
    assert (
        current_opening_status(
            _hours(
                {
                    "Friday": True,
                    "Intervals": [{"from": "22:00:00", "to": "03:00:00"}],
                },
                {
                    "Saturday": True,
                    "Intervals": [{"from": "10:00:00", "to": "18:00:00"}],
                },
            ),
            **_MOSCOW,
            now=datetime(2026, 7, 31, 22, 30, tzinfo=UTC),
        )
        is True
    )


@pytest.mark.parametrize(
    "hours",
    [
        None,
        _hours(),
        _hours({"TwentyFourHours": True}),
        _hours({"Everyday": True}),
        _hours(
            {
                "Everyday": True,
                "TwentyFourHours": True,
                "Intervals": [{"from": "10:00:00", "to": "18:00:00"}],
            }
        ),
        _hours(
            {
                "Everyday": True,
                "Intervals": [{"from": "10:00:30", "to": "18:00:00"}],
            }
        ),
        _hours(
            {
                "Everyday": True,
                "Intervals": [{"from": "10:00:00", "to": "10:00:00"}],
            }
        ),
    ],
)
def test_incomplete_or_contradictory_schedule_is_unknown(
    hours: YandexCompanyHours | None,
) -> None:
    assert current_opening_status(hours, **_MOSCOW, now=_NOW) is None
