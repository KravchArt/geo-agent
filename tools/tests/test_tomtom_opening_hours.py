"""Tests for model-facing TomTom opening-hours normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tools.geo.places_search.tomtom.opening_hours import summarize_opening_hours
from tools.geo.places_search.tomtom.schemas import (
    TomTomOpeningHours,
    TomTomTimeZone,
)

_NOW = datetime(2026, 7, 30, 9, tzinfo=UTC)
_MOSCOW = TomTomTimeZone.model_validate({"ianaId": "Europe/Moscow"})


def _hours(*ranges: tuple[str, int, int, str, int, int]) -> TomTomOpeningHours:
    return TomTomOpeningHours.model_validate(
        {
            "mode": "nextSevenDays",
            "timeRanges": [
                {
                    "startTime": {
                        "date": start_date,
                        "hour": start_hour,
                        "minute": start_minute,
                    },
                    "endTime": {
                        "date": end_date,
                        "hour": end_hour,
                        "minute": end_minute,
                    },
                }
                for (
                    start_date,
                    start_hour,
                    start_minute,
                    end_date,
                    end_hour,
                    end_minute,
                ) in ranges
            ],
        }
    )


def test_missing_hours_remain_unknown() -> None:
    summary = summarize_opening_hours(None, _MOSCOW, now=_NOW)

    assert summary.text is None
    assert summary.open_24h is None
    assert summary.is_open_now is None


def test_identical_daily_hours_are_grouped_into_one_date_span() -> None:
    ranges = tuple(
        (f"2026-{month:02d}-{day:02d}", 8, 0, f"2026-{month:02d}-{day:02d}", 22, 0)
        for month, day in (
            (7, 30),
            (7, 31),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
        )
    )

    summary = summarize_opening_hours(_hours(*ranges), _MOSCOW, now=_NOW)

    assert summary.text == "Next 7 days: Jul 30–Aug 5 08:00–22:00"
    assert summary.open_24h is False
    assert summary.is_open_now is True


def test_missing_day_in_seven_day_response_is_rendered_as_closed() -> None:
    ranges = (
        ("2026-07-30", 8, 0, "2026-07-30", 22, 0),
        ("2026-07-31", 8, 0, "2026-07-31", 22, 0),
        ("2026-08-02", 8, 0, "2026-08-02", 22, 0),
        ("2026-08-03", 8, 0, "2026-08-03", 22, 0),
        ("2026-08-04", 8, 0, "2026-08-04", 22, 0),
        ("2026-08-05", 8, 0, "2026-08-05", 22, 0),
    )

    summary = summarize_opening_hours(_hours(*ranges), _MOSCOW, now=_NOW)

    assert summary.text == ("Next 7 days: Jul 30–31 08:00–22:00; Aug 1 closed; Aug 2–5 08:00–22:00")
    assert summary.open_24h is False
    assert summary.is_open_now is True


def test_split_shifts_are_merged_and_grouped() -> None:
    ranges = tuple(
        item
        for month, day in (
            (7, 30),
            (7, 31),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
        )
        for item in (
            (f"2026-{month:02d}-{day:02d}", 9, 0, f"2026-{month:02d}-{day:02d}", 13, 0),
            (f"2026-{month:02d}-{day:02d}", 14, 0, f"2026-{month:02d}-{day:02d}", 18, 0),
        )
    )

    summary = summarize_opening_hours(_hours(*ranges), _MOSCOW, now=_NOW)

    assert summary.text == ("Next 7 days: Jul 30–Aug 5 09:00–13:00, 14:00–18:00")
    assert summary.is_open_now is True


def test_ranges_crossing_midnight_are_split_between_calendar_days() -> None:
    summary = summarize_opening_hours(
        _hours(
            ("2026-07-30", 20, 0, "2026-07-31", 2, 0),
            ("2026-07-31", 20, 0, "2026-08-01", 2, 0),
        ),
        _MOSCOW,
        now=_NOW,
    )

    assert summary.text == (
        "Next 7 days: Jul 30 20:00–24:00; "
        "Jul 31 00:00–02:00, 20:00–24:00; "
        "Aug 1 00:00–02:00; Aug 2–5 closed"
    )
    assert summary.open_24h is False
    assert summary.is_open_now is False


def test_continuous_full_window_is_the_only_confirmed_24h_schedule() -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-30", 0, 0, "2026-08-06", 0, 0)),
        _MOSCOW,
        now=_NOW,
    )

    assert summary.text == "Open 24 hours for the next 7 days"
    assert summary.open_24h is True
    assert summary.is_open_now is True


def test_empty_complete_window_means_closed_for_all_seven_days() -> None:
    summary = summarize_opening_hours(_hours(), _MOSCOW, now=_NOW)

    assert summary.text == "Next 7 days: Jul 30–Aug 5 closed"
    assert summary.open_24h is False
    assert summary.is_open_now is False


def test_missing_timezone_preserves_ranges_but_does_not_infer_closed_days() -> None:
    summary = summarize_opening_hours(
        _hours(
            ("2026-07-30", 8, 0, "2026-07-30", 22, 0),
            ("2026-07-31", 8, 0, "2026-07-31", 22, 0),
        ),
        None,
        now=_NOW,
    )

    assert summary.text == "Published hours: Jul 30–31 08:00–22:00"
    assert summary.open_24h is False
    assert summary.is_open_now is None


def test_missing_timezone_cannot_prove_24h() -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-30", 0, 0, "2026-08-06", 0, 0)),
        None,
        now=_NOW,
    )

    assert summary.text == "Published hours: Jul 30–Aug 5 open 24 hours"
    assert summary.open_24h is None
    assert summary.is_open_now is None


def test_stale_window_preserves_ranges_without_inventing_closed_days() -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-29", 8, 0, "2026-07-29", 22, 0)),
        _MOSCOW,
        now=_NOW,
    )

    assert summary.text == "Published hours: Jul 29 08:00–22:00"
    assert summary.open_24h is False
    assert summary.is_open_now is None


def test_malformed_range_does_not_look_like_a_closed_week() -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-30", 22, 0, "2026-07-30", 8, 0)),
        _MOSCOW,
        now=_NOW,
    )

    assert summary.text is None
    assert summary.open_24h is None
    assert summary.is_open_now is None


def test_poi_timezone_selects_the_correct_local_start_date() -> None:
    berlin = TomTomTimeZone.model_validate({"ianaId": "Europe/Berlin"})
    after_local_midnight = datetime(2026, 3, 28, 23, 30, tzinfo=UTC)

    summary = summarize_opening_hours(
        _hours(("2026-03-29", 10, 0, "2026-03-29", 18, 0)),
        berlin,
        now=after_local_midnight,
    )

    assert summary.text == ("Next 7 days: Mar 29 10:00–18:00; Mar 30–Apr 4 closed")
    assert summary.is_open_now is False


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 7, 30, 4, 59, tzinfo=UTC), False),
        (datetime(2026, 7, 30, 5, 0, tzinfo=UTC), True),
        (datetime(2026, 7, 30, 18, 59, tzinfo=UTC), True),
        (datetime(2026, 7, 30, 19, 0, tzinfo=UTC), False),
    ],
)
def test_current_status_uses_inclusive_open_and_exclusive_close(
    now: datetime,
    expected: bool,
) -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-30", 8, 0, "2026-07-30", 22, 0)),
        _MOSCOW,
        now=now,
    )

    assert summary.is_open_now is expected


def test_current_status_supports_an_interval_crossing_midnight() -> None:
    summary = summarize_opening_hours(
        _hours(("2026-07-30", 20, 0, "2026-07-31", 2, 0)),
        _MOSCOW,
        now=datetime(2026, 7, 30, 22, 0, tzinfo=UTC),
    )

    assert summary.is_open_now is True


def test_reference_time_must_include_a_timezone() -> None:
    with pytest.raises(
        ValueError,
        match="reference time must be timezone-aware",
    ):
        summarize_opening_hours(
            _hours(("2026-07-30", 8, 0, "2026-07-30", 22, 0)),
            _MOSCOW,
            now=datetime(2026, 7, 30, 12),
        )
