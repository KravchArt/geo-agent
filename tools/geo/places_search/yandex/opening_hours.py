"""Normalize documented Yandex availability data for current-state checks."""

from __future__ import annotations

from datetime import datetime, time

from tools.geo.opening_hours import is_open_at
from tools.geo.places_search.yandex.schemas import YandexCompanyHours

_DAY_NAMES = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
_FULL_DAY = "00:00-24:00"


def current_opening_status(
    hours: YandexCompanyHours | None,
    *,
    lat: float,
    lon: float,
    now: datetime,
) -> bool | None:
    """Evaluate documented weekly Yandex hours in the organisation's time zone."""

    expression = _opening_hours_expression(hours)
    if expression is None:
        return None
    return is_open_at(expression, lat=lat, lon=lon, now=now)


def _opening_hours_expression(hours: YandexCompanyHours | None) -> str | None:
    """Convert Yandex day/interval records into an equivalent OSM expression.

    The shared evaluator supplies coordinate-derived time zones. Any incomplete
    or contradictory availability makes the whole current status unknown: the
    ``open_now`` contract must never turn partial data into a false positive.
    """

    if hours is None or not hours.availabilities:
        return None

    intervals_by_day: dict[int, list[str]] = {}

    for availability in hours.availabilities:
        days = availability.days
        if not days:
            return None
        split_intervals: tuple[tuple[int, str], ...]
        if availability.twenty_four_hours:
            if availability.intervals:
                return None
            split_intervals = ((0, _FULL_DAY),)
        else:
            if not availability.intervals:
                return None
            normalized_intervals: list[tuple[int, str]] = []
            for interval in availability.intervals:
                split = _split_interval(interval.from_time, interval.to_time)
                if split is None:
                    return None
                normalized_intervals.extend(split)
            split_intervals = tuple(normalized_intervals)

        for day in days:
            for day_offset, scheduled_interval in split_intervals:
                target_day = (day + day_offset) % len(_DAY_NAMES)
                day_intervals = intervals_by_day.setdefault(target_day, [])
                if _FULL_DAY in day_intervals:
                    continue
                if scheduled_interval == _FULL_DAY:
                    day_intervals.clear()
                if scheduled_interval not in day_intervals:
                    day_intervals.append(scheduled_interval)

    rules = [
        f"{_DAY_NAMES[day]} {','.join(intervals)}"
        for day, intervals in sorted(intervals_by_day.items())
    ]
    return "; ".join(rules) or None


def _split_interval(
    from_time: time,
    to_time: time,
) -> tuple[tuple[int, str], ...] | None:
    # The target opening-hours grammar has minute precision. Reject uncommon
    # non-zero seconds rather than silently widening an interval.
    if from_time.second or from_time.microsecond or to_time.second or to_time.microsecond:
        return None
    if from_time == to_time:
        return None

    start = from_time.strftime("%H:%M")
    if to_time == time.min:
        return ((0, f"{start}-24:00"),)

    end = to_time.strftime("%H:%M")
    if from_time < to_time:
        return ((0, f"{start}-{end}"),)

    return (
        (0, f"{start}-24:00"),
        (1, f"00:00-{end}"),
    )
