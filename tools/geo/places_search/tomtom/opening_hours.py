"""Normalize TomTom's dated seven-day opening-hour ranges for tool output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools.geo.places_search.tomtom.schemas import (
    TomTomOpeningHours,
    TomTomTimeZone,
)

_DAYS_IN_TOMTOM_WINDOW = 7
_MINUTES_PER_DAY = 24 * 60
_MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_MinuteInterval = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TomTomOpeningHoursSummary:
    """Stable model-facing representation of TomTom's transient date ranges."""

    text: str | None
    open_24h: bool | None
    is_open_now: bool | None


def summarize_opening_hours(
    opening_hours: TomTomOpeningHours | None,
    time_zone: TomTomTimeZone | None,
    *,
    now: datetime | None = None,
) -> TomTomOpeningHoursSummary:
    """Build a compact schedule without converting dated data into weekly rules.

    ``nextSevenDays`` may include holiday exceptions, so the result deliberately
    keeps calendar dates instead of claiming that the same weekday schedule
    repeats every week.
    """

    if opening_hours is None:
        return TomTomOpeningHoursSummary(text=None, open_24h=None, is_open_now=None)

    if any(item.end_time.value <= item.start_time.value for item in opening_hours.time_ranges):
        # A malformed interval makes the provider's schedule untrustworthy. It
        # must not accidentally turn into seven apparently "closed" days.
        return TomTomOpeningHoursSummary(text=None, open_24h=None, is_open_now=None)

    reference_time = now or datetime.now(UTC)
    if reference_time.tzinfo is None:
        raise ValueError("opening-hours reference time must be timezone-aware")

    zone = _load_zone(time_zone)
    if opening_hours.mode != "nextSevenDays" or zone is None:
        return _summarize_published_ranges(opening_hours)

    first_day = reference_time.astimezone(zone).date()
    dates = tuple(first_day + timedelta(days=offset) for offset in range(_DAYS_IN_TOMTOM_WINDOW))
    window_start = datetime.combine(dates[0], time.min)
    window_end = datetime.combine(dates[-1] + timedelta(days=1), time.min)
    has_ranges_in_window = any(
        item.start_time.value < window_end and item.end_time.value > window_start
        for item in opening_hours.time_ranges
    )
    if opening_hours.time_ranges and not has_ranges_in_window:
        # A range crossing either window boundary is legitimate and gets
        # clipped below. If every published range is outside the window, the
        # payload is stale or uses an inconsistent upstream clock: preserve its
        # text, but do not infer closed days or a current status.
        return _summarize_published_ranges(opening_hours)

    intervals_by_date = _split_ranges_by_date(
        opening_hours,
        first_day=dates[0],
        last_day=dates[-1],
    )
    daily_intervals = tuple(_merge_intervals(intervals_by_date.get(day, ())) for day in dates)
    local_now = reference_time.astimezone(zone).replace(tzinfo=None)
    is_open_now = any(
        item.start_time.value <= local_now < item.end_time.value
        for item in opening_hours.time_ranges
    )

    if all(intervals == ((0, _MINUTES_PER_DAY),) for intervals in daily_intervals):
        return TomTomOpeningHoursSummary(
            text="Open 24 hours for the next 7 days",
            open_24h=True,
            is_open_now=is_open_now,
        )

    return TomTomOpeningHoursSummary(
        text=_format_schedule("Next 7 days", dates, daily_intervals),
        # The response explicitly covers all seven local calendar days. An
        # omitted day is therefore closed, not an unknown recurring schedule.
        open_24h=False,
        is_open_now=is_open_now,
    )


def _load_zone(time_zone: TomTomTimeZone | None) -> ZoneInfo | None:
    if time_zone is None:
        return None
    try:
        return ZoneInfo(time_zone.iana_id)
    except (ZoneInfoNotFoundError, ValueError):
        # Provider data remains useful even if a new or malformed time-zone ID
        # is not present in the host's IANA database.
        return None


def _summarize_published_ranges(
    opening_hours: TomTomOpeningHours,
) -> TomTomOpeningHoursSummary:
    """Keep useful hours when their complete seven-day window is unprovable."""

    intervals_by_date = _split_ranges_by_date(opening_hours)
    if not intervals_by_date:
        return TomTomOpeningHoursSummary(text=None, open_24h=None, is_open_now=None)

    dates = tuple(sorted(intervals_by_date))
    daily_intervals = tuple(_merge_intervals(intervals_by_date[day]) for day in dates)
    has_partial_day = any(intervals != ((0, _MINUTES_PER_DAY),) for intervals in daily_intervals)
    return TomTomOpeningHoursSummary(
        text=_format_schedule("Published hours", dates, daily_intervals),
        # Partial data can disprove 24/7, but it cannot prove it.
        open_24h=False if has_partial_day else None,
        is_open_now=None,
    )


def _split_ranges_by_date(
    opening_hours: TomTomOpeningHours,
    *,
    first_day: date | None = None,
    last_day: date | None = None,
) -> dict[date, list[_MinuteInterval]]:
    result: dict[date, list[_MinuteInterval]] = {}
    window_start = datetime.combine(first_day, time.min) if first_day is not None else None
    window_end = (
        datetime.combine(last_day + timedelta(days=1), time.min) if last_day is not None else None
    )

    for item in opening_hours.time_ranges:
        start = item.start_time.value
        end = item.end_time.value
        # The public summarizer rejects malformed schedules before reaching this
        # helper; keep the guard so direct internal reuse still terminates safely.
        if end <= start:
            continue
        if window_start is not None:
            start = max(start, window_start)
        if window_end is not None:
            end = min(end, window_end)
        if end <= start:
            continue

        current_day = start.date()
        while datetime.combine(current_day, time.min) < end:
            day_start = datetime.combine(current_day, time.min)
            day_end = day_start + timedelta(days=1)
            segment_start = max(start, day_start)
            segment_end = min(end, day_end)
            if segment_start < segment_end:
                result.setdefault(current_day, []).append(
                    (
                        _minute_of_day(segment_start),
                        (
                            _MINUTES_PER_DAY
                            if segment_end == day_end
                            else _minute_of_day(segment_end)
                        ),
                    )
                )
            current_day += timedelta(days=1)

    return result


def _minute_of_day(value: datetime) -> int:
    return value.hour * 60 + value.minute


def _merge_intervals(
    intervals: list[_MinuteInterval] | tuple[_MinuteInterval, ...],
) -> tuple[_MinuteInterval, ...]:
    merged: list[_MinuteInterval] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _format_schedule(
    label: str,
    dates: tuple[date, ...],
    daily_intervals: tuple[tuple[_MinuteInterval, ...], ...],
) -> str:
    groups: list[str] = []
    group_start = 0

    for index in range(1, len(dates) + 1):
        is_same_schedule = (
            index < len(dates)
            and dates[index] == dates[index - 1] + timedelta(days=1)
            and daily_intervals[index] == daily_intervals[group_start]
        )
        if is_same_schedule:
            continue

        groups.append(
            f"{_format_date_span(dates[group_start], dates[index - 1])} "
            f"{_format_intervals(daily_intervals[group_start])}"
        )
        group_start = index

    return f"{label}: {'; '.join(groups)}"


def _format_date_span(start: date, end: date) -> str:
    start_month = _MONTH_NAMES[start.month - 1]
    end_month = _MONTH_NAMES[end.month - 1]

    if start == end:
        return f"{start_month} {start.day}"
    if start.year == end.year and start.month == end.month:
        return f"{start_month} {start.day}–{end.day}"
    if start.year == end.year:
        return f"{start_month} {start.day}–{end_month} {end.day}"
    return f"{start_month} {start.day}, {start.year}–{end_month} {end.day}, {end.year}"


def _format_intervals(intervals: tuple[_MinuteInterval, ...]) -> str:
    if not intervals:
        return "closed"
    if intervals == ((0, _MINUTES_PER_DAY),):
        return "open 24 hours"
    return ", ".join(f"{_format_minute(start)}–{_format_minute(end)}" for start, end in intervals)


def _format_minute(value: int) -> str:
    if value == _MINUTES_PER_DAY:
        return "24:00"
    hours, minutes = divmod(value, 60)
    return f"{hours:02d}:{minutes:02d}"
