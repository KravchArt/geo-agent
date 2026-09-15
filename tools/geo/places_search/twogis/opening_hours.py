"""Normalize 2GIS weekly schedules for tool output and current-state checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tools.geo.opening_hours import is_open_at
from tools.geo.places_search.twogis.schemas import TwoGisSchedule

_OSM_DAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
_DISPLAY_DAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
_FULL_DAY_PERIODS = frozenset({("00:00", "24:00"), ("00:00", "00:00")})

_DailyPeriods = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class TwoGisOpeningHoursSummary:
    text: str | None
    open_24h: bool | None
    is_open_now: bool | None


def summarize_schedule(
    schedule: TwoGisSchedule | None,
    *,
    lat: float,
    lon: float,
    now: datetime,
) -> TwoGisOpeningHoursSummary:
    if schedule is None:
        return TwoGisOpeningHoursSummary(None, None, None)

    expression_rules: list[str] = []
    daily_periods: list[_DailyPeriods | None] = []
    complete_week = True
    full_week = True

    for osm_day, day in zip(
        _OSM_DAYS,
        schedule.days,
        strict=True,
    ):
        if day is None or not day.working_hours:
            daily_periods.append(None)
            complete_week = False
            full_week = False
            continue
        periods = tuple((period.from_time, period.to_time) for period in day.working_hours)
        daily_periods.append(periods)
        if len(periods) != 1 or periods[0] not in _FULL_DAY_PERIODS:
            full_week = False
        intervals = [f"{start}-{end}" for start, end in periods]
        expression_rules.append(f"{osm_day} {','.join(intervals)}")

    open_24h = True if schedule.is_24x7 or (complete_week and full_week) else None
    if complete_week and open_24h is None:
        open_24h = False

    text = "круглосуточно" if open_24h is True else _format_weekly_schedule(tuple(daily_periods))
    comment = schedule.comment or schedule.description
    if comment:
        text = f"{text}; {comment}" if text else comment

    expression = "; ".join(expression_rules) or None
    is_open_now = True if open_24h is True else is_open_at(expression, lat=lat, lon=lon, now=now)
    return TwoGisOpeningHoursSummary(text, open_24h, is_open_now)


def _format_weekly_schedule(daily_periods: tuple[_DailyPeriods | None, ...]) -> str | None:
    """Collapse adjacent days with identical hours into compact ranges."""

    groups: list[str] = []
    index = 0
    while index < len(daily_periods):
        periods = daily_periods[index]
        if periods is None:
            index += 1
            continue

        end = index
        while end + 1 < len(daily_periods) and daily_periods[end + 1] == periods:
            end += 1

        if index == 0 and end == len(_DISPLAY_DAYS) - 1:
            days = "ежедневно"
        elif index == end:
            days = _DISPLAY_DAYS[index]
        else:
            days = f"{_DISPLAY_DAYS[index]}–{_DISPLAY_DAYS[end]}"
        intervals = ", ".join(f"{start}–{finish}" for start, finish in periods)
        groups.append(f"{days} {intervals}")
        index = end + 1

    return "; ".join(groups) or None
