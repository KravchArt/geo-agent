"""Evaluate OSM-compatible opening-hours expressions at a fixed instant."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from opening_hours import (  # type: ignore[attr-defined]
    InvalidCoordinatesError,
    OpeningHours,
    ParserError,
    State,
    UnknownCountryError,
)

_UTC = ZoneInfo("UTC")


def is_open_at(
    hours_text: str | None,
    *,
    lat: float,
    lon: float,
    now: datetime,
) -> bool | None:
    """Return the confirmed current state, or ``None`` when it cannot be proved.

    Coordinates let the parser infer the place's IANA time zone and country,
    which are needed for local wall-clock rules, solar events, and public
    holiday selectors. A fixed aware ``now`` keeps every result in one tool
    call evaluated at the same instant.
    """

    if hours_text is None or not hours_text.strip():
        return None
    if now.tzinfo is None:
        raise ValueError("opening-hours reference time must be timezone-aware")

    # opening-hours-py expects an aware datetime whose tzinfo exposes an IANA
    # key. datetime.UTC has no such key, so normalize all aware inputs first.
    reference_time = now.astimezone(_UTC)

    try:
        state, _ = OpeningHours(
            hours_text,
            coords=(lat, lon),
        ).state(reference_time)
    except (
        InvalidCoordinatesError,
        ParserError,
        UnknownCountryError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return None

    if state is State.OPEN:
        return True
    if state is State.CLOSED:
        return False
    return None
