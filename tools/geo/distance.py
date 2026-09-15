"""Shared geographic distance calculations."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

_EARTH_RADIUS_M = 6_371_000


def distance_m(
    *,
    from_lat: float,
    from_lon: float,
    to_lat: float,
    to_lon: float,
) -> int:
    """Calculate the great-circle distance between two points in metres."""

    lat_delta = radians(to_lat - from_lat)
    lon_delta = radians(to_lon - from_lon)
    a = (
        sin(lat_delta / 2) ** 2
        + cos(radians(from_lat)) * cos(radians(to_lat)) * sin(lon_delta / 2) ** 2
    )
    return round(2 * _EARTH_RADIUS_M * asin(sqrt(a)))
