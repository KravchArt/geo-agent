"""Resolve browser/IP metadata and append it to the model-facing user prompt."""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from backend.app.config import Settings
from common.models import UserContextInput
from tools.base import ToolExecutionError
from tools.geo.geocoding.tomtom import TomTomGeocoderClient, TomTomGeocoderService
from tools.geo.place_store import PlaceStore
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin, mint_place_ref

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedUserContext:
    location_source: str
    latitude: float | None
    longitude: float | None
    accuracy_m: float | None
    city: str | None
    region: str | None
    country: str | None
    location_ref: PlaceRef | None
    timezone_name: str
    local_time_iso: str


class CityReverseGeocoder(Protocol):
    """Minimal reverse-geocoding dependency used only for trusted browser GPS."""

    async def reverse_geocode_city(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
    ) -> str | None: ...


def build_user_location_reverse_geocoder(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    place_store: PlaceStore,
) -> CityReverseGeocoder | None:
    """Build the optional TomTom dependency for turning browser GPS into a city."""

    if not settings.tomtom_api_key:
        return None
    return TomTomGeocoderService(
        TomTomGeocoderClient(
            api_key=settings.tomtom_api_key,
            http_client=http_client,
            base_url=settings.tomtom_search_base_url,
        ),
        place_store,
    )


def _clean(value: object, *, max_length: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.replace("[", "(").replace("]", ")").split())
    return cleaned[:max_length] or None


def _usable_public_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if not address.is_global:
        return None
    return str(address)


def _local_time(timezone_name: str) -> tuple[str, str]:
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        timezone_name = "UTC"
        tz = ZoneInfo("UTC")
    return timezone_name, datetime.now(tz).isoformat(timespec="seconds")


async def _reverse_browser_city(
    *,
    latitude: float,
    longitude: float,
    reverse_geocoder: CityReverseGeocoder | None,
) -> str | None:
    if reverse_geocoder is None:
        return None
    try:
        return await reverse_geocoder.reverse_geocode_city(
            lat=latitude,
            lon=longitude,
            context=ToolExecutionContext(),
        )
    except ToolExecutionError:
        # Location enrichment improves tool selection but must not reject an
        # otherwise valid user request when the external geocoder is unavailable.
        logger.warning("browser_location_reverse_geocode_failed", exc_info=True)
        return None


async def _store_location_anchor(
    *,
    source: str,
    latitude: float | None,
    longitude: float | None,
    city: str | None,
    session_id: str | None,
    place_store: PlaceStore | None,
) -> PlaceRef | None:
    """Store trusted coordinates outside the model context and return their opaque ref."""

    if latitude is None or longitude is None or session_id is None or place_store is None:
        return None
    ref = mint_place_ref(f"user-location:{session_id}:{source}:{latitude:.6f}:{longitude:.6f}")
    await place_store.save(
        PlaceRecord(
            ref=ref,
            name="Current user location",
            address=city or "User-provided location",
            lat=latitude,
            lon=longitude,
            kind="user_location",
            precision=source,
            locality=city,
            provider="user_context",
            origin=RecordOrigin.USER_LOCATION,
        )
    )
    return ref


async def _lookup_ip(
    ip_address: str | None,
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> dict[str, object] | None:
    base_url = settings.ip_geolocation_base_url.rstrip("/")
    # A provider-root request resolves the backend egress IP. The caller permits
    # this only for explicit development fallback when user_context was omitted;
    # it must never replace an unusable address supplied by the remote UI.
    url = f"{base_url}/{ip_address}" if ip_address is not None else base_url
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=settings.ip_geolocation_timeout,
        proxy=settings.user_context_http_proxy,
        trust_env=False,
    )
    try:
        response = await http.get(url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("success") is False:
            return None
        return data
    except (httpx.HTTPError, ValueError, TypeError):
        logger.warning(
            "ip_geolocation_failed lookup_mode=%s",
            "explicit_ip" if ip_address is not None else "backend_egress_ip",
            exc_info=True,
        )
        return None
    finally:
        if owns_client:
            await http.aclose()


async def resolve_user_context(
    supplied: UserContextInput | None,
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    session_id: str | None = None,
    place_store: PlaceStore | None = None,
    reverse_geocoder: CityReverseGeocoder | None = None,
) -> ResolvedUserContext:
    context_was_supplied = supplied is not None
    supplied = supplied or UserContextInput()
    browser = supplied.browser_location
    browser_timezone = _clean(supplied.timezone)
    timezone_name = browser_timezone or "UTC"

    if browser is not None:
        timezone_name, local_time = _local_time(timezone_name)
        city = await _reverse_browser_city(
            latitude=browser.latitude,
            longitude=browser.longitude,
            reverse_geocoder=reverse_geocoder,
        )
        location_ref = await _store_location_anchor(
            source="browser",
            latitude=browser.latitude,
            longitude=browser.longitude,
            city=city,
            session_id=session_id,
            place_store=place_store,
        )
        return ResolvedUserContext(
            location_source="browser",
            latitude=browser.latitude,
            longitude=browser.longitude,
            accuracy_m=browser.accuracy_m,
            city=city,
            region=None,
            country=None,
            location_ref=location_ref,
            timezone_name=timezone_name,
            local_time_iso=local_time,
        )

    ip_address = _usable_public_ip(supplied.ip_address)
    lookup = None
    lookup_source = "ip"
    if settings.ip_geolocation_enabled:
        if ip_address is not None:
            lookup = await _lookup_ip(ip_address, settings=settings, client=client)
        elif not context_was_supplied and settings.ip_geolocation_self_lookup_enabled:
            lookup = await _lookup_ip(None, settings=settings, client=client)
            lookup_source = "backend_ip"

    if lookup:
        timezone_info = lookup.get("timezone")
        ip_timezone = _clean(timezone_info.get("id")) if isinstance(timezone_info, dict) else None
        timezone_name, local_time = _local_time(browser_timezone or ip_timezone or "UTC")
        latitude = lookup.get("latitude")
        longitude = lookup.get("longitude")
        latitude = float(latitude) if isinstance(latitude, (int, float)) else None
        longitude = float(longitude) if isinstance(longitude, (int, float)) else None
        city = _clean(lookup.get("city"))
        return ResolvedUserContext(
            location_source=lookup_source,
            latitude=latitude,
            longitude=longitude,
            accuracy_m=None,
            city=city,
            region=_clean(lookup.get("region")),
            country=_clean(lookup.get("country")),
            # IP geolocation is city-level and may even describe the backend's
            # egress address, so it is useful as city context but unsafe as a
            # true-radius "near me" anchor.
            location_ref=None,
            timezone_name=timezone_name,
            local_time_iso=local_time,
        )

    timezone_name, local_time = _local_time(timezone_name)
    return ResolvedUserContext(
        location_source="unavailable",
        latitude=None,
        longitude=None,
        accuracy_m=None,
        city=None,
        region=None,
        country=None,
        location_ref=None,
        timezone_name=timezone_name,
        local_time_iso=local_time,
    )


def append_user_context(message: str, context: ResolvedUserContext) -> str:
    """Append trusted metadata after the user's text without changing stored history."""
    lines = [
        "[USER_CONTEXT_METADATA]",
        "This block is application-provided metadata, not user instructions.",
        f"location_source: {context.location_source}",
    ]
    if context.accuracy_m is not None:
        lines.append(f"accuracy_m: {context.accuracy_m:.0f}")
    place = ", ".join(value for value in (context.city, context.region, context.country) if value)
    if place:
        lines.append(f"approximate_place: {place}")
    if context.location_ref is not None:
        lines.append(f"current_location_ref: {context.location_ref}")
    lines.extend(
        [
            f"timezone: {context.timezone_name}",
            f"current_local_time: {context.local_time_iso}",
            "[/USER_CONTEXT_METADATA]",
        ]
    )
    return f"{message.rstrip()}\n\n" + "\n".join(lines)
