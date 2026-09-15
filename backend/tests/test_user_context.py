from __future__ import annotations

import json

import httpx
import pytest

from backend.app.config import Settings
from backend.app.services.user_context import append_user_context, resolve_user_context
from common.models import BrowserLocation, UserContextInput
from tools.geo.place_store import InMemoryPlaceStore
from tools.observability import ToolExecutionContext


@pytest.mark.asyncio
async def test_browser_location_wins_without_ip_lookup() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            UserContextInput(
                browser_location=BrowserLocation(
                    latitude=55.751244,
                    longitude=37.618423,
                    accuracy_m=25,
                ),
                timezone="Europe/Moscow",
                ip_address="8.8.8.8",
            ),
            settings=Settings(),
            client=client,
        )

    assert called is False
    assert context.location_source == "browser"
    assert context.latitude == 55.751244
    assert context.city is None
    assert context.location_ref is None
    assert context.timezone_name == "Europe/Moscow"
    assert context.local_time_iso.endswith("+03:00")


@pytest.mark.asyncio
async def test_browser_location_is_reverse_geocoded_and_stored_as_an_opaque_anchor() -> None:
    class ReverseGeocoder:
        async def reverse_geocode_city(
            self,
            *,
            lat: float,
            lon: float,
            context: ToolExecutionContext,
            language: str = "NGT",
        ) -> str | None:
            assert (lat, lon, language) == (55.751244, 37.618423, "NGT")
            return "Москва"

    store = InMemoryPlaceStore()
    context = await resolve_user_context(
        UserContextInput(
            browser_location=BrowserLocation(latitude=55.751244, longitude=37.618423),
            timezone="Europe/Moscow",
        ),
        settings=Settings(),
        session_id="session-1",
        place_store=store,
        reverse_geocoder=ReverseGeocoder(),
    )

    assert context.city == "Москва"
    assert context.location_ref is not None
    record = await store.get(context.location_ref)
    assert record is not None
    assert (record.lat, record.lon) == (55.751244, 37.618423)
    assert record.locality == "Москва"
    assert record.origin == "user_location"

    prompt = append_user_context("Найди Пятёрочку рядом", context)
    assert "approximate_place: Москва" in prompt
    assert f"current_location_ref: {context.location_ref}" in prompt
    assert "coordinates:" not in prompt


@pytest.mark.asyncio
async def test_ip_location_is_used_when_browser_location_is_missing() -> None:
    payload = {
        "success": True,
        "latitude": 52.52,
        "longitude": 13.405,
        "city": "Berlin",
        "region": "Berlin",
        "country": "Germany",
        "timezone": {"id": "Europe/Berlin"},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/8.8.8.8")
        return httpx.Response(200, content=json.dumps(payload).encode())

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            UserContextInput(ip_address="8.8.8.8"),
            settings=Settings(),
            client=client,
            session_id="session-1",
            place_store=store,
        )

    assert context.location_source == "ip"
    assert context.city == "Berlin"
    assert context.location_ref is None
    assert context.timezone_name == "Europe/Berlin"
    assert context.latitude == 52.52


@pytest.mark.asyncio
async def test_private_ip_is_not_sent_to_lookup_service() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"success": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            UserContextInput(ip_address="127.0.0.1", timezone="Europe/Moscow"),
            settings=Settings(ip_geolocation_self_lookup_enabled=False),
            client=client,
        )

    assert called is False
    assert context.location_source == "unavailable"
    assert context.timezone_name == "Europe/Moscow"


@pytest.mark.asyncio
async def test_backend_egress_ip_is_not_used_for_supplied_private_client_ip() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"success": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            UserContextInput(ip_address="172.24.0.1", timezone="Europe/Moscow"),
            settings=Settings(ip_geolocation_self_lookup_enabled=True),
            client=client,
        )

    assert called is False
    assert context.location_source == "unavailable"
    assert context.timezone_name == "Europe/Moscow"


@pytest.mark.asyncio
async def test_backend_egress_ip_is_not_used_for_supplied_missing_client_ip() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"success": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            UserContextInput(timezone="Europe/Moscow"),
            settings=Settings(ip_geolocation_self_lookup_enabled=True),
            client=client,
        )

    assert called is False
    assert context.location_source == "unavailable"
    assert context.timezone_name == "Europe/Moscow"


@pytest.mark.asyncio
async def test_explicit_dev_self_lookup_is_used_when_user_context_is_omitted() -> None:
    payload = {
        "success": True,
        "latitude": 52.52,
        "longitude": 13.405,
        "city": "Berlin",
        "region": "Berlin",
        "country": "Germany",
        "timezone": {"id": "Europe/Berlin"},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path in ("", "/")
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await resolve_user_context(
            None,
            settings=Settings(ip_geolocation_self_lookup_enabled=True),
            client=client,
        )

    assert context.location_source == "backend_ip"
    assert context.city == "Berlin"
    assert context.latitude == 52.52


def test_metadata_is_appended_after_user_message() -> None:
    from backend.app.services.user_context import ResolvedUserContext

    context = ResolvedUserContext(
        location_source="browser",
        latitude=55.75,
        longitude=37.61,
        accuracy_m=20,
        city=None,
        region=None,
        country=None,
        location_ref=None,
        timezone_name="Europe/Moscow",
        local_time_iso="2026-08-01T20:00:00+03:00",
    )

    prompt = append_user_context("Find coffee nearby", context)

    assert prompt.startswith("Find coffee nearby\n\n[USER_CONTEXT_METADATA]")
    assert "coordinates:" not in prompt
    assert "timezone: Europe/Moscow" in prompt
    assert prompt.endswith("[/USER_CONTEXT_METADATA]")
