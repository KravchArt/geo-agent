"""Tests for TomTom-backed internal geocoding."""

from __future__ import annotations

import httpx
import pytest

from tools.base import ToolExecutionError, ToolFailureKind
from tools.geo.geocoding import (
    GeocodedPlaceResolver,
    GeocodePlaceInput,
    ToponymKind,
)
from tools.geo.geocoding.tomtom import TomTomGeocoderClient, TomTomGeocoderService
from tools.geo.place_store import InMemoryPlaceStore
from tools.observability import ToolExecutionContext


def _berlin(
    *,
    result_id: str,
    country: str,
    country_code: str,
    lat: float,
    lon: float,
    score: float | None = None,
    country_subdivision: str | None = None,
    country_secondary_subdivision: str | None = None,
) -> dict[str, object]:
    address: dict[str, object] = {
        "municipality": "Berlin",
        "country": country,
        "countryCode": country_code,
        "freeformAddress": f"Berlin, {country}",
    }
    if country_subdivision is not None:
        address["countrySubdivision"] = country_subdivision
    if country_secondary_subdivision is not None:
        address["countrySecondarySubdivision"] = country_secondary_subdivision
    result: dict[str, object] = {
        "type": "Geography",
        "id": result_id,
        "entityType": "Municipality",
        "matchConfidence": {"score": 1.0},
        "address": address,
        "position": {"lat": lat, "lon": lon},
        "boundingBox": {
            "topLeftPoint": {"lat": lat + 0.2, "lon": lon - 0.2},
            "btmRightPoint": {"lat": lat - 0.2, "lon": lon + 0.2},
        },
    }
    if score is not None:
        result["score"] = score
    return result


async def test_geocode_maps_municipality_and_persists_bounds() -> None:
    payload = {
        "results": [
            _berlin(
                result_id="DE/GEO/p0/1",
                country="Germany",
                country_code="DE",
                lat=52.52,
                lon=13.405,
                score=2.513,
                country_subdivision="Berlin",
            )
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            store,
        )
        result = await service.geocode(
            GeocodePlaceInput(query="Berlin"),
            ToolExecutionContext(),
        )

    assert result.best is not None
    assert result.best.name == "Berlin"
    assert result.best.address == "Berlin, Germany"
    assert result.best.kind is ToponymKind.LOCALITY
    assert result.best.municipality == "Berlin"
    assert result.best.country_subdivision == "Berlin"
    assert result.best.country_code == "DE"
    assert result.best.provider_entity_type == "Municipality"
    assert result.best.relevance_score == 2.513
    assert result.best.match_confidence_score == 1.0
    record = await store.get(result.best.ref)
    assert record is not None
    assert record.provider == "tomtom_geocoder"
    assert record.address == "Berlin, Germany"
    assert record.locality == "Berlin"
    assert record.bounds is not None
    assert round(record.bounds.west, 3) == 13.205
    assert round(record.bounds.south, 2) == 52.32
    assert round(record.bounds.east, 3) == 13.605
    assert round(record.bounds.north, 2) == 52.72


async def test_geocode_removes_terminal_transliteration_apostrophe() -> None:
    payload = {
        "results": [
            _berlin(
                result_id="RU/GEO/p0/tver",
                country="Russia",
                country_code="RU",
                country_subdivision="Tver Oblast",
                lat=56.8596,
                lon=35.9119,
            )
        ]
    }
    payload["results"][0]["address"]["municipality"] = "Tver'"  # type: ignore[index]
    payload["results"][0]["address"]["freeformAddress"] = "Tver', Tver Oblast, Russia"  # type: ignore[index]

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            store,
        )
        result = await service.geocode(
            GeocodePlaceInput(query="Tver"),
            ToolExecutionContext(),
        )

    assert result.best is not None
    assert result.best.name == "Tver"
    assert result.best.municipality == "Tver"
    assert result.best.address == "Tver, Tver Oblast, Russia"
    record = await store.get(result.best.ref)
    assert record is not None
    assert record.name == "Tver"
    assert record.locality == "Tver"


async def test_geocode_locality_address_includes_structured_region_and_country() -> None:
    payload = {
        "results": [
            _berlin(
                result_id="US/GEO/p0/berlin-nh",
                country="United States",
                country_code="US",
                country_subdivision="New Hampshire",
                country_secondary_subdivision="Coos County",
                lat=44.4687,
                lon=-71.1851,
            )
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            InMemoryPlaceStore(),
        )
        result = await service.geocode(
            GeocodePlaceInput(query="Berlin"),
            ToolExecutionContext(),
        )

    assert result.best is not None
    assert result.best.address == "Berlin, Coos County, New Hampshire, United States"
    assert result.best.country_secondary_subdivision == "Coos County"


async def test_bounded_locality_uses_first_ranked_tomtom_result() -> None:
    payload = {
        "results": [
            _berlin(
                result_id="DE/GEO/p0/1",
                country="Germany",
                country_code="DE",
                lat=52.52,
                lon=13.405,
            ),
            _berlin(
                result_id="US/GEO/p0/2",
                country="United States",
                country_code="US",
                lat=44.4687,
                lon=-71.1851,
            ),
        ]
    }

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload, request=request)

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            store,
        )
        resolver = GeocodedPlaceResolver(geocoder=service, place_store=store)
        resolved = await resolver.geocode_bounded_locality(
            "Berlin",
            ToolExecutionContext(),
        )

    assert resolved is not None
    assert resolved.record.address == "Berlin, Germany"
    assert resolved.record.lat == 52.52
    assert resolved.record.lon == 13.405
    assert requests[0].url.params["entityTypeSet"] == ("Municipality,CountrySecondarySubdivision")


async def test_bounded_locality_accepts_city_region_classified_as_secondary_subdivision() -> None:
    payload = {
        "results": [
            {
                "type": "Geography",
                "id": "KZ/GEO/p0/almaty",
                "entityType": "CountrySecondarySubdivision",
                "matchConfidence": {"score": 0.875},
                "address": {
                    "countrySecondarySubdivision": "Алматы",
                    "countrySubdivision": "Казахстан",
                    "country": "Казахстан",
                    "countryCode": "KZ",
                    "freeformAddress": "Алматы",
                },
                "position": {"lat": 43.238949, "lon": 76.889709},
                "boundingBox": {
                    "topLeftPoint": {"lat": 43.4, "lon": 76.7},
                    "btmRightPoint": {"lat": 43.1, "lon": 77.1},
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["entityTypeSet"] == ("Municipality,CountrySecondarySubdivision")
        return httpx.Response(200, json=payload, request=request)

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            store,
        )
        resolver = GeocodedPlaceResolver(geocoder=service, place_store=store)
        resolved = await resolver.geocode_bounded_locality("Almaty", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record.name == "Алматы"
    assert resolved.record.locality == "Алматы"
    assert resolved.record.bounds is not None


async def test_bounded_locality_prefers_localized_municipality_over_city_region() -> None:
    payload = {
        "results": [
            _berlin(
                result_id="RU/GEO/p0/astrakhan-city",
                country="Russia",
                country_code="RU",
                country_subdivision="Southern Federal District",
                country_secondary_subdivision="Astrakhan Oblast",
                lat=46.3497,
                lon=48.0408,
                score=2.45,
            ),
            {
                "type": "Geography",
                "id": "RU/GEO/p0/astrakhan-region",
                "entityType": "CountrySecondarySubdivision",
                "matchConfidence": {"score": 0.92},
                "address": {
                    "countrySecondarySubdivision": "Astrakhan Oblast",
                    "countrySubdivision": "Southern Federal District",
                    "country": "Russia",
                    "countryCode": "RU",
                    "freeformAddress": "Astrakhan Oblast, Russia",
                },
                "position": {"lat": 46.15, "lon": 48.15},
                "boundingBox": {
                    "topLeftPoint": {"lat": 47.3, "lon": 46.8},
                    "btmRightPoint": {"lat": 45.5, "lon": 49.5},
                },
            },
        ]
    }
    # Simulate TomTom falling back to the local spelling despite an English
    # query. This must not let the broader region replace the municipality.
    payload["results"][0]["address"]["municipality"] = "Астрахань"  # type: ignore[index]
    payload["results"][0]["address"]["freeformAddress"] = "Астрахань, Россия"  # type: ignore[index]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    store = InMemoryPlaceStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            store,
        )
        resolver = GeocodedPlaceResolver(geocoder=service, place_store=store)
        resolved = await resolver.geocode_bounded_locality("Astrakhan", ToolExecutionContext())

    assert resolved is not None
    assert resolved.record.name == "Астрахань"
    assert resolved.record.lat == 46.3497
    assert resolved.record.lon == 48.0408


async def test_reverse_geocode_returns_structured_address() -> None:
    payload = {
        "addresses": [
            {
                "id": "DE/PAD/1",
                "address": {
                    "municipality": "Berlin",
                    "municipalitySubdivision": "Mitte",
                    "neighbourhood": "Dorotheenstadt",
                    "countrySubdivision": "Berlin",
                    "countrySecondarySubdivision": "Berlin",
                    "postalCode": "10117",
                    "country": "Deutschland",
                    "countryCode": "DE",
                    "countryCodeISO3": "DEU",
                    "freeformAddress": "Unter den Linden 6, 10117 Berlin",
                },
                "position": "52.5186,13.3980",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "entityType" not in request.url.params
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            InMemoryPlaceStore(),
        )
        result = await service.reverse_geocode(
            lat=52.5186,
            lon=13.398,
            context=ToolExecutionContext(),
            language="de-DE",
        )

    assert result.address == "Unter den Linden 6, 10117 Berlin"
    assert result.municipality == "Berlin"
    assert result.municipality_subdivision == "Mitte"
    assert result.neighbourhood == "Dorotheenstadt"
    assert result.country_subdivision == "Berlin"
    assert result.country_secondary_subdivision == "Berlin"
    assert result.postal_code == "10117"
    assert result.country == "Deutschland"
    assert result.country_code == "DE"
    assert result.country_code_iso3 == "DEU"


async def test_reverse_geocode_city_requests_municipality_geography() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["entityType"] == "Municipality"
        return httpx.Response(
            200,
            json={
                "addresses": [
                    {
                        "entityType": "Municipality",
                        "address": {
                            "municipality": "Berlin",
                            "country": "Germany",
                            "countryCode": "DE",
                        },
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            InMemoryPlaceStore(),
        )
        city = await service.reverse_geocode_city(
            lat=52.52,
            lon=13.405,
            context=ToolExecutionContext(),
        )

    assert city == "Berlin"


async def test_reverse_geocode_returns_empty_output_when_no_address_exists() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"addresses": []}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            InMemoryPlaceStore(),
        )
        result = await service.reverse_geocode(
            lat=0,
            lon=0,
            context=ToolExecutionContext(),
        )

    assert result.municipality is None
    assert result.address is None


async def test_reverse_geocode_rejects_invalid_response_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"addresses": [{"address": {"municipality": 123}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        service = TomTomGeocoderService(
            TomTomGeocoderClient(api_key="test-key", http_client=http_client),
            InMemoryPlaceStore(),
        )
        with pytest.raises(ToolExecutionError) as exc_info:
            await service.reverse_geocode(
                lat=52.52,
                lon=13.405,
                context=ToolExecutionContext(),
            )

    assert str(exc_info.value) == "TomTom reverse geocoder gave wrong answer schema"
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
