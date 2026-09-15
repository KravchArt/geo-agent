"""Tests for the Yandex-backed geocoding service."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.schemas import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ToponymKind,
)
from tools.geo.geocoding.yandex.client import YandexGeocoderClient
from tools.geo.geocoding.yandex.service import YandexGeocoderService
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, PlaceRef


class FakePlaceStore:
    def __init__(self) -> None:
        self.records: dict[str, PlaceRecord] = {}
        self.save_calls = 0
        self.save_many_calls: list[list[PlaceRecord]] = []

    async def save(self, record: PlaceRecord) -> None:
        self.save_calls += 1
        self.records[record.ref] = record

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        batch = list(records)
        self.save_many_calls.append(batch)
        self.records.update({record.ref: record for record in batch})

    async def get(self, ref: PlaceRef) -> PlaceRecord | None:
        return self.records.get(ref)


def test_build_query_does_not_duplicate_same_city() -> None:
    assert (
        YandexGeocoderService._build_query(
            GeocodePlaceInput(
                query="Frankfurt am Main",
                city="Frankfurt am Main",
            )
        )
        == "Frankfurt am Main"
    )


async def test_geocode_saves_coordinates_and_returns_safe_match():
    """Verify that geocode saves coordinates and returns safe match."""

    payload: dict[str, object] = {
        "response": {
            "GeoObjectCollection": {
                "featureMember": [
                    {
                        "GeoObject": {
                            "name": "Красная площадь",
                            "uri": "ymapsbm1://geo?oid=123",
                            "metaDataProperty": {
                                "GeocoderMetaData": {
                                    "kind": "house",
                                    "text": "Россия, Москва, Красная площадь",
                                    "precision": "exact",
                                    "Address": {
                                        "Components": [
                                            {"kind": "country", "name": "Россия"},
                                            {"kind": "locality", "name": "Москва"},
                                        ],
                                    },
                                },
                            },
                            "boundedBy": {
                                "Envelope": {
                                    "lowerCorner": "37.619000 55.752000",
                                    "upperCorner": "37.622000 55.755000",
                                },
                            },
                            "Point": {
                                "pos": "37.6208 55.7539",
                            },
                        },
                    },
                ],
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["geocode"] == "Москва, Красная площадь"
        assert request.url.params["results"] == "3"
        return httpx.Response(200, json=payload)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(
            api_key="test-key",
            http_client=http_client,
        )
        service = YandexGeocoderService(
            client=client,
            place_store=store,
        )

        context = ToolExecutionContext()
        result = await service.geocode(
            GeocodePlaceInput(
                query="Красная площадь",
                city="Москва",
                limit=3,
            ),
            context,
        )

    assert result.ambiguous is False
    assert result.best == PlaceMatch(
        ref=result.matches[0].ref,
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        kind=ToponymKind.HOUSE,
        precision="exact",
    )

    stored_record = await store.get(result.matches[0].ref)

    assert stored_record is not None
    assert stored_record.provider == "yandex_geocoder"
    assert stored_record.lat == 55.7539
    assert stored_record.lon == 37.6208
    assert stored_record.locality == "Москва"
    assert stored_record.bounds == GeoBounds(
        west=37.619,
        south=55.752,
        east=37.622,
        north=55.755,
    )
    assert store.save_calls == 0
    assert len(store.save_many_calls) == 1
    assert [record.ref for record in store.save_many_calls[0]] == [result.matches[0].ref]
    assert len(context.upstream_calls) == 1
    assert context.upstream_calls[0].provider == "yandex_geocoder"


async def test_geocode_returns_empty_output_when_yandex_finds_nothing():
    """Verify that geocode returns empty output when Yandex finds nothing."""

    payload: dict[str, object] = {
        "response": {
            "GeoObjectCollection": {},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = YandexGeocoderClient(
            api_key="test-key",
            http_client=http_client,
        )
        service = YandexGeocoderService(
            client=client,
            place_store=store,
        )

        result = await service.geocode(
            GeocodePlaceInput(query="Несуществующее место"),
            ToolExecutionContext(),
        )

    assert result == GeocodePlaceOutput()
    assert store.records == {}


async def test_geocode_removes_duplicate_yandex_candidates():
    """Verify that geocode removes duplicate Yandex candidates."""

    geo_object = {
        "name": "Красная площадь",
        "uri": "ymapsbm1://geo?oid=123",
        "metaDataProperty": {
            "GeocoderMetaData": {
                "kind": "house",
                "text": "Россия, Москва, Красная площадь",
            },
        },
        "Point": {
            "pos": "37.6208 55.7539",
        },
    }
    payload = {
        "response": {
            "GeoObjectCollection": {
                "featureMember": [
                    {"GeoObject": geo_object},
                    {"GeoObject": geo_object},
                ],
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = YandexGeocoderService(
            client=YandexGeocoderClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
        )

        result = await service.geocode(
            GeocodePlaceInput(query="Красная площадь"),
            ToolExecutionContext(),
        )

    assert len(result.matches) == 1
    assert result.ambiguous is False
    assert len(store.records) == 1


async def test_geocode_keeps_unknown_kind_in_store_but_hides_it_from_model():
    """Verify that geocode keeps unknown kind in store but hides it from model."""

    payload: dict[str, object] = {
        "response": {
            "GeoObjectCollection": {
                "featureMember": [
                    {
                        "GeoObject": {
                            "name": "Шереметьево",
                            "metaDataProperty": {
                                "GeocoderMetaData": {
                                    "kind": "airport",
                                    "text": "Московская область, аэропорт Шереметьево",
                                },
                            },
                            "Point": {
                                "pos": "37.4146 55.9726",
                            },
                        },
                    },
                ],
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = YandexGeocoderService(
            client=YandexGeocoderClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
        )

        result = await service.geocode(
            GeocodePlaceInput(query="Шереметьево"),
            ToolExecutionContext(),
        )

    assert result.best is not None
    assert result.best.kind is None

    stored_record = await store.get(result.best.ref)

    assert stored_record is not None
    assert stored_record.kind == "airport"


async def test_geocode_propagates_yandex_http_error():
    """Verify that geocode propagates Yandex HTTP error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = YandexGeocoderService(
            client=YandexGeocoderClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await service.geocode(
                GeocodePlaceInput(query="Красная площадь"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.RATE_LIMITED
    assert store.records == {}


async def test_geocode_rejects_invalid_yandex_payload_without_saving_records():
    """Verify that geocode rejects invalid Yandex payload without saving records."""

    payload: dict[str, object] = {
        "response": {
            "GeoObjectCollection": {
                "featureMember": [
                    {
                        "GeoObject": {
                            "name": "Красная площадь",
                            "metaDataProperty": {
                                "GeocoderMetaData": {
                                    "kind": "house",
                                    "text": "Россия, Москва, Красная площадь",
                                },
                            },
                            # Point намеренно отсутствует.
                        },
                    },
                ],
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    store = FakePlaceStore()
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        service = YandexGeocoderService(
            client=YandexGeocoderClient(
                api_key="test-key",
                http_client=http_client,
            ),
            place_store=store,
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await service.geocode(
                GeocodePlaceInput(query="Красная площадь"),
                ToolExecutionContext(),
            )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "Yandex geocoder gave wrong answer schema"
    assert exc_info.value.provider == "yandex_geocoder"
    assert exc_info.value.status_code is None
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
    assert store.records == {}
