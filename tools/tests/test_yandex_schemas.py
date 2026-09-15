"""Tests for Yandex Geocoder response models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools.geo.geocoding.yandex.schemas import (
    YandexEnvelope,
    YandexGeocoderMetaData,
    YandexGeocoderResponse,
    YandexGeoObject,
    YandexPoint,
)


def test_yandex_point_parses_yandex_coordinate_order():
    """Verify that Yandex point parses Yandex coordinate order."""

    point = YandexPoint.model_validate({"pos": "37.6208 55.7539"})

    assert point.lon == 37.6208
    assert point.lat == 55.7539


@pytest.mark.parametrize(
    "pos",
    [
        "37.6208",
        "37.6208 55.7539 10",
        "not-a-coordinate",
        "181 55",
        "37 -91",
    ],
)
def test_yandex_point_rejects_invalid_coordinates(pos: str):
    """Verify that Yandex point rejects invalid coordinates."""

    with pytest.raises(ValidationError):
        YandexPoint.model_validate({"pos": pos})


def test_yandex_envelope_parses_documented_corner_order():
    """Verify that Yandex envelope parses documented corner order."""

    envelope = YandexEnvelope.model_validate(
        {
            "lowerCorner": "36.803101 55.142174",
            "upperCorner": "37.967427 56.021251",
        },
    )

    assert envelope.west == 36.803101
    assert envelope.south == 55.142174
    assert envelope.east == 37.967427
    assert envelope.north == 56.021251


def test_yandex_envelope_rejects_inverted_corners():
    """Verify that Yandex envelope rejects inverted corners."""

    with pytest.raises(ValidationError, match="lowerCorner longitude"):
        YandexEnvelope.model_validate(
            {
                "lowerCorner": "38.0 55.0",
                "upperCorner": "37.0 56.0",
            },
        )


def test_yandex_geocoder_metadata_parses_documented_fields():
    """Verify that Yandex geocoder metadata parses documented fields."""

    metadata = YandexGeocoderMetaData.model_validate(
        {
            "kind": "house",
            "text": "Россия, Москва, Красная площадь",
            "precision": "exact",
        },
    )

    assert metadata.kind == "house"
    assert metadata.text == "Россия, Москва, Красная площадь"
    assert metadata.precision == "exact"


def test_yandex_geo_object_parses_nested_yandex_response():
    """Verify that Yandex geo object parses nested Yandex response."""

    candidate = YandexGeoObject.model_validate(
        {
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
    )

    assert candidate.name == "Красная площадь"
    assert candidate.uri == "ymapsbm1://geo?oid=123"
    assert candidate.geocoder_metadata.text == "Россия, Москва, Красная площадь"
    assert candidate.geocoder_metadata.locality == "Москва"
    assert candidate.bounds is not None
    assert candidate.bounds.west == 37.619
    assert candidate.point.lat == 55.7539


def test_yandex_geocoder_response_parses_candidates():
    """Verify that Yandex geocoder response parses candidates."""

    response = YandexGeocoderResponse.model_validate(
        {
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
                                "Point": {
                                    "pos": "37.6208 55.7539",
                                },
                            },
                        },
                    ],
                },
            },
        },
    )

    assert len(response.candidates) == 1
    assert response.candidates[0].name == "Красная площадь"


def test_yandex_geocoder_response_accepts_no_candidates():
    """Verify that Yandex geocoder response accepts no candidates."""

    response = YandexGeocoderResponse.model_validate(
        {
            "response": {
                "GeoObjectCollection": {},
            },
        },
    )

    assert response.candidates == []
