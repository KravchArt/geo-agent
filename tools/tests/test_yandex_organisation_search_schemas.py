from __future__ import annotations

import pytest

from tools.geo.places_search.yandex.schemas import (
    YandexCompanyMetaData,
    YandexOrganisationSearchResponse,
)


def _open_24h(availabilities: list[dict[str, bool]]) -> bool | None:
    company = YandexCompanyMetaData.model_validate(
        {
            "id": "company-id",
            "name": "Кофейня",
            "Address": {"formatted": "Москва"},
            "Hours": {"Availabilities": availabilities},
        }
    )
    return company.open_24h


def test_parses_documented_organisation_fields():
    """Verify that parses documented organisation fields."""

    response = YandexOrganisationSearchResponse.model_validate(
        {
            "type": "FeatureCollection",
            "properties": {
                "ResponseMetaData": {
                    "SearchResponse": {
                        "found": 1,
                    },
                },
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "CompanyMetaData": {
                            "id": "171100494916",
                            "name": "Кофейня на Никольской",
                            "Address": {
                                "formatted": "Россия, Москва, Никольская улица, 17",
                            },
                            "url": "https://example.test/",
                            "Categories": [
                                {"name": "Кофейня"},
                                {"name": "Кафе"},
                            ],
                            "Phones": [
                                {"formatted": "+7 (495) 123-45-67"},
                            ],
                            "Hours": {
                                "text": "ежедневно, круглосуточно",
                                "Availabilities": [
                                    {"Everyday": True, "TwentyFourHours": True},
                                ],
                            },
                            "Features": [
                                {"id": "wheelchair_access", "value": True},
                                {"id": "ramp", "value": False},
                            ],
                        },
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [37.6231, 55.7581],
                    },
                },
            ],
        },
    )

    assert response.found == 1
    assert len(response.organisations) == 1

    organisation = response.organisations[0]

    assert organisation.company.id == "171100494916"
    assert organisation.company.address.formatted == ("Россия, Москва, Никольская улица, 17")
    assert organisation.company.open_24h is True
    assert organisation.point.lon == 37.6231
    assert organisation.point.lat == 55.7581


def test_parses_documented_working_hour_intervals() -> None:
    company = YandexCompanyMetaData.model_validate(
        {
            "id": "company-id",
            "name": "Кофейня",
            "Address": {"formatted": "Москва"},
            "Hours": {
                "Availabilities": [
                    {
                        "Everyday": True,
                        "Intervals": [
                            {"from": "10:00:00", "to": "21:30:00"},
                        ],
                    }
                ]
            },
        }
    )

    assert company.hours is not None
    interval = company.hours.availabilities[0].intervals[0]
    assert interval.from_time.isoformat() == "10:00:00"
    assert interval.to_time.isoformat() == "21:30:00"


@pytest.mark.parametrize("value", ["25:00:00", "not-a-time", 10])
def test_rejects_invalid_working_hour_interval(value: object) -> None:
    with pytest.raises(ValueError, match="working-hours interval"):
        YandexCompanyMetaData.model_validate(
            {
                "id": "company-id",
                "name": "Кофейня",
                "Address": {"formatted": "Москва"},
                "Hours": {
                    "Availabilities": [
                        {
                            "Everyday": True,
                            "Intervals": [{"from": value, "to": "21:30:00"}],
                        }
                    ]
                },
            }
        )


@pytest.mark.parametrize(
    ("availabilities", "expected"),
    [
        ([{"Everyday": True, "TwentyFourHours": True}], True),
        ([{"Sunday": True, "TwentyFourHours": True}], False),
        (
            [
                {"Weekdays": True, "TwentyFourHours": True},
                {"Weekend": True, "TwentyFourHours": True},
            ],
            True,
        ),
        (
            [
                {"Weekdays": True, "TwentyFourHours": True},
                {"Weekend": True},
            ],
            False,
        ),
        ([{"TwentyFourHours": True}], None),
        ([], None),
    ],
)
def test_open_24h_requires_confirmed_full_week_coverage(
    availabilities: list[dict[str, bool]],
    expected: bool | None,
) -> None:
    """Verify that open 24h requires confirmed full week coverage."""

    assert _open_24h(availabilities) is expected
