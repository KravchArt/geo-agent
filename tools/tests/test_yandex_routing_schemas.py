"""Schema tests for Yandex routing responses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools.geo.routing import TrafficType, TransportMode
from tools.geo.routing.yandex.schemas import (
    YandexDistanceMatrixResponse,
    YandexRouteResponse,
)


def test_route_response_parses_documented_subset_and_ignores_geometry() -> None:
    """Verify that route response parses documented subset and ignores geometry."""

    response = YandexRouteResponse.model_validate(
        {
            "traffic_type": "realtime",
            "route": {
                "legs": [
                    {
                        "status": "OK",
                        "steps": [
                            {
                                "length": 120.5,
                                "duration": 60.25,
                                "mode": "walking",
                                "polyline": {"points": [[55.7, 37.5], [55.8, 37.6]]},
                            }
                        ],
                    }
                ],
                "flags": {"hasTolls": True, "hasNonTransactionalTolls": False},
            },
            "optimization": {"waypoints_order": [0, 2, 1]},
        }
    )

    assert response.traffic_type is TrafficType.REALTIME
    assert response.route.legs[0].steps[0].mode is TransportMode.WALKING
    assert response.route.flags.has_tolls is True
    assert response.optimization is not None
    assert response.optimization.waypoints_order == [0, 2, 1]


def test_unknown_route_status_is_rejected() -> None:
    """Verify that unknown route status is rejected."""

    with pytest.raises(ValidationError):
        YandexRouteResponse.model_validate(
            {
                "route": {
                    "legs": [
                        {
                            "status": "PARTIAL",
                            "steps": [],
                        }
                    ]
                }
            }
        )


@pytest.mark.parametrize("field", ["length", "duration"])
def test_route_step_rejects_non_finite_numbers(field: str) -> None:
    """Verify that route step rejects non finite numbers."""

    step = {
        "length": 120.0,
        "duration": 60.0,
        "mode": "walking",
    }
    step[field] = float("inf")

    with pytest.raises(ValidationError):
        YandexRouteResponse.model_validate(
            {
                "route": {
                    "legs": [
                        {
                            "status": "OK",
                            "steps": [step],
                        }
                    ]
                }
            }
        )


def test_successful_matrix_element_requires_distance_and_duration() -> None:
    """Verify that successful matrix element requires distance and duration."""

    with pytest.raises(ValidationError, match="requires distance and duration"):
        YandexDistanceMatrixResponse.model_validate(
            {
                "rows": [
                    {
                        "elements": [
                            {
                                "status": "OK",
                                "distance": {"value": 100},
                            }
                        ]
                    }
                ]
            }
        )


def test_failed_matrix_element_may_omit_costs() -> None:
    """Verify that failed matrix element may omit costs."""

    response = YandexDistanceMatrixResponse.model_validate(
        {"rows": [{"elements": [{"status": "FAIL"}]}]}
    )

    element = response.rows[0].elements[0]
    assert element.status == "FAIL"
    assert element.distance is None
    assert element.duration is None
