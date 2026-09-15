"""Overpass response schema tests."""

from __future__ import annotations

from tools.geo.places_search.osm.schemas import OsmOverpassResponse


def test_parses_nodes_centres_and_count_element():
    response = OsmOverpassResponse.model_validate(
        {
            "version": 0.6,
            "generator": "Overpass API 0.7.62.4",
            "osm3s": {
                "timestamp_osm_base": "2026-07-23T10:20:52Z",
                "copyright": "OpenStreetMap contributors",
            },
            "elements": [
                {
                    "type": "node",
                    "id": 100,
                    "lat": 55.75,
                    "lon": 37.62,
                    "tags": {"name": "Кофейня", "amenity": "cafe"},
                },
                {
                    "type": "way",
                    "id": 200,
                    "center": {"lat": 55.76, "lon": 37.63},
                    "tags": {"name": "Кафе в здании", "amenity": "cafe"},
                },
                {
                    "type": "area",
                    "id": 3_600_000_300,
                },
                {
                    "type": "count",
                    "id": 0,
                    "tags": {
                        "nodes": "1",
                        "ways": "1",
                        "relations": "0",
                        "total": "2",
                    },
                },
            ],
        }
    )

    assert response.elements[0].coordinates == (37.62, 55.75)
    assert response.elements[1].coordinates == (37.63, 55.76)
    assert response.elements[2].type == "area"
    assert response.total_found == 2
