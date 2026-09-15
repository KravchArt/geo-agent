"""Selection and hydration rules for client-side map pins."""

from __future__ import annotations

from backend.app.services.map_places import (
    eligible_place_refs,
    load_map_places,
    missing_listed_place_names,
    reusable_search_areas,
    selected_eligible_place_refs,
    selected_search_anchor_refs,
)
from tools.base import ToolResult
from tools.geo.place_store import InMemoryPlaceStore
from tools.refs import PlaceRecord, RecordOrigin

FIRST = "plc_a1b2c3d4e5"
SECOND = "plc_f6e7d8c9b0"
OTHER = "plc_0123456789"


def _places_result() -> ToolResult:
    return ToolResult(
        tool_name="places_search",
        ok=True,
        data={
            "places": [
                {"ref": FIRST, "id": "1", "name": "First", "address": "One"},
            ],
            "area": {"ref": OTHER, "name": "Moscow", "address": "Moscow"},
            "anchor": OTHER,
            "resolved": [
                {
                    "client_id": "candidate",
                    "input_name": "Second",
                    "status": "ambiguous",
                    "options": [{"ref": SECOND, "id": "2", "name": "Second", "address": "Two"}],
                }
            ],
        },
    )


def test_only_actual_places_search_candidates_are_eligible() -> None:
    refs = eligible_place_refs(
        [
            _places_result(),
            ToolResult(tool_name="routing_tool", ok=True, data={"route": {}}),
        ]
    )

    assert refs == {FIRST, SECOND}


def test_only_final_answer_selection_becomes_ordered_map_refs() -> None:
    refs = selected_eligible_place_refs(
        [SECOND, OTHER, SECOND, FIRST],
        [
            _places_result(),
            ToolResult(
                tool_name="places_search",
                ok=True,
                data={
                    "places": [
                        {
                            "ref": "plc_1111111111",
                            "id": "intermediate",
                            "name": "Intermediate",
                            "address": "Hidden",
                        }
                    ]
                },
            ),
        ],
    )

    assert refs == [SECOND, FIRST]


def test_anchor_is_selected_only_when_its_near_search_contributed_a_pin() -> None:
    assert selected_search_anchor_refs([FIRST], [_places_result()]) == [OTHER]
    assert selected_search_anchor_refs([SECOND], [_places_result()]) == []


def test_concrete_places_named_without_refs_are_reported() -> None:
    missing = missing_listed_place_names(
        "Нашлись First и Second.",
        [SECOND],
        [_places_result()],
    )

    assert missing == ["First"]


def test_answer_without_concrete_place_names_does_not_require_map_refs() -> None:
    assert (
        missing_listed_place_names(
            "Вот несколько подходящих вариантов.",
            [],
            [_places_result()],
        )
        == []
    )


def test_no_inline_selection_means_no_map_refs() -> None:
    assert selected_eligible_place_refs([], [_places_result()]) == []


def test_search_area_is_retained_only_as_private_reusable_context() -> None:
    result = _places_result()

    areas = reusable_search_areas(
        [result, ToolResult(tool_name="routing_tool", ok=True, data={"route": {}})]
    )

    assert [area.model_dump(mode="json") for area in areas] == [
        {"ref": OTHER, "name": "Moscow", "address": "Moscow"}
    ]
    assert OTHER not in eligible_place_refs([result])


def test_failed_or_malformed_search_does_not_create_private_area_context() -> None:
    assert (
        reusable_search_areas(
            [
                ToolResult(tool_name="places_search", ok=False, error="failed"),
                ToolResult(tool_name="places_search", ok=True, data={"area": {"name": "Moscow"}}),
            ]
        )
        == []
    )


def test_empty_places_search_result_has_no_eligible_refs() -> None:
    result = ToolResult(tool_name="places_search", ok=True, data={"places": []})

    assert eligible_place_refs([result]) == frozenset()


async def test_selected_refs_are_hydrated_from_the_trusted_place_store() -> None:
    store = InMemoryPlaceStore()
    await store.save(
        PlaceRecord(
            ref=SECOND,
            name="Second",
            address="Two",
            lat=55.75,
            lon=37.62,
            kind="cafe",
            origin=RecordOrigin.PLACES_SEARCH,
        )
    )

    places = await load_map_places([SECOND, FIRST], place_store=store)

    assert [place.ref for place in places] == [SECOND]
    assert places[0].latitude == 55.75
    assert places[0].longitude == 37.62
    assert places[0].marker_role == "result"


async def test_nearby_search_anchor_is_hydrated_as_a_distinct_map_role() -> None:
    store = InMemoryPlaceStore()
    await store.save_many(
        [
            PlaceRecord(
                ref=FIRST,
                name="Pharmacy",
                address="One",
                lat=56.20,
                lon=43.80,
                kind="pharmacy",
                origin=RecordOrigin.PLACES_SEARCH,
            ),
            PlaceRecord(
                ref=OTHER,
                name="Cafe Molot",
                address="Anchor",
                lat=56.21,
                lon=43.81,
                kind="cafe",
                origin=RecordOrigin.PLACES_SEARCH,
            ),
        ]
    )

    places = await load_map_places(
        [FIRST],
        anchor_refs=[OTHER],
        place_store=store,
    )

    assert [place.ref for place in places] == [FIRST, OTHER]
    assert [place.marker_role for place in places] == ["result", "search_anchor"]
