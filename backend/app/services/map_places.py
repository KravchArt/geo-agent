"""Turn a model's inline-selected ``plc_`` refs into verified client map data.

The model never receives coordinates and places ``[[plc_...]]`` markers beside
the concrete places selected for its final answer. This module accepts only refs
emitted by a successful ``places_search`` call in the current request, then
expands them from the trusted place store.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import ValidationError

from common.models import MapPlace
from tools.base import ToolResult
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.schemas import PlacesSearchOutput, ResolvedSearchArea


def reusable_search_areas(results: Iterable[ToolResult]) -> list[ResolvedSearchArea]:
    """Collect private area refs that a later conversation turn can reuse.

    Search areas are tool scope, not user-facing map pins.  Keep them out of
    :func:`eligible_place_refs`, while retaining their opaque refs separately
    for follow-up area searches and ``mode=near`` calls with a new textual ``near``.
    """

    areas_by_ref: dict[str, ResolvedSearchArea] = {}
    for result in results:
        if result.tool_name != "places_search" or not result.ok or result.data is None:
            continue
        try:
            output = PlacesSearchOutput.model_validate(result.data)
        except ValidationError:
            continue
        if output.area is not None:
            areas_by_ref[output.area.ref] = output.area
    return list(areas_by_ref.values())


def eligible_place_refs(results: Iterable[ToolResult]) -> frozenset[str]:
    """Return selectable refs from successful ``places_search`` results only.

    Area and anchor refs describe search scope rather than a user-facing place,
    so they are intentionally excluded. ``resolve`` options are included: the
    model may legitimately show an ambiguity to the user as a choice.
    """

    refs: set[str] = set()
    for result in results:
        if result.tool_name != "places_search" or not result.ok or result.data is None:
            continue
        try:
            output = PlacesSearchOutput.model_validate(result.data)
        except ValidationError:
            # ToolResult data has already been validated at execution time. This
            # guard makes a malformed historical/custom tool result fail closed.
            continue

        refs.update(place.ref for place in output.places)
        for resolution in output.resolved:
            if resolution.place is not None:
                refs.add(resolution.place.ref)
            refs.update(option.ref for option in resolution.options)
    return frozenset(refs)


def missing_listed_place_names(
    answer: str,
    selected_refs: Sequence[str],
    results: Iterable[ToolResult],
) -> list[str]:
    """Find concrete search results named in prose but missing an inline ref."""

    normalized_answer = " ".join(answer.casefold().split())
    selected = set(selected_refs)
    missing: list[str] = []
    for result in results:
        if result.tool_name != "places_search" or not result.ok or result.data is None:
            continue
        try:
            output = PlacesSearchOutput.model_validate(result.data)
        except ValidationError:
            continue
        places = list(output.places)
        for resolution in output.resolved:
            if resolution.place is not None:
                places.append(resolution.place)
            places.extend(resolution.options)
        for place in places:
            normalized_name = " ".join(place.name.casefold().split())
            if (
                place.ref not in selected
                and len(normalized_name) >= 3
                and normalized_name in normalized_answer
                and place.name not in missing
            ):
                missing.append(place.name)
    return missing


def selected_eligible_place_refs(
    selected_refs: Sequence[str],
    results: Iterable[ToolResult],
) -> list[str]:
    """Keep the model's final selection order, dropping untrusted and duplicate refs.

    Tool calls may contain broad or intermediate searches. Only refs explicitly
    present in the final answer become pins, and only when a successful
    ``places_search`` in this request actually returned them.
    """

    eligible_refs = eligible_place_refs(results)
    seen: set[str] = set()
    selected: list[str] = []
    for ref in selected_refs:
        if ref in eligible_refs and ref not in seen:
            seen.add(ref)
            selected.append(ref)
    return selected


def selected_search_anchor_refs(
    selected_refs: Sequence[str],
    results: Iterable[ToolResult],
) -> list[str]:
    """Return anchors for nearby searches that contributed visible result pins."""

    selected = set(selected_refs)
    seen: set[str] = set()
    anchors: list[str] = []
    for result in results:
        if result.tool_name != "places_search" or not result.ok or result.data is None:
            continue
        try:
            output = PlacesSearchOutput.model_validate(result.data)
        except ValidationError:
            continue
        if output.anchor is None or not selected.intersection(place.ref for place in output.places):
            continue
        if output.anchor not in seen:
            seen.add(output.anchor)
            anchors.append(output.anchor)
    return anchors


async def load_map_places(
    selected_refs: Sequence[str],
    *,
    anchor_refs: Sequence[str] = (),
    place_store: PlaceStore | None,
) -> list[MapPlace]:
    """Expand already-validated refs into the client-facing map payload.

    A missing record is ignored defensively. Validation happens before this
    function; omitting an expired record is safer than emitting an unverified
    coordinate or failing an otherwise useful textual answer.
    """

    if place_store is None:
        return []

    anchor_set = set(anchor_refs)
    ordered_refs = list(dict.fromkeys([*selected_refs, *anchor_refs]))
    places: list[MapPlace] = []
    for ref in ordered_refs:
        record = await place_store.get(ref)
        if record is None:
            continue
        places.append(
            MapPlace(
                ref=record.ref,
                name=record.name,
                address=record.address,
                latitude=record.lat,
                longitude=record.lon,
                kind=record.kind,
                marker_role="search_anchor" if ref in anchor_set else "result",
            )
        )
    return places
