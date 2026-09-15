"""Echo-grounding — the check that a cited ref actually came from a tool."""

from __future__ import annotations

from backend.app.services.grounding import extract_refs, verify_answer_refs
from tools.geo.place_store import InMemoryPlaceStore
from tools.refs import PlaceRecord, RecordOrigin, SourceRecord
from tools.web.memory_store import InMemorySourceStore

REAL_PLACE = "plc_a1b2c3d4e5"
REAL_SOURCE = "src_9f8e7d6c5b"
INVENTED = "plc_deadbeef00"


async def _stores() -> tuple[InMemoryPlaceStore, InMemorySourceStore]:
    places = InMemoryPlaceStore()
    await places.save(
        PlaceRecord(
            ref=REAL_PLACE,
            name="Кофемания",
            address="Москва",
            lat=55.75,
            lon=37.62,
            origin=RecordOrigin.PLACES_SEARCH,
        )
    )
    sources = InMemorySourceStore()
    await sources.save_many(
        [
            SourceRecord(
                ref=REAL_SOURCE,
                url="https://example.com/a",
                title="t",
                domain="example.com",
                snippet="s",
            )
        ]
    )
    return places, sources


def test_extract_refs_finds_them_inside_prose():
    places, sources = extract_refs(
        f"Go to {REAL_PLACE}, details in {REAL_SOURCE}. Also {REAL_PLACE} again."
    )
    assert places == [REAL_PLACE]  # de-duplicated
    assert sources == [REAL_SOURCE]


def test_extract_refs_ignores_lookalikes():
    # Truncated / wrong-charset refs are not refs.
    places, sources = extract_refs("plc_a1b2c3d4 and plc_ZZZZZZZZZZ and 55.75,37.62")
    assert places == []
    assert sources == []


async def test_answer_citing_real_refs_is_grounded():
    places, sources = await _stores()

    report = await verify_answer_refs(
        f"Try {REAL_PLACE}. Source: {REAL_SOURCE}.",
        place_store=places,
        source_store=sources,
    )

    assert report.grounded is True
    assert report.total_refs == 2
    assert report.unknown_refs == []


async def test_invented_ref_is_caught():
    # This is the whole point: the model cannot fabricate a place.
    places, sources = await _stores()

    report = await verify_answer_refs(
        f"Try {REAL_PLACE} and {INVENTED}.", place_store=places, source_store=sources
    )

    assert report.grounded is False
    assert report.unknown_refs == [INVENTED]


async def test_answer_without_refs_is_vacuously_grounded():
    places, sources = await _stores()

    report = await verify_answer_refs("Prague is lovely.", place_store=places, source_store=sources)

    assert report.grounded is True
    assert report.total_refs == 0


async def test_missing_store_fails_closed():
    # Without a store we cannot prove the ref — so we must not assume it is fine.
    report = await verify_answer_refs(f"Go to {REAL_PLACE}.", place_store=None, source_store=None)

    assert report.grounded is False
    assert report.unknown_refs == [REAL_PLACE]
