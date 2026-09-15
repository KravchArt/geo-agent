"""Tests for the place-record storage contract."""

from __future__ import annotations

from collections.abc import Sequence

from tools.geo.place_store import PlaceStore
from tools.refs import PlaceRecord, PlaceRef, RecordOrigin


class FakePlaceStore:
    def __init__(self) -> None:
        self._records: dict[str, PlaceRecord] = {}

    async def save(self, record: PlaceRecord) -> None:
        self._records[record.ref] = record

    async def save_many(self, records: Sequence[PlaceRecord]) -> None:
        self._records.update({record.ref: record for record in records})

    async def get(self, ref: PlaceRef) -> PlaceRecord | None:
        return self._records.get(ref)


async def test_fake_place_store_matches_protocol():
    """Verify that fake place store matches protocol."""

    store = FakePlaceStore()

    assert isinstance(store, PlaceStore)


async def test_place_store_saves_and_returns_record():
    """Verify that place store saves and returns record."""

    store = FakePlaceStore()
    record = PlaceRecord(
        ref="plc_a1b2c3d4e5",
        name="Красная площадь",
        address="Москва, Красная площадь",
        lat=55.7539,
        lon=37.6208,
        origin=RecordOrigin.GEOCODE,
    )

    await store.save(record)

    assert await store.get(record.ref) == record
    assert await store.get("plc_0000000000") is None


async def test_place_store_saves_a_batch_of_records():
    """Verify that place store saves a batch of records."""

    store = FakePlaceStore()
    records = [
        PlaceRecord(
            ref="plc_a1b2c3d4e5",
            name="Красная площадь",
            address="Москва, Красная площадь",
            lat=55.7539,
            lon=37.6208,
            origin=RecordOrigin.GEOCODE,
        ),
        PlaceRecord(
            ref="plc_b2c3d4e5f6",
            name="ВДНХ",
            address="Москва, проспект Мира, 119",
            lat=55.8298,
            lon=37.6337,
            origin=RecordOrigin.GEOCODE,
        ),
    ]

    await store.save_many(records)

    assert await store.get(records[0].ref) == records[0]
    assert await store.get(records[1].ref) == records[1]
