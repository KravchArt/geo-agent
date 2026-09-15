"""refs — the handle format that keeps coordinates and URLs away from the model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode
from tools.refs import GeoBounds, PlaceRecord, PlaceRef, RecordOrigin, SourceRef, mint_place_ref

PLACE = "plc_a1b2c3d4e5"
SOURCE = "src_9f8e7d6c5b"


class _Place(BaseModel):
    ref: PlaceRef


class _Source(BaseModel):
    ref: SourceRef


def test_valid_refs_are_accepted():
    """Verify that valid refs are accepted."""

    assert _Place(ref=PLACE).ref == PLACE
    assert _Source(ref=SOURCE).ref == SOURCE


@pytest.mark.parametrize(
    "bad",
    [
        "plc_a1b2c3d4e",  # truncated — must FAIL, not resolve to something else
        "plc_a1b2c3d4e55",  # too long
        "plc_A1B2C3D4E5",  # uppercase
        "plc_g1b2c3d4e5",  # not hex
        "a1b2c3d4e5",  # no prefix
        "plc-a1b2c3d4e5",  # wrong separator
        "55.7539,37.6208",  # the exact thing we are trying to ban
        "",
    ],
)
def test_malformed_place_refs_are_rejected(bad):
    """Verify that malformed place refs are rejected."""

    with pytest.raises(ValidationError):
        _Place(ref=bad)


def test_a_source_ref_cannot_be_used_where_a_place_is_expected():
    """Verify that a source ref cannot be used where a place is expected."""

    # The prefix is a type tag: passing a web page where a point is expected is a
    # schema error, not a runtime surprise.
    with pytest.raises(ValidationError):
        _Place(ref=SOURCE)
    with pytest.raises(ValidationError):
        _Source(ref=PLACE)


def test_place_record_is_where_coordinates_actually_live():
    """Verify that place record is where coordinates actually live."""

    record = PlaceRecord(
        ref=PLACE,
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        lat=55.7539,
        lon=37.6208,
        kind="locality",
        locality="Москва",
        bounds=GeoBounds(
            west=36.803101,
            south=55.142174,
            east=37.967427,
            north=56.021251,
        ),
        origin=RecordOrigin.GEOCODE,
    )
    assert PlaceRecord.model_validate(record.model_dump(mode="json")) == record


def test_place_record_rejects_impossible_coordinates():
    """Verify that place record rejects impossible coordinates."""

    with pytest.raises(ValidationError):
        PlaceRecord(
            ref=PLACE,
            name="nowhere",
            address="nowhere",
            lat=91.0,
            lon=37.6208,
            origin=RecordOrigin.GEOCODE,
        )


def test_geo_bounds_reject_inverted_corners():
    """Verify that geo bounds reject inverted corners."""

    with pytest.raises(ValidationError, match="west must be less than east"):
        GeoBounds(west=38.0, south=55.0, east=37.0, north=56.0)


def test_unknown_ref_has_its_own_error_code():
    """Verify that unknown ref has its own error code."""

    # An expired or invented ref must be reported, never guessed at.
    assert ToolErrorCode.UNKNOWN_REF.value == "unknown_ref"


def test_mint_place_ref_is_deterministic():
    """Verify that mint place ref is deterministic."""

    identity = "yandex:uri:ymapsbm1://geo?oid=123"

    assert mint_place_ref(identity) == mint_place_ref(identity)


def test_mint_place_ref_normalizes_surrounding_whitespace():
    """Verify that mint place ref normalizes surrounding whitespace."""

    identity = "yandex:uri:ymapsbm1://geo?oid=123"

    assert mint_place_ref(f"  {identity}  ") == mint_place_ref(identity)


def test_mint_place_ref_creates_valid_place_ref():
    """Verify that mint place ref creates valid place ref."""

    ref = mint_place_ref("yandex:uri:ymapsbm1://geo?oid=123")

    assert _Place(ref=ref).ref == ref


def test_mint_place_ref_rejects_empty_identity():
    """Verify that mint place ref rejects empty identity."""

    with pytest.raises(ValueError, match="place identity cannot be empty"):
        mint_place_ref("   ")
