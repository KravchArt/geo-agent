"""internal geocoder — schema-level tests (no public tool, no API)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tools.geo.geocoding import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ToponymKind,
)

REF = "plc_a1b2c3d4e5"


def test_json_schema_is_generated_for_the_llm():
    """Verify that JSON schema is generated for the LLM."""

    schema = GeocodePlaceInput.model_json_schema()
    assert set(schema["properties"]) == {"query", "city", "limit", "locality_only"}
    assert schema["required"] == ["query"]


def test_query_is_required_and_non_empty():
    """Verify that query is required and non empty."""

    GeocodePlaceInput.model_validate({"query": "Красная площадь"})
    with pytest.raises(ValidationError):
        GeocodePlaceInput.model_validate({"query": "   "})
    with pytest.raises(ValidationError):
        GeocodePlaceInput.model_validate({"city": "Москва"})


def test_whitespace_is_normalized_so_equivalent_calls_match():
    """Verify that whitespace is normalized so equivalent calls match."""

    a = GeocodePlaceInput.model_validate({"query": "  Красная   площадь ", "city": " Москва "})
    b = GeocodePlaceInput.model_validate({"query": "Красная площадь", "city": "Москва"})
    assert a.model_dump() == b.model_dump()


def test_output_hands_back_a_ref_and_never_coordinates():
    """Verify that output hands back a ref and never coordinates."""

    match = PlaceMatch(
        ref=REF,
        name="Красная площадь",
        address="Россия, Москва, Красная площадь",
        kind=ToponymKind.LOCALITY,
    )
    # The whole point: there is nowhere in the model-facing schema to put a
    # coordinate, so there is nothing for the model to copy wrong.
    assert "lat" not in PlaceMatch.model_fields
    assert "lon" not in PlaceMatch.model_fields

    output = GeocodePlaceOutput(best=match, matches=[match])
    assert GeocodePlaceOutput.model_validate(output.model_dump(mode="json")) == output


def test_output_can_report_ambiguity_instead_of_guessing():
    """Verify that output can report ambiguity instead of guessing."""

    # "Пушкинская" is a metro station in several cities — the tool must offer the
    # choice, not silently pick one.
    output = GeocodePlaceOutput(
        matches=[
            PlaceMatch(ref="plc_1111111111", name="Пушкинская", address="Москва", kind="metro"),
            PlaceMatch(
                ref="plc_2222222222", name="Пушкинская", address="Санкт-Петербург", kind="metro"
            ),
        ],
        ambiguous=True,
    )
    assert output.best is None
    assert output.ambiguous is True
    assert len(output.matches) == 2


def test_nothing_found_is_not_an_error():
    """Verify that nothing found is not an error."""

    empty = GeocodePlaceOutput()
    assert empty.best is None
    assert empty.matches == []
    assert empty.ambiguous is False


def test_malformed_ref_is_rejected_in_the_output_too():
    """Verify that malformed ref is rejected in the output too."""

    with pytest.raises(ValidationError):
        PlaceMatch(ref="55.7539,37.6208", name="x", address="y")
