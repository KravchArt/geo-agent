"""places_search — schema-level tests (no tool, no API)."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from tools.geo.places_search import (
    PLACES_SEARCH_SPEC,
    Place,
    PlaceCategory,
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
)
from tools.geo.places_search.category_normalization import normalized_category_alias_keys
from tools.geo.places_search.schemas import PLACES_SEARCH_LLM_PARAMETERS

ANCHOR = "plc_a1b2c3d4e5"
AREA = "plc_b2c3d4e5f6"
PLACE = "plc_c3d4e5f6a7"


def test_model_facing_places_schema_contains_no_cyrillic_text() -> None:
    serialized = json.dumps(PLACES_SEARCH_LLM_PARAMETERS, ensure_ascii=False)
    assert re.search(r"[А-Яа-яЁё]", serialized) is None


def test_model_facing_places_schema_preserves_free_text_language_and_script() -> None:
    properties = PLACES_SEARCH_LLM_PARAMETERS["properties"]

    assert "Only `category` gets the canonical English enum" in properties["query"]["description"]
    assert (
        "never translate, transliterate, anglicize, or localize"
        in (properties["area"]["description"])
    )
    assert (
        "never translate, transliterate, anglicize, or localize"
        in (properties["near"]["description"])
    )

    organisation_properties = properties["organisations"]["items"]["properties"]
    assert (
        "source spelling, language, and script" in (organisation_properties["name"]["description"])
    )
    assert (
        "Never translate or transliterate it"
        in (organisation_properties["address_hint"]["description"])
    )

    assert "`category` is the only field normalized to an English enum" in (
        PLACES_SEARCH_SPEC.description
    )
    assert "compare every free-text argument with its source text" in (
        PLACES_SEARCH_SPEC.description
    )


def test_json_schema_is_generated_for_backend_validation():
    """The complete Pydantic schema remains the authoritative validator."""

    schema = PlacesSearchInput.model_json_schema()
    assert set(schema["properties"]) == {
        "mode",
        "query",
        "category",
        "area",
        "near",
        "organisations",
        "radius_m",
        "open_24h",
        "open_now",
        "min_rating",
        "limit",
    }
    assert set(schema["required"]) == {"mode"}
    min_rating = schema["properties"]["min_rating"]
    numeric_rating_schema = next(
        option for option in min_rating["anyOf"] if option.get("type") == "number"
    )
    assert numeric_rating_schema["minimum"] == 0
    assert numeric_rating_schema["maximum"] == 5
    assert "2GIS coverage" in min_rating["description"]
    assert "web_search" in min_rating["description"]
    assert "raw OpenStreetMap tags" in schema["properties"]["category"]["description"]
    assert "open now" in schema["properties"]["open_24h"]["description"]
    open_now_description = schema["properties"]["open_now"]["description"]
    assert "tool execution time" in open_now_description
    assert "Keep phrases such as 'open now' out of query" in open_now_description
    assert "distinct from open_24h" in open_now_description
    assert "Already processed search target" in schema["properties"]["query"]["description"]
    assert "not the user's full sentence" in schema["properties"]["query"]["description"]
    assert "never translate or transliterate it" in schema["properties"]["query"]["description"]
    assert "Locality scope" in schema["properties"]["area"]["description"]
    assert "never translate or transliterate them" in schema["properties"]["area"]["description"]
    near_description = schema["properties"]["near"]["description"]
    assert "never reduce it to a bare proper name" in near_description
    assert "never translate or transliterate it" in near_description
    assert "Manhattan district" in near_description
    assert "Oxford Circus station" in near_description


def test_tool_spec_points_to_places_search_contract():
    """Verify that tool spec points to places search contract."""

    assert PLACES_SEARCH_SPEC.name == "places_search"
    assert PLACES_SEARCH_SPEC.input_model is PlacesSearchInput
    assert PLACES_SEARCH_SPEC.output_model is PlacesSearchOutput
    assert PLACES_SEARCH_SPEC.description
    assert "ordinary place discovery" in PLACES_SEARCH_SPEC.description
    assert "authoritative contract" in PLACES_SEARCH_SPEC.description
    assert "cannot filter by average check, price or budget" in PLACES_SEARCH_SPEC.description
    assert "only supported attribute filters are opening hours" in PLACES_SEARCH_SPEC.description
    assert "rating filtering requires 2GIS coverage" in PLACES_SEARCH_SPEC.description
    assert "use only `mode=resolve` on those candidates" in PLACES_SEARCH_SPEC.description
    assert "resolve them in exactly one batch" in PLACES_SEARCH_SPEC.description
    assert "same web criterion with additional or different candidates" in (
        PLACES_SEARCH_SPEC.description
    )
    assert "verified subset even when it contains fewer results" in PLACES_SEARCH_SPEC.description
    assert "Never use `mode=area` or `mode=near`" in PLACES_SEARCH_SPEC.description
    assert "place type or category matches the entity type" in PLACES_SEARCH_SPEC.description
    assert "`query` is required" in PLACES_SEARCH_SPEC.description
    assert "Before every places_search call, classify the target" in PLACES_SEARCH_SPEC.description
    assert "always set the matching `category` enum" in PLACES_SEARCH_SPEC.description
    assert "query='pharmacies' and category='pharmacy'" in PLACES_SEARCH_SPEC.description
    assert "query='Starbucks' with no category" in PLACES_SEARCH_SPEC.description
    assert "it is only the anchor for discovering other places" in PLACES_SEARCH_SPEC.description
    assert "Never make a separate mode=resolve call solely" in PLACES_SEARCH_SPEC.description
    assert "near='Cafe Molot'" in PLACES_SEARCH_SPEC.description
    assert "category='pharmacy', and radius_m=3000" in PLACES_SEARCH_SPEC.description
    assert "make one places_search call per category" in PLACES_SEARCH_SPEC.description
    assert "never merge multiple categories into one query" in PLACES_SEARCH_SPEC.description
    assert "original language and script" in PLACES_SEARCH_SPEC.description
    assert "solely to obtain endpoint refs for a route" in PLACES_SEARCH_SPEC.description
    assert "routing_tool resolves textual route endpoints" in PLACES_SEARCH_SPEC.description
    assert "When the tool returns one or more results, use them for the part" in (
        PLACES_SEARCH_SPEC.description
    )
    assert "do not call places_search again with the same search intent" in (
        PLACES_SEARCH_SPEC.description.lower()
    )
    assert "distinct unresolved part" in PLACES_SEARCH_SPEC.description
    assert "materially different search parameters" in PLACES_SEARCH_SPEC.description
    assert "search_mode_accuracy" in PLACES_SEARCH_SPEC.eval_metrics
    assert PLACES_SEARCH_SPEC.answer_fields == (
        "places",
        "resolved.place",
        "resolved.options",
    )

    schema = PLACES_SEARCH_SPEC.input_model.model_json_schema()
    assert set(schema["properties"]) == {
        "mode",
        "query",
        "category",
        "area",
        "near",
        "organisations",
        "radius_m",
        "open_24h",
        "open_now",
        "min_rating",
        "limit",
    }


def test_llm_contract_is_compact_but_keeps_the_calling_convention():
    """The model gets concise instructions; Pydantic keeps the strict rules."""

    assert PLACES_SEARCH_SPEC.llm_parameters == PLACES_SEARCH_LLM_PARAMETERS
    assert set(PLACES_SEARCH_LLM_PARAMETERS["properties"]) == {
        "mode",
        "query",
        "category",
        "area",
        "near",
        "organisations",
        "radius_m",
        "open_24h",
        "open_now",
        "min_rating",
        "limit",
    }
    assert PLACES_SEARCH_LLM_PARAMETERS["required"] == ["mode"]
    assert PLACES_SEARCH_LLM_PARAMETERS["allOf"] == [
        {
            "if": {
                "properties": {"mode": {"enum": ["area", "near"]}},
                "required": ["mode"],
            },
            "then": {"required": ["query"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "near"}},
                "required": ["mode"],
            },
            "then": {"required": ["near"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "area"}},
                "required": ["mode"],
            },
            "then": {"required": ["area"]},
        },
        {
            "if": {
                "properties": {"mode": {"const": "resolve"}},
                "required": ["mode"],
            },
            "then": {"required": ["area", "organisations"]},
        },
    ]
    assert "$defs" not in PLACES_SEARCH_LLM_PARAMETERS
    assert PLACES_SEARCH_LLM_PARAMETERS["properties"]["category"]["enum"] == [
        category.value for category in PlaceCategory
    ]
    assert (
        "Required for area and near modes"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["query"]["description"])
    )
    assert (
        "Use `area` to discover a set of places or branches in a city"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["mode"]["description"])
    )
    assert (
        "where the locality has 2GIS coverage"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["min_rating"]["description"])
    )
    assert (
        "Resolution selects the first provider-ranked candidate"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["organisations"]["description"])
    )
    assert (
        "anchor ambiguity belongs to near mode"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["organisations"]["description"])
    )
    assert (
        "selected value as area"
        in (PLACES_SEARCH_LLM_PARAMETERS["properties"]["area"]["description"])
    )
    mode_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["mode"]["description"]
    assert "Choose by the expected result" in mode_description
    assert "find Starbucks stores in Chicago" in mode_description
    assert "phone of Starbucks on Oxford Street" in mode_description
    query_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["query"]["description"]
    assert "ownership, provenance" in query_description
    assert "must not be simulated by reformulating query across repeated calls" in query_description
    category_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["category"]["description"]
    assert "Never choose the nearest enum" in category_description
    assert "`artisan workshop` is neither `gift_shop` nor `arts_centre`" in category_description
    assert "make one places_search call per category" in category_description
    assert "exactly one category and its matching generic query" in category_description
    assert "never merge multiple categories into one query" in category_description
    area_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["area"]["description"]
    assert "reuse its valid ref" in area_description
    assert "area='München', not area='Munich'" in area_description
    assert "Omit area with an already resolved near plc_ ref" in area_description
    near_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["near"]["description"]
    assert "present the select_anchor clarification" in near_description
    assert "belongs directly in this field" in near_description
    assert "resolves textual anchors internally" in near_description
    min_rating_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["min_rating"]["description"]
    assert "only when the user requests a rating threshold" in min_rating_description
    organisations_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["organisations"][
        "description"
    ]
    assert "never convert it to area discovery" in organisations_description
    assert "copy that qualifier into `address_hint`" in organisations_description
    assert "Never resolve a bare organisation name" in organisations_description
    assert "omit `address_hint` only when the evidence provides none" in organisations_description
    assert "include all chosen candidates in one resolve call" in organisations_description
    assert "never call resolve again for the same criterion" in organisations_description
    assert "return the verified subset" in organisations_description
    address_hint_description = PLACES_SEARCH_LLM_PARAMETERS["properties"]["organisations"]["items"][
        "properties"
    ]["address_hint"]["description"]
    assert "Required whenever evidence provides" in address_hint_description
    assert "omit only when no such qualifier is available" in address_hint_description


def test_area_mode_requires_unified_area():
    """Area accepts locality text or a reusable ref in one field."""

    PlacesSearchInput.model_validate({"mode": "area", "query": "кофейни", "area": "Москва"})
    PlacesSearchInput.model_validate({"mode": "area", "query": "кофейни", "area": AREA})

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate({"mode": "area", "query": "кофейни"})

    assert exc_info.value.errors()[0]["type"] == "places_search_area_required"


def test_near_mode_requires_a_ref_or_textual_anchor():
    """Verify that near mode requires a ref or textual anchor."""

    PlacesSearchInput.model_validate({"mode": "near", "query": "кофейни", "near": ANCHOR})
    PlacesSearchInput.model_validate(
        {
            "mode": "near",
            "query": "кофейни",
            "near": "Красная площадь",
            "area": "Москва",
        },
    )
    PlacesSearchInput.model_validate(
        {
            "mode": "near",
            "query": "кофейни",
            "near": "Красная площадь",
            "area": AREA,
        },
    )

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate(
            {
                "mode": "near",
                "query": "кофейни",
                "near": "Красная площадь",
            }
        )

    assert exc_info.value.errors()[0]["type"] == "places_search_near_area_required"

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate({"mode": "near", "query": "кофейни"})

    assert exc_info.value.errors()[0]["type"] == "places_search_near_anchor_required"

    PlacesSearchInput.model_validate(
        {"mode": "near", "query": "кофейни", "near": "Красная площадь", "area": "Москва"}
    )


def test_near_mode_rejects_coordinates_outright():
    """Verify that near mode rejects coordinates outright."""

    with pytest.raises(ValidationError):
        PlacesSearchInput.model_validate(
            {"mode": "near", "query": "кофейни", "near": "55.7539,37.6208"}
        )


def test_near_mode_existing_anchor_rejects_redundant_locality_scope():
    """An existing anchor ref already carries its backend locality."""

    for redundant_scope in ({"city": "Москва"}, {"area_ref": AREA}):
        with pytest.raises(ValidationError) as exc_info:
            PlacesSearchInput.model_validate(
                {
                    "mode": "near",
                    "query": "кофейни",
                    "near": ANCHOR,
                    **redundant_scope,
                }
            )

        assert exc_info.value.errors()[0]["type"] == "places_search_near_ref_scope_forbidden"


def test_area_mode_rejects_an_anchor():
    """Verify that area mode rejects an anchor."""

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate(
            {"mode": "area", "query": "кофейни", "city": "Москва", "near": ANCHOR}
        )

    assert exc_info.value.errors()[0]["type"] == "places_search_area_near_forbidden"


def test_area_mode_rejects_a_textual_anchor():
    """Verify that area mode rejects a textual anchor."""

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate(
            {
                "mode": "area",
                "query": "кофейни",
                "city": "Москва",
                "near_query": "Красная площадь",
            },
        )

    assert exc_info.value.errors()[0]["type"] == "places_search_area_near_forbidden"


def test_whitespace_is_normalized_so_equivalent_calls_match():
    """Verify that whitespace is normalized so equivalent calls match."""

    a = PlacesSearchInput.model_validate({"mode": "area", "query": "  кофейни ", "city": "Москва "})
    b = PlacesSearchInput.model_validate({"mode": "area", "query": "кофейни", "city": "Москва"})
    assert a.model_dump() == b.model_dump()


def test_model_can_select_a_controlled_category_hint():
    """Verify category hints are typed and optional for name searches."""

    category_search = PlacesSearchInput.model_validate(
        {
            "mode": "area",
            "query": "кофейни",
            "category": "coffee_shop",
            "city": "Москва",
        }
    )
    name_search = PlacesSearchInput.model_validate(
        {
            "mode": "area",
            "query": "Кофикс",
            "city": "Москва",
        }
    )

    assert category_search.category is PlaceCategory.COFFEE_SHOP
    assert name_search.category is None

    with pytest.raises(ValidationError):
        PlacesSearchInput.model_validate(
            {
                "mode": "area",
                "query": "что-нибудь",
                "category": "amenity=cafe",
                "city": "Москва",
            }
        )


@pytest.mark.parametrize(
    ("query", "model_category", "expected"),
    [
        ("кафе", "coffee_shop", PlaceCategory.CAFE),
        ("кофейни", "cafe", PlaceCategory.COFFEE_SHOP),
        ("пабы", "bar", PlaceCategory.PUB),
        ("бары", "pub", PlaceCategory.BAR),
        ("пиццерии", "restaurant", PlaceCategory.PIZZERIA),
        ("фастфуд", "restaurant", PlaceCategory.FAST_FOOD),
        ("аптеки", "clinic", PlaceCategory.PHARMACY),
        ("гостиницы", "hostel", PlaceCategory.HOTEL),
        ("bookstores", "library", PlaceCategory.BOOKSTORE),
    ],
)
def test_obvious_generic_query_corrects_model_category(
    query: str,
    model_category: str,
    expected: PlaceCategory,
) -> None:
    args = PlacesSearchInput(
        mode="area",
        query=query,
        category=model_category,
        city="Москва",
    )

    assert args.category is expected


def test_category_normalization_covers_every_advertised_category() -> None:
    assert normalized_category_alias_keys() == frozenset(
        category.value for category in PlaceCategory
    )


def test_missing_category_keeps_explicit_named_search_semantics() -> None:
    args = PlacesSearchInput(mode="area", query="Кафе", city="Москва")

    assert args.category is None


def test_generic_discovery_defaults_to_ten_places():
    args = PlacesSearchInput.model_validate(
        {
            "mode": "area",
            "query": "кофейни",
            "category": "coffee_shop",
            "city": "Москва",
        }
    )

    assert args.limit == 10
    assert "default 10" in PLACES_SEARCH_LLM_PARAMETERS["properties"]["limit"]["description"]


@pytest.mark.parametrize(
    ("category", "requested_limit", "expected_limit"),
    [
        (None, 3, 3),
        (None, 20, 5),
        (PlaceCategory.COFFEE_SHOP, 20, 20),
    ],
)
def test_named_searches_are_capped_without_limiting_category_discovery(
    category: PlaceCategory | None,
    requested_limit: int,
    expected_limit: int,
) -> None:
    args = PlacesSearchInput(
        mode="area",
        query="Кофикс" if category is None else "кофейни",
        category=category,
        city="Москва",
        limit=requested_limit,
    )

    assert args.result_limit == expected_limit


def test_model_can_select_flower_shop_category():
    category_search = PlacesSearchInput.model_validate(
        {
            "mode": "area",
            "query": "магазины цветов",
            "category": "flower_shop",
            "city": "Москва",
        }
    )

    assert category_search.category is PlaceCategory.FLOWER_SHOP


def test_open_now_is_distinct_from_open_24h() -> None:
    args = PlacesSearchInput.model_validate(
        {
            "mode": "area",
            "query": "кофейни",
            "city": "Москва",
            "open_now": True,
        }
    )

    assert args.open_now is True
    assert args.open_24h is False


def test_unknown_field_is_rejected():
    """Verify that unknown field is rejected."""

    with pytest.raises(ValidationError):
        PlacesSearchInput.model_validate(
            {"mode": "area", "query": "кофейни", "city": "Москва", "sort_by": "rating"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("radius_m", 50), ("radius_m", 50_000), ("limit", 0), ("limit", 100)],
)
def test_bounds_are_enforced(field, value):
    """Verify that bounds are enforced."""

    args = {"mode": "near", "query": "кафе", "near": ANCHOR, field: value}
    with pytest.raises(ValidationError):
        PlacesSearchInput.model_validate(args)


def test_output_carries_refs_and_no_coordinates():
    """Verify that output carries refs and no coordinates."""

    # Places come back as refs, ready to be handed straight to routing_tool.
    assert "lat" not in Place.model_fields
    assert "lon" not in Place.model_fields
    assert "url" not in Place.model_fields
    is_open_now_schema = Place.model_json_schema()["properties"]["is_open_now"]
    assert "confirmed the place was open" in is_open_now_schema["description"]
    assert "True is guaranteed" in is_open_now_schema["description"]

    output = PlacesSearchOutput(
        places=[
            Place(
                ref=PLACE,
                id="1069243553",
                name="Кофемания",
                address="Россия, Москва, Никольская улица, 10",
                categories=["Кофейня"],
                phones=["+7 (495) 786-45-64"],
                hours_text="круглосуточно",
                open_24h=True,
                is_open_now=True,
                accessibility=["wheelchair_access"],
                distance_m=200,
            )
        ],
        truncated=True,
        area=ResolvedSearchArea(
            ref=AREA,
            name="Москва",
            address="Россия, Москва",
        ),
        anchor=ANCHOR,
    )
    assert output.returned_count == 1
    # Must survive the JSONB round-trip used by tool_call.result.
    assert PlacesSearchOutput.model_validate(output.model_dump(mode="json")) == output


def test_empty_result_is_not_an_error():
    """Verify that empty result is not an error."""

    empty = PlacesSearchOutput()
    assert empty.places == []
    assert empty.area is None
    assert empty.returned_count == 0
    assert "provider_total_found" not in PlacesSearchOutput.model_fields


def test_json_schema_contains_mode_dependent_constraints():
    """Verify that JSON schema contains mode dependent constraints."""

    schema = PlacesSearchInput.model_json_schema()

    assert "allOf" in schema
    assert len(schema["allOf"]) == 3

    area_rule, near_rule, resolve_rule = schema["allOf"]

    assert area_rule["if"]["properties"]["mode"]["const"] == "area"
    assert area_rule["then"]["required"] == ["query", "area"]

    assert near_rule["if"]["properties"]["mode"]["const"] == "near"
    assert near_rule["then"]["required"] == ["query", "near"]
    assert near_rule["then"]["oneOf"][0]["properties"]["near"]["pattern"]
    assert near_rule["then"]["oneOf"][1]["required"] == ["area"]

    assert resolve_rule["if"]["properties"]["mode"]["const"] == "resolve"
    assert resolve_rule["then"]["required"] == ["organisations", "area"]


def test_resolve_mode_accepts_correlated_organisation_batch() -> None:
    args = PlacesSearchInput.model_validate(
        {
            "mode": "resolve",
            "city": "Berlin",
            "organisations": [
                {
                    "client_id": "candidate_1",
                    "name": "Alpha Restaurant",
                    "address_hint": "Mitte",
                }
            ],
        }
    )

    assert args.query is None
    assert args.organisations[0].client_id == "candidate_1"
    assert args.organisations[0].address_hint == "Mitte"


def test_resolve_mode_accepts_selected_area_ref() -> None:
    args = PlacesSearchInput.model_validate(
        {
            "mode": "resolve",
            "area_ref": "plc_a1b2c3d4e5",
            "organisations": [{"client_id": "candidate_1", "name": "Alpha Restaurant"}],
        }
    )

    assert args.city is None
    assert args.area_ref == "plc_a1b2c3d4e5"


def test_resolve_mode_treats_blank_address_hint_as_omitted() -> None:
    args = PlacesSearchInput.model_validate(
        {
            "mode": "resolve",
            "city": "Berlin",
            "organisations": [
                {
                    "client_id": "candidate_1",
                    "name": "Alpha Restaurant",
                    "address_hint": "   ",
                }
            ],
        }
    )

    assert args.organisations[0].address_hint is None


def test_near_mode_rejects_legacy_ref_and_query_together():
    """The migration adapter does not silently choose between two anchors."""

    with pytest.raises(ValidationError) as exc_info:
        PlacesSearchInput.model_validate(
            {
                "mode": "near",
                "query": "кофейни",
                "near": ANCHOR,
                "near_query": "Красная площадь",
            }
        )

    assert exc_info.value.errors()[0]["type"] == "value_error"
