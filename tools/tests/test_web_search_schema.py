"""web_search — schema-level tests (no tool, no API)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tools.web.search import (
    MAX_SNIPPET_CHARS,
    WEB_SEARCH_LLM_PARAMETERS,
    WEB_SEARCH_SPEC,
    TimeRange,
    WebResult,
    WebSearchInput,
    WebSearchOutput,
    WebTopic,
)

SRC = "src_9f8e7d6c5b"


def test_json_schema_is_generated_for_backend_validation():
    """The complete Pydantic schema remains the authoritative validator."""

    schema = WebSearchInput.model_json_schema()
    assert set(schema["properties"]) == {
        "query",
        "topic",
        "time_range",
        "include_domains",
        "exclude_domains",
    }
    assert schema["required"] == ["query"]
    assert "subject, location, and time constraint" in schema["properties"]["query"]["description"]
    assert "publication-recency" in schema["properties"]["time_range"]["description"]
    # Provider tuning must not leak into the model's schema.
    for provider_knob in ("search_depth", "include_answer", "include_raw_content", "api_key"):
        assert provider_knob not in schema["properties"]


def test_tool_spec_points_to_web_search_contract():
    """Verify that tool spec points to web search contract."""

    assert WEB_SEARCH_SPEC.name == "web_search"
    assert WEB_SEARCH_SPEC.input_model is WebSearchInput
    assert WEB_SEARCH_SPEC.output_model is WebSearchOutput
    assert WEB_SEARCH_SPEC.description
    assert "untrusted data" in WEB_SEARCH_SPEC.description
    assert "directly verified by a map tool" in WEB_SEARCH_SPEC.description
    assert "query_correctness" in WEB_SEARCH_SPEC.eval_metrics
    assert "citation_correctness" in WEB_SEARCH_SPEC.eval_metrics
    assert WEB_SEARCH_SPEC.answer_fields == ("results",)

    schema = WEB_SEARCH_SPEC.input_model.model_json_schema()
    assert schema["required"] == ["query"]
    assert "topic" in schema["properties"]
    assert "time_range" in schema["properties"]
    assert "include_domains" in schema["properties"]
    assert "exclude_domains" in schema["properties"]


def test_llm_contract_is_compact_but_keeps_the_calling_convention():
    """The model gets concise instructions; Pydantic keeps strict validation."""

    assert WEB_SEARCH_SPEC.llm_parameters == WEB_SEARCH_LLM_PARAMETERS
    assert set(WEB_SEARCH_LLM_PARAMETERS["properties"]) == {
        "query",
        "topic",
        "time_range",
        "include_domains",
        "exclude_domains",
    }
    assert WEB_SEARCH_LLM_PARAMETERS["required"] == ["query"]
    assert "$defs" not in WEB_SEARCH_LLM_PARAMETERS
    assert WEB_SEARCH_LLM_PARAMETERS["properties"]["topic"]["enum"] == [
        topic.value for topic in WebTopic
    ]
    assert WEB_SEARCH_LLM_PARAMETERS["properties"]["time_range"]["enum"] == [
        time_range.value for time_range in TimeRange
    ]


def test_query_is_required_and_normalized():
    """Verify that query is required and normalized."""

    a = WebSearchInput.model_validate({"query": "  музей   космонавтики  ремонт "})
    b = WebSearchInput.model_validate({"query": "музей космонавтики ремонт"})
    assert a.model_dump() == b.model_dump()

    with pytest.raises(ValidationError):
        WebSearchInput.model_validate({"query": "   "})


def test_defaults_are_conservative():
    """Verify that defaults are conservative."""

    params = WebSearchInput.model_validate({"query": "что посмотреть в Праге"})
    assert params.topic is WebTopic.GENERAL
    assert params.time_range is None


def test_result_count_is_internal_and_cannot_be_set_by_the_model():
    """Every provider uses the fixed internal cap; callers cannot override it."""

    assert "max_results" not in WebSearchInput.model_fields
    assert "max_results" not in WEB_SEARCH_LLM_PARAMETERS["properties"]
    with pytest.raises(ValidationError):
        WebSearchInput.model_validate({"query": "x", "max_results": 5})


def test_domains_are_normalized_to_hosts():
    """Verify that domains are normalized to hosts."""

    params = WebSearchInput.model_validate(
        {"query": "часы работы", "include_domains": ["https://mos.ru/museum/", "MOS.RU"]}
    )
    assert params.include_domains == ["mos.ru", "mos.ru"]


def test_a_non_domain_is_rejected():
    """Verify that a non domain is rejected."""

    with pytest.raises(ValidationError, match="not a valid domain"):
        WebSearchInput.model_validate({"query": "x", "exclude_domains": ["не домен"]})


def test_news_topic_and_time_range_are_available_for_events():
    """Verify that news topic and time range are available for events."""

    params = WebSearchInput.model_validate(
        {"query": "события в Москве", "topic": "news", "time_range": "week"}
    )
    assert params.topic is WebTopic.NEWS
    assert params.time_range is TimeRange.WEEK


def test_results_are_refs_never_urls_or_provider_metadata():
    """Verify that results are refs never URLs or provider metadata."""

    assert set(WebResult.model_fields) == {
        "ref",
        "title",
        "domain",
        "snippet",
        "published_date",
    }
    assert set(WebSearchOutput.model_fields) == {"query", "results"}


def test_output_round_trips_as_json():
    """Verify that output round trips as JSON."""

    output = WebSearchOutput(
        query="музей космонавтики ремонт",
        results=[
            WebResult(
                ref=SRC,
                title="Музей космонавтики закрыт на реконструкцию",
                domain="kosmo-museum.ru",
                snippet="С 1 сентября музей закрыт на реконструкцию до конца года.",
                published_date=date(2026, 6, 30),
            )
        ],
    )
    assert WebSearchOutput.model_validate(output.model_dump(mode="json")) == output


def test_snippet_must_be_non_empty_and_bounded():
    """Verify that snippet must be non empty and bounded."""

    common = {
        "ref": SRC,
        "title": "Источник",
        "domain": "example.com",
    }

    with pytest.raises(ValidationError):
        WebResult.model_validate({**common, "snippet": ""})

    with pytest.raises(ValidationError):
        WebResult.model_validate({**common, "snippet": "x" * (MAX_SNIPPET_CHARS + 1)})


def test_a_place_ref_is_not_a_source():
    """Verify that a place ref is not a source."""

    with pytest.raises(ValidationError):
        WebResult(ref="plc_a1b2c3d4e5", title="x", domain="a.ru", snippet="y")


def test_empty_result_is_not_an_error():
    """Verify that empty result is not an error."""

    empty = WebSearchOutput(query="совсем ничего")
    assert empty.results == []


def test_tool_description_explains_not_found_handling():
    assert "`status=NO_RESULTS`" in WEB_SEARCH_SPEC.description
    assert "do not retry or reformulate" in WEB_SEARCH_SPEC.description
    assert "another appropriate available tool" in WEB_SEARCH_SPEC.description
    assert "no answer was found" in WEB_SEARCH_SPEC.description
