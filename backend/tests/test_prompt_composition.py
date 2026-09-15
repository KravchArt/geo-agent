"""The system prompt must describe exactly the tools the model can actually call.

The tool list on the wire is built from configured providers, so it shrinks when
a key is missing. If the prompt still documents the missing tool, the model tries
it and burns a turn on "unknown tool" — so both come from the same source.
"""

from __future__ import annotations

import re

import pytest

from backend.app.prompts.censor import CENSOR_SYSTEM_PROMPT
from backend.app.prompts.orchestrator import (
    DOCUMENTED_TOOLS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_orchestrator_prompt,
)
from backend.app.prompts.scoper import SCOPER_SYSTEM_PROMPT


def test_all_documented_tools_are_discoverable():
    # Guards the split: if the prompt's section headings are reformatted, this
    # fails loudly instead of silently producing a prompt with no tools.
    assert set(DOCUMENTED_TOOLS) == {"places_search", "routing_tool", "web_search"}


@pytest.mark.parametrize(
    "available",
    [
        ["web_search"],
        ["places_search"],
        ["places_search", "routing_tool"],
        list(DOCUMENTED_TOOLS),
    ],
)
def test_unavailable_tools_are_never_mentioned(available):
    prompt = build_orchestrator_prompt(available)

    for name in available:
        assert f"`{name}`" in prompt
    for missing in set(DOCUMENTED_TOOLS) - set(available):
        assert f"`{missing}`" not in prompt


def test_cross_references_are_pruned_but_useful_guidance_survives():
    # A cross-tool sentence must disappear without removing the independent
    # guidance for the tool that remains available.
    prompt = build_orchestrator_prompt(["web_search"])

    assert "`places_search`" not in prompt
    assert "Use for fresh or non-map facts" in prompt


def test_places_section_does_not_duplicate_web_search_scope() -> None:
    prompt = " ".join(build_orchestrator_prompt(list(DOCUMENTED_TOOLS)).split())

    assert "Use for place discovery and map facts" in prompt
    assert "`web_search` only for non-map or temporary facts" not in prompt
    assert "Use for fresh or non-map facts" in prompt


def test_sections_are_renumbered_without_gaps():
    prompt = build_orchestrator_prompt(["web_search"])

    assert "5.1. `web_search`" in prompt
    assert "5.2. Choosing and combining tools" in prompt


def test_no_tools_says_so_instead_of_listing_none():
    prompt = build_orchestrator_prompt([])

    for name in DOCUMENTED_TOOLS:
        assert f"`{name}`" not in prompt
    assert "No tools are available" in prompt
    # The behavioural rules still apply.
    assert "Write the final answer" in prompt


def test_empty_tool_results_are_not_retried_with_reformulated_arguments() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "If a tool returns no result or `not_found`" in prompt
    assert "paraphrasing, broadening, narrowing, or otherwise reformulating" in prompt
    assert "State that nothing was found" in prompt
    assert "only after the user provides new information" in prompt


def test_clarification_selection_uses_the_schema_indicated_field() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "pass the chosen value unchanged in its schema-specified field" in prompt
    assert "not necessarily a ref" in prompt


def test_the_agent_role_survives_every_combination():
    for available in ([], ["web_search"], list(DOCUMENTED_TOOLS)):
        prompt = build_orchestrator_prompt(available)
        assert "You are a geo agent" in prompt


def test_final_answer_style_is_lively_and_allows_recommendation_language() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "natural, lively, and engaging style" in prompt
    assert '"popular" or "recommended" is allowed' in prompt
    assert "concrete factual claims grounded in tool results" in prompt
    assert "neutral style" not in prompt


def test_system_prompts_contain_no_cyrillic_text() -> None:
    assert re.search(r"[А-Яа-яЁё]", ORCHESTRATOR_SYSTEM_PROMPT) is None
    assert re.search(r"[А-Яа-яЁё]", SCOPER_SYSTEM_PROMPT) is None
    assert re.search(r"[А-Яа-яЁё]", CENSOR_SYSTEM_PROMPT) is None
    for available in ([], ["places_search"], list(DOCUMENTED_TOOLS)):
        assert re.search(r"[А-Яа-яЁё]", build_orchestrator_prompt(available)) is None


def test_final_answer_contract_is_compact_and_always_present() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "user-facing Markdown, without a JSON envelope" in prompt
    assert "`[[plc_...]]`" in prompt
    assert "Include refs only for places that are actually present" in prompt


def test_tool_sections_defer_argument_contracts_to_advertised_schemas():
    prompt = build_orchestrator_prompt(list(DOCUMENTED_TOOLS))

    assert "The advertised JSON schema is the authoritative contract" not in prompt
    assert "Carry every user-stated constraint into the first tool call" not in prompt
    assert "Arguments:" not in prompt
    assert "Rules for `mode=" not in prompt


def test_routing_calling_details_are_not_duplicated_in_the_prompt() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search", "routing_tool"]).split())

    assert "named organisations, and POIs directly to this tool" not in prompt
    assert "it resolves every textual route point itself" not in prompt
    assert "Do not call `places_search` first merely to resolve a route point" not in prompt
    assert "If a valid `plc_...` ref is already available" not in prompt


def test_routing_prompt_keeps_only_tool_selection_policy() -> None:
    prompt = " ".join(build_orchestrator_prompt(["routing_tool"]).split())

    assert "measured route, travel time, route-cost ranking, or a meeting point" in prompt
    assert "Never estimate route measurements manually" in prompt
    assert "`query` and `area`" not in prompt
    assert "`departure_time`" not in prompt
    assert "`length_m`" not in prompt


def test_web_candidates_are_batch_resolved_with_correlation_ids() -> None:
    prompt = build_orchestrator_prompt(["web_search", "places_search", "routing_tool"])

    assert "`places_search(mode=resolve)`" in prompt
    assert "`client_id`" in prompt
    assert "Keep price, menu, availability" not in prompt


def test_web_facts_require_minimal_primary_source_citations() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search"]).split())

    assert "Use for fresh or non-map facts" in prompt
    assert "cite each fact supported by web data with its `src_...` ref" not in prompt


def test_web_attribution_is_not_presented_as_direct_map_data() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert (
        "Do not present a source's attribution as data directly verified by a map tool"
        not in prompt
    )


def test_places_search_call_details_are_not_duplicated_in_system_prompt() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "first make one `places_search` call that includes every criterion" not in prompt
    assert "Choose the mode by the expected result" not in prompt
    assert "For every generic place-type request, set `category`" not in prompt


def test_follow_up_named_place_reuses_conversation_city_and_resolves_first() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "specific named establishment is itself the requested result" in prompt
    assert "use `places_search(mode=resolve)` first for its map details" in prompt
    assert "only if resolution fails or the requested field is unavailable" in prompt


def test_named_nearby_anchor_goes_directly_to_near_mode() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "only the anchor for discovering other places nearby" in prompt
    assert "call `places_search(mode=near)` directly" in prompt
    assert "never call resolve solely to prepare an anchor ref" in prompt


def test_follow_up_inherits_an_established_conversation_locality() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Conversation locality continuity" in prompt
    assert "This is established evidence from the dialogue, not inventing a city" in prompt
    assert "most recent city that the user explicitly stated" in prompt
    assert "do not ask for it again" in prompt


def test_web_first_recommendations_use_only_evidenced_candidates() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "call `web_search` first" in prompt
    assert "Do not add a date, year, or time range the user did not request" in prompt
    assert "only candidates explicitly named by the evidence" in prompt
    assert "rather than rediscovering places" in prompt
    assert "extract up to ten distinct evidenced candidates" in prompt
    assert "aim to return five clearly qualifying, successfully resolved places" in prompt
    assert "rather than padding it with unresolved or wrong-type organisations" in prompt
    assert "must have explicit evidence of a retail function" in prompt
    assert "the only permitted `places_search` call" in prompt
    assert "one `mode=resolve` batch of at most ten candidates" in prompt
    assert "additional or different candidates" in prompt
    assert "return the verified subset even if fewer places qualify" in prompt
    assert "do not use `mode=area` or `mode=near`" in prompt
    assert "cannot preserve the web-only criterion" in prompt
    assert "resolved place type or category matches the entity type" in prompt


def test_web_facts_keep_their_source_citations_after_place_resolution() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "Every final-answer claim that depends on web evidence" in prompt
    assert "exact returned `[[src_...]]` ref" in prompt
    assert "does not replace the citation" in prompt


def test_non_map_identity_and_style_criteria_use_web_search_first() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "ownership and identity qualifiers" in prompt
    assert '"family-run artisan workshops in Florence"' in prompt
    assert "call `web_search` first" in prompt
    assert "Do not approximate `artisan workshop` with `gift_shop` or `arts_centre`" in prompt
    assert "repeated `places_search` calls" in prompt


def test_successful_web_search_is_used_before_a_distinct_follow_up_search() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "After successful `web_search`, use its evidence or its named candidates" in prompt
    assert "Do not repeat or rephrase that search" in prompt
    assert "distinct unresolved sub-question" in prompt


def test_single_criterion_recommendation_has_a_fixed_tool_sequence() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "call `web_search` first with every user constraint" in prompt
    assert "Batch evidenced candidates through `places_search(mode=resolve)`" in prompt
    assert "join map facts to their evidence" in prompt


def test_places_search_mode_details_are_owned_by_tool_schema() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Choose the mode by the expected result" not in prompt
    assert "select_anchor clarification" not in prompt


def test_resolve_does_not_repeat_processed_client_ids() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Reuse a completed resolved, not_found, or ambiguous outcome" not in prompt


def test_same_name_ambiguous_branches_are_presented_as_addresses() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Same-name ambiguous branches are addresses under one organisation heading" not in prompt


def test_cityless_named_location_is_resolved_before_places_search() -> None:
    prompt = " ".join(build_orchestrator_prompt(["web_search", "places_search"]).split())

    assert "Never invent a city from general knowledge" in prompt
    assert "call `web_search` first with the original location phrase" in prompt
    assert "must not override a separately named street or address" in prompt
    assert prompt.index("Choosing and combining tools") < prompt.index(
        "call `web_search` first with the original location phrase"
    )


def test_city_context_rules_survive_without_web_search() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Never invent a city from general knowledge" in prompt
    assert "ACTIVE CONVERSATION GEO CONTEXT" in prompt
    assert "must not override a separately named street or address" in prompt
    assert "original location phrase" not in prompt


def test_places_search_prompt_enforces_free_text_language_preservation() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Language preservation applies before tool execution" in prompt
    assert "Never translate, transliterate, anglicize, localize" in prompt
    assert "HARD LANGUAGE CONSTRAINT" in prompt
    assert "`category` may contain its required English enum" in prompt
    assert "An English or Latin-script replacement" in prompt
    assert "Immediately before emitting the function call" in prompt
    assert "canonical English value only in `category`" in prompt
    assert "München" not in prompt


def test_rating_filter_falls_back_to_web_search_outside_twogis_coverage() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search", "web_search"]).split())

    assert "`unsupported_filter`" in prompt
    assert "call `web_search` with the same place type, rating, and locality" in prompt


def test_area_clarification_reuses_selected_ref() -> None:
    prompt = build_orchestrator_prompt(["places_search"])

    assert "`area_ref` for `select_area`" not in prompt
    assert "never convert it to an `area` discovery" not in prompt


def test_area_argument_details_are_owned_by_tool_schema() -> None:
    prompt = " ".join(build_orchestrator_prompt(["places_search"]).split())

    assert "Pass locality text only for the first search there" not in prompt
    assert "Omit area with an already resolved near plc_ ref" not in prompt


def test_twogis_area_clarification_reuses_qualified_city_query() -> None:
    prompt = build_orchestrator_prompt(["places_search"])

    assert "`select_area_query`, use the exact selected value as `city`" not in prompt
