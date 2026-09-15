"""Blank optional settings must mean "unset", not the empty string.

An env file spells an unset optional as a bare `FOO=`, which pydantic reads as
"". Left as a value it reached httpx as `proxy=""` and the app failed to start —
so a fresh checkout that copied .env.example could not boot at all.
"""

from __future__ import annotations

import httpx
import pytest

from backend.app.config import Settings


def _settings(**overrides: object) -> Settings:
    # _env_file=None: assert the behaviour, not the developer's local .env.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "tools_http_proxy",
        "llm_http_proxy",
        "llm_api_key",
        "llm_reasoning_effort",
        "llm_reasoning_max_tokens",
        "scope_base_url",
        "scope_api_key",
        "scope_model",
        "exa_api_key",
        "firecrawl_api_key",
        "redis_password",
    ],
)
def test_blank_optional_becomes_none(field):
    assert getattr(_settings(**{field: ""}), field) is None
    assert getattr(_settings(**{field: "   "}), field) is None


def test_configured_values_survive():
    settings = _settings(tools_http_proxy="http://127.0.0.1:12334")

    assert settings.tools_http_proxy == "http://127.0.0.1:12334"


def test_agent_sampling_defaults():
    settings = _settings()

    assert settings.llm_temperature == 0.2
    assert settings.llm_top_p == 0.9
    assert settings.llm_reasoning_effort is None
    assert settings.llm_reasoning_max_tokens is None


def test_reasoning_effort_is_configurable() -> None:
    assert _settings(llm_reasoning_effort="low").llm_reasoning_effort == "low"


def test_reasoning_max_tokens_is_configurable() -> None:
    assert _settings(llm_reasoning_max_tokens=800).llm_reasoning_max_tokens == 800


def test_reasoning_effort_and_max_tokens_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _settings(llm_reasoning_effort="low", llm_reasoning_max_tokens=800)


def test_blank_proxy_builds_a_working_client():
    # The regression itself: this raised "Unknown scheme for proxy URL URL('')".
    settings = _settings(tools_http_proxy="")

    httpx.AsyncClient(proxy=settings.tools_http_proxy, trust_env=False)


def test_required_strings_are_left_alone():
    # Only optionals are blanked; a required field keeps whatever it was given so
    # a genuine misconfiguration still surfaces instead of silently defaulting.
    assert _settings(postgres_host="").postgres_host == ""


def test_geocoding_country_scope_contains_agreed_regions() -> None:
    allowed = set(_settings().geocoding_allowed_country_codes)

    assert {"RU", "KZ", "UZ", "KG", "TJ", "AM", "GE", "AZ"} <= allowed
    assert {"GB", "PT", "PL", "CH", "IT", "XK"} <= allowed
    assert "TM" not in allowed


def test_geocoding_country_scope_normalises_and_deduplicates_codes() -> None:
    settings = _settings(geocoding_allowed_country_codes=[" ru ", "PT", "RU"])

    assert settings.geocoding_allowed_country_codes == ["RU", "PT"]
