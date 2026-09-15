"""Application settings loaded from environment (Pydantic Settings).

Single source of truth for configuration. Values come from the process
environment (and, for local dev, an optional ``.env`` file). Secrets are NEVER
committed — see ``.env.example`` for the full list of variables.
"""

from __future__ import annotations

from functools import lru_cache
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _accepts_none(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation)


def _default_text_place_resolution_providers() -> list[Literal["twogis", "tomtom"]]:
    return ["tomtom"]


def _default_geocoding_allowed_country_codes() -> list[str]:
    """Return the product's supported city-resolution countries.

    Cyprus and Turkey are included under the broad, political definition of
    Europe. ``XK`` is Kosovo's commonly used (but non-ISO) provider code.
    Turkmenistan is deliberately absent: it was not part of the agreed Central
    Asia scope.
    """

    return [
        "AD",  # Andorra
        "AL",  # Albania
        "AM",  # Armenia
        "AT",  # Austria
        "AZ",  # Azerbaijan
        "BA",  # Bosnia and Herzegovina
        "BE",  # Belgium
        "BG",  # Bulgaria
        "BY",  # Belarus
        "CH",  # Switzerland
        "CY",  # Cyprus
        "CZ",  # Czechia
        "DE",  # Germany
        "DK",  # Denmark
        "EE",  # Estonia
        "ES",  # Spain
        "FI",  # Finland
        "FR",  # France
        "GB",  # United Kingdom
        "GE",  # Georgia
        "GR",  # Greece
        "HR",  # Croatia
        "HU",  # Hungary
        "IE",  # Ireland
        "IS",  # Iceland
        "IT",  # Italy
        "KG",  # Kyrgyzstan
        "KZ",  # Kazakhstan
        "LI",  # Liechtenstein
        "LT",  # Lithuania
        "LU",  # Luxembourg
        "LV",  # Latvia
        "MC",  # Monaco
        "MD",  # Moldova
        "ME",  # Montenegro
        "MK",  # North Macedonia
        "MT",  # Malta
        "NL",  # Netherlands
        "NO",  # Norway
        "PL",  # Poland
        "PT",  # Portugal
        "RO",  # Romania
        "RS",  # Serbia
        "RU",  # Russia
        "SE",  # Sweden
        "SI",  # Slovenia
        "SK",  # Slovakia
        "SM",  # San Marino
        "TJ",  # Tajikistan
        "TR",  # Turkey
        "UA",  # Ukraine
        "UZ",  # Uzbekistan
        "VA",  # Vatican City
        "XK",  # Kosovo
    ]


class Settings(BaseSettings):
    """Typed application settings.

    Env var names are the upper-cased field names (case-insensitive), e.g.
    ``POSTGRES_HOST`` -> ``postgres_host``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Runtime ---
    app_env: Literal["dev", "prod"] = "dev"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --- PostgreSQL ---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "geoagent"
    postgres_password: str = "geoagent"
    postgres_db: str = "geoagent"

    # --- Redis ---
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_session_ttl: int = Field(default=3600, ge=1)
    agent_request_lock_ttl: int = Field(default=600, ge=1)
    redis_echo_ttl: int = Field(default=1800, ge=1)
    #: How long a minted ref (plc_/src_) stays resolvable. Must outlive a
    #: conversation, or the model's own refs stop resolving mid-dialogue.
    redis_ref_ttl: int = Field(default=86_400, ge=1)

    # --- LLM ---
    llm_mode: Literal["mock", "vllm"] = "mock"
    llm_base_url: str = "http://vllm:8000/v1"
    llm_model: str = "geoagent-model"
    llm_api_key: str | None = None
    llm_timeout: int = Field(default=60, ge=1)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    #: Optional OpenRouter-compatible reasoning budget. Leave unset for local
    #: servers that do not implement the unified ``reasoning`` request field.
    llm_reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    #: Optional explicit reasoning-token budget. OpenRouter forwards this as
    #: ``reasoning.max_tokens``; whether it is a hard limit depends on the
    #: selected model/provider implementation.
    llm_reasoning_max_tokens: int | None = Field(default=None, ge=1)
    #: Outbound proxy for the model endpoint, e.g. ``http://127.0.0.1:12334``.
    #: Needed when the API is remote (OpenRouter) and direct egress is blocked;
    #: leave empty for a local server, or every call to localhost is proxied too.
    llm_http_proxy: str | None = None

    # --- Tool providers ---
    # Places search and routing have stable runtime priorities with sequential
    # fallback; configuration controls which providers are enabled.
    places_search_providers: list[Literal["yandex", "tomtom", "twogis"]] = Field(
        default_factory=list
    )
    routing_providers: list[Literal["yandex", "graphhopper", "osrm", "twogis"]] = Field(
        default_factory=list
    )
    #: One named-place resolution chain shared by places-search anchors and
    #: textual routing points. The TomTom geocoder remains the final fallback.
    text_place_resolution_providers: list[Literal["twogis", "tomtom"]] = Field(
        default_factory=_default_text_place_resolution_providers
    )
    web_search_providers: list[Literal["exa", "tavily", "firecrawl"]] = Field(default_factory=list)
    # The first limit applies to each network request; the second caps the full
    # handler, including internal geocoding and all subsequent provider calls.
    tools_http_timeout: int = Field(default=15, ge=1)
    tools_execution_timeout: int = Field(default=45, ge=1)
    #: 2GIS is the first places provider and has a TomTom fallback.  Keep its
    #: individual attempt short so one stalled catalog request does not consume
    #: the whole user-visible search budget.
    dgis_catalog_timeout: int = Field(default=3, ge=1)
    #: Outbound proxy for tool traffic (Yandex/Overpass/Tavily/Exa/Firecrawl), e.g.
    #: ``http://127.0.0.1:12334``. Set explicitly rather than read from
    #: HTTP_PROXY/ALL_PROXY: routing is a deployment decision, and an ambient
    #: ``socks://`` value would stop httpx from starting at all.
    tools_http_proxy: str | None = None
    #: Country whitelist used for bare city resolution before score comparison.
    #: It prevents unrelated cities in the Americas and elsewhere from winning
    #: a close TomTom ranking for the product's supported geography.
    geocoding_allowed_country_codes: list[str] = Field(
        default_factory=_default_geocoding_allowed_country_codes
    )

    # --- User context enrichment ---
    ip_geolocation_enabled: bool = True
    ip_geolocation_self_lookup_enabled: bool = False
    ip_geolocation_base_url: str = Field(default="https://ipwho.is", min_length=1)
    ip_geolocation_timeout: float = Field(default=3.0, gt=0)
    user_context_http_proxy: str | None = None

    # Retained for deployments that still call the legacy Yandex geocoder
    # directly; runtime place resolution now uses TOMTOM_API_KEY.
    yandex_geocoder_api_key: str | None = None
    yandex_organisation_search_api_key: str | None = None
    yandex_routing_api_key: str | None = None
    tomtom_api_key: str | None = None
    tomtom_search_base_url: str = Field(
        default="https://api.tomtom.com/search/2",
        min_length=1,
    )
    dgis_api_key: str | None = None
    dgis_catalog_base_url: str = Field(
        default="https://catalog.api.2gis.com",
        min_length=1,
    )
    dgis_routing_base_url: str = Field(
        default="https://routing.api.2gis.com",
        min_length=1,
    )
    graphhopper_api_key: str | None = None
    graphhopper_base_url: str = Field(
        default="https://graphhopper.com/api/1",
        min_length=1,
    )
    routing_snap_warning_distance_m: int = Field(default=250, ge=1)
    osrm_base_url: str = Field(
        default="https://router.project-osrm.org",
        min_length=1,
    )
    osrm_walking_base_url: str = Field(
        default="https://routing.openstreetmap.de/routed-foot",
        min_length=1,
    )
    osrm_user_agent: str = Field(default="GeoAgent/0.0.0", min_length=1)
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    firecrawl_api_key: str | None = None

    # --- Scope gate ---
    # rule_based (default) is instant regex and needs no LLM; llm asks the model
    # to classify with SCOPER_SYSTEM_PROMPT and falls back to the regex gate when
    # the model is unreachable or answers unparseably; no_scoper disables scope
    # filtering while retaining an explicit audit record.
    #: classifier adds the fine-tuned e5 encoder in front of `llm`: it decides
    #: outright outside the grey zone and defers to the LLM scoper inside it.
    scope_provider: Literal["rule_based", "llm", "classifier", "no_scoper"] = "rule_based"
    #: Point the classifier at its OWN endpoint — a small model on CPU is enough
    #: for a yes/no verdict, and it keeps this off the critical GPU. None reuses
    #: the main LLM backend.
    scope_base_url: str | None = None
    scope_api_key: str | None = None
    #: Model name at that endpoint; None reuses LLM_MODEL.
    scope_model: str | None = None

    # --- Scope classifier (SCOPE_PROVIDER=classifier) ---
    #: Root of a vLLM server started with `--runner pooling`; the gate posts to
    #: its /classify endpoint. Required when the provider is `classifier`.
    scope_classifier_base_url: str | None = None
    scope_classifier_api_key: str | None = None
    scope_classifier_model: str | None = None
    #: Defaults below mirror artifacts/scope_bert_meta.json — change them together
    #: with the weights or the verdict silently shifts.
    #: e5 was fine-tuned with this prefix; dropping it degrades quality quietly.
    scope_classifier_prefix: str = "query: "
    #: Calibration temperature. Applied to the logit, not to the raw probability.
    scope_classifier_temperature: float = Field(default=2.332827091217041, gt=0.0)
    #: Below `grey_low` the request is out of scope, above `grey_high` it is in.
    #: Between them the classifier abstains and the LLM scoper decides. Setting
    #: both to the same value disables the cascade and makes it a plain threshold.
    scope_classifier_grey_low: float = Field(default=0.4, ge=0.0, le=1.0)
    scope_classifier_grey_high: float = Field(default=0.6, ge=0.0, le=1.0)
    scope_classifier_timeout: int = Field(default=5, gt=0)
    #: The encoder may admit a request alone, but not turn one away alone: every
    #: rejection is re-decided by the LLM scoper, which reads the recent turns.
    #: This is where the encoder is blind — it scores "а что рядом?" like a
    #: fragment — and where the error is expensive, because it is paid by a real
    #: user. The cost is one small chat completion per rejected turn. Set false to
    #: let the encoder reject on its own, leaving the grey zone as the only path
    #: to the scoper.
    scope_classifier_confirm_reject: bool = False

    # --- Scope OOD check (distance to the training corpus) ---
    #: The grey zone catches requests the classifier is unsure about. It cannot
    #: catch requests on topics the classifier never saw: those do not land near
    #: the threshold, they land confidently on one side of it. Measured on
    #: production-shaped logs, three of four false positives scored 0.89-0.98 and
    #: all of them were among the requests farthest from the training corpus.
    #: off = disabled; shadow = distance is measured and logged but never changes
    #: a verdict; enforce = far requests are deferred to the LLM scoper.
    #: Start in `shadow`: the flag rate below was measured on a QA corpus whose
    #: topics sit close to the training data, and open traffic will flag more.
    scope_ood_mode: Literal["off", "shadow", "enforce"] = "off"
    #: Path to the .npz built by the classifier repo. Must come from the same
    #: checkpoint that serves /classify — the fine-tuned encoder's space is not
    #: the base model's, and mixing them yields plausible, meaningless distances.
    scope_ood_index_path: str | None = None
    #: Endpoint serving that checkpoint with `--runner pooling --task embed`.
    #: None reuses SCOPE_CLASSIFIER_BASE_URL, which only works if that server also
    #: exposes /v1/embeddings.
    scope_ood_base_url: str | None = None
    scope_ood_model: str | None = None
    #: Defer a request when its distance to the corpus is worse than this quantile
    #: of the training rows' own distances. Cost measured on 196 labelled
    #: production-shaped requests: 0.01 defers 2.6% of traffic and catches 1 of 4
    #: false positives; 0.05 defers 11.7% and catches 3 of 4.
    scope_ood_quantile: float = Field(default=0.01, gt=0.0, lt=1.0)
    #: Once a session has produced one in-scope verdict, stop classifying it and
    #: send every later turn straight to the orchestrator. This exists because the
    #: classifier is contextless: it would reject "а что рядом?" on its own.
    #: The cost is real — after one in-scope turn the session is ungated until its
    #: Redis TTL expires. Set false to classify every turn.
    scope_session_unlock: bool = True

    # --- Censorship gate ---
    # rule_based (default) is instant regex and needs no server — the right choice
    # locally and whenever the orchestrator runs against a hosted API. model asks
    # a classifier over a plain OpenAI chat completion (CENSOR_SYSTEM_PROMPT) and
    # falls back to the regex gate when the model is unreachable or unparseable.
    #: guardian = IBM Granite Guardian, which answers Yes/No through its own chat
    #: template instead of a prompt; model = the prompt-based judge.
    censorship_provider: Literal["rule_based", "model", "guardian"] = "rule_based"
    #: Point the classifier at its OWN endpoint, for the same reason the scope gate
    #: has one: a one-word verdict must not queue behind the main model. None
    #: reuses the main LLM backend.
    censorship_base_url: str | None = None
    censorship_api_key: str | None = None
    #: Model name at that endpoint; None reuses LLM_MODEL.
    censorship_model: str | None = None

    # --- Derived connection strings ---
    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @model_validator(mode="before")
    @classmethod
    def _blank_optional_values_are_unset(cls, data: Any) -> Any:
        """Treat ``FOO=`` in .env as "not configured", not as the empty string.

        Env files spell an unset optional as a bare ``FOO=``, which arrives here
        as ``""``. Left alone it is a *value*: ``TOOLS_HTTP_PROXY=`` became
        ``httpx.AsyncClient(proxy="")``, which raises "Unknown scheme for proxy
        URL" and the app never finished starting — so anyone who copied
        .env.example got a container that could not boot.
        """
        if not isinstance(data, dict):
            return data
        optional = {
            name for name, field in cls.model_fields.items() if _accepts_none(field.annotation)
        }
        return {
            key: (
                None if key in optional and isinstance(value, str) and not value.strip() else value
            )
            for key, value in data.items()
        }

    @field_validator("geocoding_allowed_country_codes")
    @classmethod
    def _normalise_geocoding_country_codes(cls, values: list[str]) -> list[str]:
        """Accept country-code configuration once in a canonical form."""

        normalized: list[str] = []
        for value in values:
            code = value.strip().upper()
            if len(code) != 2 or not code.isalpha():
                raise ValueError("geocoding country codes must be two letters")
            if code not in normalized:
                normalized.append(code)
        if not normalized:
            raise ValueError("geocoding_allowed_country_codes cannot be empty")
        return normalized

    @model_validator(mode="after")
    def _configuration_guardrails(self) -> Settings:
        """Fail fast on invalid production and enabled-provider configuration."""

        if self.llm_reasoning_effort is not None and self.llm_reasoning_max_tokens is not None:
            raise ValueError(
                "LLM_REASONING_EFFORT and LLM_REASONING_MAX_TOKENS are mutually exclusive"
            )

        if self.app_env == "prod":
            if self.app_debug:
                raise ValueError("APP_DEBUG must be false in prod")

            if self.llm_mode == "vllm" and not self.llm_api_key:
                raise ValueError("LLM_API_KEY is required in prod when LLM_MODE=vllm")

        if self.scope_provider == "classifier":
            if not self.scope_classifier_base_url:
                raise ValueError("SCOPE_PROVIDER=classifier requires SCOPE_CLASSIFIER_BASE_URL")
            if not self.scope_classifier_model:
                raise ValueError("SCOPE_PROVIDER=classifier requires SCOPE_CLASSIFIER_MODEL")

        if self.scope_classifier_grey_low > self.scope_classifier_grey_high:
            raise ValueError("SCOPE_CLASSIFIER_GREY_LOW must not exceed SCOPE_CLASSIFIER_GREY_HIGH")

        if self.scope_ood_mode != "off":
            if self.scope_provider != "classifier":
                raise ValueError("SCOPE_OOD_MODE requires SCOPE_PROVIDER=classifier")
            if not self.scope_ood_index_path:
                raise ValueError("SCOPE_OOD_MODE requires SCOPE_OOD_INDEX_PATH")
            if not (self.scope_ood_base_url or self.scope_classifier_base_url):
                raise ValueError("SCOPE_OOD_MODE requires SCOPE_OOD_BASE_URL")
            if not (self.scope_ood_model or self.scope_classifier_model):
                raise ValueError("SCOPE_OOD_MODE requires SCOPE_OOD_MODEL")

        if "yandex" in self.places_search_providers:
            required_keys = {
                "YANDEX_ORGANISATION_SEARCH_API_KEY": self.yandex_organisation_search_api_key,
            }
            missing_keys = [name for name, value in required_keys.items() if not value]

            if missing_keys:
                raise ValueError(
                    "missing required provider keys for places_search provider yandex: "
                    + ", ".join(missing_keys)
                )

        if "tomtom" in self.places_search_providers:
            required_keys = {"TOMTOM_API_KEY": self.tomtom_api_key}
            missing_keys = [name for name, value in required_keys.items() if not value]
            if missing_keys:
                raise ValueError(
                    "missing required provider keys for places_search provider tomtom: "
                    + ", ".join(missing_keys)
                )

        if "twogis" in self.places_search_providers:
            required_keys = {
                "DGIS_API_KEY": self.dgis_api_key,
                "TOMTOM_API_KEY": self.tomtom_api_key,
            }
            missing_keys = [name for name, value in required_keys.items() if not value]
            if missing_keys:
                raise ValueError(
                    "missing required provider keys for places_search provider twogis: "
                    + ", ".join(missing_keys)
                )

        if (
            (self.places_search_providers or self.routing_providers)
            and "twogis" in self.text_place_resolution_providers
            and not self.dgis_api_key
        ):
            raise ValueError("DGIS_API_KEY is required when 2GIS text-place resolution is enabled")

        if "yandex" in self.routing_providers:
            required_keys = {
                "YANDEX_ROUTING_API_KEY": self.yandex_routing_api_key,
                "TOMTOM_API_KEY": self.tomtom_api_key,
            }
            missing_keys = [name for name, value in required_keys.items() if not value]

            if missing_keys:
                raise ValueError(
                    "missing required provider keys for routing provider yandex: "
                    + ", ".join(missing_keys)
                )

        if "graphhopper" in self.routing_providers:
            required_keys = {
                "GRAPHHOPPER_API_KEY": self.graphhopper_api_key,
                "TOMTOM_API_KEY": self.tomtom_api_key,
            }
            missing_keys = [name for name, value in required_keys.items() if not value]
            if missing_keys:
                raise ValueError(
                    "missing required provider keys for routing provider graphhopper: "
                    + ", ".join(missing_keys)
                )

        if "twogis" in self.routing_providers:
            required_keys = {
                "DGIS_API_KEY": self.dgis_api_key,
                "TOMTOM_API_KEY": self.tomtom_api_key,
            }
            missing_keys = [name for name, value in required_keys.items() if not value]
            if missing_keys:
                raise ValueError(
                    "missing required provider keys for routing provider twogis: "
                    + ", ".join(missing_keys)
                )

        if "osrm" in self.routing_providers and not self.tomtom_api_key:
            raise ValueError("TOMTOM_API_KEY is required when OSRM routing is enabled")

        if self.places_search_providers and not self.tomtom_api_key:
            raise ValueError("TOMTOM_API_KEY is required for internal place geocoding")

        if "tavily" in self.web_search_providers and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when Tavily web search is enabled")

        if "exa" in self.web_search_providers and not self.exa_api_key:
            raise ValueError("EXA_API_KEY is required when Exa web search is enabled")

        if "firecrawl" in self.web_search_providers and not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY is required when Firecrawl web search is enabled")

        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
