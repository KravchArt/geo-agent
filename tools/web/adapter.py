"""Shared provider-response validation and safe result materialization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.refs import SourceRecord, mint_source_ref
from tools.web.search import MAX_SNIPPET_CHARS, WebResult, WebSearchOutput
from tools.web.source_store import SourceStore

ResponseT = TypeVar("ResponseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class WebSearchCandidate:
    """Provider-neutral result before URL safety and source-ref materialization."""

    title: str | None
    url: str
    snippet: str
    published_date: date | None = None


def validate_search_response(
    payload: dict[str, Any],
    *,
    response_model: type[ResponseT],
    provider: str,
) -> ResponseT:
    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise ToolExecutionError(
            error_code=ToolErrorCode.UPSTREAM_ERROR,
            public_message="web search provider returned invalid data",
            provider=provider,
            failure_kind=ToolFailureKind.INVALID_SCHEMA,
            retryable=False,
        ) from exc


async def materialize_search_results(
    *,
    provider: str,
    query: str,
    max_results: int,
    candidates: Iterable[WebSearchCandidate],
    source_store: SourceStore,
) -> WebSearchOutput:
    """Apply the cross-provider source-safety invariants and persist refs."""

    results: list[WebResult] = []
    records: list[SourceRecord] = []
    seen_urls: set[str] = set()

    for candidate in candidates:
        if len(results) >= max_results:
            break

        normalized = _normalize_url(candidate.url)
        if normalized is None:
            continue

        normalized_url, domain = normalized
        if normalized_url in seen_urls:
            continue

        title = (candidate.title or "").strip()
        snippet = candidate.snippet.strip()[:MAX_SNIPPET_CHARS]

        if not title or not snippet:
            continue

        seen_urls.add(normalized_url)

        source_ref = mint_source_ref(f"{provider}:url:{normalized_url}")
        record = SourceRecord(
            ref=source_ref,
            url=normalized_url,
            title=title,
            domain=domain,
            snippet=snippet,
            published_date=candidate.published_date,
        )
        records.append(record)
        results.append(
            WebResult(
                ref=source_ref,
                title=title,
                domain=domain,
                snippet=snippet,
                published_date=candidate.published_date,
            )
        )

    if records:
        await source_store.save_many(records)

    return WebSearchOutput(
        query=query,
        results=results,
    )


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None

    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _normalize_url(value: str) -> tuple[str, str] | None:
    parsed = urlsplit(value)

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None

    domain = parsed.hostname.lower()
    normalized_url = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    return normalized_url, domain
