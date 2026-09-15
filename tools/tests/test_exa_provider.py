"""Tests for the Exa-backed web search provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.observability import ToolExecutionContext
from tools.refs import SourceRecord, SourceRef
from tools.web.exa.client import ExaSearchClient
from tools.web.exa.provider import ExaWebSearchProvider
from tools.web.search import MAX_SNIPPET_CHARS, WebSearchInput


class FakeExaClient:
    provider = "exa"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[WebSearchInput] = []

    async def search(
        self,
        args: WebSearchInput,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        self.calls.append(args)
        return self._payload


class FakeSourceStore:
    def __init__(self) -> None:
        self.records: dict[str, SourceRecord] = {}
        self.save_many_calls: list[list[SourceRecord]] = []

    async def save_many(self, sources: Sequence[SourceRecord]) -> None:
        batch = list(sources)
        self.save_many_calls.append(batch)
        self.records.update({record.ref: record for record in batch})

    async def get(self, ref: SourceRef) -> SourceRecord | None:
        return self.records.get(ref)


def make_service(
    payload: dict[str, Any],
    source_store: FakeSourceStore,
) -> ExaWebSearchProvider:
    return ExaWebSearchProvider(
        client=cast(ExaSearchClient, FakeExaClient(payload)),
        source_store=source_store,
    )


async def test_search_returns_safe_results_and_saves_sources_once() -> None:
    """Verify that search returns safe results and saves sources once."""

    payload: dict[str, Any] = {
        "requestId": "request-1",
        "resolvedSearchType": "",
        "costDollars": {"total": 0.007},
        "results": [
            {
                "title": " Coffee guide ",
                "url": "https://Example.COM/guide#best-places",
                "publishedDate": "2026-07-12T01:36:32.547Z",
                "highlights": [" First relevant passage. ", "Second passage."],
                "highlightScores": [0.91, 0.88],
            },
            {
                "title": "Official city guide",
                "url": "https://mos.ru/culture",
                "highlights": ["Official city information"],
            },
        ],
    }
    store = FakeSourceStore()
    service = make_service(payload, store)

    result = await service.search(
        WebSearchInput(query="coffee in moscow"),
        ToolExecutionContext(),
    )

    assert result.query == "coffee in moscow"
    assert len(result.results) == 2
    assert result.results[0].ref.startswith("src_")
    assert result.results[0].title == "Coffee guide"
    assert result.results[0].domain == "example.com"
    assert result.results[0].snippet == "First relevant passage.\n\nSecond passage."
    assert "score" not in result.results[0].model_dump()
    assert result.results[0].published_date is not None
    assert result.results[0].published_date.isoformat() == "2026-07-12"

    assert len(store.save_many_calls) == 1
    assert len(store.save_many_calls[0]) == 2
    stored_record = await store.get(result.results[0].ref)
    assert stored_record is not None
    assert stored_record.url == "https://example.com/guide"


async def test_search_skips_unsafe_duplicate_and_contentless_results() -> None:
    """Verify that search skips unsafe duplicate and contentless results."""

    payload: dict[str, Any] = {
        "requestId": "request-1",
        "results": [
            {
                "title": "Unsafe",
                "url": "javascript:alert(1)",
                "highlights": ["Must not be returned"],
            },
            {
                "title": "First valid result",
                "url": "https://example.com/article#first-fragment",
                "highlights": ["Useful snippet"],
            },
            {
                "title": "Duplicate result",
                "url": "https://example.com/article#second-fragment",
                "highlights": ["Duplicate snippet"],
            },
            {
                "title": "Credentials in URL",
                "url": "https://user:password@example.com/private",
                "highlights": ["Must not be returned"],
            },
            {
                "title": "No extracted contents",
                "url": "https://example.org/no-content",
                "highlights": [],
            },
            {
                "title": None,
                "url": "https://example.org/no-title",
                "highlights": ["Must not be returned"],
            },
        ],
    }
    store = FakeSourceStore()
    service = make_service(payload, store)

    result = await service.search(
        WebSearchInput(query="coffee in moscow"),
        ToolExecutionContext(),
    )

    assert [item.title for item in result.results] == ["First valid result"]
    assert len(store.save_many_calls) == 1
    assert [record.url for record in store.save_many_calls[0]] == [
        "https://example.com/article",
    ]


async def test_search_truncates_combined_highlights() -> None:
    """Verify that search truncates combined highlights."""

    payload: dict[str, Any] = {
        "requestId": "request-1",
        "results": [
            {
                "title": "Long article",
                "url": "https://example.com/article",
                "highlights": ["x" * 1_500, "y" * 1_000],
            }
        ],
    }
    store = FakeSourceStore()
    service = make_service(payload, store)

    result = await service.search(
        WebSearchInput(query="coffee in moscow"),
        ToolExecutionContext(),
    )

    assert len(result.results[0].snippet) == MAX_SNIPPET_CHARS
    combined = f"{'x' * 1_500}\n\n{'y' * 1_000}"
    assert result.results[0].snippet == combined[:MAX_SNIPPET_CHARS]


async def test_search_does_not_save_when_no_result_has_usable_evidence() -> None:
    """Verify that search does not save when no result has usable evidence."""

    payload: dict[str, Any] = {
        "requestId": "request-1",
        "results": [
            {
                "title": "No highlights",
                "url": "https://example.com/article",
            }
        ],
    }
    store = FakeSourceStore()
    service = make_service(payload, store)

    result = await service.search(
        WebSearchInput(query="coffee in moscow"),
        ToolExecutionContext(),
    )

    assert result.results == []
    assert store.save_many_calls == []


async def test_search_maps_invalid_provider_schema_without_saving_records() -> None:
    """Verify that search maps invalid provider schema without saving records."""

    payload: dict[str, Any] = {
        "results": [],
    }
    store = FakeSourceStore()
    service = make_service(payload, store)

    with pytest.raises(ToolExecutionError) as exc_info:
        await service.search(
            WebSearchInput(query="coffee in moscow"),
            ToolExecutionContext(),
        )

    assert exc_info.value.error_code is ToolErrorCode.UPSTREAM_ERROR
    assert str(exc_info.value) == "web search provider returned invalid data"
    assert exc_info.value.provider == "exa"
    assert exc_info.value.status_code is None
    assert exc_info.value.failure_kind is ToolFailureKind.INVALID_SCHEMA
    assert exc_info.value.retryable is False
    assert store.save_many_calls == []
