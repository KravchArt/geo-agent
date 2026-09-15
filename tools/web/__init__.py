"""Web-search contracts, coordination and local stores."""

from tools.web.coordinator import WebSearchCoordinator
from tools.web.memory_store import InMemorySourceStore
from tools.web.provider import WebSearchProvider
from tools.web.search import (
    WEB_SEARCH_SPEC,
    TimeRange,
    WebResult,
    WebSearchInput,
    WebSearchOutput,
    WebTopic,
)
from tools.web.source_store import SourceStore

__all__ = [
    "WEB_SEARCH_SPEC",
    "InMemorySourceStore",
    "SourceStore",
    "TimeRange",
    "WebResult",
    "WebSearchCoordinator",
    "WebSearchInput",
    "WebSearchOutput",
    "WebSearchProvider",
    "WebTopic",
]
