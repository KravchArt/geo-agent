"""Organisation-search provider contract and domain errors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tools.base import ToolClarification, ToolErrorCode, ToolExecutionError
from tools.geo.places_search.schemas import PlacesSearchInput, PlacesSearchOutput
from tools.observability import ToolExecutionContext
from tools.refs import PlaceRef


class UnknownPlaceRefError(ToolExecutionError):
    """The supplied anchor ref is absent from the place store."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.UNKNOWN_REF,
            f"Place ref {ref} was not found. Pass the anchor text as near to resolve it again.",
        )


class AnchorNotFoundError(ToolExecutionError):
    """Search-scope resolution found no anchor for the textual request."""

    def __init__(self, query: str) -> None:
        super().__init__(
            ToolErrorCode.NOT_FOUND,
            f"Could not resolve the nearby-search anchor: {query!r}",
        )


class AmbiguousSearchAnchorError(ToolExecutionError):
    """Neither geocoding nor named-POI search selected one anchor."""

    def __init__(
        self,
        query: str,
        *,
        clarification: ToolClarification | None = None,
    ) -> None:
        super().__init__(
            ToolErrorCode.INVALID_INPUT,
            f"Nearby-search anchor is ambiguous: {query!r}. "
            "Specify a more precise name, object type, or full address.",
            clarification=clarification,
        )


class SearchAreaNotFoundError(ToolExecutionError):
    """Shared place resolution found no bounded locality for area search."""

    def __init__(self, city: str) -> None:
        super().__init__(
            ToolErrorCode.NOT_FOUND,
            f"Could not resolve a bounded city search area: {city!r}",
        )


class AmbiguousSearchAreaError(ToolExecutionError):
    """More than one bounded locality matches the requested city."""

    def __init__(
        self,
        city: str,
        *,
        clarification: ToolClarification | None = None,
    ) -> None:
        super().__init__(
            ToolErrorCode.INVALID_INPUT,
            f"City is ambiguous: {city!r}. Specify its region or country.",
            clarification=clarification,
        )


class UnknownSearchAreaRefError(ToolExecutionError):
    """The supplied area ref is absent from the place store."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.UNKNOWN_REF,
            f"Search area ref {ref} was not found. Pass city to resolve it again.",
        )


class InvalidSearchAreaRefError(ToolExecutionError):
    """The supplied ref does not identify a bounded locality."""

    def __init__(self, ref: PlaceRef) -> None:
        super().__init__(
            ToolErrorCode.INVALID_INPUT,
            f"Place ref {ref} does not identify a bounded city search area.",
        )


@runtime_checkable
class PlacesSearchProvider(Protocol):
    """One concrete backend that consumes a coordinator-prepared search scope."""

    provider: str
    #: Whether this adapter can verify current status from schedule plus local time.
    supports_open_now: bool
    #: The provider can turn a textual ``mode=area, area=...`` value into its
    #: own native scope. Other providers require the coordinator's shared
    #: preparation into a Redis-backed area ref first.
    resolves_area_natively: bool

    async def search(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Search within a prepared area or around a prepared anchor ref."""
        ...


@runtime_checkable
class FirstAddressPlacesSearchProvider(Protocol):
    """Optional best-effort provider path used by ``mode=resolve``."""

    async def search_first_address(
        self,
        args: PlacesSearchInput,
        context: ToolExecutionContext,
    ) -> PlacesSearchOutput:
        """Return the provider-ranked first card with an address."""
        ...
