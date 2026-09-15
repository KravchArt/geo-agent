"""Place-search contracts and local provider implementations."""

from tools.geo.places_search.coordinator import PlacesSearchCoordinator
from tools.geo.places_search.provider import (
    AmbiguousSearchAnchorError,
    PlacesSearchProvider,
    UnknownPlaceRefError,
)
from tools.geo.places_search.resolution import (
    PlacesSearchScopeLoader,
    PlacesSearchScopeResolver,
)
from tools.geo.places_search.schemas import (
    PLACES_SEARCH_SPEC,
    OrganisationCandidate,
    OrganisationResolution,
    OrganisationResolutionStatus,
    Place,
    PlaceCategory,
    PlacesSearchInput,
    PlacesSearchOutput,
    ResolvedSearchArea,
    SearchMode,
)

__all__ = [
    "PLACES_SEARCH_SPEC",
    "AmbiguousSearchAnchorError",
    "OrganisationCandidate",
    "OrganisationResolution",
    "OrganisationResolutionStatus",
    "Place",
    "PlaceCategory",
    "PlacesSearchCoordinator",
    "PlacesSearchInput",
    "PlacesSearchOutput",
    "PlacesSearchProvider",
    "PlacesSearchScopeLoader",
    "PlacesSearchScopeResolver",
    "ResolvedSearchArea",
    "SearchMode",
    "UnknownPlaceRefError",
]
