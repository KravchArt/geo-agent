"""Internal geocoder contracts, candidate matching, and provider integrations."""

from tools.geo.geocoding.matching import select_unique_place_match
from tools.geo.geocoding.resolver import (
    GeocodedPlaceResolver,
    PlaceResolutionContractError,
)
from tools.geo.geocoding.schemas import (
    MAX_MATCHES,
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ReverseGeocodeOutput,
    ToponymKind,
)
from tools.geo.geocoding.service import GeocoderService

__all__ = [
    "MAX_MATCHES",
    "GeocodePlaceInput",
    "GeocodePlaceOutput",
    "GeocodedPlaceResolver",
    "GeocoderService",
    "PlaceMatch",
    "PlaceResolutionContractError",
    "ReverseGeocodeOutput",
    "ToponymKind",
    "select_unique_place_match",
]
