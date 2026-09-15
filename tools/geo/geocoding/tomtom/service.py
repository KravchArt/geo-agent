"""TomTom-backed implementation of the internal geocoding service."""

from __future__ import annotations

import re

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.schemas import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ReverseGeocodeOutput,
    ToponymKind,
)
from tools.geo.geocoding.tomtom.client import TomTomGeocoderClient
from tools.geo.geocoding.tomtom.schemas import (
    TomTomGeocodeResponse,
    TomTomGeocodeResult,
    TomTomReverseGeocodeResponse,
)
from tools.geo.place_store import PlaceStore
from tools.geo.text_place_query import compose_scoped_place_query, same_locality_anchor
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref

_TRAILING_TRANSLITERATION_APOSTROPHE = re.compile(r"(?<=\w)'$")


class TomTomGeocoderService:
    """Resolve addresses and geographic entities with TomTom."""

    provider = "tomtom_geocoder"

    def __init__(self, client: TomTomGeocoderClient, place_store: PlaceStore) -> None:
        self._client = client
        self._place_store = place_store

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        query = compose_scoped_place_query(query=args.query, city=args.city, city_first=False)
        payload = await self._client.search(
            query,
            limit=args.limit,
            context=context,
            # Some city-regions are classified by TomTom as a
            # CountrySecondarySubdivision rather than a Municipality (for
            # example English "Almaty").  Both are allowed only for the
            # dedicated city-scope lookup; ordinary address geocoding keeps
            # its full result set and existing type semantics.
            entity_type_set=(
                "Municipality,CountrySecondarySubdivision" if args.locality_only else None
            ),
        )
        try:
            response = TomTomGeocodeResponse.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "TomTom geocoder gave wrong answer schema",
                provider=self._client.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc

        candidates = response.results
        if args.locality_only and any(
            candidate.entity_type == "Municipality"
            and same_locality_anchor(self._name(candidate), args.query)
            for candidate in candidates
        ):
            # A city can be returned twice: as its actual Municipality and as
            # a much broader CountrySecondarySubdivision with the same label
            # (for example Moscow).  The latter is a fallback only for cities
            # such as Almaty, where TomTom supplies no Municipality at all.
            candidates = [
                candidate
                for candidate in candidates
                if candidate.entity_type != "CountrySecondarySubdivision"
            ]

        matches: list[PlaceMatch] = []
        records: list[PlaceRecord] = []
        seen_refs: set[str] = set()
        for candidate in candidates:
            ref = mint_place_ref(f"tomtom:geocode:{candidate.id}")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            kind = self._kind(candidate, locality_only=args.locality_only)
            name = self._name(candidate)
            address = self._address(candidate, name=name, kind=kind)
            locality = self._locality(candidate, name=name, kind=kind)
            precision = self._precision(candidate)
            bounds = self._bounds(candidate)
            records.append(
                PlaceRecord(
                    ref=ref,
                    name=name,
                    address=address,
                    lat=candidate.position.lat,
                    lon=candidate.position.lon,
                    kind=kind.value if kind is not None else candidate.type,
                    precision=precision,
                    locality=locality,
                    bounds=bounds,
                    provider=self.provider,
                    provider_uri=None,
                    origin=RecordOrigin.GEOCODE,
                )
            )
            matches.append(
                PlaceMatch(
                    ref=ref,
                    name=name,
                    address=address,
                    kind=kind,
                    precision=precision,
                    municipality=locality,
                    country_secondary_subdivision=_clean_display_label(
                        candidate.address.country_secondary_subdivision
                    ),
                    country_subdivision=_clean_display_label(candidate.address.country_subdivision),
                    country_code=candidate.address.country_code,
                    provider_entity_type=candidate.entity_type,
                    relevance_score=candidate.score,
                    match_confidence_score=(
                        candidate.match_confidence.score
                        if candidate.match_confidence is not None
                        else None
                    ),
                )
            )

        if records:
            await self._place_store.save_many(records)
        return GeocodePlaceOutput(
            best=matches[0] if matches else None,
            matches=matches,
            ambiguous=len(matches) > 1,
        )

    async def reverse_geocode(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
    ) -> ReverseGeocodeOutput:
        """Resolve coordinates into the nearest structured address."""

        return await self._reverse_geocode(
            lat=lat,
            lon=lon,
            context=context,
            language=language,
            entity_type=None,
        )

    async def reverse_geocode_city(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str = "NGT",
    ) -> str | None:
        """Resolve coordinates directly to TomTom's municipality geography."""

        result = await self._reverse_geocode(
            lat=lat,
            lon=lon,
            context=context,
            language=language,
            entity_type="Municipality",
        )
        return result.municipality

    async def _reverse_geocode(
        self,
        *,
        lat: float,
        lon: float,
        context: ToolExecutionContext,
        language: str,
        entity_type: str | None,
    ) -> ReverseGeocodeOutput:
        payload = await self._client.reverse_search(
            lat=lat,
            lon=lon,
            context=context,
            language=language,
            entity_type=entity_type,
        )
        try:
            response = TomTomReverseGeocodeResponse.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "TomTom reverse geocoder gave wrong answer schema",
                provider=self._client.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc

        if not response.addresses:
            return ReverseGeocodeOutput()

        address = response.addresses[0].address
        return ReverseGeocodeOutput(
            address=address.freeform_address,
            municipality=address.municipality or address.local_name,
            municipality_subdivision=address.municipality_subdivision,
            neighbourhood=address.neighbourhood,
            country_subdivision=address.country_subdivision,
            country_secondary_subdivision=address.country_secondary_subdivision,
            postal_code=address.postal_code,
            country=address.country,
            country_code=address.country_code,
            country_code_iso3=address.country_code_iso3,
        )

    @staticmethod
    def _kind(
        candidate: TomTomGeocodeResult,
        *,
        locality_only: bool,
    ) -> ToponymKind | None:
        if candidate.type == "Point Address":
            return ToponymKind.HOUSE
        if candidate.type in {"Street", "Cross Street", "Address Range"}:
            return ToponymKind.STREET
        if candidate.type != "Geography":
            return None
        if candidate.entity_type == "Municipality":
            return ToponymKind.LOCALITY
        if locality_only and candidate.entity_type == "CountrySecondarySubdivision":
            return ToponymKind.LOCALITY
        if candidate.entity_type in {
            "MunicipalitySubdivision",
            "MunicipalitySecondarySubdivision",
            "Neighbourhood",
        }:
            return ToponymKind.DISTRICT
        return None

    @staticmethod
    def _locality(
        candidate: TomTomGeocodeResult,
        *,
        name: str,
        kind: ToponymKind | None,
    ) -> str | None:
        """Return the canonical locality label stored with a city-scope result."""

        if kind is not ToponymKind.LOCALITY:
            return _clean_display_label(candidate.address.municipality)
        return _clean_display_label(
            candidate.address.municipality
            or candidate.address.local_name
            or candidate.address.country_secondary_subdivision
            or name
        )

    @staticmethod
    def _name(candidate: TomTomGeocodeResult) -> str:
        address = candidate.address
        if candidate.type == "Geography":
            name = (
                address.local_name
                or address.municipality
                or address.municipality_subdivision
                or address.freeform_address
                or candidate.id
            )
        elif candidate.type in {"Street", "Cross Street", "Address Range"}:
            name = address.street_name or address.freeform_address or candidate.id
        elif address.street_name is not None and address.street_number is not None:
            name = f"{address.street_name}, {address.street_number}"
        else:
            name = address.freeform_address or address.street_name or candidate.id
        return _clean_display_label(name) or candidate.id

    @staticmethod
    def _address(
        candidate: TomTomGeocodeResult,
        *,
        name: str,
        kind: ToponymKind | None,
    ) -> str:
        """Prefer structured administrative context for locality choices."""

        address = candidate.address
        if kind is not ToponymKind.LOCALITY:
            return _clean_display_label(address.freeform_address) or name

        parts = (
            _clean_display_label(address.municipality or address.local_name) or name,
            _clean_display_label(address.country_secondary_subdivision),
            _clean_display_label(address.country_subdivision),
            _clean_display_label(address.country),
        )
        unique_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            if part is None:
                continue
            cleaned = " ".join(part.split())
            normalized = cleaned.casefold()
            if cleaned and normalized not in seen:
                seen.add(normalized)
                unique_parts.append(cleaned)
        return ", ".join(unique_parts) or _clean_display_label(address.freeform_address) or name

    @staticmethod
    def _precision(candidate: TomTomGeocodeResult) -> str | None:
        confidence = candidate.match_confidence
        if confidence is None:
            return None
        return "exact" if confidence.score >= 0.999 else "other"

    @staticmethod
    def _bounds(candidate: TomTomGeocodeResult) -> GeoBounds | None:
        bounds = candidate.bounding_box
        if bounds is None:
            return None
        return GeoBounds(
            west=bounds.top_left.lon,
            south=bounds.bottom_right.lat,
            east=bounds.bottom_right.lon,
            north=bounds.top_left.lat,
        )


def _clean_display_label(value: str | None) -> str | None:
    """Remove TomTom's terminal apostrophe used for a transliterated soft sign."""

    if value is None:
        return None
    return _TRAILING_TRANSLITERATION_APOSTROPHE.sub("", value)
