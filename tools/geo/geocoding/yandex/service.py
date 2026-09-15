"""Yandex-backed implementation of the internal geocoding service."""

from __future__ import annotations

from pydantic import ValidationError

from tools.base import ToolErrorCode, ToolExecutionError, ToolFailureKind
from tools.geo.geocoding.schemas import (
    GeocodePlaceInput,
    GeocodePlaceOutput,
    PlaceMatch,
    ToponymKind,
)
from tools.geo.geocoding.yandex.client import YandexGeocoderClient
from tools.geo.geocoding.yandex.schemas import (
    YandexGeocoderResponse,
    YandexGeoObject,
)
from tools.geo.place_store import PlaceStore
from tools.geo.text_place_query import compose_scoped_place_query
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, mint_place_ref


class YandexGeocoderService:
    """Resolve places with Yandex and persist hidden coordinates."""

    provider = "yandex_geocoder"

    def __init__(
        self,
        client: YandexGeocoderClient,
        place_store: PlaceStore,
    ) -> None:
        self._client = client
        self._place_store = place_store

    async def geocode(
        self,
        args: GeocodePlaceInput,
        context: ToolExecutionContext,
    ) -> GeocodePlaceOutput:
        query = self._build_query(args)
        payload = await self._client.search(query, limit=args.limit, context=context)
        try:
            response = YandexGeocoderResponse.model_validate(payload)
        except ValidationError as exc:
            raise ToolExecutionError(
                ToolErrorCode.UPSTREAM_ERROR,
                "Yandex geocoder gave wrong answer schema",
                provider=self._client.provider,
                failure_kind=ToolFailureKind.INVALID_SCHEMA,
                retryable=False,
            ) from exc

        matches: list[PlaceMatch] = []
        records: list[PlaceRecord] = []
        seen_refs: set[str] = set()

        for candidate in response.candidates:
            ref = mint_place_ref(self._identity(candidate))
            if ref in seen_refs:
                continue

            seen_refs.add(ref)
            records.append(
                PlaceRecord(
                    ref=ref,
                    name=candidate.name,
                    address=candidate.geocoder_metadata.text,
                    lat=candidate.point.lat,
                    lon=candidate.point.lon,
                    kind=candidate.geocoder_metadata.kind,
                    precision=candidate.geocoder_metadata.precision,
                    locality=candidate.geocoder_metadata.locality,
                    bounds=self._to_bounds(candidate),
                    provider=self.provider,
                    provider_uri=candidate.uri,
                    origin=RecordOrigin.GEOCODE,
                )
            )
            matches.append(
                PlaceMatch(
                    ref=ref,
                    name=candidate.name,
                    address=candidate.geocoder_metadata.text,
                    kind=self._to_toponym_kind(candidate.geocoder_metadata.kind),
                    precision=candidate.geocoder_metadata.precision,
                ),
            )

        if records:
            await self._place_store.save_many(records)

        return GeocodePlaceOutput(
            best=matches[0] if matches else None,
            matches=matches,
            ambiguous=len(matches) > 1,
        )

    @staticmethod
    def _build_query(args: GeocodePlaceInput) -> str:
        # Forward geocoding works best with the normal address hierarchy:
        # locality first, then street/place and house.
        return compose_scoped_place_query(
            query=args.query,
            city=args.city,
            city_first=True,
        )

    @staticmethod
    def _identity(candidate: YandexGeoObject) -> str:
        if candidate.uri is not None:
            return f"yandex:uri:{candidate.uri}"

        return (
            "yandex:coordinates:"
            f"{candidate.point.lon:.6f},"
            f"{candidate.point.lat:.6f}:"
            f"{candidate.geocoder_metadata.text}"
        )

    @staticmethod
    def _to_bounds(candidate: YandexGeoObject) -> GeoBounds | None:
        if candidate.bounds is None:
            return None

        return GeoBounds(
            west=candidate.bounds.west,
            south=candidate.bounds.south,
            east=candidate.bounds.east,
            north=candidate.bounds.north,
        )

    @staticmethod
    def _to_toponym_kind(kind: str) -> ToponymKind | None:
        try:
            return ToponymKind(kind)
        except ValueError:
            return None
