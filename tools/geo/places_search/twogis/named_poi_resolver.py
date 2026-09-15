"""Coverage-aware 2GIS lookup for resolving one textual search anchor."""

from __future__ import annotations

from tools.base import ToolClarification, ToolClarificationOption
from tools.geo.distance import distance_m as calculate_distance_m
from tools.geo.errors import AmbiguousPlaceError
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.tomtom.matching import normalized_tokens
from tools.geo.places_search.twogis.client import TwoGisSearchClient
from tools.geo.places_search.twogis.matching import build_named_candidates, has_explicit_address
from tools.geo.places_search.twogis.schemas import TwoGisItem, TwoGisItemsResponse
from tools.geo.text_place_query import (
    compose_scoped_place_query,
    twogis_response_locale,
    uses_cyrillic,
)
from tools.geo.text_place_resolution import PlaceResolutionArea
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds, PlaceRecord, RecordOrigin, ResolvedPlace, mint_place_ref

_ANCHOR_EQUIVALENCE_DISTANCE_M = 50
_MAX_CLARIFICATION_OPTIONS = 3


class TwoGisNamedPoiResolver:
    """Resolve a POI, address, or toponym when 2GIS covers its locality."""

    provider = "twogis"
    requires_city_bounds = False

    def __init__(
        self,
        *,
        client: TwoGisSearchClient,
        place_store: PlaceStore,
    ) -> None:
        self._client = client
        self._place_store = place_store

    async def resolve_named_poi(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Check coverage, persist the first valid item in provider order."""

        response = await self._search(
            query=query,
            city=city,
            context=context,
            city_bounds=city_bounds,
            area=area,
            native_city_scope=False,
        )
        if response is None:
            return None
        items = _provider_items(
            response.result.items,
            query=query,
            city=city,
        )
        if not items:
            return None

        items = _prefer_unique_name_contained_in_query(items, query=query)
        groups = _anchor_groups(items)
        if len(groups) > 1:
            representatives = [group[0] for group in groups[:_MAX_CLARIFICATION_OPTIONS]]
            records = [_place_record(item, city=city) for item in representatives]
            await self._place_store.save_many(records)
            raise AmbiguousPlaceError(
                query,
                clarification=ToolClarification(
                    kind="select_anchor",
                    question=_clarification_question(query, groups),
                    options=[
                        ToolClarificationOption(
                            value=record.ref,
                            label=_clarification_label(item),
                            description=record.address,
                        )
                        for item, record in zip(representatives, records, strict=True)
                    ],
                ),
            )

        item = groups[0][0]
        record = _place_record(item, city=city)
        await self._place_store.save(record)
        return ResolvedPlace(ref=record.ref, record=record)

    async def resolve_first_address(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None = None,
        area: PlaceResolutionArea | None = None,
    ) -> ResolvedPlace | None:
        """Return the first provider card with a point and explicit address."""

        response = await self._search(
            query=query,
            city=city,
            context=context,
            city_bounds=city_bounds,
            area=area,
            native_city_scope=True,
        )
        if response is None:
            return None
        item = next(
            (
                item
                for item in response.result.items
                if item.point is not None and has_explicit_address(item)
            ),
            None,
        )
        if item is None:
            return None
        record = _place_record(item, city=city)
        await self._place_store.save(record)
        return ResolvedPlace(ref=record.ref, record=record)

    async def _search(
        self,
        *,
        query: str,
        city: str,
        context: ToolExecutionContext,
        city_bounds: GeoBounds | None,
        area: PlaceResolutionArea | None,
        native_city_scope: bool,
    ) -> TwoGisItemsResponse | None:
        del city_bounds

        if area is not None and area.lat is not None and area.lon is not None:
            region = await self._client.find_region_at_point(
                lon=area.lon,
                lat=area.lat,
                context=context,
            )
        else:
            region = await self._client.find_region(city, context)
        if region is None:
            return None

        if native_city_scope:
            locale = twogis_response_locale(
                query,
                country_code=region.country_code,
            )
            if area is not None and area.lat is not None and area.lon is not None:
                resolved_city = await self._client.find_city_at_point(
                    lon=area.lon,
                    lat=area.lat,
                    expected_region_id=region.id,
                    locale=locale,
                    context=context,
                )
            else:
                resolved_city = await self._client.find_city(
                    city,
                    context=context,
                    expected_region_id=region.id,
                    country_code=region.country_code,
                )
            if resolved_city is None:
                return None
            return await self._client.search_places(
                query=query,
                city_id=resolved_city.id,
                page_size=10,
                locale=locale,
                context=context,
            )

        return await self._client.search_places(
            query=compose_scoped_place_query(
                query=query,
                city=city,
                city_first=False,
            ),
            page_size=10,
            locale=twogis_response_locale(
                query,
                country_code=region.country_code,
            ),
            context=context,
        )


def _provider_items(
    items: list[TwoGisItem],
    *,
    query: str,
    city: str,
) -> list[TwoGisItem]:
    """Keep valid provider items without changing their relative order."""

    return [candidate.item for candidate in build_named_candidates(items, query=query, city=city)]


def _prefer_unique_name_contained_in_query(
    items: list[TwoGisItem],
    *,
    query: str,
) -> list[TwoGisItem]:
    """Prefer the first result when it is the sole fully represented name.

    A natural-language anchor may wrap an organisation name in qualifiers and
    inflection, for example ``офис Яндекса на Красной Розе``.  The catalog can
    return both ``Яндекс`` and ``Яндекс, фирменный магазин``.  Only the former
    has its complete name represented in the request; choosing it avoids a
    false clarification while preserving provider order and real ambiguity
    when several items have names fully represented by the query.
    """

    query_tokens = normalized_tokens(query)
    exact_items = [
        item
        for item in items
        if (name_tokens := normalized_tokens(item.name))
        and all(
            any(_same_lexeme(name_token, query_token) for query_token in query_tokens)
            for name_token in name_tokens
        )
    ]
    return [items[0]] if exact_items == [items[0]] else items


def _same_lexeme(left: str, right: str) -> bool:
    """Match an exact token or a conservative inflected suffix variant."""

    if left == right:
        return True
    return min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left))


def _anchor_groups(items: list[TwoGisItem]) -> list[list[TwoGisItem]]:
    """Group items that are equivalent coordinates for a nearby-search anchor."""

    groups: list[list[TwoGisItem]] = []
    for item in items:
        group = next(
            (
                existing
                for existing in groups
                if any(_same_anchor_location(item, member) for member in existing)
            ),
            None,
        )
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    return groups


def _same_anchor_location(left: TwoGisItem, right: TwoGisItem) -> bool:
    left_building = (
        left.structured_address.building_id if left.structured_address is not None else None
    )
    right_building = (
        right.structured_address.building_id if right.structured_address is not None else None
    )
    if left_building is not None and left_building == right_building:
        return True

    assert left.point is not None and right.point is not None
    return (
        calculate_distance_m(
            from_lat=left.point.lat,
            from_lon=left.point.lon,
            to_lat=right.point.lat,
            to_lon=right.point.lon,
        )
        <= _ANCHOR_EQUIVALENCE_DISTANCE_M
    )


def _clarification_question(query: str, groups: list[list[TwoGisItem]]) -> str:
    network_names, network_group_counts = _network_metadata(groups)
    network = next(
        (
            network_names[key] or query
            for key, count in network_group_counts.items()
            if count == len(groups)
        ),
        None,
    )
    if network is not None:
        if uses_cyrillic(query):
            return f"Какой объект сети {network!r} вы имели в виду?"
        return f"Which {network!r} location do you mean?"
    if uses_cyrillic(query):
        return f"Какой объект по запросу {query!r} вы имели в виду?"
    return f"Which place matching {query!r} do you mean?"


def _network_metadata(
    groups: list[list[TwoGisItem]],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int]]:
    network_names: dict[tuple[str, str], str] = {}
    network_group_counts: dict[tuple[str, str], int] = {}
    for group in groups:
        seen_in_group: set[tuple[str, str]] = set()
        for item in group:
            entity_kind: str | None = None
            entity_id: str | None = None
            entity_name: str | None = None
            if item.brand is not None and item.brand.id is not None:
                entity_kind = "brand"
                entity_id = item.brand.id
                entity_name = item.brand.name
            elif item.org is not None and item.org.id is not None:
                entity_kind = "org"
                entity_id = item.org.id
                entity_name = item.org.name
            if entity_kind is None or entity_id is None:
                continue
            key = (entity_kind, entity_id)
            network_names[key] = entity_name or ""
            seen_in_group.add(key)
        for key in seen_in_group:
            network_group_counts[key] = network_group_counts.get(key, 0) + 1
    return network_names, network_group_counts


def _clarification_label(item: TwoGisItem) -> str:
    """Return a human-readable identity, not only a short catalogue alias."""

    item_tokens = set(normalized_tokens(item.name))
    if item.org is not None and item.org.name is not None:
        organisation_tokens = set(normalized_tokens(item.org.name))
        if item_tokens < organisation_tokens:
            return item.org.name

    if item.name_ex is not None and item.name_ex.legal_name is not None:
        legal_name = item.name_ex.legal_name
        if normalized_tokens(legal_name) != normalized_tokens(item.name):
            return f"{item.name}, {legal_name}"

    if item.category_names:
        return f"{item.name} ({item.category_names[0]})"
    return item.name


def _place_record(item: TwoGisItem, *, city: str) -> PlaceRecord:
    assert item.point is not None  # Selected items require a routable point.
    return PlaceRecord(
        ref=mint_place_ref(f"twogis:item:{item.id}"),
        name=item.name,
        address=item.address,
        lat=item.point.lat,
        lon=item.point.lon,
        locality=item.locality or city,
        provider="twogis",
        provider_id=item.id,
        origin=RecordOrigin.PLACES_SEARCH,
    )
