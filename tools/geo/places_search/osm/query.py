"""Safe Overpass QL construction for OpenStreetMap place search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.geo.places_search.schemas import PlaceCategory
from tools.refs import GeoBounds

_NAME_TAGS = ("name", "name:ru", "name:en", "brand", "operator")


@dataclass(frozen=True, slots=True)
class OsmTagConstraint:
    """One additional controlled condition on an OSM tag."""

    key: str
    value: str
    operator: Literal["=", "~", "!~"] = "="


@dataclass(frozen=True, slots=True)
class OsmTagFilter:
    """One controlled OSM tag selector and its additional conditions."""

    key: str
    value: str
    constraints: tuple[OsmTagConstraint, ...] = ()


_PUBLIC_ACCESS = OsmTagConstraint("access", "^(private|no)$", "!~")
_COFFEE_CUISINE = OsmTagConstraint("cuisine", "(^|;)coffee_shop(;|$)", "~")
_PIZZA_CUISINE = OsmTagConstraint("cuisine", "(^|;)pizza(;|$)", "~")
_SWIMMING_SPORT = OsmTagConstraint("sport", "(^|;)swimming(;|$)", "~")
_NON_SUBWAY_STATION = OsmTagConstraint(
    "station",
    "^(subway|light_rail|monorail)$",
    "!~",
)


_CATEGORY_FILTERS: dict[PlaceCategory, tuple[OsmTagFilter, ...]] = {
    PlaceCategory.AIRPORT: (OsmTagFilter("aeroway", "aerodrome"),),
    PlaceCategory.ALCOHOL_STORE: (OsmTagFilter("shop", "alcohol"),),
    PlaceCategory.ARTS_CENTRE: (OsmTagFilter("amenity", "arts_centre"),),
    PlaceCategory.ATM: (OsmTagFilter("amenity", "atm"),),
    PlaceCategory.ATTRACTION: (OsmTagFilter("tourism", "attraction"),),
    PlaceCategory.BAKERY: (OsmTagFilter("shop", "bakery"),),
    PlaceCategory.BANK: (OsmTagFilter("amenity", "bank"),),
    PlaceCategory.BAR: (OsmTagFilter("amenity", "bar"),),
    PlaceCategory.BEAUTY_SALON: (OsmTagFilter("shop", "beauty"),),
    PlaceCategory.BICYCLE_RENTAL: (OsmTagFilter("amenity", "bicycle_rental"),),
    PlaceCategory.BICYCLE_STORE: (OsmTagFilter("shop", "bicycle"),),
    PlaceCategory.BOOKSTORE: (OsmTagFilter("shop", "books"),),
    PlaceCategory.BUS_STATION: (OsmTagFilter("amenity", "bus_station"),),
    PlaceCategory.BUTCHER: (OsmTagFilter("shop", "butcher"),),
    PlaceCategory.CAFE: (OsmTagFilter("amenity", "cafe"),),
    PlaceCategory.CAR_DEALER: (OsmTagFilter("shop", "car"),),
    PlaceCategory.CAR_RENTAL: (OsmTagFilter("amenity", "car_rental"),),
    PlaceCategory.CAR_REPAIR: (OsmTagFilter("shop", "car_repair"),),
    PlaceCategory.CAR_WASH: (OsmTagFilter("amenity", "car_wash"),),
    PlaceCategory.CASINO: (OsmTagFilter("amenity", "casino"),),
    PlaceCategory.CHARGING_STATION: (
        OsmTagFilter("amenity", "charging_station", (_PUBLIC_ACCESS,)),
    ),
    PlaceCategory.CINEMA: (OsmTagFilter("amenity", "cinema"),),
    PlaceCategory.CLINIC: (
        OsmTagFilter("amenity", "clinic"),
        OsmTagFilter("healthcare", "clinic"),
    ),
    PlaceCategory.CLOTHING_STORE: (OsmTagFilter("shop", "clothes"),),
    PlaceCategory.COFFEE_SHOP: (OsmTagFilter("amenity", "cafe", (_COFFEE_CUISINE,)),),
    PlaceCategory.COMMUNITY_CENTRE: (OsmTagFilter("amenity", "community_centre"),),
    PlaceCategory.COMPUTER_STORE: (OsmTagFilter("shop", "computer"),),
    PlaceCategory.CONFECTIONERY: (OsmTagFilter("shop", "confectionery"),),
    PlaceCategory.CONVENIENCE_STORE: (OsmTagFilter("shop", "convenience"),),
    PlaceCategory.COSMETICS_STORE: (OsmTagFilter("shop", "cosmetics"),),
    PlaceCategory.DENTIST: (
        OsmTagFilter("amenity", "dentist"),
        OsmTagFilter("healthcare", "dentist"),
    ),
    PlaceCategory.DEPARTMENT_STORE: (OsmTagFilter("shop", "department_store"),),
    PlaceCategory.DOCTORS: (
        OsmTagFilter("amenity", "doctors"),
        OsmTagFilter("healthcare", "doctor"),
    ),
    PlaceCategory.DRY_CLEANING: (OsmTagFilter("shop", "dry_cleaning"),),
    PlaceCategory.ELECTRONICS_STORE: (OsmTagFilter("shop", "electronics"),),
    PlaceCategory.FAST_FOOD: (OsmTagFilter("amenity", "fast_food"),),
    PlaceCategory.FIRE_STATION: (OsmTagFilter("amenity", "fire_station"),),
    PlaceCategory.FITNESS_CENTRE: (OsmTagFilter("leisure", "fitness_centre"),),
    PlaceCategory.FLOWER_SHOP: (OsmTagFilter("shop", "florist"),),
    PlaceCategory.FOOD_COURT: (OsmTagFilter("amenity", "food_court"),),
    PlaceCategory.FUEL: (OsmTagFilter("amenity", "fuel"),),
    PlaceCategory.FURNITURE_STORE: (OsmTagFilter("shop", "furniture"),),
    PlaceCategory.GALLERY: (OsmTagFilter("tourism", "gallery"),),
    PlaceCategory.GARDEN_CENTRE: (OsmTagFilter("shop", "garden_centre"),),
    PlaceCategory.GIFT_SHOP: (OsmTagFilter("shop", "gift"),),
    PlaceCategory.GREENGROCER: (OsmTagFilter("shop", "greengrocer"),),
    PlaceCategory.GUEST_HOUSE: (OsmTagFilter("tourism", "guest_house"),),
    PlaceCategory.HAIRDRESSER: (OsmTagFilter("shop", "hairdresser"),),
    PlaceCategory.HARDWARE_STORE: (
        OsmTagFilter("shop", "hardware"),
        OsmTagFilter("shop", "doityourself"),
    ),
    PlaceCategory.HOSPITAL: (
        OsmTagFilter("amenity", "hospital"),
        OsmTagFilter("healthcare", "hospital"),
    ),
    PlaceCategory.HOSTEL: (OsmTagFilter("tourism", "hostel"),),
    PlaceCategory.HOTEL: (
        OsmTagFilter("tourism", "hotel"),
        OsmTagFilter("tourism", "motel"),
    ),
    PlaceCategory.ICE_CREAM: (OsmTagFilter("amenity", "ice_cream"),),
    PlaceCategory.JEWELRY_STORE: (OsmTagFilter("shop", "jewelry"),),
    PlaceCategory.KINDERGARTEN: (OsmTagFilter("amenity", "kindergarten"),),
    PlaceCategory.LAUNDRY: (OsmTagFilter("shop", "laundry"),),
    PlaceCategory.LIBRARY: (OsmTagFilter("amenity", "library"),),
    PlaceCategory.MALL: (OsmTagFilter("shop", "mall"),),
    PlaceCategory.MARKETPLACE: (OsmTagFilter("amenity", "marketplace"),),
    PlaceCategory.MOBILE_PHONE_STORE: (OsmTagFilter("shop", "mobile_phone"),),
    PlaceCategory.MUSEUM: (OsmTagFilter("tourism", "museum"),),
    PlaceCategory.MUSIC_VENUE: (OsmTagFilter("amenity", "music_venue"),),
    PlaceCategory.NIGHTCLUB: (OsmTagFilter("amenity", "nightclub"),),
    PlaceCategory.OPTICIAN: (OsmTagFilter("shop", "optician"),),
    PlaceCategory.PARK: (OsmTagFilter("leisure", "park"),),
    PlaceCategory.PARKING: (OsmTagFilter("amenity", "parking", (_PUBLIC_ACCESS,)),),
    PlaceCategory.PET_STORE: (OsmTagFilter("shop", "pet"),),
    PlaceCategory.PHARMACY: (OsmTagFilter("amenity", "pharmacy"),),
    PlaceCategory.PIZZERIA: (
        OsmTagFilter("amenity", "restaurant", (_PIZZA_CUISINE,)),
        OsmTagFilter("amenity", "fast_food", (_PIZZA_CUISINE,)),
        OsmTagFilter("amenity", "cafe", (_PIZZA_CUISINE,)),
    ),
    PlaceCategory.PLACE_OF_WORSHIP: (OsmTagFilter("amenity", "place_of_worship"),),
    PlaceCategory.PLAYGROUND: (OsmTagFilter("leisure", "playground", (_PUBLIC_ACCESS,)),),
    PlaceCategory.POLICE: (OsmTagFilter("amenity", "police"),),
    PlaceCategory.POST_OFFICE: (OsmTagFilter("amenity", "post_office"),),
    PlaceCategory.PUB: (OsmTagFilter("amenity", "pub"),),
    PlaceCategory.RESTAURANT: (OsmTagFilter("amenity", "restaurant"),),
    PlaceCategory.SCHOOL: (OsmTagFilter("amenity", "school"),),
    PlaceCategory.SHOE_STORE: (OsmTagFilter("shop", "shoes"),),
    PlaceCategory.SPORTS_CENTRE: (OsmTagFilter("leisure", "sports_centre"),),
    PlaceCategory.SPORTS_STORE: (OsmTagFilter("shop", "sports"),),
    PlaceCategory.STATIONERY_STORE: (OsmTagFilter("shop", "stationery"),),
    PlaceCategory.SUPERMARKET: (OsmTagFilter("shop", "supermarket"),),
    PlaceCategory.SWIMMING_POOL: (
        OsmTagFilter("leisure", "swimming_pool", (_PUBLIC_ACCESS,)),
        OsmTagFilter(
            "leisure",
            "sports_centre",
            (_SWIMMING_SPORT, _PUBLIC_ACCESS),
        ),
        OsmTagFilter(
            "leisure",
            "sports_hall",
            (_SWIMMING_SPORT, _PUBLIC_ACCESS),
        ),
    ),
    PlaceCategory.TAXI: (OsmTagFilter("amenity", "taxi"),),
    PlaceCategory.THEATRE: (OsmTagFilter("amenity", "theatre"),),
    PlaceCategory.TOY_STORE: (OsmTagFilter("shop", "toys"),),
    PlaceCategory.TRAIN_STATION: (OsmTagFilter("railway", "station", (_NON_SUBWAY_STATION,)),),
    PlaceCategory.TRAVEL_AGENCY: (OsmTagFilter("shop", "travel_agency"),),
    PlaceCategory.UNIVERSITY: (OsmTagFilter("amenity", "university"),),
    PlaceCategory.VETERINARY: (OsmTagFilter("amenity", "veterinary"),),
    PlaceCategory.VIEWPOINT: (OsmTagFilter("tourism", "viewpoint"),),
    PlaceCategory.ZOO: (OsmTagFilter("tourism", "zoo"),),
}


def _ql_string(value: str) -> str:
    """Escape a literal for an Overpass QL double-quoted string."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _tag_filter_ql(tag_filter: OsmTagFilter) -> str:
    parts = [
        f'["{_ql_string(tag_filter.key)}"="{_ql_string(tag_filter.value)}"]',
    ]
    parts.extend(
        (f'["{_ql_string(constraint.key)}"{constraint.operator}"{_ql_string(constraint.value)}"]')
        for constraint in tag_filter.constraints
    )
    return "".join(parts)


def _scope(
    *,
    bbox: GeoBounds | None,
    center: tuple[float, float] | None,
    radius_m: int | None,
) -> str:
    if bbox is not None:
        if center is not None or radius_m is not None:
            raise ValueError("bbox cannot be combined with center and radius")
        return f"({bbox.south:.6f},{bbox.west:.6f},{bbox.north:.6f},{bbox.east:.6f})"

    if center is None or radius_m is None:
        raise ValueError("center and radius must be passed together")
    if not 1 <= radius_m <= 100_000:
        raise ValueError("radius must be between 1 and 100000 metres")

    lon, lat = center
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("center coordinates are out of range")

    return f"(around:{radius_m},{lat:.6f},{lon:.6f})"


def _selectors(
    *,
    cleaned_text: str,
    category: PlaceCategory | None,
    open_24h: bool,
    open_now: bool,
    spatial_filter: str,
) -> list[str]:
    if open_24h:
        hours_filter = '["opening_hours"="24/7"]'
    elif open_now:
        # Overpass can cheaply require a schedule tag, but evaluating the OSM
        # syntax correctly needs local time, holidays, and sometimes solar
        # events. The provider performs that exact check after this fetch.
        hours_filter = '["opening_hours"]'
    else:
        hours_filter = ""

    if category is not None:
        return [
            (f"nwr{_tag_filter_ql(tag_filter)}{hours_filter}{spatial_filter};")
            for tag_filter in _CATEGORY_FILTERS[category]
        ]

    # Without a category the public contract treats ``text`` as an exact place
    # name. Equality also lets Overpass use its tag index; a case-insensitive
    # regex here turns even a city-sized search into a scan and often times out.
    escaped_text = _ql_string(cleaned_text)
    return [
        (f'nwr["{_ql_string(key)}"="{escaped_text}"]{hours_filter}{spatial_filter};')
        for key in _NAME_TAGS
    ]


def _selector_block(selectors: list[str], *, indent: str) -> str:
    return f"\n{indent}".join(selectors)


def _boundary_area_block(boundary_name: str, bbox: GeoBounds) -> str:
    cleaned_name = " ".join(boundary_name.split())
    if not cleaned_name:
        raise ValueError("boundary_name must not be blank")

    escaped_name = _ql_string(cleaned_name)
    bbox_filter = _scope(bbox=bbox, center=None, radius_m=None)
    selectors = [
        f'rel{relation_filter}["{key}"="{escaped_name}"]{bbox_filter};'
        for relation_filter in (
            '["boundary"="administrative"]',
            ('["type"="multipolygon"]["place"~"^(city|town|village|municipality)$"]'),
        )
        for key in ("name", "name:ru", "official_name", "short_name")
    ]
    return (
        f"(\n  {_selector_block(selectors, indent='  ')}\n)"
        "->.searchBoundaries;\n"
        ".searchBoundaries map_to_area ->.searchArea;\n"
        ".searchArea out ids;\n"
    )


def build_overpass_query(
    *,
    text: str,
    category: PlaceCategory | None,
    limit: int,
    timeout_s: int,
    open_24h: bool,
    open_now: bool = False,
    bbox: GeoBounds | None = None,
    center: tuple[float, float] | None = None,
    radius_m: int | None = None,
    boundary_name: str | None = None,
) -> str:
    """Return one bounded Overpass query without accepting arbitrary QL."""

    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        raise ValueError("text must not be blank")
    if not 1 <= limit <= 1_000:
        raise ValueError("limit must be between 1 and 1000")
    if not 1 <= timeout_s <= 180:
        raise ValueError("timeout must be between 1 and 180 seconds")
    if boundary_name is not None and bbox is None:
        raise ValueError("boundary_name can only be used with bbox")

    if boundary_name is not None:
        assert bbox is not None
        spatial_filter = "(area.searchArea)"
        boundary_block = _boundary_area_block(boundary_name, bbox)
    else:
        spatial_filter = _scope(bbox=bbox, center=center, radius_m=radius_m)
        boundary_block = ""

    selectors = _selectors(
        cleaned_text=cleaned_text,
        category=category,
        open_24h=open_24h,
        open_now=open_now,
        spatial_filter=spatial_filter,
    )
    body = f"(\n  {_selector_block(selectors, indent='  ')}\n)->.matches;"

    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f"{boundary_block}"
        f"{body}\n"
        f".matches out tags center {limit};\n"
        ".matches out count;"
    )


def osm_category_tags(category: PlaceCategory) -> tuple[OsmTagFilter, ...]:
    """Expose the controlled mapping for normalization and tests."""

    return _CATEGORY_FILTERS[category]
