"""Safe Overpass QL construction tests."""

from __future__ import annotations

from tools.geo.places_search.osm.query import (
    build_overpass_query,
    osm_category_tags,
)
from tools.geo.places_search.schemas import PlaceCategory
from tools.refs import GeoBounds


def test_coffee_shop_area_query_maps_boundary_to_overpass_area():
    query = build_overpass_query(
        text="кофейни",
        category=PlaceCategory.COFFEE_SHOP,
        limit=50,
        timeout_s=12,
        open_24h=False,
        bbox=GeoBounds(west=36.8, south=55.1, east=38.0, north=56.0),
        boundary_name="Москва",
    )

    for key in ("name", "name:ru", "official_name", "short_name"):
        assert (
            f'rel["boundary"="administrative"]["{key}"="Москва"]'
            "(55.100000,36.800000,56.000000,38.000000);" in query
        )
        assert (
            'rel["type"="multipolygon"]'
            '["place"~"^(city|town|village|municipality)$"]'
            f'["{key}"="Москва"]'
            "(55.100000,36.800000,56.000000,38.000000);" in query
        )
    assert ".searchBoundaries map_to_area ->.searchArea;" in query
    assert ".searchArea out ids;" in query
    assert ('nwr["amenity"="cafe"]["cuisine"~"(^|;)coffee_shop(;|$)"](area.searchArea);') in query
    assert "out body geom" not in query
    assert '["addr:city"' not in query
    assert '["addr:place"' not in query
    assert "кофейни" not in query
    assert ".matches out tags center 50;" in query
    assert ".matches out count;" in query


def test_near_query_uses_lat_lon_order_and_24h_filter():
    query = build_overpass_query(
        text="аптеки",
        category=PlaceCategory.PHARMACY,
        limit=25,
        timeout_s=10,
        open_24h=True,
        center=(37.6208, 55.7539),
        radius_m=1_000,
    )

    assert (
        'nwr["amenity"="pharmacy"]["opening_hours"="24/7"](around:1000,55.753900,37.620800);'
    ) in query


def test_open_now_query_requires_schedule_for_local_evaluation():
    query = build_overpass_query(
        text="рестораны",
        category=PlaceCategory.RESTAURANT,
        limit=50,
        timeout_s=12,
        open_24h=False,
        open_now=True,
        center=(37.6208, 55.7539),
        radius_m=1_000,
    )

    assert (
        'nwr["amenity"="restaurant"]["opening_hours"](around:1000,55.753900,37.620800);'
    ) in query
    assert '["opening_hours"="24/7"]' not in query


def test_name_query_uses_exact_match_and_escapes_overpass_string_metacharacters():
    query = build_overpass_query(
        text='A.[x] "quoted"); out;',
        category=None,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=500,
    )

    assert query.count("nwr[") == 5
    assert 'nwr["name"="A.[x] \\"quoted\\"); out;"]' in query
    assert query.count('\\"quoted\\"); out;"]') == 5
    assert '["name"~' not in query


def test_cafe_remains_broad_for_general_cafe_requests():
    query = build_overpass_query(
        text="кафе",
        category=PlaceCategory.CAFE,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=500,
    )

    assert 'nwr["amenity"="cafe"](around:500,55.000000,37.000000);' in query
    assert '["cuisine"' not in query


def test_pizzeria_matches_supported_amenities_with_pizza_cuisine():
    query = build_overpass_query(
        text="пиццерии",
        category=PlaceCategory.PIZZERIA,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=500,
    )

    for amenity in ("restaurant", "fast_food", "cafe"):
        assert (
            f'nwr["amenity"="{amenity}"]["cuisine"~"(^|;)pizza(;|$)"]'
            "(around:500,55.000000,37.000000);"
        ) in query


def test_public_place_categories_exclude_explicitly_private_objects():
    for category, key, value in (
        (PlaceCategory.CHARGING_STATION, "amenity", "charging_station"),
        (PlaceCategory.PARKING, "amenity", "parking"),
        (PlaceCategory.PLAYGROUND, "leisure", "playground"),
    ):
        query = build_overpass_query(
            text=category.value,
            category=category,
            limit=20,
            timeout_s=12,
            open_24h=False,
            center=(37.0, 55.0),
            radius_m=500,
        )

        assert (
            f'nwr["{key}"="{value}"]["access"!~"^(private|no)$"](around:500,55.000000,37.000000);'
        ) in query


def test_swimming_pool_includes_public_pool_facilities():
    query = build_overpass_query(
        text="бассейны",
        category=PlaceCategory.SWIMMING_POOL,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=500,
    )

    assert (
        'nwr["leisure"="swimming_pool"]["access"!~"^(private|no)$"]'
        "(around:500,55.000000,37.000000);"
    ) in query
    for leisure in ("sports_centre", "sports_hall"):
        assert (
            f'nwr["leisure"="{leisure}"]["sport"~"(^|;)swimming(;|$)"]'
            '["access"!~"^(private|no)$"](around:500,55.000000,37.000000);'
        ) in query


def test_every_public_category_has_an_osm_mapping():
    for category in PlaceCategory:
        assert osm_category_tags(category), category


def test_new_common_categories_use_canonical_osm_tags():
    expected = {
        PlaceCategory.FLOWER_SHOP: (("shop", "florist"),),
        PlaceCategory.GARDEN_CENTRE: (("shop", "garden_centre"),),
        PlaceCategory.PET_STORE: (("shop", "pet"),),
        PlaceCategory.CAR_WASH: (("amenity", "car_wash"),),
        PlaceCategory.NIGHTCLUB: (("amenity", "nightclub"),),
    }

    for category, tag_pairs in expected.items():
        filters = osm_category_tags(category)
        assert tuple((item.key, item.value) for item in filters) == tag_pairs


def test_flower_shop_query_searches_florists_not_names():
    query = build_overpass_query(
        text="магазины цветов",
        category=PlaceCategory.FLOWER_SHOP,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=1_000,
    )

    assert 'nwr["shop"="florist"](around:1000,55.000000,37.000000);' in query
    assert "магазины цветов" not in query


def test_train_station_excludes_explicit_subway_stations():
    query = build_overpass_query(
        text="железнодорожные вокзалы",
        category=PlaceCategory.TRAIN_STATION,
        limit=20,
        timeout_s=12,
        open_24h=False,
        center=(37.0, 55.0),
        radius_m=1_000,
    )

    assert (
        'nwr["railway"="station"]["station"!~"^(subway|light_rail|monorail)$"]'
        "(around:1000,55.000000,37.000000);"
    ) in query
