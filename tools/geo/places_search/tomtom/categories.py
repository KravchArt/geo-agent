"""Reviewed public-category mapping for TomTom POI search endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from tools.geo.places_search.schemas import PlaceCategory


@dataclass(frozen=True, slots=True)
class TomTomCategorySpec:
    """Reviewed mapping between one public category and TomTom's taxonomy."""

    query: str
    classification_codes: frozenset[str]
    category_ids: tuple[int, ...] = ()
    category_names: frozenset[str] = frozenset()
    supports_queryless_nearby: bool = True


def _spec(
    query: str,
    *classification_codes: str,
    category_ids: tuple[int, ...] = (),
    category_names: tuple[str, ...] = (),
    supports_queryless_nearby: bool = True,
) -> TomTomCategorySpec:
    return TomTomCategorySpec(
        query=query,
        classification_codes=frozenset(classification_codes),
        category_ids=category_ids,
        category_names=frozenset(category_names),
        supports_queryless_nearby=supports_queryless_nearby,
    )


# Category Search accepts provider category names rather than our enum values.
# Keeping the translation explicit makes taxonomy drift visible in review and
# lets tests verify that every public PlaceCategory remains supported.
# Numeric IDs come from TomTom's official StandardCategoryID catalogue:
# https://developer.tomtom.com/assets/downloads/tomtom-sdks/ios/api-reference/0.35.0/
# TomTomSDKCommon/Enums/StandardCategoryID.html
# A four-digit ID denotes a parent category; response records can carry a
# seven-digit descendant whose first four digits identify that parent.
TOMTOM_CATEGORY_SPECS: Final = MappingProxyType(
    {
        PlaceCategory.AIRPORT: _spec("airport", "AIRPORT", category_ids=(7383,)),
        PlaceCategory.ALCOHOL_STORE: _spec(
            "food drinks: wine spirits",
            "SHOP",
            category_ids=(9361025,),
        ),
        PlaceCategory.ARTS_CENTRE: _spec(
            "cultural center",
            "CULTURAL_CENTER",
            category_ids=(7319,),
        ),
        # TomTom's official standard POI category identifier for ATM.
        PlaceCategory.ATM: _spec(
            "cash dispenser",
            "CASH_DISPENSER",
            category_ids=(7397,),
        ),
        PlaceCategory.ATTRACTION: _spec(
            "important tourist attraction",
            "IMPORTANT_TOURIST_ATTRACTION",
            category_ids=(7376,),
        ),
        PlaceCategory.BAKERY: _spec(
            "food drinks: bakers",
            "SHOP",
            category_ids=(9361018,),
        ),
        PlaceCategory.BANK: _spec(
            "bank",
            "BANK",
            category_ids=(7328,),
        ),
        PlaceCategory.BAR: _spec(
            "bar",
            "CAFE_PUB",
            "NIGHTLIFE",
            category_ids=(9379004,),
        ),
        PlaceCategory.BEAUTY_SALON: _spec(
            "beauty salon",
            "SHOP",
            category_ids=(9361067,),
        ),
        # TomTom has no bicycle-specific standard IDs. Its equipment-rental
        # and sports-shop categories are too broad for queryless Nearby Search.
        PlaceCategory.BICYCLE_RENTAL: _spec("equipment rental", "COMPANY"),
        PlaceCategory.BICYCLE_STORE: _spec("sports equipment clothing", "SHOP"),
        PlaceCategory.BOOKSTORE: _spec(
            "book shops",
            "SHOP",
            category_ids=(9361002,),
        ),
        PlaceCategory.BUS_STATION: _spec(
            "bus station",
            "PUBLIC_TRANSPORT_STOP",
            # 9942 contains bus stops, taxi stands, tram stops, and coach
            # stops. It is useful as an upstream filter only when combined
            # with the semantic "bus station" query; using it in queryless
            # Nearby Search would mostly return ordinary bus stops.
            category_ids=(9942,),
            category_names=("bus station",),
            supports_queryless_nearby=False,
        ),
        PlaceCategory.BUTCHER: _spec(
            "food drinks: butchers",
            "SHOP",
            category_ids=(9361019,),
        ),
        PlaceCategory.CAFE: _spec("café", "CAFE_PUB", category_ids=(9376002,)),
        PlaceCategory.CAR_DEALER: _spec(
            "automotive dealer",
            "AUTOMOTIVE_DEALER",
            category_ids=(9910002,),
        ),
        PlaceCategory.CAR_RENTAL: _spec(
            "rent-a-car facility",
            "RENT_A_CAR_FACILITY",
            category_ids=(7312,),
        ),
        PlaceCategory.CAR_REPAIR: _spec(
            "general car repair servicing",
            "REPAIR_FACILITY",
            category_ids=(7310004,),
        ),
        PlaceCategory.CAR_WASH: _spec(
            "car wash",
            "CAR_WASH",
            category_ids=(9155002,),
        ),
        PlaceCategory.CASINO: _spec("casino", "CASINO", category_ids=(7341,)),
        PlaceCategory.CHARGING_STATION: _spec(
            "electric vehicle station",
            "ELECTRIC_VEHICLE_STATION",
            category_ids=(7309,),
        ),
        PlaceCategory.CINEMA: _spec("cinema", "CINEMA", category_ids=(7342002,)),
        PlaceCategory.CLINIC: _spec(
            "clinic",
            "HEALTH_CARE_SERVICE",
            category_ids=(7321002,),
        ),
        PlaceCategory.CLOTHING_STORE: _spec(
            "clothing accessories: general",
            "SHOP",
            category_ids=(9361006,),
        ),
        PlaceCategory.COFFEE_SHOP: _spec(
            "coffee shop",
            "CAFE_PUB",
            category_ids=(9376006,),
        ),
        PlaceCategory.COMMUNITY_CENTRE: _spec(
            "community center",
            "COMMUNITY_CENTER",
            category_ids=(7363,),
        ),
        PlaceCategory.COMPUTER_STORE: _spec(
            "electrical, office it: computer computer supplies",
            "SHOP",
            category_ids=(9361012,),
        ),
        PlaceCategory.CONFECTIONERY: _spec(
            "specialty foods",
            "SHOP",
            category_ids=(9361061,),
        ),
        PlaceCategory.CONVENIENCE_STORE: _spec(
            "convenience stores",
            "SHOP",
            category_ids=(9361009,),
        ),
        PlaceCategory.COSMETICS_STORE: _spec(
            "beauty supplies",
            "SHOP",
            category_ids=(9361050,),
        ),
        PlaceCategory.DENTIST: _spec("dentist", "DENTIST", category_ids=(9374,)),
        PlaceCategory.DEPARTMENT_STORE: _spec(
            "department store",
            "DEPARTMENT_STORE",
            category_ids=(7327,),
        ),
        PlaceCategory.DOCTORS: _spec("doctor", "DOCTOR", category_ids=(9373,)),
        PlaceCategory.DRY_CLEANING: _spec(
            "dry cleaners",
            "SHOP",
            category_ids=(9361010,),
        ),
        PlaceCategory.ELECTRONICS_STORE: _spec(
            "electrical, office it: consumer electronics",
            "SHOP",
            category_ids=(9361013,),
        ),
        PlaceCategory.FAST_FOOD: _spec(
            "fast food",
            "RESTAURANT",
            category_ids=(7315015,),
        ),
        PlaceCategory.FIRE_STATION: _spec(
            "fire station/brigade",
            "FIRE_STATION_BRIGADE",
            category_ids=(7392,),
        ),
        PlaceCategory.FITNESS_CENTRE: _spec(
            "fitness club center",
            "SPORTS_CENTER",
            category_ids=(7320002,),
        ),
        PlaceCategory.FLOWER_SHOP: _spec(
            "florists",
            "SHOP",
            category_ids=(9361017,),
        ),
        PlaceCategory.FOOD_COURT: _spec(
            "restaurant area",
            "RESTAURANT_AREA",
            category_ids=(9359,),
        ),
        PlaceCategory.FUEL: _spec(
            "petrol station",
            "PETROL_STATION",
            category_ids=(7311,),
        ),
        PlaceCategory.FURNITURE_STORE: _spec(
            "furniture/home furnishings",
            "SHOP",
            category_ids=(9361054,),
        ),
        PlaceCategory.GALLERY: _spec(
            "antique/art",
            "SHOP",
            category_ids=(9361049,),
        ),
        PlaceCategory.GARDEN_CENTRE: _spec(
            "house garden: garden centers services",
            "SHOP",
            category_ids=(9361032,),
        ),
        PlaceCategory.GIFT_SHOP: _spec(
            "gifts, cards, novelties souvenirs",
            "SHOP",
            category_ids=(9361026,),
        ),
        PlaceCategory.GREENGROCER: _spec(
            "food drinks: green grocers",
            "SHOP",
            category_ids=(9361022,),
        ),
        PlaceCategory.GUEST_HOUSE: _spec(
            "guest house",
            "HOTEL_MOTEL",
            category_ids=(7314002,),
        ),
        PlaceCategory.HAIRDRESSER: _spec(
            "hairdressers barbers",
            "SHOP",
            category_ids=(9361027,),
        ),
        PlaceCategory.HARDWARE_STORE: _spec(
            "hardware",
            "SHOP",
            category_ids=(9361069,),
        ),
        PlaceCategory.HOSPITAL: _spec(
            "hospital",
            "HEALTH_CARE_SERVICE",
            "HOSPITAL_POLYCLINIC",
            category_ids=(7321,),
        ),
        PlaceCategory.HOSTEL: _spec(
            "hostel",
            "HOTEL_MOTEL",
            category_ids=(7314004,),
        ),
        PlaceCategory.HOTEL: _spec(
            "hotel",
            "HOTEL_MOTEL",
            category_ids=(7314003,),
        ),
        PlaceCategory.ICE_CREAM: _spec(
            "ice cream parlor",
            "RESTAURANT",
            category_ids=(7315078,),
        ),
        PlaceCategory.JEWELRY_STORE: _spec(
            "jewelry, clocks watches",
            "SHOP",
            category_ids=(9361036,),
        ),
        PlaceCategory.KINDERGARTEN: _spec(
            "pre school",
            "SCHOOL",
            category_ids=(7372004,),
        ),
        PlaceCategory.LAUNDRY: _spec("laundry", "SHOP", category_ids=(9361045,)),
        PlaceCategory.LIBRARY: _spec("library", "LIBRARY", category_ids=(9913,)),
        PlaceCategory.MALL: _spec("mall", "SHOPPING_CENTER", category_ids=(7373,)),
        PlaceCategory.MARKETPLACE: _spec("market", "MARKET", category_ids=(7332,)),
        PlaceCategory.MOBILE_PHONE_STORE: _spec(
            "mobile phone shop",
            "SHOP",
            category_ids=(9361075,),
        ),
        PlaceCategory.MUSEUM: _spec("museum", "MUSEUM", category_ids=(7317,)),
        PlaceCategory.MUSIC_VENUE: _spec(
            "music center",
            "THEATER",
            category_ids=(7318003,),
        ),
        PlaceCategory.NIGHTCLUB: _spec(
            "discotheque",
            "NIGHTLIFE",
            category_ids=(9379002,),
        ),
        PlaceCategory.OPTICIAN: _spec(
            "opticians",
            "SHOP",
            category_ids=(9361038,),
        ),
        PlaceCategory.PARK: _spec(
            "park",
            "PARK_RECREATION_AREA",
            category_ids=(9362008,),
        ),
        PlaceCategory.PARKING: _spec(
            "parking lot",
            "OPEN_PARKING_AREA",
            "PARKING_GARAGE",
            category_ids=(7313, 7369),
        ),
        PlaceCategory.PET_STORE: _spec(
            "pet supplies",
            "SHOP",
            category_ids=(9361064,),
        ),
        PlaceCategory.PHARMACY: _spec(
            "pharmacy",
            "PHARMACY",
            category_ids=(7326,),
        ),
        PlaceCategory.PIZZERIA: _spec(
            "pizza",
            "RESTAURANT",
            category_ids=(7315036,),
        ),
        PlaceCategory.PLACE_OF_WORSHIP: _spec(
            "place of worship",
            "PLACE_OF_WORSHIP",
            category_ids=(7339,),
        ),
        PlaceCategory.PLAYGROUND: _spec(
            "amusement place",
            "AMUSEMENT_PARK",
            category_ids=(9902004,),
        ),
        PlaceCategory.POLICE: _spec(
            "police station",
            "POLICE_STATION",
            category_ids=(7322,),
        ),
        PlaceCategory.POST_OFFICE: _spec(
            "post office",
            "POST_OFFICE",
            category_ids=(7324,),
        ),
        PlaceCategory.PUB: _spec(
            "pub",
            "CAFE_PUB",
            "NIGHTLIFE",
            category_ids=(9376003,),
        ),
        PlaceCategory.RESTAURANT: _spec(
            "restaurant",
            "RESTAURANT",
            category_ids=(7315,),
        ),
        PlaceCategory.SCHOOL: _spec(
            "school",
            "SCHOOL",
            category_ids=(7372,),
        ),
        PlaceCategory.SHOE_STORE: _spec(
            "clothing accessories: footwear shoe repairs",
            "SHOP",
            category_ids=(9361005,),
        ),
        PlaceCategory.SPORTS_CENTRE: _spec(
            "sports center",
            "SPORTS_CENTER",
            category_ids=(7320,),
        ),
        PlaceCategory.SPORTS_STORE: _spec(
            "sports equipment clothing",
            "SHOP",
            category_ids=(9361039,),
        ),
        PlaceCategory.STATIONERY_STORE: _spec(
            "electrical, office it: office equipment",
            "SHOP",
            category_ids=(9361014,),
        ),
        PlaceCategory.SUPERMARKET: _spec(
            "supermarkets hypermarkets",
            "MARKET",
            category_ids=(7332005,),
        ),
        PlaceCategory.SWIMMING_POOL: _spec(
            "swimming pool",
            "SWIMMING_POOL",
            category_ids=(7338,),
        ),
        PlaceCategory.TAXI: _spec(
            "taxi stand",
            "PUBLIC_TRANSPORT_STOP",
            category_ids=(9942003,),
        ),
        PlaceCategory.THEATRE: _spec("theater", "THEATER", category_ids=(7318,)),
        PlaceCategory.TOY_STORE: _spec(
            "toys games",
            "SHOP",
            category_ids=(9361040,),
        ),
        PlaceCategory.TRAIN_STATION: _spec(
            "railway station",
            "RAILWAY_STATION",
            "PUBLIC_TRANSPORT_STOP",
            category_ids=(7380,),
        ),
        PlaceCategory.TRAVEL_AGENCY: _spec(
            "travel agents",
            "SHOP",
            category_ids=(9361041,),
        ),
        PlaceCategory.UNIVERSITY: _spec(
            "college/university",
            "COLLEGE_UNIVERSITY",
            category_ids=(7377,),
        ),
        PlaceCategory.VETERINARY: _spec(
            "veterinarian",
            "VETERINARIAN",
            category_ids=(9375,),
        ),
        PlaceCategory.VIEWPOINT: _spec(
            "scenic/panoramic view",
            "SCENIC_PANORAMIC_VIEW",
            category_ids=(7337,),
        ),
        PlaceCategory.ZOO: _spec(
            "zoo",
            "ZOOS_ARBORETA_BOTANICAL_GARDEN",
            category_ids=(9927003,),
        ),
    }
)


def tomtom_category_spec(category: PlaceCategory) -> TomTomCategorySpec:
    """Return the exhaustive TomTom mapping for a public category."""

    return TOMTOM_CATEGORY_SPECS[category]
