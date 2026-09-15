"""Reviewed mapping from public place categories to the 2GIS rubricator."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from tools.geo.places_search.schemas import PlaceCategory
from tools.geo.text_place_query import twogis_response_locale

_RUSSIAN_RUBRIC_CATALOG_COUNTRIES = frozenset(
    {"am", "az", "by", "cn", "ge", "kg", "kz", "ru", "tj", "uz"}
)


@dataclass(frozen=True, slots=True)
class TwoGisCategorySpec:
    """One reviewed search phrase and its semantically accepted 2GIS aliases."""

    query: str
    aliases: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class TwoGisRubricLookup:
    query: str
    locale: str
    allowed_aliases: tuple[str, ...] | None = ()


def _spec(query: str, *aliases: str) -> TwoGisCategorySpec:
    return TwoGisCategorySpec(query=query, aliases=aliases)


def _unsupported(query: str) -> TwoGisCategorySpec:
    # The query is retained for the live diagnostic matrix, but production
    # refuses to guess a rubric when 2GIS has no equivalent category.
    return TwoGisCategorySpec(query=query, aliases=None)


# Aliases were reviewed against the Moscow 2GIS rubricator. Common numeric
# rubric IDs are deliberately not stored: the API documents IDs as regional,
# so the adapter resolves these stable aliases to IDs for the selected project.
TWOGIS_CATEGORY_SPECS: Final = MappingProxyType(
    {
        PlaceCategory.AIRPORT: _spec("аэропорты", "aehroporty"),
        PlaceCategory.ALCOHOL_STORE: _spec("алкомаркеты", "alkogolnye_napitki"),
        PlaceCategory.ARTS_CENTRE: _unsupported("арт-центры"),
        PlaceCategory.ATM: _spec("банкоматы", "bankomaty"),
        PlaceCategory.ATTRACTION: _unsupported("достопримечательности"),
        PlaceCategory.BAKERY: _spec("пекарни", "pekarni"),
        PlaceCategory.BANK: _spec("банки", "banki"),
        PlaceCategory.BAR: _spec("бары", "bary"),
        PlaceCategory.BEAUTY_SALON: _spec(
            "салоны красоты",
            "parikmakherskie_uslugi",
            "nogtevojj_servis",
            "vizazhisty",
            "oformlenie_brovejj_i_resnic",
        ),
        PlaceCategory.BICYCLE_RENTAL: _unsupported("прокат велосипедов"),
        PlaceCategory.BICYCLE_STORE: _spec("магазин велосипедов", "velosipedy"),
        PlaceCategory.BOOKSTORE: _spec("книжные магазины", "knigi"),
        PlaceCategory.BUS_STATION: _spec("автостанции", "avtovokzaly"),
        PlaceCategory.BUTCHER: _spec("мясные магазины", "myaso_i_polufabrikaty"),
        PlaceCategory.CAFE: _spec("кафе", "kafe"),
        PlaceCategory.CAR_DEALER: _spec(
            "автосалоны",
            "prodazha_legkovykh_avtomobilejj",
        ),
        PlaceCategory.CAR_RENTAL: _spec("прокат машин", "prokat_avtotransporta"),
        PlaceCategory.CAR_REPAIR: _spec("ремонт автомобилей", "legkovojj_avtoservis"),
        PlaceCategory.CAR_WASH: _spec("автомойки", "avtomojjki"),
        PlaceCategory.CASINO: _unsupported("казино"),
        PlaceCategory.CHARGING_STATION: _spec(
            "электрозаправки",
            "stancii_zaryadki_ehlektromobilejj",
        ),
        PlaceCategory.CINEMA: _spec("кинотеатры", "kinoteatry"),
        PlaceCategory.CLINIC: _spec(
            "медицинские клиники",
            "mnogoprofilnye_medicinskie_centry",
        ),
        PlaceCategory.CLOTHING_STORE: _spec(
            "магазины одежды",
            "muzhskaya_odezhda",
            "zhenskaya_odezhda",
            "detskaya_odezhda",
            "dzhinsovaya_odezhda",
            "verkhnyaya_odezhda",
            "trikotazhnye_izdeliya",
        ),
        PlaceCategory.COFFEE_SHOP: _spec("кофейни", "kofejjni"),
        PlaceCategory.COMMUNITY_CENTRE: _spec("дома культуры", "doma_kultury"),
        PlaceCategory.COMPUTER_STORE: _spec("компьютерные магазины", "kompyutery"),
        PlaceCategory.CONFECTIONERY: _spec(
            "кондитерские",
            "kafe_konditerskie",
            "konditerskie_izdeliya",
        ),
        PlaceCategory.CONVENIENCE_STORE: _spec(
            "продуктовый магазин",
            "produktovye_magaziny",
        ),
        PlaceCategory.COSMETICS_STORE: _spec(
            "магазины косметики",
            "kosmetika_i_parfyumeriya",
        ),
        PlaceCategory.DENTIST: _spec(
            "стоматологи",
            "stomatologicheskie_polikliniki",
            "detskie_stomatologicheskie_polikliniki",
            "chastnye_stomatologii",
            "chastnye_detskie_stomatologii",
        ),
        PlaceCategory.DEPARTMENT_STORE: _unsupported("универмаги"),
        PlaceCategory.DOCTORS: _unsupported("врачи"),
        PlaceCategory.DRY_CLEANING: _spec(
            "химчистки",
            "khimchistki_odezhdy_i_tekstilya",
            "khimchistki_obuvi",
        ),
        PlaceCategory.ELECTRONICS_STORE: _spec(
            "электроника",
            "audiotekhnika_i_videotekhnika",
            "bytovaya_tekhnika",
        ),
        PlaceCategory.FAST_FOOD: _spec("быстрое питание", "bystroe_pitanie"),
        PlaceCategory.FIRE_STATION: _spec("пожарные части", "pozharnaya_okhrana"),
        PlaceCategory.FITNESS_CENTRE: _spec("фитнес клубы", "fitnes_kluby"),
        PlaceCategory.FLOWER_SHOP: _spec("магазины цветов", "cvety"),
        PlaceCategory.FOOD_COURT: _spec("фудкорт", "fudmolly"),
        PlaceCategory.FUEL: _spec("АЗС", "zapravochnye_stancii"),
        PlaceCategory.FURNITURE_STORE: _spec("магазины мебели", "mebelnye_magaziny"),
        PlaceCategory.GALLERY: _spec("галереи", "khudozhestvennye_vystavki"),
        PlaceCategory.GARDEN_CENTRE: _spec(
            "садовые центры",
            "pitomniki_rastenijj",
            "semena_i_posadochnyjj_material",
        ),
        PlaceCategory.GIFT_SHOP: _spec("сувенирные магазины", "suveniry"),
        PlaceCategory.GREENGROCER: _spec("овощной магазин", "ovoshhi_i_frukty"),
        PlaceCategory.GUEST_HOUSE: _spec(
            "гостевые дома",
            "arenda_kottedzhejj_gostevykh_domov",
        ),
        PlaceCategory.HAIRDRESSER: _spec(
            "парикмахерские",
            "parikmakherskie_uslugi",
            "detskie_parikmakherskie",
            "barbershopy",
        ),
        PlaceCategory.HARDWARE_STORE: _spec("хозяйственные магазины", "khoztovary"),
        PlaceCategory.HOSPITAL: _spec("больницы", "bolnicy"),
        PlaceCategory.HOSTEL: _spec("хостелы", "khostely"),
        PlaceCategory.HOTEL: _spec("гостиницы", "gostinicy"),
        PlaceCategory.ICE_CREAM: _spec("мороженое", "morozhenoe"),
        PlaceCategory.JEWELRY_STORE: _spec("ювелирные магазины", "yuvelirnye_izdeliya"),
        PlaceCategory.KINDERGARTEN: _spec(
            "детские сады",
            "detskie_sady",
            "chastnye_detskie_sady",
        ),
        PlaceCategory.LAUNDRY: _spec("прачечные", "prachechnye"),
        PlaceCategory.LIBRARY: _spec("библиотеки", "biblioteki"),
        PlaceCategory.MALL: _spec(
            "торговые центры",
            "torgovye_centry",
            "torgovo_razvlekatelnye_centry",
        ),
        PlaceCategory.MARKETPLACE: _spec("рынки", "rynki"),
        PlaceCategory.MOBILE_PHONE_STORE: _spec(
            "магазины телефонов",
            "mobilnye_telefony",
        ),
        PlaceCategory.MUSEUM: _spec("музеи", "muzei"),
        PlaceCategory.MUSIC_VENUE: _spec("концертные площадки", "koncertnye_zaly"),
        PlaceCategory.NIGHTCLUB: _spec("ночные клубы", "nochnye_kluby"),
        PlaceCategory.OPTICIAN: _spec("салоны оптики", "optika"),
        PlaceCategory.PARK: _spec("парки", "parki"),
        PlaceCategory.PARKING: _spec("парковки", "parkingi", "avtostoyanki"),
        PlaceCategory.PET_STORE: _spec("зоомагазины", "zootovary"),
        PlaceCategory.PHARMACY: _spec("аптеки", "apteki"),
        PlaceCategory.PIZZERIA: _spec("пиццерии", "piccerii"),
        PlaceCategory.PLACE_OF_WORSHIP: _spec(
            "религиозные организации",
            "religioznye_organizacii",
        ),
        PlaceCategory.PLAYGROUND: _spec("детские площадки", "detskie_ploshhadki"),
        PlaceCategory.POLICE: _spec(
            "полиция",
            "otdeleniya_policii",
            "uchastkovye_punkty_policii",
        ),
        PlaceCategory.POST_OFFICE: _spec("почта", "pochta"),
        PlaceCategory.PUB: _unsupported("пабы"),
        PlaceCategory.RESTAURANT: _spec("рестораны", "restorany"),
        PlaceCategory.SCHOOL: _spec("общеобразовательные школы", "shkoly"),
        PlaceCategory.SHOE_STORE: _spec("магазины обуви", "obuvnye_magaziny"),
        PlaceCategory.SPORTS_CENTRE: _unsupported("спортивные комплексы"),
        PlaceCategory.SPORTS_STORE: _spec(
            "спортивные магазины",
            "sportivnyjj_inventar",
            "sportivnaya_odezhda_i_obuv",
        ),
        PlaceCategory.STATIONERY_STORE: _spec("магазины канцтоваров", "kanctovary"),
        PlaceCategory.SUPERMARKET: _spec("супермаркеты", "supermarkety"),
        PlaceCategory.SWIMMING_POOL: _spec("бассейны", "bassejjny"),
        PlaceCategory.TAXI: _spec("такси", "taksi"),
        PlaceCategory.THEATRE: _spec("театры", "teatry"),
        PlaceCategory.TOY_STORE: _spec("магазины игрушек", "igrushki"),
        PlaceCategory.TRAIN_STATION: _spec(
            "жд вокзалы",
            "zheleznodorozhnye_vokzaly",
            "prigorodnye_vokzaly",
        ),
        PlaceCategory.TRAVEL_AGENCY: _spec("туристические агентства", "turagentstva"),
        PlaceCategory.UNIVERSITY: _spec("университеты", "universitety"),
        PlaceCategory.VETERINARY: _spec("ветеринарные клиники", "veterinarnye_kliniki"),
        PlaceCategory.VIEWPOINT: _spec("смотровые площадки", "smotrovye_ploshhadki"),
        PlaceCategory.ZOO: _spec("зоопарки", "zoopark"),
    }
)


def twogis_category_spec(category: PlaceCategory) -> TwoGisCategorySpec:
    return TWOGIS_CATEGORY_SPECS[category]


def twogis_rubric_lookup(
    *,
    category: PlaceCategory,
    query: str,
    country_code: str | None,
) -> TwoGisRubricLookup:
    """Build the category-directory query used by the production provider."""

    response_locale = twogis_response_locale(query, country_code=country_code)
    if country_code is not None and country_code.casefold() in _RUSSIAN_RUBRIC_CATALOG_COUNTRIES:
        spec = twogis_category_spec(category)
        return TwoGisRubricLookup(
            query=spec.query,
            locale=twogis_response_locale(spec.query, country_code=country_code),
            allowed_aliases=spec.aliases,
        )
    return TwoGisRubricLookup(query=query, locale=response_locale)
