"""Deterministic correction of model-selected generic place categories."""

from __future__ import annotations

import re

_RU_ALIASES: dict[str, tuple[str, ...]] = {
    "airport": ("аэропорт", "аэропорты"),
    "alcohol_store": ("алкомаркет", "алкомаркеты", "алкогольный магазин", "магазины алкоголя"),
    "arts_centre": ("арт центр", "арт центры", "центр искусств", "центры искусств"),
    "atm": ("банкомат", "банкоматы"),
    "attraction": ("достопримечательность", "достопримечательности"),
    "bakery": ("пекарня", "пекарни"),
    "bank": ("банк", "банки"),
    "bar": ("бар", "бары"),
    "beauty_salon": ("салон красоты", "салоны красоты"),
    "bicycle_rental": ("велопрокат", "велопрокаты", "прокат велосипедов"),
    "bicycle_store": ("веломагазин", "веломагазины", "магазин велосипедов"),
    "bookstore": ("книжный магазин", "книжные магазины"),
    "bus_station": ("автовокзал", "автовокзалы", "автостанция", "автостанции"),
    "butcher": ("мясная лавка", "мясные лавки", "мясной магазин", "мясные магазины"),
    "cafe": ("кафе",),
    "car_dealer": ("автосалон", "автосалоны"),
    "car_rental": ("автопрокат", "прокат автомобилей", "прокат машин"),
    "car_repair": ("автосервис", "автосервисы", "ремонт автомобилей"),
    "car_wash": ("автомойка", "автомойки"),
    "casino": ("казино",),
    "charging_station": (
        "зарядная станция",
        "зарядные станции",
        "электрозаправка",
        "электрозаправки",
    ),
    "cinema": ("кинотеатр", "кинотеатры"),
    "clinic": ("клиника", "клиники"),
    "clothing_store": ("магазин одежды", "магазины одежды"),
    "coffee_shop": ("кофейня", "кофейни"),
    "community_centre": (
        "общественный центр",
        "общественные центры",
        "дом культуры",
        "дома культуры",
    ),
    "computer_store": ("компьютерный магазин", "компьютерные магазины"),
    "confectionery": ("кондитерская", "кондитерские"),
    "convenience_store": ("магазин у дома", "магазины у дома"),
    "cosmetics_store": ("магазин косметики", "магазины косметики"),
    "dentist": ("стоматология", "стоматологии", "стоматолог", "стоматологи"),
    "department_store": ("универмаг", "универмаги"),
    "doctors": ("врач", "врачи", "доктор", "доктора"),
    "dry_cleaning": ("химчистка", "химчистки"),
    "electronics_store": ("магазин электроники", "магазины электроники"),
    "fast_food": ("фастфуд", "фаст фуд", "быстрое питание"),
    "fire_station": ("пожарная часть", "пожарные части"),
    "fitness_centre": ("фитнес центр", "фитнес центры", "фитнес клуб", "фитнес клубы"),
    "flower_shop": ("цветочный магазин", "цветочные магазины", "магазин цветов", "магазины цветов"),
    "food_court": ("фудкорт", "фудкорты", "фуд корт", "фуд корты"),
    "fuel": ("азс", "заправка", "заправки", "автозаправка", "автозаправки"),
    "furniture_store": (
        "мебельный магазин",
        "мебельные магазины",
        "магазин мебели",
        "магазины мебели",
    ),
    "gallery": ("галерея", "галереи"),
    "garden_centre": ("садовый центр", "садовые центры"),
    "gift_shop": (
        "магазин подарков",
        "магазины подарков",
        "сувенирный магазин",
        "сувенирные магазины",
    ),
    "greengrocer": ("овощной магазин", "овощные магазины", "магазин овощей", "магазины овощей"),
    "guest_house": ("гостевой дом", "гостевые дома"),
    "hairdresser": ("парикмахерская", "парикмахерские"),
    "hardware_store": (
        "строительный магазин",
        "строительные магазины",
        "хозяйственный магазин",
        "хозяйственные магазины",
    ),
    "hospital": ("больница", "больницы"),
    "hostel": ("хостел", "хостелы"),
    "hotel": ("отель", "отели", "гостиница", "гостиницы"),
    "ice_cream": ("кафе мороженое", "мороженое"),
    "jewelry_store": ("ювелирный магазин", "ювелирные магазины"),
    "kindergarten": ("детский сад", "детские сады"),
    "laundry": ("прачечная", "прачечные"),
    "library": ("библиотека", "библиотеки"),
    "mall": ("торговый центр", "торговые центры", "трц"),
    "marketplace": ("рынок", "рынки"),
    "mobile_phone_store": (
        "салон связи",
        "салоны связи",
        "магазин телефонов",
        "магазины телефонов",
    ),
    "museum": ("музей", "музеи"),
    "music_venue": ("концертная площадка", "концертные площадки"),
    "nightclub": ("ночной клуб", "ночные клубы"),
    "optician": ("оптика", "салон оптики", "салоны оптики"),
    "park": ("парк", "парки"),
    "parking": ("парковка", "парковки"),
    "pet_store": ("зоомагазин", "зоомагазины"),
    "pharmacy": ("аптека", "аптеки"),
    "pizzeria": ("пиццерия", "пиццерии"),
    "place_of_worship": (
        "храм",
        "храмы",
        "церковь",
        "церкви",
        "мечеть",
        "мечети",
        "синагога",
        "синагоги",
    ),
    "playground": ("детская площадка", "детские площадки"),
    "police": ("отделение полиции", "отделения полиции", "полиция"),
    "post_office": ("почтовое отделение", "почтовые отделения", "почта"),
    "pub": ("паб", "пабы"),
    "restaurant": ("ресторан", "рестораны"),
    "school": ("школа", "школы"),
    "shoe_store": ("обувной магазин", "обувные магазины", "магазин обуви", "магазины обуви"),
    "sports_centre": ("спортивный центр", "спортивные центры", "спорткомплекс", "спорткомплексы"),
    "sports_store": (
        "спортивный магазин",
        "спортивные магазины",
        "магазин спорттоваров",
        "магазины спорттоваров",
    ),
    "stationery_store": (
        "канцелярский магазин",
        "канцелярские магазины",
        "магазин канцтоваров",
        "магазины канцтоваров",
    ),
    "supermarket": ("супермаркет", "супермаркеты"),
    "swimming_pool": ("бассейн", "бассейны"),
    "taxi": ("такси", "стоянка такси", "стоянки такси"),
    "theatre": ("театр", "театры"),
    "toy_store": ("магазин игрушек", "магазины игрушек"),
    "train_station": (
        "железнодорожный вокзал",
        "железнодорожные вокзалы",
        "жд вокзал",
        "жд вокзалы",
    ),
    "travel_agency": (
        "турагентство",
        "турагентства",
        "туристическое агентство",
        "туристические агентства",
    ),
    "university": ("университет", "университеты", "вуз", "вузы"),
    "veterinary": ("ветклиника", "ветклиники", "ветеринарная клиника", "ветеринарные клиники"),
    "viewpoint": ("смотровая площадка", "смотровые площадки"),
    "zoo": ("зоопарк", "зоопарки"),
}


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^0-9a-zа-я]+", " ", value.casefold().replace("ё", "е")).split())


def _build_alias_index() -> dict[str, str]:
    result: dict[str, str] = {}
    for category, russian_aliases in _RU_ALIASES.items():
        english = category.replace("_", " ")
        english_aliases = (english, english if english.endswith("s") else f"{english}s")
        for alias in (*english_aliases, *russian_aliases):
            normalized = _normalize(alias)
            existing = result.setdefault(normalized, category)
            if existing != category:
                raise RuntimeError(f"ambiguous place-category alias: {alias}")
    return result


_CATEGORY_BY_EXACT_QUERY = _build_alias_index()


def normalized_category_value(query: str | None, category: str | None) -> str | None:
    """Correct a supplied category when the generic query has one obvious meaning.

    ``None`` remains ``None`` because omission explicitly means a named-place
    search; an organisation can itself be called "Кафе" or "Park".
    """

    if query is None or category is None:
        return category
    return _CATEGORY_BY_EXACT_QUERY.get(_normalize(query), category)


def normalized_category_alias_keys() -> frozenset[str]:
    """Expose covered enum values for a schema completeness test."""

    return frozenset(_RU_ALIASES)


def russian_category_query(category: str) -> str:
    """Return a stable Russian lookup phrase for one canonical category."""

    aliases = _RU_ALIASES.get(category)
    if aliases is None:
        raise ValueError(f"unsupported canonical place category: {category}")
    return aliases[-1]


def english_category_query(category: str) -> str:
    """Return the canonical English lookup phrase used in model input."""

    if category not in _RU_ALIASES:
        raise ValueError(f"unsupported canonical place category: {category}")
    return category.replace("_", " ")


def category_for_exact_query(query: str) -> str | None:
    """Return the canonical category represented by one exact localized label."""

    return _CATEGORY_BY_EXACT_QUERY.get(_normalize(query))
