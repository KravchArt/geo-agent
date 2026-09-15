"""Small deterministic transliteration helpers for model-facing geo labels."""

from __future__ import annotations

_CYRILLIC_TO_LATIN = {
    "а": "a",
    "ә": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "ғ": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ng",
    "о": "o",
    "ө": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ұ": "u",
    "ү": "u",
    "ф": "f",
    "х": "kh",
    "һ": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "є": "ie",
    "і": "i",
    "ї": "i",
}


def transliterate_cyrillic(value: str) -> str:
    """Transliterate supported Cyrillic letters while preserving other text."""

    result: list[str] = []
    for character in value:
        replacement = _CYRILLIC_TO_LATIN.get(character.casefold())
        if replacement is None:
            result.append(character)
        elif character.isupper():
            result.append(replacement.capitalize())
        else:
            result.append(replacement)
    return "".join(result)
