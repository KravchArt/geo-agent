"""Standalone visual preview of the chat-with-map layout.

Run locally after installing the UI dependencies:

    streamlit run ui/map_preview.py

This file has no API calls and no LLM dependency. It demonstrates the exact map
component used by ``ui/app.py`` with static, non-sensitive Moscow example data.
"""

from __future__ import annotations

import streamlit as st

from ui.map_view import MapMarker, render_map

USER_LOCATION = {
    "name": "Ваше местоположение",
    "address": "Пример точки около центра Москвы",
    "latitude": 55.7558,
    "longitude": 37.6173,
    "color": [37, 99, 235, 230],
    "radius_m": 65,
}

SELECTED_PLACES = [
    {
        "name": "Государственная Третьяковская галерея",
        "address": "Лаврушинский пер., 10",
        "latitude": 55.7414,
        "longitude": 37.6208,
        "color": [220, 38, 38, 220],
        "radius_m": 50,
    },
    {
        "name": "Парк Горького",
        "address": "Крымский Вал, 9",
        "latitude": 55.7299,
        "longitude": 37.6010,
        "color": [220, 38, 38, 220],
        "radius_m": 50,
    },
    {
        "name": "Дом культуры «ГЭС-2»",
        "address": "Болотная наб., 15",
        "latitude": 55.7448,
        "longitude": 37.6115,
        "color": [220, 38, 38, 220],
        "radius_m": 50,
    },
]


def _map_markers() -> list[MapMarker]:
    """Build the same verified marker shape used by the production UI."""

    return [
        {
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "label": str(place["name"]),
            "address": str(place["address"]),
            "color": list(place["color"]),
            "radius_m": int(place["radius_m"]),
        }
        for place in [USER_LOCATION, *SELECTED_PLACES]
    ]


st.set_page_config(page_title="GeoAgent — map preview", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoAgent — превью карты")
st.caption("Статический пример: три места, выбранные моделью, и точка пользователя.")

chat_column, map_column = st.columns([3, 2], gap="large")

with chat_column:
    with st.chat_message("user"):
        st.markdown("Куда сходить в Москве днём, если хочется искусства и прогулки?")

    with st.chat_message("assistant"):
        st.markdown(
            """Я бы выбрал три точки, которые удобно объединить в одну прогулку:

1. **Третьяковская галерея** — классическое искусство в Лаврушинском переулке.
2. **ГЭС-2** — современное культурное пространство на Болотной набережной.
3. **Парк Горького** — прогулка у Москвы-реки после музеев.

На карте показаны только эти три места: остальные результаты поиска не попадают
в интерфейс, пока модель не выберет их для ответа."""
        )
        st.caption("В реальном запросе список пинов приходит из `map.places` API-ответа.")

with map_column:
    st.subheader("Карта")
    render_map(_map_markers())
    st.caption("🔵 Пользователь · 🔴 Места из финального ответа")
    for place in SELECTED_PLACES:
        st.markdown(f"- **{place['name']}** · {place['address']}")
