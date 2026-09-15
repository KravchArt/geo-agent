"""Shared, compact map renderer for the GeoAgent UI and visual preview."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import pandas as pd
import pydeck as pdk
import streamlit as st
from pydeck.types import String


class MapMarker(TypedDict):
    """A fully client-side marker; coordinates already passed backend validation."""

    latitude: float
    longitude: float
    label: str
    address: str
    color: list[int]
    radius_m: int


def render_map(markers: Sequence[MapMarker], *, height: int = 560) -> None:
    """Render labelled, compact pins with a tooltip over Streamlit's Carto base map."""

    if not markers:
        raise ValueError("render_map requires at least one marker")

    data = pd.DataFrame(markers)
    required_columns = {"latitude", "longitude", "label", "address", "color", "radius_m"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"map marker is missing required fields: {missing}")
    # TextLayer supports line breaks, which keeps the name prominent while
    # retaining the address directly at the marker instead of only in a tooltip.
    data["map_label"] = data["label"] + "\n" + data["address"]
    view = pdk.ViewState(
        latitude=float(data["latitude"].mean()),
        longitude=float(data["longitude"].mean()),
        zoom=13 if len(data) == 1 else 11.5,
        pitch=0,
    )
    deck = pdk.Deck(
        initial_view_state=view,
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=data,
                get_position="[longitude, latitude]",
                get_fill_color="color",
                get_radius="radius_m",
                radius_min_pixels=4,
                radius_max_pixels=8,
                pickable=True,
            ),
            pdk.Layer(
                "TextLayer",
                data=data,
                get_position="[longitude, latitude]",
                get_text="map_label",
                get_color=[31, 41, 55],
                get_size=13,
                get_pixel_offset=[0, -12],
                # pydeck distinguishes literal strings from DataFrame column
                # names, so these constants must use ``String`` explicitly.
                get_alignment_baseline=String("bottom"),
                get_text_anchor=String("middle"),
            ),
        ],
        tooltip={"text": "{label}\n{address}"},
    )
    st.pydeck_chart(deck, width="stretch", height=height)
