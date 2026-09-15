"""HTTP client for Yandex Maps Geocoder API."""

from __future__ import annotations

from typing import Any

import httpx

from tools.geo.text_place_query import yandex_response_locale
from tools.geo.yandex_http import get_yandex_json
from tools.observability import ToolExecutionContext


class YandexGeocoderClient:
    """Call Yandex Maps Geocoder API for forward geocoding."""

    _URL = "https://geocode-maps.yandex.ru/v1/"
    provider = "yandex_geocoder"

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client

    async def search(
        self,
        query: str,
        *,
        limit: int,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Return the raw JSON response for a text place query."""

        return await get_yandex_json(
            http_client=self._http_client,
            url=self._URL,
            params={
                "apikey": self._api_key,
                "geocode": query,
                "lang": yandex_response_locale(query),
                "results": str(limit),
                "format": "json",
            },
            provider=self.provider,
            operation="search",
            context=context,
        )
