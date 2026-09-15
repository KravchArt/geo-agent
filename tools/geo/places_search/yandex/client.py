"""HTTP client for Yandex Organization Search API."""

from __future__ import annotations

from typing import Any

import httpx

from tools.geo.yandex_http import get_yandex_json
from tools.observability import ToolExecutionContext
from tools.refs import GeoBounds


class YandexOrganisationSearchClient:
    """Call Yandex Organization Search API with type=biz."""

    _URL = "https://search-maps.yandex.ru/v1/"
    provider = "yandex_organisation_search"

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client

    async def search(
        self,
        *,
        text: str,
        limit: int,
        center: tuple[float, float] | None = None,
        span: tuple[float, float] | None = None,
        bbox: GeoBounds | None = None,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Return raw Yandex JSON for an organisation search.

        ``center`` is ``(lon, lat)``.
        ``span`` is ``(longitude_span, latitude_span)`` in degrees.
        """

        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")

        if (center is None) != (span is None):
            raise ValueError("center and span must be passed together")
        if bbox is not None and center is not None:
            raise ValueError("bbox cannot be combined with center and span")

        params: dict[str, str] = {
            "apikey": self._api_key,
            "text": text,
            "type": "biz",
            "lang": "ru_RU",
            "results": str(limit),
        }

        if center is not None and span is not None:
            lon, lat = center
            lon_span, lat_span = span

            params["ll"] = f"{lon:.6f},{lat:.6f}"
            params["spn"] = f"{lon_span:.6f},{lat_span:.6f}"
            params["rspn"] = "1"

        if bbox is not None:
            params["bbox"] = f"{bbox.west:.6f},{bbox.south:.6f}~{bbox.east:.6f},{bbox.north:.6f}"
            params["rspn"] = "1"

        return await get_yandex_json(
            http_client=self._http_client,
            url=self._URL,
            params=params,
            provider=self.provider,
            operation="search",
            context=context,
        )
