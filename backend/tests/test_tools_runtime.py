from __future__ import annotations

import pytest

from backend.app.config import get_settings
from backend.app.main import app
from tools import ToolExecutor


async def test_lifespan_builds_runtime_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLACES_SEARCH_PROVIDERS", "[]")
    monkeypatch.setenv("ROUTING_PROVIDERS", "[]")
    monkeypatch.setenv("WEB_SEARCH_PROVIDERS", "[]")

    get_settings.cache_clear()

    try:
        async with app.router.lifespan_context(app):
            assert isinstance(app.state.tool_executor, ToolExecutor)

            assert not app.state.tools

            assert app.state.place_store is not None
            assert app.state.source_store is not None
            assert app.state.tools_http_client.is_closed is False

        assert app.state.tools_http_client.is_closed is True

        result = await app.state.tool_executor.run(
            "routing_tool",
            {
                "mode": "route",
                "waypoints": [
                    "plc_a1b2c3d4e5",
                    "plc_b2c3d4e5f6",
                ],
            },
        )

        assert result.ok is False
        assert result.error == "unknown tool: routing_tool"

    finally:
        # Не оставляем тестовый Settings для следующих тестов.
        get_settings.cache_clear()
