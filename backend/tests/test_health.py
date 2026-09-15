"""/health endpoint test — real Postgres + Redis connectivity."""

from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_health_ok(db_ready, redis_client, app_client):
    resp = await app_client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"postgres": True, "redis": True}
    assert body["llm_mode"] == "mock"


async def test_root(app_client):
    resp = await app_client.get("/")

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "geoagent"
    assert body["llm_mode"] == "mock"
