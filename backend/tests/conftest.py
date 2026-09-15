"""Shared pytest fixtures.

Test defaults point at localhost so a plain `uv run pytest` works against
`docker compose up` (ports published to localhost) and against CI service
containers (also reachable on localhost).

Integration tests SKIP when a service is unreachable — UNLESS
`REQUIRE_INTEGRATION=1` (set in CI), in which case they FAIL. This guarantees CI
actually exercises Postgres/Redis instead of silently skipping.

Resources are never shared between tests: the app owns its engine/Redis client
via its lifespan, and every fixture below closes what it opens. That keeps tests
order-independent (a connection bound to one test's event loop is dead in the
next).
"""

from __future__ import annotations

import os

# Must run before any Settings() is constructed. Real environment variables win
# over .env, so this pins the suite to deterministic, offline backends no matter
# what the developer has configured locally — otherwise `app_client` boots the
# real lifespan and the gates start calling a hosted model mid-test.
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_MODE", "mock")
os.environ.setdefault("SCOPE_PROVIDER", "rule_based")
os.environ.setdefault("CENSORSHIP_PROVIDER", "rule_based")
os.environ.setdefault("PLACES_SEARCH_PROVIDERS", "[]")
os.environ.setdefault("ROUTING_PROVIDERS", "[]")
os.environ.setdefault("WEB_SEARCH_PROVIDERS", "[]")

import pytest
from httpx import ASGITransport, AsyncClient

REQUIRE_INTEGRATION = os.getenv("REQUIRE_INTEGRATION") == "1"


def fail_or_skip(resource: str, exc: Exception) -> None:
    """Fail in CI, skip locally, when a service is unreachable."""
    msg = f"{resource} not reachable: {exc!r}"
    if REQUIRE_INTEGRATION:
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture
async def redis_client():
    """A test-owned Redis client (also the reachability gate)."""
    from backend.app.redis.client import RedisClient

    client = RedisClient.from_settings()
    try:
        await client.ping()
    except Exception as exc:
        await client.close()
        fail_or_skip("redis", exc)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def db_ready():
    """Postgres reachability gate, on a throwaway engine of its own."""
    from backend.app.config import get_settings
    from backend.app.db.session import create_engine, ping_database

    engine = create_engine(get_settings())
    try:
        await ping_database(engine)
    except Exception as exc:
        await engine.dispose()
        fail_or_skip("postgres", exc)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def migrated_db():
    """Ensure the current Alembic schema exists for integration tests that write data."""
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    yield


@pytest.fixture
async def app_client():
    """HTTP client against the app with its **real lifespan** run.

    Running the lifespan is what populates ``app.state`` (engine, sessionmaker,
    Redis) — and, on exit, disposes it. Without this the handlers would have no
    resources, and leaked ones would poison the next test.
    """
    from backend.app.main import app

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
