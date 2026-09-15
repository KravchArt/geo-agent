"""Async SQLAlchemy engine + session management.

The engine is **owned by the application lifespan** and lives on ``app.state``
(see :mod:`backend.app.main`). There is deliberately no process-wide singleton:
an engine binds to the event loop that created it, so a module-level one breaks
silently in any other loop (tests, workers, background tasks).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Build a new async engine. Caller owns it and must ``dispose()`` it."""
    return create_async_engine(settings.database_url, pool_pre_ping=True, future=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def ping_database(engine: AsyncEngine) -> bool:
    """Return True if a trivial query succeeds against Postgres."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: a session from the app-owned sessionmaker."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session
