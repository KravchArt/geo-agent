"""Migration test: the from-scratch schema applies cleanly on an empty DB.

Runs synchronously (no active event loop) so Alembic's own ``asyncio.run`` inside
env.py works. Uses a dedicated engine for verification rather than the app
singletons.
"""

from __future__ import annotations

import asyncio

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.config import get_settings
from backend.tests.conftest import fail_or_skip

EXPECTED_TABLES = {
    "conversation",
    "conversation_message",
    "request",
    "react_trace",
    "tool_call",
    "model_response",
    "metrics",
    "pipeline_stage_metric",
    "llm_call_metric",
    "upstream_call_metric",
}


async def _fetch_tables(url: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            names = await conn.run_sync(lambda c: inspect(c).get_table_names())
    finally:
        await engine.dispose()
    return set(names)


async def _fetch_columns(url: str, table_name: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            columns = await conn.run_sync(lambda c: inspect(c).get_columns(table_name))
    finally:
        await engine.dispose()
    return {str(column["name"]) for column in columns}


def test_migration_applies_on_clean_db():
    settings = get_settings()
    url = settings.database_url

    # Connectivity gate (skip locally / fail in CI when Postgres is down).
    try:
        asyncio.run(_fetch_tables(url))
    except Exception as exc:
        fail_or_skip("postgres", exc)

    cfg = Config("alembic.ini")

    # Guarantee a clean slate, then apply from scratch to head.
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    tables = asyncio.run(_fetch_tables(url))
    assert tables >= EXPECTED_TABLES, f"missing tables: {EXPECTED_TABLES - tables}"
    tool_call_columns = asyncio.run(_fetch_columns(url, "tool_call"))
    assert tool_call_columns >= {
        "error_code",
        "provider",
        "status_code",
        "failure_kind",
        "provider_code",
        "retryable",
    }
    conversation_message_columns = asyncio.run(_fetch_columns(url, "conversation_message"))
    assert conversation_message_columns >= {"map_places", "search_areas"}

    # Deliberately leave the schema AT HEAD. Re-runs still start clean because the
    # downgrade above does that; tearing the schema down here instead would wipe
    # the developer's database, since tests and the local app share one Postgres —
    # every `pytest` would leave a running stack with no tables.
