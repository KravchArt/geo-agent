from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from backend.app.db.models import Metrics, Request
from backend.app.main import app
from backend.app.services.conversations import (
    _place_context_from_json,
    _search_area_context_from_json,
)


def test_hidden_place_context_keeps_refs_but_never_coordinates() -> None:
    context, count = _place_context_from_json(
        [
            {
                "ref": "plc_77a91385f1",
                "name": "Frank by Баста, реберная",
                "address": "Москва, Мясницкая улица, 24",
                "latitude": 55.762841,
                "longitude": 37.635262,
            }
        ],
        limit=20,
    )

    assert count == 1
    assert "Frank by Баста, реберная" in context
    assert "Мясницкая улица, 24" in context
    assert "plc_77a91385f1" in context
    assert "55.762841" not in context
    assert "37.635262" not in context


def test_hidden_search_area_context_keeps_reusable_ref() -> None:
    context, count = _search_area_context_from_json(
        [
            {
                "ref": "plc_77a91385f1",
                "name": "Москва",
                "address": "Москва, Россия",
            }
        ],
        limit=20,
    )

    assert count == 1
    assert context == ("- Search area: Москва — Москва, Россия → plc_77a91385f1")


@pytest.mark.integration
async def test_history_api_lists_restores_owns_and_deletes_conversation(
    db_ready, redis_client, migrated_db, app_client
):
    client_id = f"client-{uuid.uuid4()}"
    other_client_id = f"client-{uuid.uuid4()}"
    session_id = f"conversation-{uuid.uuid4()}"
    message = "Plan one day in Prague"
    headers = {"X-Client-ID": client_id}

    chat = await app_client.post(
        "/api/v1/chat",
        headers=headers,
        json={"session_id": session_id, "message": message},
    )
    assert chat.status_code == 200

    listing = await app_client.get(
        "/api/v1/conversations?limit=10&offset=0",
        headers=headers,
    )
    assert listing.status_code == 200
    assert listing.headers["cache-control"] == "private, no-store"
    assert listing.headers["vary"] == "X-Client-ID"
    list_body = listing.json()
    assert list_body["limit"] == 10
    assert list_body["offset"] == 0
    assert list_body["items"] == [
        {
            "session_id": session_id,
            "title": message,
            "created_at": list_body["items"][0]["created_at"],
            "updated_at": list_body["items"][0]["updated_at"],
            "message_count": 2,
        }
    ]

    detail = await app_client.get(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "private, no-store"
    assert detail.headers["vary"] == "X-Client-ID"
    body = detail.json()
    assert body["session_id"] == session_id
    assert body["message_count"] == 2
    assert [item["role"] for item in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == message
    assert body["messages"][0]["sources"] == []
    assert body["messages"][0]["status"] is None
    assert body["messages"][1]["content"] == chat.json()["answer"]
    assert body["messages"][1]["status"] == "completed"
    assert body["messages"][1]["rejection_reason"] is None

    for method in (app_client.get, app_client.delete):
        hidden = await method(
            f"/api/v1/conversations/{session_id}",
            headers={"X-Client-ID": other_client_id},
        )
        assert hidden.status_code == 404
        assert hidden.json() == {"detail": "Conversation not found"}
        assert hidden.headers["cache-control"] == "private, no-store"
        assert hidden.headers["vary"] == "X-Client-ID"

    request_count_before = len(await _requests_for_session(session_id))
    denied_chat = await app_client.post(
        "/api/v1/chat",
        headers={"X-Client-ID": other_client_id},
        json={"session_id": session_id, "message": "Find a hotel in Prague"},
    )
    assert denied_chat.status_code == 404
    assert len(await _requests_for_session(session_id)) == request_count_before

    deleted = await app_client.delete(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert await redis_client.get_history(session_id) == []
    assert await redis_client.get_session_value(session_id, "last_answer") is None

    missing = await app_client.get(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assert missing.status_code == 404
    # Product history deletion deliberately does not erase the audit trail.
    assert len(await _requests_for_session(session_id)) == request_count_before


@pytest.mark.integration
async def test_history_api_requires_client_header(db_ready, migrated_db, app_client):
    assert (await app_client.get("/api/v1/conversations")).status_code == 422
    assert (await app_client.get("/api/v1/conversations/nonexistent")).status_code == 422
    assert (await app_client.delete("/api/v1/conversations/nonexistent")).status_code == 422


@pytest.mark.integration
async def test_unowned_legacy_conversation_cannot_be_claimed(db_ready, migrated_db, app_client):
    session_id = f"legacy-{uuid.uuid4()}"
    first = await app_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Plan one day in Prague"},
    )
    assert first.status_code == 200

    attacker_headers = {"X-Client-ID": f"client-{uuid.uuid4()}"}
    claim = await app_client.post(
        "/api/v1/chat",
        headers=attacker_headers,
        json={"session_id": session_id, "message": "Find a hotel in Prague"},
    )
    assert claim.status_code == 404
    assert (
        await app_client.get(
            f"/api/v1/conversations/{session_id}",
            headers=attacker_headers,
        )
    ).status_code == 404

    # The original headerless client keeps its backward-compatible chat flow.
    follow_up = await app_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Find another place nearby"},
    )
    assert follow_up.status_code == 200


@pytest.mark.integration
async def test_owned_conversation_does_not_inherit_ownerless_redis_history(
    db_ready, redis_client, migrated_db, app_client
):
    session_id = f"predictable-{uuid.uuid4()}"
    await redis_client.append_history(session_id, "user", "private legacy question")
    await redis_client.append_history(session_id, "assistant", "private legacy answer")

    message = "Plan one day in Prague"
    response = await app_client.post(
        "/api/v1/chat",
        headers={"X-Client-ID": f"client-{uuid.uuid4()}"},
        json={"session_id": session_id, "message": message},
    )
    assert response.status_code == 200

    async with app.state.sessionmaker() as db:
        request_row = await db.scalar(
            select(Request).where(
                Request.session_id == session_id,
                Request.user_query == message,
            )
        )
        assert request_row is not None
        metrics = await db.scalar(select(Metrics).where(Metrics.request_id == request_row.id))
        assert metrics is not None
        assert metrics.extra is not None
        assert metrics.extra["observability"]["counts"]["history_turns"] == 0


@pytest.mark.integration
async def test_rejected_answer_is_saved_as_public_assistant_message(
    db_ready, migrated_db, app_client
):
    client_id = f"client-{uuid.uuid4()}"
    session_id = f"conversation-{uuid.uuid4()}"
    headers = {"X-Client-ID": client_id}

    chat = await app_client.post(
        "/api/v1/chat",
        headers=headers,
        json={"session_id": session_id, "message": "Write a Python sorting function"},
    )
    assert chat.status_code == 200
    assert chat.json()["status"] == "rejected"

    detail = await app_client.get(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assistant = detail.json()["messages"][1]
    assert assistant["content"] == chat.json()["answer"]
    assert assistant["status"] == "rejected"
    assert assistant["rejection_reason"] == "out_of_scope"


@pytest.mark.integration
async def test_postgres_history_is_used_after_redis_session_is_deleted(
    db_ready, redis_client, migrated_db, app_client
):
    client_id = f"client-{uuid.uuid4()}"
    session_id = f"conversation-{uuid.uuid4()}"
    headers = {"X-Client-ID": client_id}

    first = await app_client.post(
        "/api/v1/chat",
        headers=headers,
        json={"session_id": session_id, "message": "Find cafes in Prague"},
    )
    assert first.status_code == 200
    await redis_client.delete_session(session_id)

    second_message = "Find another cafe nearby"
    second = await app_client.post(
        "/api/v1/chat",
        headers=headers,
        json={"session_id": session_id, "message": second_message},
    )
    assert second.status_code == 200

    async with app.state.sessionmaker() as db:
        request_row = await db.scalar(
            select(Request).where(
                Request.session_id == session_id,
                Request.user_query == second_message,
            )
        )
        assert request_row is not None
        metrics = await db.scalar(select(Metrics).where(Metrics.request_id == request_row.id))
        assert metrics is not None
        assert metrics.extra is not None
        observability = metrics.extra["observability"]
        assert observability["counts"]["history_turns"] == 2

    detail = await app_client.get(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assert detail.json()["message_count"] == 4


async def _requests_for_session(session_id: str) -> list[Request]:
    async with app.state.sessionmaker() as db:
        return list(
            (await db.scalars(select(Request).where(Request.session_id == session_id))).all()
        )
