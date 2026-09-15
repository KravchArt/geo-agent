"""Durable conversation ownership, listing and transcript persistence."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Conversation, ConversationMessage
from common.models import (
    ConversationDetail,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationSummary,
    ConversationTurn,
    MapData,
    MapPlace,
    SourceCitation,
)
from tools.geo.places_search.schemas import ResolvedSearchArea

_DEFAULT_TITLE = "New conversation"
_TITLE_LIMIT = 80
_HISTORY_PLACE_CONTEXT_MAX = 20


class ConversationNotFoundError(LookupError):
    """The conversation is absent or belongs to another client."""


def _title_from_message(message: str) -> str:
    normalized = " ".join(message.split())
    if not normalized:
        return _DEFAULT_TITLE
    if len(normalized) <= _TITLE_LIMIT:
        return normalized
    return normalized[: _TITLE_LIMIT - 1].rstrip() + "…"


def _sources_from_json(items: object) -> list[SourceCitation]:
    if not isinstance(items, list):
        return []
    sources: list[SourceCitation] = []
    for item in items:
        try:
            sources.append(SourceCitation.model_validate(item))
        except ValidationError:
            # A malformed historical citation must not make the whole transcript
            # unavailable. New writes are validated by SourceCitation beforehand.
            continue
    return sources


def _map_from_json(items: object) -> MapData | None:
    """Restore persisted pins without letting malformed old rows break history."""

    if not isinstance(items, list):
        return None
    places: list[MapPlace] = []
    for item in items:
        try:
            places.append(MapPlace.model_validate(item))
        except ValidationError:
            continue
    return MapData(places=places) if places else None


def _place_context_from_json(items: object, *, limit: int) -> tuple[str, int]:
    """Build model-only follow-up context without exposing coordinates.

    Public assistant text deliberately hides ``plc_`` refs. Keeping a small
    name/address-to-ref index lets a follow-up reuse an exact earlier place
    rather than ambiguously resolving its textual address again.
    """

    map_data = _map_from_json(items)
    if map_data is None or limit <= 0:
        return "", 0

    places = [place for place in map_data.places if place.marker_role == "result"][:limit]
    lines = [f"- {place.name} — {place.address} → {place.ref}" for place in places]
    return "\n".join(lines), len(places)


def _search_area_context_from_json(items: object, *, limit: int) -> tuple[str, int]:
    """Render private reusable localities without exposing them through the API."""

    if not isinstance(items, list) or limit <= 0:
        return "", 0

    areas: list[ResolvedSearchArea] = []
    for item in items:
        if len(areas) >= limit:
            break
        try:
            areas.append(ResolvedSearchArea.model_validate(item))
        except ValidationError:
            continue
    lines = [f"- Search area: {area.name} — {area.address} → {area.ref}" for area in areas]
    return "\n".join(lines), len(areas)


class ConversationStore:
    """Postgres-backed product store; observability remains in its own tables."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def ensure_chat_access(
        self,
        session_id: str,
        client_id: str | None,
    ) -> Conversation:
        """Create a conversation or verify ownership before expensive LLM work.

        Legacy headerless chats create an unowned conversation which remains
        headerless: knowledge of a possibly predictable legacy ``session_id`` is
        not sufficient proof to claim and read it through the history API. The
        row lock also serializes concurrent turns in one conversation, so a later
        turn cannot read stale context while the previous one is running.
        """

        await self._db.execute(
            insert(Conversation)
            .values(id=session_id, client_id=client_id, title=_DEFAULT_TITLE)
            .on_conflict_do_nothing(index_elements=[Conversation.id])
        )
        conversation = await self._db.scalar(
            select(Conversation).where(Conversation.id == session_id).with_for_update()
        )
        if conversation is None:
            raise ConversationNotFoundError

        if client_id != conversation.client_id:
            raise ConversationNotFoundError

        await self._db.flush()
        return conversation

    async def list_conversations(
        self,
        client_id: str,
        *,
        limit: int,
        offset: int,
    ) -> ConversationListResponse:
        message_count = (
            select(func.count(ConversationMessage.id))
            .where(ConversationMessage.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery()
        )
        rows = (
            await self._db.execute(
                select(Conversation, message_count.label("message_count"))
                .where(Conversation.client_id == client_id)
                .order_by(Conversation.updated_at.desc(), Conversation.id)
                .limit(limit)
                .offset(offset)
            )
        ).all()
        items = [
            ConversationSummary(
                session_id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                message_count=int(count),
            )
            for conversation, count in rows
        ]
        return ConversationListResponse(items=items, limit=limit, offset=offset)

    async def get_conversation(self, session_id: str, client_id: str) -> ConversationDetail:
        rows = (
            await self._db.execute(
                select(Conversation, ConversationMessage)
                .outerjoin(
                    ConversationMessage,
                    ConversationMessage.conversation_id == Conversation.id,
                )
                .where(
                    Conversation.id == session_id,
                    Conversation.client_id == client_id,
                )
                .order_by(ConversationMessage.sequence_no)
            )
        ).all()
        if not rows:
            raise ConversationNotFoundError

        conversation = rows[0][0]
        messages = [
            self._message_response(message)
            for _conversation, message in rows
            if message is not None
        ]
        return ConversationDetail(
            session_id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=len(messages),
            messages=messages,
        )

    async def delete_conversation(self, session_id: str, client_id: str) -> None:
        conversation = await self._owned_conversation(session_id, client_id)
        await self._db.delete(conversation)
        await self._db.flush()

    async def recent_history(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> list[ConversationTurn]:
        newest_first = (
            await self._db.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == session_id)
                .order_by(ConversationMessage.sequence_no.desc())
                .limit(limit)
            )
        ).all()
        # Newest results win because follow-ups normally refer to the last list
        # the user saw. A bounded index prevents old map results from consuming
        # an unbounded part of the model context.
        remaining_place_context = _HISTORY_PLACE_CONTEXT_MAX
        place_context_by_message_id: dict[uuid.UUID, str] = {}
        for row in newest_first:
            if row.role != "assistant" or remaining_place_context <= 0:
                continue
            context_parts: list[str] = []
            area_context, area_count = _search_area_context_from_json(
                row.search_areas,
                limit=remaining_place_context,
            )
            if area_context:
                context_parts.append(area_context)
                remaining_place_context -= area_count
            place_context, place_count = _place_context_from_json(
                row.map_places,
                limit=remaining_place_context,
            )
            if place_context:
                context_parts.append(place_context)
                remaining_place_context -= place_count
            if context_parts:
                place_context_by_message_id[row.id] = "\n".join(context_parts)

        history: list[ConversationTurn] = []
        for row in reversed(newest_first):
            if row.role not in ("user", "assistant") or not row.content.strip():
                continue
            turn: ConversationTurn = {"role": row.role, "content": row.content}
            private_context = place_context_by_message_id.get(row.id)
            if private_context:
                # Private key: public conversation responses still expose only
                # content, sources and map data.
                turn["_place_context"] = private_context
            history.append(turn)
        return history

    async def persist_exchange(
        self,
        *,
        session_id: str,
        request_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
        sources: Sequence[SourceCitation],
        map_places: Sequence[MapPlace],
        search_areas: Sequence[ResolvedSearchArea],
        status: Literal["completed", "rejected"],
        rejection_reason: Literal["out_of_scope", "censorship", "output_censorship"] | None,
    ) -> None:
        """Append the public request/answer pair in the caller's transaction."""

        conversation = await self._db.scalar(
            select(Conversation).where(Conversation.id == session_id).with_for_update()
        )
        if conversation is None:
            raise ConversationNotFoundError

        last_sequence = await self._db.scalar(
            select(func.max(ConversationMessage.sequence_no)).where(
                ConversationMessage.conversation_id == session_id
            )
        )
        next_sequence = int(last_sequence or 0) + 1
        if next_sequence == 1:
            conversation.title = _title_from_message(user_message)
        # Unlike now(), clock_timestamp() reflects completion time rather than
        # the start of a potentially long model transaction.
        conversation.updated_at = func.clock_timestamp()

        self._db.add_all(
            [
                ConversationMessage(
                    conversation_id=session_id,
                    request_id=request_id,
                    sequence_no=next_sequence,
                    role="user",
                    content=user_message,
                    sources=[],
                    map_places=[],
                    search_areas=[],
                ),
                ConversationMessage(
                    conversation_id=session_id,
                    request_id=request_id,
                    sequence_no=next_sequence + 1,
                    role="assistant",
                    content=assistant_message,
                    sources=[source.model_dump(mode="json") for source in sources],
                    map_places=[place.model_dump(mode="json") for place in map_places],
                    search_areas=[area.model_dump(mode="json") for area in search_areas],
                    status=status,
                    rejection_reason=rejection_reason,
                ),
            ]
        )
        await self._db.flush()

    async def _owned_conversation(self, session_id: str, client_id: str) -> Conversation:
        conversation = await self._db.scalar(
            select(Conversation).where(
                Conversation.id == session_id,
                Conversation.client_id == client_id,
            )
        )
        if conversation is None:
            raise ConversationNotFoundError
        return conversation

    @staticmethod
    def _message_response(row: ConversationMessage) -> ConversationMessageResponse:
        return ConversationMessageResponse(
            id=str(row.id),
            role=cast(Literal["user", "assistant"], row.role),
            content=row.content,
            sources=_sources_from_json(row.sources),
            map=_map_from_json(row.map_places),
            status=cast(Literal["completed", "rejected"] | None, row.status),
            rejection_reason=cast(
                Literal["out_of_scope", "censorship", "output_censorship"] | None,
                row.rejection_reason,
            ),
            created_at=row.created_at,
        )
