"""ORM models — logging/observability tables.

These tables record what the agent did, for debugging and evaluation:

    request ─┬─1:1─ react_trace ─┬─1:N─ tool_call
             │                   └─1:N─ model_response
             └─1:1─ metrics

Phase 0 ships the schema + a from-scratch Alembic migration only. No writer
logic yet — the orchestrator populates these in a later phase.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from backend.app.db.base import Base


class Conversation(Base):
    """A durable, user-visible chat conversation."""

    __tablename__ = "conversation"
    __table_args__ = (Index("ix_conversation_client_updated", "client_id", "updated_at"),)

    # Keep the existing wire identifier instead of forcing UUIDs: integrations and
    # evaluation scripts already use arbitrary strings up to 128 characters.
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Nullable preserves the old POST /chat contract. Unowned conversations are
    # usable by that legacy client, but are never exposed through history APIs.
    client_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.sequence_no",
    )


class Request(Base):
    """A single top-level user request to the agent."""

    __tablename__ = "request"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    user_query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    trace: Mapped[ReactTrace | None] = relationship(
        back_populates="request", uselist=False, cascade="all, delete-orphan"
    )
    metrics: Mapped[Metrics | None] = relationship(
        back_populates="request", uselist=False, cascade="all, delete-orphan"
    )
    stage_metrics: Mapped[list[PipelineStageMetric]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )


class ConversationMessage(Base):
    """One public user/assistant message in a durable conversation transcript."""

    __tablename__ = "conversation_message"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        UniqueConstraint(
            "conversation_id",
            "sequence_no",
            name="uq_conversation_message_conversation_sequence",
        ),
        UniqueConstraint(
            "request_id",
            "role",
            name="uq_conversation_message_request_role",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), index=True
    )
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("request.id", ondelete="SET NULL"), nullable=True, index=True
    )
    sequence_no: Mapped[int] = mapped_column(BigInteger)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    #: Verified map pins selected by the model for this specific assistant turn.
    map_places: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    #: Private resolved localities reusable by later model tool calls. Never public API data.
    search_areas: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class GateCheckLog(Base):
    """Result of a preflight gate evaluation for a request."""

    __tablename__ = "gate_check_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("request.id", ondelete="CASCADE"), index=True
    )
    gate_name: Mapped[str] = mapped_column(String(128), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verdict: Mapped[str] = mapped_column(String(32))
    passed: Mapped[bool] = mapped_column(default=False)
    reason: Mapped[str] = mapped_column(Text)
    matched_rules: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    phase: Mapped[str] = mapped_column(String(32), default="input", index=True)
    call_index: Mapped[int] = mapped_column(Integer, default=1)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error_type: Mapped[str | None] = mapped_column(String(256), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReactTrace(Base):
    """The ReAct loop executed for a request (1:1 with request)."""

    __tablename__ = "react_trace"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("request.id", ondelete="CASCADE"), unique=True, index=True
    )
    num_steps: Mapped[int] = mapped_column(Integer, default=0)
    # Full serialized ReActTrace (see common.models.ReActTrace).
    raw_trace: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped[Request] = relationship(back_populates="trace")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )
    model_responses: Mapped[list[ModelResponse]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )


class ToolCall(Base):
    """A single tool invocation (request + response) within a trace."""

    __tablename__ = "tool_call"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("react_trace.id", ondelete="CASCADE"), index=True
    )
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    # echo-grounding hash: maps a tool call/result to its Redis-cached payload.
    tool_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    error_code: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_kind: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    provider_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retryable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    upstream_call_count: Mapped[int] = mapped_column(Integer, default=0)
    upstream_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_bytes: Mapped[int] = mapped_column(Integer, default=0)
    model_tokens_estimate: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    trace: Mapped[ReactTrace] = relationship(back_populates="tool_calls")
    upstream_calls: Mapped[list[UpstreamCallMetric]] = relationship(
        back_populates="tool_call", cascade="all, delete-orphan"
    )


class ModelResponse(Base):
    """A single LLM call (prompt + completion) within a trace."""

    __tablename__ = "model_response"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("react_trace.id", ondelete="CASCADE"), index=True
    )
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(16))  # mock | vllm
    prompt: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    trace: Mapped[ReactTrace] = relationship(back_populates="model_responses")


class Metrics(Base):
    """Aggregate metrics for a request (1:1 with request)."""

    __tablename__ = "metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("request.id", ondelete="CASCADE"), unique=True, index=True
    )
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    num_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    num_model_calls: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(default=False)
    llm_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tool_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    upstream_call_count: Mapped[int] = mapped_column(Integer, default=0)
    successful_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    regenerations: Mapped[int] = mapped_column(Integer, default=0)
    gate_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    num_gate_calls: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped[Request] = relationship(back_populates="metrics")


class PipelineStageMetric(Base):
    """One measured execution of a named request-pipeline stage."""

    __tablename__ = "pipeline_stage_metric"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("request.id", ondelete="CASCADE"), index=True
    )
    stage_name: Mapped[str] = mapped_column(String(128), index=True)
    occurrence: Mapped[int] = mapped_column(Integer, default=1)
    latency_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(256), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped[Request] = relationship(back_populates="stage_metrics")


class LLMCallMetric(Base):
    """One physical LLM request made during the ReAct loop."""

    __tablename__ = "llm_call_metric"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("react_trace.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_type: Mapped[str | None] = mapped_column(String(256), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UpstreamCallMetric(Base):
    """One external provider/API call performed inside a tool invocation."""

    __tablename__ = "upstream_call_metric"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tool_call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tool_call.id", ondelete="CASCADE"), index=True
    )
    call_index: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    operation: Mapped[str] = mapped_column(String(128), index=True)
    latency_ms: Mapped[int] = mapped_column(Integer)
    parallel_group: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_kind: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retryable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tool_call: Mapped[ToolCall] = relationship(back_populates="upstream_calls")
