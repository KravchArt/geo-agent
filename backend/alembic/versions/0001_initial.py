"""initial schema: logging/observability tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-11

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "request",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="pending"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_request"),
    )
    op.create_index("ix_request_session_id", "request", ["session_id"])

    op.create_table(
        "react_trace",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("num_steps", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("raw_trace", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["request.id"],
            name="fk_react_trace_request_id_request",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_react_trace"),
        sa.UniqueConstraint("request_id", name="uq_react_trace_request_id"),
    )
    op.create_index("ix_react_trace_request_id", "react_trace", ["request_id"])

    op.create_table(
        "tool_call",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("tool_hash", sa.String(length=64), nullable=True),
        sa.Column("arguments", postgresql.JSONB(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ok"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["react_trace.id"],
            name="fk_tool_call_trace_id_react_trace",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tool_call"),
    )
    op.create_index("ix_tool_call_trace_id", "tool_call", ["trace_id"])
    op.create_index("ix_tool_call_tool_name", "tool_call", ["tool_name"])
    op.create_index("ix_tool_call_tool_hash", "tool_call", ["tool_hash"])

    op.create_table(
        "model_response",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("prompt", postgresql.JSONB(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("usage", postgresql.JSONB(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["react_trace.id"],
            name="fk_model_response_trace_id_react_trace",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_response"),
    )
    op.create_index("ix_model_response_trace_id", "model_response", ["trace_id"])

    op.create_table(
        "metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("total_latency_ms", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("num_tool_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("num_model_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "success", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("extra", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["request.id"],
            name="fk_metrics_request_id_request",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_metrics"),
        sa.UniqueConstraint("request_id", name="uq_metrics_request_id"),
    )
    op.create_index("ix_metrics_request_id", "metrics", ["request_id"])


def downgrade() -> None:
    op.drop_table("metrics")
    op.drop_table("model_response")
    op.drop_table("tool_call")
    op.drop_table("react_trace")
    op.drop_table("request")
