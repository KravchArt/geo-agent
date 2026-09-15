"""add durable conversation history

Revision ID: 0005_conversation_history
Revises: 0004_tool_error_observability
Create Date: 2026-08-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_conversation_history"
down_revision: str | None = "0004_tool_error_observability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("client_id", sa.String(length=128), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=False),
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
        sa.PrimaryKeyConstraint("id", name="pk_conversation"),
    )
    op.create_index("ix_conversation_client_id", "conversation", ["client_id"])
    op.create_index(
        "ix_conversation_client_updated",
        "conversation",
        ["client_id", "updated_at"],
    )

    op.create_table(
        "conversation_message",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sequence_no", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("rejection_reason", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_conversation_message_role",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name="fk_conversation_message_conversation_id_conversation",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["request.id"],
            name="fk_conversation_message_request_id_request",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversation_message"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence_no",
            name="uq_conversation_message_conversation_sequence",
        ),
        sa.UniqueConstraint(
            "request_id",
            "role",
            name="uq_conversation_message_request_role",
        ),
    )
    op.create_index(
        "ix_conversation_message_conversation_id",
        "conversation_message",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_message_request_id",
        "conversation_message",
        ["request_id"],
    )


def downgrade() -> None:
    op.drop_table("conversation_message")
    op.drop_table("conversation")
