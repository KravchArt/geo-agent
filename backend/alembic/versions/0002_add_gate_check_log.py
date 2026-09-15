"""add gate check log table

Revision ID: 0002_gate_check_log
Revises: 0001_initial
Create Date: 2026-07-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_gate_check_log"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gate_check_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("gate_name", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("matched_rules", postgresql.JSONB(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("raw_response", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["request.id"],
            name="fk_gate_check_log_request_id_request",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_gate_check_log"),
    )
    op.create_index("ix_gate_check_log_request_id", "gate_check_log", ["request_id"])
    op.create_index("ix_gate_check_log_gate_name", "gate_check_log", ["gate_name"])


def downgrade() -> None:
    op.drop_index("ix_gate_check_log_gate_name", table_name="gate_check_log")
    op.drop_index("ix_gate_check_log_request_id", table_name="gate_check_log")
    op.drop_table("gate_check_log")
