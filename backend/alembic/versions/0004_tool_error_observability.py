"""add queryable tool error observability fields

Revision ID: 0004_tool_error_observability
Revises: 0003_detailed_observability
Create Date: 2026-08-06
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_tool_error_observability"
down_revision: str | None = "0003_detailed_observability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tool_call", sa.Column("error_code", sa.String(length=128), nullable=True))
    op.add_column("tool_call", sa.Column("provider", sa.String(length=128), nullable=True))
    op.add_column("tool_call", sa.Column("status_code", sa.Integer(), nullable=True))
    op.add_column("tool_call", sa.Column("failure_kind", sa.String(length=128), nullable=True))
    op.add_column("tool_call", sa.Column("provider_code", sa.String(length=128), nullable=True))
    op.add_column("tool_call", sa.Column("retryable", sa.Boolean(), nullable=True))
    op.create_index("ix_tool_call_error_code", "tool_call", ["error_code"])
    op.create_index("ix_tool_call_provider", "tool_call", ["provider"])
    op.create_index("ix_tool_call_failure_kind", "tool_call", ["failure_kind"])


def downgrade() -> None:
    op.drop_index("ix_tool_call_failure_kind", table_name="tool_call")
    op.drop_index("ix_tool_call_provider", table_name="tool_call")
    op.drop_index("ix_tool_call_error_code", table_name="tool_call")
    op.drop_column("tool_call", "retryable")
    op.drop_column("tool_call", "provider_code")
    op.drop_column("tool_call", "failure_kind")
    op.drop_column("tool_call", "status_code")
    op.drop_column("tool_call", "provider")
    op.drop_column("tool_call", "error_code")
