"""persist selected map places with assistant messages

Revision ID: 0006_conversation_map_places
Revises: 0005_conversation_history
Create Date: 2026-08-11
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_conversation_map_places"
down_revision: str | None = "0005_conversation_history"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversation_message",
        sa.Column(
            "map_places",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.alter_column("conversation_message", "map_places", server_default=None)


def downgrade() -> None:
    op.drop_column("conversation_message", "map_places")
