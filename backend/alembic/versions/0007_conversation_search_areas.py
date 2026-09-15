"""persist private reusable search areas with assistant messages

Revision ID: 0007_conversation_search_areas
Revises: 0006_conversation_map_places
Create Date: 2026-08-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_conversation_search_areas"
down_revision: str | None = "0006_conversation_map_places"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversation_message",
        sa.Column(
            "search_areas",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.alter_column("conversation_message", "search_areas", server_default=None)


def downgrade() -> None:
    op.drop_column("conversation_message", "search_areas")
