"""Add key_preview column to api_keys table.

Revision ID: 004
Revises: 003
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("key_preview", sa.String(20)))


def downgrade() -> None:
    op.drop_column("api_keys", "key_preview")
