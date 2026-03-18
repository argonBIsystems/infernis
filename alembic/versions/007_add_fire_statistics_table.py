"""Add fire_statistics table for location fire risk profiles.

Revision ID: 007
Revises: 006
Create Date: 2026-03-17
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fire_statistics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cell_id", sa.String(30), nullable=False),
        sa.Column("fires_10yr_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fires_10yr_nearest_km", sa.Float(), nullable=True),
        sa.Column("fires_10yr_largest_ha", sa.Float(), nullable=True),
        sa.Column("fires_10yr_causes", JSONB(), nullable=True),
        sa.Column("fires_30yr_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fires_30yr_nearest_km", sa.Float(), nullable=True),
        sa.Column("fires_30yr_largest_ha", sa.Float(), nullable=True),
        sa.Column("fires_30yr_causes", JSONB(), nullable=True),
        sa.Column("fires_all_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fires_all_nearest_km", sa.Float(), nullable=True),
        sa.Column("fires_all_largest_ha", sa.Float(), nullable=True),
        sa.Column("fires_all_causes", JSONB(), nullable=True),
        sa.Column("fires_all_record_start", sa.Integer(), nullable=True),
        sa.Column("susceptibility_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("susceptibility_percentile", sa.Integer(), nullable=True),
        sa.Column("susceptibility_label", sa.String(20), nullable=True),
        sa.Column("susceptibility_basis", sa.String(20), nullable=True),
        sa.Column("exposure_percentile", sa.Integer(), nullable=True),
        sa.Column("mean_return_years", sa.Float(), nullable=True),
        sa.Column("typical_severity", sa.String(20), nullable=True),
        sa.Column("dominant_cause", sa.String(20), nullable=True),
        sa.Column("computed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cell_id"),
    )
    op.create_index("ix_fire_statistics_cell_id", "fire_statistics", ["cell_id"])


def downgrade() -> None:
    op.drop_index("ix_fire_statistics_cell_id", table_name="fire_statistics")
    op.drop_table("fire_statistics")
