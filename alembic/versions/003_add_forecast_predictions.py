"""Add forecast_predictions table for multi-day forecasts.

Revision ID: 003
Revises: 002
Create Date: 2026-02-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecast_predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cell_id", sa.String(30), nullable=False),
        sa.Column("base_date", sa.Date(), nullable=False),
        sa.Column("lead_day", sa.SmallInteger(), nullable=False),
        sa.Column("valid_date", sa.Date(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("danger_level", sa.SmallInteger(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("fwi_components", JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_forecast_cell_valid",
        "forecast_predictions",
        ["cell_id", "valid_date"],
    )
    op.create_index(
        "ix_forecast_base_lead",
        "forecast_predictions",
        ["base_date", "lead_day"],
    )


def downgrade() -> None:
    op.drop_index("ix_forecast_base_lead", table_name="forecast_predictions")
    op.drop_index("ix_forecast_cell_valid", table_name="forecast_predictions")
    op.drop_table("forecast_predictions")
