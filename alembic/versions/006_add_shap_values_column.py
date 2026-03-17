"""Add shap_values JSONB column to predictions table.

Revision ID: 006
Revises: 005
Create Date: 2026-03-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("shap_values", JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions", "shap_values")
