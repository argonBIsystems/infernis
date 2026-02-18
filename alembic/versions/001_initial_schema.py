"""Initial schema - grid_cells, predictions, pipeline_runs, api_keys, fire_history.

Revision ID: 001
Revises: None
Create Date: 2026-02-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # grid_cells
    op.create_table(
        "grid_cells",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cell_id", sa.String(30), unique=True, nullable=False),
        sa.Column("geom", Geometry("POLYGON", srid=3005), nullable=False),
        sa.Column("centroid", Geometry("POINT", srid=4326), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("bec_zone", sa.String(10)),
        sa.Column("fuel_type", sa.String(5)),
        sa.Column("elevation_m", sa.Float()),
        sa.Column("slope_deg", sa.Float()),
        sa.Column("aspect_deg", sa.Float()),
        sa.Column("hillshade", sa.Float()),
    )
    op.create_index("ix_grid_cells_cell_id", "grid_cells", ["cell_id"])
    op.create_index("ix_grid_cells_centroid", "grid_cells", ["centroid"], postgresql_using="gist")
    op.create_index("ix_grid_cells_bec_zone", "grid_cells", ["bec_zone"])

    # predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("cell_id", sa.String(30), nullable=False),
        sa.Column("prediction_date", sa.Date(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("ffmc", sa.Float()),
        sa.Column("dmc", sa.Float()),
        sa.Column("dc", sa.Float()),
        sa.Column("isi", sa.Float()),
        sa.Column("bui", sa.Float()),
        sa.Column("fwi", sa.Float()),
        sa.Column("temperature_c", sa.Float()),
        sa.Column("rh_pct", sa.Float()),
        sa.Column("wind_kmh", sa.Float()),
        sa.Column("precip_24h_mm", sa.Float()),
        sa.Column("soil_moisture", sa.Float()),
        sa.Column("ndvi", sa.Float()),
        sa.Column("snow_cover", sa.Boolean()),
        sa.Column("features", JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_predictions_cell_id", "predictions", ["cell_id"])
    op.create_index("ix_predictions_cell_date", "predictions", ["cell_id", "prediction_date"], unique=True)
    op.create_index("ix_predictions_date", "predictions", ["prediction_date"])
    op.create_index("ix_predictions_level", "predictions", ["prediction_date", "level"])

    # pipeline_runs
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("cells_processed", sa.Integer(), server_default="0"),
        sa.Column("error_message", sa.Text()),
        sa.Column("model_version", sa.String(50)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_pipeline_runs_date", "pipeline_runs", ["run_date"])
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"])

    # api_keys
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("daily_limit", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("requests_today", sa.Integer(), server_default="0"),
        sa.Column("last_reset", sa.Date()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
    )
    op.create_index("ix_api_keys_hash", "api_keys", ["key_hash"])
    op.create_index("ix_api_keys_tier", "api_keys", ["tier"])

    # fire_history
    op.create_table(
        "fire_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("fire_id", sa.String(50), unique=True, nullable=False),
        sa.Column("fire_name", sa.String(200)),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date()),
        sa.Column("end_date", sa.Date()),
        sa.Column("cause", sa.String(50)),
        sa.Column("size_ha", sa.Float()),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("geom", Geometry("POINT", srid=4326)),
        sa.Column("source", sa.String(30)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_fire_history_geom", "fire_history", ["geom"], postgresql_using="gist")
    op.create_index("ix_fire_history_year", "fire_history", ["year"])


def downgrade() -> None:
    op.drop_table("fire_history")
    op.drop_table("api_keys")
    op.drop_table("pipeline_runs")
    op.drop_table("predictions")
    op.drop_table("grid_cells")
    op.execute("DROP EXTENSION IF EXISTS postgis")
