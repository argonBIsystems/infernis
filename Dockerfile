FROM python:3.11-slim

WORKDIR /app

# System deps for GDAL, GEOS, PROJ (required by rasterio, shapely, pyproj)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[dashboard]"

# Copy models for both 1km and 5km resolutions
COPY models/fire_core_1km_v1.json models/fire_core_1km_v1.json
COPY models/bec_calibration_1km.json models/bec_calibration_1km.json
COPY models/training_metrics_1km.json models/training_metrics_1km.json
COPY models/fire_core_v1.json models/fire_core_v1.json
COPY models/bec_calibration.json models/bec_calibration.json

# Cached grid with topography (avoids GEE call on every restart)
COPY data/processed/bc_grid_5km.parquet data/processed/bc_grid_5km.parquet

COPY alembic/ alembic/
COPY alembic.ini .
COPY static/ static/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn infernis.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
