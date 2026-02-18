# INFERNIS

> *Intelligence forged in fire.*

Open-source wildfire risk prediction engine for British Columbia, Canada. INFERNIS ingests weather reanalysis, satellite imagery, soil moisture, vegetation indices, topography, and fuel classifications through an automated daily pipeline, then outputs fire risk scores via a REST API at 1km resolution.

## Highlights

- **2.1 million grid cells** covering all of BC at 1km resolution
- **Hybrid ML ensemble**: XGBoost + U-Net CNN with per-BEC-zone calibration
- **Full FWI system**: Vectorized Canadian Fire Weather Index computation
- **Multi-day forecasts**: Up to 10-day risk forecasts using HRDPS/GDPS NWP models
- **REST API**: JSON endpoints for point queries, area grids, heatmaps, and forecasts
- **Automated pipeline**: Daily data fetch, prediction, and cache update via APScheduler

## Model Performance

| Model | AUC-ROC | Avg Precision | Brier Score |
|-------|---------|---------------|-------------|
| XGBoost (1km, 28 features) | 0.974 | 0.794 | 0.036 |
| CNN FireUNet (1km spatial) | 0.815 | -- | -- |
| Walk-forward backtest (6 years) | 0.90-0.93 | 0.43-0.59 | 0.04-0.08 |

## Quick Start

```bash
# Clone and install
git clone https://github.com/argonBIsystems/infernis.git
cd infernis
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (CDS, GEE, etc.)

# Start infrastructure (PostgreSQL + Redis)
docker-compose up -d db redis

# Run database migrations
alembic upgrade head

# Generate BC grid (optional: skip GEE topography with --skip-topo)
python scripts/generate_grid.py --resolution 1

# Start the API server
uvicorn infernis.main:app --host 0.0.0.0 --port 8000
```

The server starts the daily pipeline automatically. Visit `http://localhost:8000/v1/docs` for interactive API documentation.

## Grid Resolution

INFERNIS supports two grid resolutions, controlled by the `INFERNIS_GRID_RESOLUTION_KM` environment variable in `.env`:

| Resolution | Cells | Pipeline Time | Use Case |
|------------|-------|---------------|----------|
| `1.0` (1km) | ~2.1M | ~5-12 min | Full precision, local development, research |
| `5.0` (5km) | ~84K | ~30s | Lightweight, demos, cost-sensitive deployments |

```bash
# .env — set resolution (default: 1.0)
INFERNIS_GRID_RESOLUTION_KM=1.0

# Regenerate the grid after changing resolution
python scripts/generate_grid.py --resolution 1   # or --resolution 5
```

Both resolutions use the same model architecture. Pre-trained weights are resolution-specific (`fire_core_1km_v1.json` vs `fire_core_v1.json` for 5km).

## Pre-trained Models

Pre-trained model weights are available for download (not included in the repo due to size):

| Model | Size | Download |
|-------|------|----------|
| XGBoost 1km (`fire_core_1km_v1.json`) | 19 MB | *Coming soon* |
| CNN 1km (`heatmap_1km_v1.pt`) | 119 MB | *Coming soon* |
| BEC calibration (`bec_calibration_1km.json`) | 1.5 KB | *Coming soon* |

Place downloaded models in the `models/` directory. Alternatively, train your own using the included training scripts (see Training below).

## API

```
GET /v1/risk/{lat}/{lon}          Point fire risk query
GET /v1/forecast/{lat}/{lon}      Multi-day forecast (up to 10 days)
GET /v1/risk/grid?bbox=...        Area risk query (GeoJSON)
GET /v1/risk/heatmap?bbox=...     Visual risk heatmap (PNG)
GET /v1/risk/zones                BC zone risk summary
GET /v1/fwi/{lat}/{lon}           Raw FWI components
GET /v1/conditions/{lat}/{lon}    Weather/environment conditions
GET /v1/history/{lat}/{lon}       Historical fire records
GET /v1/status                    System health
GET /v1/coverage                  Grid metadata
```

Authentication via `X-API-Key` header. Free tier: 50 requests/day. See [API Reference](docs/API_REFERENCE.md) for full documentation.

## How It Works

1. **Data Pipeline** -- Daily fetch of ERA5 weather reanalysis (with 7-day fallback for data lag), MODIS satellite imagery via Google Earth Engine, and weather station data.

2. **FWI Computation** -- Vectorized Canadian Fire Weather Index system computing all 6 components (FFMC, DMC, DC, ISI, BUI, FWI) for 2.1M cells in a single numpy call.

3. **XGBoost Classifier** -- Gradient-boosted tree model trained on 10 years of historical fire data (2015-2024) with 28 features spanning FWI, weather, vegetation, topography, and soil moisture.

4. **CNN Spatial Model** -- U-Net architecture (FireUNet) processing daily raster snapshots of BC to capture spatial fire spread patterns and topographic effects.

5. **Risk Fusion** -- Weighted ensemble of XGBoost + CNN scores with per-BEC-zone logistic calibration, outputting a 6-level danger classification (VERY_LOW through EXTREME).

6. **Forecast Engine** -- Multi-day risk forecasts using HRDPS (2.5km, days 1-2) and GDPS (15km, days 3-10) numerical weather prediction with FWI roll-forward and confidence decay.

## Training

Train your own models from scratch:

```bash
# Full training pipeline (feature engineering + XGBoost + calibration)
python scripts/train.py process --data-dir data/raw --output data/processed/features
python scripts/train.py build --features data/processed/features --output data/processed/training_data.parquet
python scripts/train.py train --data data/processed/training_data.parquet --output models/
python scripts/train.py evaluate --model models/fire_core_v1.json --data data/processed/training_data.parquet

# Train CNN heatmap model
python scripts/train_heatmap.py --data-dir data/processed/heatmap --epochs 30

# Per-BEC-zone calibration
python scripts/calibrate_bec.py --data data/processed/training_data.parquet --output models/bec_calibration.json

# Walk-forward backtesting
python scripts/backtest.py backtest --data data/processed/training_data.parquet --output reports/backtest.json
```

## Project Structure

```
src/infernis/
  api/              REST API routes, auth, Firebase dashboard
  db/               SQLAlchemy ORM, PostGIS, migrations
  grid/             BC grid generator (1km/5km, BC Albers EPSG:3005)
  models/           Pydantic schemas, danger level enums
  pipelines/        Daily pipeline, ERA5, GEE, HRDPS/GDPS, forecasting
  services/         Vectorized FWI (CFFDRS), Redis cache
  training/         XGBoost trainer, FireUNet CNN, risk fuser, backtester

tests/              190+ tests across 24 test files
docs/               White paper, technical architecture, API reference
```

## Documentation

| Document | Description |
|----------|-------------|
| [White Paper](docs/WHITE_PAPER.md) | Problem statement, wildfire science, methodology |
| [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) | System design, database schema, pipeline flows |
| [API Reference](docs/API_REFERENCE.md) | Full endpoint documentation with examples |

## Tech Stack

- **Runtime**: Python 3.11+, FastAPI, Uvicorn
- **ML**: XGBoost 2.1, PyTorch 2.x (MPS/CUDA), scikit-learn
- **FWI**: Custom vectorized CFFDRS (numpy)
- **Geospatial**: GeoPandas, Rasterio, Shapely, pyproj
- **Data**: ERA5 (CDS API), Google Earth Engine, NASA FIRMS
- **Database**: PostgreSQL 16 + PostGIS 3.4, Redis 7
- **Deploy**: Docker, Railway, GitHub Actions

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

```bash
# Run tests
python -m pytest tests/ -q

# Format code
ruff format src/ tests/
ruff check --fix src/ tests/
```

## License

[Apache License 2.0](LICENSE)

Built in British Columbia, Canada.
