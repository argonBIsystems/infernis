<p align="center">
  <img src="brand/infernis-logo.svg" alt="INFERNIS" width="420"/>
</p>

<p align="center">
  <em>Open-source wildfire risk prediction for British Columbia</em>
</p>

<p align="center">
  <a href="https://infernis.ca">Live API</a> &bull;
  <a href="https://api.infernis.ca/v1/docs">API Docs</a> &bull;
  <a href="https://api.infernis.ca/v1/demo/risk">Try Demo</a> &bull;
  <a href="docs/WHITE_PAPER.md">White Paper</a> &bull;
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/API-Live-22C55E" alt="API Status"/>
  <img src="https://img.shields.io/badge/Grid-84K%20cells-F97316" alt="Grid Cells"/>
</p>

---

INFERNIS ingests weather forecasts, satellite imagery, soil moisture, vegetation indices, topography, and fuel classifications through an automated daily pipeline, then outputs fire risk scores via a REST API. The hosted API runs at 5 km resolution (~84K cells); the engine supports 1 km resolution (~2.1M cells) for self-hosted deployments.

## Highlights

- **84,535 grid cells** covering all of British Columbia, updated daily at 2 PM Pacific
- **10-day forecasts** using real NWP data (ECCC GEM model via Open-Meteo)
- **Full FWI system** — vectorized Canadian Fire Weather Index (all 6 components)
- **XGBoost + CNN ensemble** with per-BEC-zone calibration (0.974 AUC-ROC)
- **REST API** — point queries, area grids, PNG heatmaps, multi-day forecasts
- **21 open data sources** — ERA5, MODIS, VIIRS, HRDPS, GDPS, CLDN, and more
- **Demo endpoints** — explore response shapes without an API key

## Try It Now

No API key needed — hit the demo endpoints to see what INFERNIS returns:

```bash
# All 6 danger levels with realistic mock data
curl https://api.infernis.ca/v1/demo/risk | python -m json.tool

# Single level
curl https://api.infernis.ca/v1/demo/risk/high | python -m json.tool

# 10-day forecast showing a drying event
curl https://api.infernis.ca/v1/demo/forecast | python -m json.tool
```

For live data, [sign up for a free API key](https://infernis.ca) (50 requests/day):

```bash
# Real-time fire risk for Kamloops
curl -H "X-API-Key: YOUR_KEY" https://api.infernis.ca/v1/risk/50.67/-120.33

# 10-day forecast for Williams Lake
curl -H "X-API-Key: YOUR_KEY" https://api.infernis.ca/v1/forecast/52.13/-122.14
```

## Model Performance

| Model | AUC-ROC | Avg Precision | Brier Score |
|-------|---------|---------------|-------------|
| XGBoost (1 km, 28 features) | 0.974 | 0.794 | 0.036 |
| CNN FireUNet (1 km spatial) | 0.815 | -- | -- |
| Walk-forward backtest (6 years) | 0.90–0.93 | 0.43–0.59 | 0.04–0.08 |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/argonBIsystems/infernis.git
cd infernis
./scripts/dev_setup.sh    # creates venv, installs deps, copies .env

# Start databases and run migrations
make db-up
make migrate

# Start the API
make dev
```

Visit `http://localhost:8000/v1/docs` for interactive API docs. The daily pipeline starts automatically.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development setup guide.

## API Endpoints

| Endpoint | Description | Auth |
|----------|-------------|------|
| `GET /v1/risk/{lat}/{lon}` | Point fire risk query | API Key |
| `GET /v1/forecast/{lat}/{lon}` | Multi-day forecast (up to 10 days) | API Key |
| `GET /v1/risk/grid?bbox=...` | Area risk query (GeoJSON) | API Key |
| `GET /v1/risk/heatmap?bbox=...` | Visual risk heatmap (PNG) | API Key |
| `GET /v1/risk/zones` | BC zone risk summary | API Key |
| `GET /v1/fwi/{lat}/{lon}` | Raw FWI components | API Key |
| `GET /v1/conditions/{lat}/{lon}` | Weather/environment conditions | API Key |
| `GET /v1/status` | Pipeline health | Public |
| `GET /v1/coverage` | Grid metadata | Public |
| `GET /v1/demo/risk` | Sample data at all danger levels | Public |
| `GET /v1/demo/risk/{level}` | Sample data for one level | Public |
| `GET /v1/demo/forecast` | Sample 10-day forecast | Public |

See [API Reference](docs/API_REFERENCE.md) for full documentation with request/response examples.

## Danger Levels

| Level | Score Range | Color | Description |
|-------|------------|-------|-------------|
| VERY_LOW | 0.00–0.05 | ![#22C55E](https://via.placeholder.com/12/22C55E/22C55E.png) `#22C55E` | Minimal risk |
| LOW | 0.05–0.15 | ![#3B82F6](https://via.placeholder.com/12/3B82F6/3B82F6.png) `#3B82F6` | Low risk |
| MODERATE | 0.15–0.35 | ![#EAB308](https://via.placeholder.com/12/EAB308/EAB308.png) `#EAB308` | Elevated — monitor conditions |
| HIGH | 0.35–0.60 | ![#F97316](https://via.placeholder.com/12/F97316/F97316.png) `#F97316` | Significant risk — fire bans likely |
| VERY_HIGH | 0.60–0.80 | ![#EF4444](https://via.placeholder.com/12/EF4444/EF4444.png) `#EF4444` | Extreme caution |
| EXTREME | 0.80–1.00 | ![#1A0000](https://via.placeholder.com/12/1A0000/1A0000.png) `#1A0000` | Immediate danger |

## How It Works

1. **Data Pipeline** — Daily fetch of ERA5 weather, MODIS/VIIRS satellite imagery via Google Earth Engine, Open-Meteo NWP forecasts, and lightning data.

2. **FWI Computation** — Vectorized Canadian Fire Weather Index system computing all 6 components (FFMC, DMC, DC, ISI, BUI, FWI) for every grid cell.

3. **XGBoost Classifier** — Gradient-boosted model trained on 10 years of historical fire data (2015–2024) with 28 features: FWI, weather, vegetation, topography, soil moisture, and lightning.

4. **CNN Spatial Model** — U-Net architecture (FireUNet) processing daily raster snapshots to capture spatial fire spread patterns.

5. **Risk Fusion** — Weighted ensemble with per-BEC-zone logistic calibration, outputting a 6-level danger classification.

6. **Forecast Engine** — 10-day fire risk forecasts using ECCC's GEM model (HRDPS 2.5 km for days 1–2, GDPS for days 3–10) with FWI roll-forward and confidence decay.

## Grid Resolution

| Resolution | Cells | Pipeline Time | Use Case |
|------------|-------|---------------|----------|
| 5 km | ~84K | ~30s | Hosted API, lightweight deployments |
| 1 km | ~2.1M | ~5–12 min | Full precision, research, self-hosted |

```bash
# Set in .env (default: 1.0)
INFERNIS_GRID_RESOLUTION_KM=5.0

# Regenerate grid after changing resolution
python scripts/generate_grid.py --resolution 5
```

## Training

Train your own models from scratch using the included scripts:

```bash
# Feature engineering → XGBoost training → calibration
python scripts/train.py process --data-dir data/raw --output data/processed/features
python scripts/train.py build --features data/processed/features --output data/processed/training_data.parquet
python scripts/train.py train --data data/processed/training_data.parquet --output models/
python scripts/train.py evaluate --model models/fire_core_v1.json --data data/processed/training_data.parquet

# CNN heatmap model
python scripts/train_heatmap.py --data-dir data/processed/heatmap --epochs 30

# Per-BEC-zone calibration
python scripts/calibrate_bec.py --data data/processed/training_data.parquet --output models/bec_calibration.json

# Walk-forward backtesting
python scripts/backtest.py backtest --data data/processed/training_data.parquet --output reports/backtest.json
```

Pre-trained weights are not included in the repo (see [Pre-trained Models](#pre-trained-models) below).

## Pre-trained Models

| Model | Size | Status |
|-------|------|--------|
| XGBoost 1 km (`fire_core_1km_v1.json`) | 19 MB | *Coming soon* |
| CNN 1 km (`heatmap_1km_v1.pt`) | 119 MB | *Coming soon* |
| BEC calibration (`bec_calibration_1km.json`) | 1.5 KB | *Coming soon* |

Place downloaded models in the `models/` directory.

## Project Structure

```
src/infernis/
  api/              REST API routes, auth middleware
  db/               SQLAlchemy ORM, PostGIS engine
  grid/             BC grid generator (1 km / 5 km, EPSG:3005)
  models/           Pydantic schemas, danger level enums
  pipelines/        Daily pipeline, ERA5, GEE, Open-Meteo, HRDPS/GDPS, forecasting
  services/         Vectorized FWI (CFFDRS), Redis cache
  training/         XGBoost trainer, FireUNet CNN, risk fuser, backtester
  main.py           FastAPI app entry point
  admin.py          CLI tools (key management, grid init, pipeline runner)

scripts/
  download/         21 data download scripts (ERA5, MODIS, CLDN, DEM, etc.)
  train.py          Model training pipeline
  backtest.py       Historical backtesting
  dev_setup.sh      One-command development setup

tests/              Test suite (mirrors src/ structure)
docs/               White paper, architecture, API reference
brand/              Logo SVGs, icon, brand guidelines
```

## Documentation

| Document | Description |
|----------|-------------|
| [White Paper](docs/WHITE_PAPER.md) | Problem statement, wildfire science, methodology |
| [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) | System design, database schema, pipeline flows |
| [API Reference](docs/API_REFERENCE.md) | Full endpoint documentation with examples |
| [Brand Guidelines](brand/BRAND.md) | Logo, colors, danger level palette, usage |

## Tech Stack

- **Runtime**: Python 3.11+, FastAPI, Uvicorn
- **ML**: XGBoost 2.1, PyTorch 2.x (MPS/CUDA), scikit-learn
- **FWI**: Custom vectorized CFFDRS (numpy)
- **Geospatial**: GeoPandas, Rasterio, Shapely, pyproj
- **Weather**: Open-Meteo (GEM seamless), ERA5 (CDS API), HRDPS/GDPS (GRIB2 fallback)
- **Satellite**: Google Earth Engine (MODIS, VIIRS), NASA FIRMS
- **Database**: PostgreSQL 16 + PostGIS 3.4, Redis 7
- **Deploy**: Docker, Railway, GitHub Actions CI

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

```bash
make test      # Run tests
make fmt       # Format with ruff
```

**Good places to start:**
- New data source integrations
- Model improvements and feature engineering
- API endpoint enhancements
- Performance optimizations
- Documentation

## License

[Apache License 2.0](LICENSE)

---

<p align="center">
  Built in British Columbia, Canada<br/>
  <a href="https://argonbi.com">Argon BI Systems Inc.</a>
</p>
