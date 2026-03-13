<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/infernis-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="brand/infernis-logo-light.svg">
    <img src="brand/infernis-logo-dark.svg" alt="INFERNIS" width="320"/>
  </picture>
</p>

<p align="center">
  <em>Open-source wildfire risk prediction for British Columbia</em>
</p>

<p align="center">
  <a href="https://infernis.ca">Live API</a> &bull;
  <a href="https://api.infernis.ca/v1/docs">Swagger Docs</a> &bull;
  <a href="https://api.infernis.ca/v1/demo/risk">Try Demo</a> &bull;
  <a href="docs/WHITE_PAPER.md">White Paper</a> &bull;
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/Hosted%20API-Live-22C55E" alt="API Status"/>
</p>

---

## What is INFERNIS?

INFERNIS is an open-source wildfire risk prediction engine for British Columbia. It runs an automated daily pipeline that ingests weather data, satellite imagery, soil moisture, vegetation indices, topography, and fuel classifications, then outputs calibrated fire risk scores through a REST API.

There are **two ways to use INFERNIS**:

### 1. Use the hosted API (easiest)

A free, live instance runs at [api.infernis.ca](https://api.infernis.ca/v1/docs) at **5 km resolution** (~84,535 grid cells covering all of BC). It updates daily at 2 PM Pacific. Sign up at [infernis.ca](https://infernis.ca) for a free API key (50 requests/day).

| What you get | Details |
|---|---|
| Resolution | 5 km (~84,535 cells) |
| Model | XGBoost, 24 features, AUC-ROC 0.942 (CV) / 0.946 (test) |
| Update frequency | Daily at 2 PM Pacific |
| Forecast | Up to 10 days (ECCC GEM model via Open-Meteo) |
| Free tier | 50 requests/day, all endpoints |
| Base URL | `https://api.infernis.ca/v1/` |
| Auth | `X-API-Key` header |

### 2. Run your own instance (this repo)

Clone this repo, download the raw data with the included scripts, train your own models, and deploy your own API at **1 km resolution** (~2.1M cells). This gives you full control, higher precision, and no rate limits.

| What you get | Details |
|---|---|
| Resolution | 1 km (~2,113,524 cells) — configurable |
| Model | XGBoost, 28 features, AUC-ROC 0.974 (test) |
| CNN available | FireUNet spatial model (AUC 0.815), trained on MPS/CUDA |
| Training data | 21 download scripts, 10 fire seasons (2015–2024) |
| Requirements | Python 3.11+, PostgreSQL 16 + PostGIS 3.4, Redis 7, ~546 GB raw data |

## Try the Hosted API

No API key needed for demo endpoints:

```bash
# Sample risk data at all 6 danger levels
curl https://api.infernis.ca/v1/demo/risk | python -m json.tool

# Sample 10-day forecast showing a drying event
curl https://api.infernis.ca/v1/demo/forecast | python -m json.tool
```

With a free API key ([sign up at infernis.ca](https://infernis.ca)):

```bash
# Real-time fire risk for Kamloops
curl -H "X-API-Key: YOUR_KEY" https://api.infernis.ca/v1/risk/50.67/-120.33

# 10-day forecast for Williams Lake
curl -H "X-API-Key: YOUR_KEY" https://api.infernis.ca/v1/forecast/52.13/-122.14

# GeoJSON grid for the Okanagan
curl -H "X-API-Key: YOUR_KEY" "https://api.infernis.ca/v1/risk/grid?bbox=49.0,-120.5,50.5,-119.0"

# PNG heatmap
curl -H "X-API-Key: YOUR_KEY" "https://api.infernis.ca/v1/risk/heatmap?bbox=49.0,-120.5,50.5,-119.0" -o heatmap.png
```

### Hosted API Endpoints

| Endpoint | Description | Auth |
|----------|-------------|------|
| `GET /v1/risk/{lat}/{lon}` | Point fire risk (score, FWI, weather, context) | API Key |
| `GET /v1/forecast/{lat}/{lon}` | Multi-day forecast (up to 10 days) | API Key |
| `GET /v1/risk/grid?bbox=s,w,n,e` | Area risk as GeoJSON FeatureCollection | API Key |
| `GET /v1/risk/heatmap?bbox=s,w,n,e` | Fire risk as PNG image | API Key |
| `GET /v1/risk/zones` | Risk summary per BEC zone | API Key |
| `GET /v1/fwi/{lat}/{lon}` | Raw FWI components (FFMC, DMC, DC, ISI, BUI, FWI) | API Key |
| `GET /v1/conditions/{lat}/{lon}` | Current weather and environment conditions | API Key |
| `GET /v1/status` | Pipeline health and last run time | Public |
| `GET /v1/coverage` | Grid metadata and BC boundaries | Public |
| `GET /v1/demo/risk` | Mock data at all 6 danger levels | Public |
| `GET /v1/demo/risk/{level}` | Mock data for one danger level | Public |
| `GET /v1/demo/forecast` | Mock 10-day forecast | Public |

Full documentation with request/response examples: [API Reference](docs/API_REFERENCE.md) | [Swagger UI](https://api.infernis.ca/v1/docs)

## Run Your Own Instance

### Setup

```bash
git clone https://github.com/argonBIsystems/infernis.git
cd infernis
./scripts/dev_setup.sh    # creates venv, installs deps, copies .env

# Start PostgreSQL + Redis
make db-up
make migrate

# Generate the 1km BC grid (~2.1M cells)
python scripts/generate_grid.py --resolution 1

# Start the API (pipeline runs automatically at 2 PM Pacific)
make dev
```

### Download training data

The repo includes 21 download scripts for all open data sources. Each script targets a specific source — run them individually or all at once:

```bash
# Download everything (~546 GB when complete)
python scripts/download/download_all.py

# Or download specific sources
python scripts/download/01_era5.py          # ERA5 weather reanalysis (ECMWF)
python scripts/download/02_gee_satellite.py # MODIS NDVI, snow, LAI (Google Earth Engine)
python scripts/download/03_cnfdb.py         # Historical fire records (NRCan)
python scripts/download/17_dem.py           # Canadian Digital Elevation Model
python scripts/download/18_cldn.py          # Lightning detection (ECCC)
python scripts/download/21_bc_bec.py        # Biogeoclimatic zones (BC Gov)
```

Some scripts require API keys (CDS, GEE, NASA Earthdata, FIRMS). See `.env.example` for what's needed.

### Train models

```bash
# Process raw data into feature matrices
python scripts/train.py process --data-dir data/raw --output data/processed/features

# Build training dataset
python scripts/train.py build --features data/processed/features --output data/processed/training_data.parquet

# Train XGBoost model
python scripts/train.py train --data data/processed/training_data.parquet --output models/

# Evaluate
python scripts/train.py evaluate --model models/fire_core_1km_v1.json --data data/processed/training_data.parquet

# Train CNN heatmap model (requires GPU or Apple Silicon MPS)
python scripts/train_heatmap.py --data-dir data/processed/heatmap --epochs 30

# Per-BEC-zone calibration
python scripts/calibrate_bec.py --data data/processed/training_data.parquet --output models/bec_calibration.json

# Walk-forward backtesting (train on years N, test on year N+1)
python scripts/backtest.py backtest --data data/processed/training_data.parquet --output reports/backtest.json
```

### Grid resolution

Set `INFERNIS_GRID_RESOLUTION_KM` in `.env`:

| Resolution | Cells | Pipeline time | Notes |
|------------|-------|---------------|-------|
| 1 km | ~2.1M | ~5–12 min | Default for self-hosted. Full precision. |
| 5 km | ~84K | ~30s | Used by hosted API. Good for demos. |

```bash
# Switch to 5km (lighter, matches hosted API)
INFERNIS_GRID_RESOLUTION_KM=5.0
python scripts/generate_grid.py --resolution 5
```

The 5 km model (`fire_core_v1.json`, 24 features) and 1 km model (`fire_core_1km_v1.json`, 28 features) are resolution-specific. The pipeline auto-selects the right model based on grid resolution.

## Model Performance

### 5 km model (hosted API)

| Metric | CV (5-fold) | Test |
|--------|-------------|------|
| AUC-ROC | 0.942 | 0.946 |
| Avg Precision | 0.629 | 0.645 |
| Brier Score | 0.092 | 0.048 |
| Features | 24 | — |
| Training samples | 1,139,112 | — |

### 1 km model (self-hosted)

| Metric | Test |
|--------|------|
| AUC-ROC | 0.974 |
| Avg Precision | 0.794 |
| Brier Score | 0.036 |
| Features | 28 |
| Training samples | 298,606 |

### CNN FireUNet (1 km only)

| Metric | Test |
|--------|------|
| AUC-ROC | 0.815 |
| Epochs trained | 24 (early stopped) |
| Training time | ~3 hours on MPS |

### Walk-forward backtest (6 seasons, 2019–2024)

AUC-ROC: 0.90–0.93 | Avg Precision: 0.43–0.59 | Brier: 0.04–0.08

## Danger Levels

| Level | Score | Color | Description |
|-------|-------|-------|-------------|
| VERY_LOW | 0.00–0.05 | `#22C55E` | Minimal risk |
| LOW | 0.05–0.15 | `#3B82F6` | Low risk |
| MODERATE | 0.15–0.35 | `#EAB308` | Elevated — monitor conditions |
| HIGH | 0.35–0.60 | `#F97316` | Significant risk |
| VERY_HIGH | 0.60–0.80 | `#EF4444` | Severe danger |
| EXTREME | 0.80–1.00 | `#1A0000` | Immediate danger |

## How It Works

1. **Data Pipeline** — Daily fetch of ERA5 weather reanalysis, MODIS/VIIRS satellite imagery via Google Earth Engine, Open-Meteo NWP forecasts, and CLDN lightning density grids.

2. **FWI Computation** — Vectorized Canadian Fire Weather Index system (CFFDRS standard equations) computing all 6 components (FFMC, DMC, DC, ISI, BUI, FWI) for every grid cell.

3. **XGBoost Classifier** — Gradient-boosted model trained on 10 fire seasons (2015–2024). The 5 km model uses 24 features; the 1 km model uses 28 features (adds 4 soil moisture depth layers).

4. **CNN Spatial Model** — U-Net architecture (FireUNet) processes daily raster snapshots to capture spatial fire spread patterns. Available for 1 km resolution only.

5. **Risk Calibration** — Per-BEC-zone logistic calibration across BC's 14 biogeoclimatic zones, outputting a 6-level danger classification.

6. **Forecast Engine** — Up to 10-day risk forecasts using ECCC's GEM model (HRDPS 2.5 km for days 1–2, GDPS for days 3–10) with FWI roll-forward and 0.95/day confidence decay.

## Project Structure

```
src/infernis/
  api/              REST API routes, auth middleware
  db/               SQLAlchemy ORM, PostGIS engine
  grid/             BC grid generator (1 km / 5 km, EPSG:3005)
  models/           Pydantic schemas, danger level enums
  pipelines/        Daily pipeline, ERA5, GEE, Open-Meteo, HRDPS/GDPS, lightning, forecasting
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
brand/              Logo, brand guidelines
```

## Documentation

| Document | Description |
|----------|-------------|
| [White Paper](docs/WHITE_PAPER.md) | Wildfire science, methodology, data fusion approach |
| [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) | System design, database schema, pipeline flows |
| [API Reference](docs/API_REFERENCE.md) | Endpoint docs with request/response examples |
| [Brand Guidelines](brand/BRAND.md) | Logo, colors, typography |

## Tech Stack

- **Runtime**: Python 3.11+, FastAPI, Uvicorn
- **ML**: XGBoost 2.1, PyTorch 2.x (MPS/CUDA), scikit-learn
- **FWI**: Custom vectorized CFFDRS (numpy)
- **Geospatial**: GeoPandas, Rasterio, Shapely, pyproj
- **Weather**: Open-Meteo (GEM seamless), ERA5 (CDS API)
- **Satellite**: Google Earth Engine (MODIS, VIIRS), NASA FIRMS
- **Database**: PostgreSQL 16 + PostGIS 3.4, Redis 7
- **Deploy**: Docker, Railway, GitHub Actions CI

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

```bash
make test      # Run tests
make fmt       # Format with ruff
```

## License

[Apache License 2.0](LICENSE)

---

<p align="center">
  Built in British Columbia, Canada &bull; <a href="https://argonbi.com">Argon BI Systems Inc.</a>
</p>
