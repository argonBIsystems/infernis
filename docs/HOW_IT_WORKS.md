# How INFERNIS Works

> A technical overview of the wildfire prediction engine for developers and evaluators.

---

## What INFERNIS Does

INFERNIS predicts daily fire ignition risk for every 5km grid cell across British Columbia (84,535 cells). It answers: **"What's the probability of a fire starting near this location today?"**

It does NOT predict:
- Whether a fire is currently burning (use `/v1/fires/near` for active incidents)
- How a fire will spread (FBP fire behavior metrics are provided but not simulation)
- Dollar-value losses (this is occurrence probability, not damage modeling)

---

## Data Pipeline

Every day at 2:00 PM Pacific Time, the automated pipeline:

1. **Fetches weather** from Open-Meteo (ECCC's GEM model) — temperature, humidity, wind, precipitation for all of BC via a coarse 50km weather grid (~1,500 points), then interpolated to the 5km prediction grid using KDTree nearest-neighbor.

2. **Computes FWI** (Canadian Fire Weather Index System) — six moisture/fire behavior indices (FFMC, DMC, DC, ISI, BUI, FWI) rolled forward daily using yesterday's state. Validated against NRCan's canonical `cffdrs_py` implementation. Includes diurnal FFMC adjustment for afternoon peak conditions.

3. **Fetches satellite data** from Google Earth Engine — MODIS NDVI (vegetation greenness), snow cover, and LAI (leaf area index).

4. **Fetches lightning data** from MSC Datamart — 24h and 72h flash density.

5. **Assembles 28-feature vector** per cell:
   - FWI components (6): ffmc, dmc, dc, isi, bui, fwi
   - Weather (10): temperature, humidity, wind speed/direction, precipitation, soil moisture (4 depths), evapotranspiration
   - Vegetation (3): ndvi, snow_cover, lai
   - Topography (5): elevation, slope, aspect, hillshade, distance to road
   - Temporal (2): day-of-year sine/cosine encoding
   - Lightning (2): 24h and 72h flash density

6. **Runs XGBoost inference** — gradient-boosted decision tree model trained on 298,606 labeled samples (2015–2024), 10:1 negative:positive ratio, with 10km spatial and 7-day temporal exclusion buffers. Outputs a 0–1 fire occurrence probability per cell.

7. **Applies BEC-zone calibration** — 14 independent logistic regression calibrations (one per biogeoclimatic zone) adjust raw XGBoost scores to account for zone-specific fire regimes.

8. **Computes SHAP explainability** — XGBoost native TreeSHAP (`pred_contribs=True`) provides per-feature contributions explaining why risk is high or low at each cell.

9. **Computes FBP fire behavior** — rate of spread, head fire intensity, crown fraction burned, fire type, and flame length via `cffdrs_py` for all 14 combustible fuel types.

10. **Writes results** to Redis (API cache) and PostgreSQL (historical retention).

11. **Runs 10-day forecast** — rolls FWI forward using Open-Meteo GEM forecast weather, with confidence decay (0.95 per lead day). Processed in 20K-cell chunks to manage memory.

---

## Model Performance

| Metric | Value | Context |
|--------|-------|---------|
| AUC-ROC | 0.974 | Cross-validated on 1km grid |
| Brier Score | 0.036 | Well-calibrated probabilities |
| Average Precision | 0.794 | Good at finding actual fires |
| Temporal Backtest | 0.90–0.93 AUC | Walk-forward on held-out 2019–2024 seasons |

The model was validated using temporal cross-validation (no future data leakage) and walk-forward backtesting against each fire season independently.

---

## Location Risk Profile

The `/v1/risk/profile` endpoint provides a comprehensive assessment combining three time horizons:

### Historical Fire Exposure
- Fires within 10km radius across three windows: 10-year, 30-year, and all-time
- Source: CNFDB (Canadian National Fire Database) + BC Fire Perimeters + BC Fire Incidents
- 225,000 fire records for BC dating back to 1919

### Structural Susceptibility
- Empirical fire occurrence rate computed from 2015–2024 training data
- Hierarchical fallback: cell-level (≥2 fires) → BEC+fuel group → BEC zone average
- Percentile ranking against all 84,535 BC cells

### Composite Risk Rating
- Weighted blend: 30% susceptibility + 30% historical exposure + 40% current daily score
- Always reflects today's conditions alongside long-term structural risk
- An underwriter sees both "this area has burned 55 times in 10 years" and "right now conditions are MODERATE"

### Seasonal Risk Curve
- Monthly fire frequency index (0–1 scale) by BEC zone
- Based on 225K fire records showing when fires occur throughout the year
- Example: Interior Douglas-fir zone peaks in August, fire season April–October

---

## API Endpoints

### Core Risk
| Endpoint | Description |
|----------|-------------|
| `GET /v1/risk/{lat}/{lon}` | Today's risk score, FWI, weather, fire behavior |
| `GET /v1/forecast/{lat}/{lon}` | 10-day forecast with confidence decay |
| `GET /v1/risk/profile/{lat}/{lon}` | Full risk profile: historical + susceptibility + composite |
| `GET /v1/explain/{lat}/{lon}` | SHAP feature contributions explaining the risk score |
| `GET /v1/trends/{lat}/{lon}` | Today vs historical baseline, velocity |

### Insurance
| Endpoint | Description |
|----------|-------------|
| `POST /v1/insurance/portfolio` | Bulk assess up to 1,000 properties |

### Aggregation
| Endpoint | Description |
|----------|-------------|
| `GET /v1/risk/zones` | Province-wide risk by BEC zone |
| `GET /v1/risk/grid?bbox=...` | GeoJSON grid for bounding box |
| `POST /v1/risk/batch` | Multi-point query (up to 50 locations) |

### Context
| Endpoint | Description |
|----------|-------------|
| `GET /v1/fwi/{lat}/{lon}` | Raw FWI components |
| `GET /v1/conditions/{lat}/{lon}` | Weather + environmental conditions |
| `GET /v1/fires/near/{lat}/{lon}` | Active fires from BC Wildfire Service |
| `GET /v1/risk/history/{lat}/{lon}` | Daily risk history (up to 30 days) |

### System
| Endpoint | Description |
|----------|-------------|
| `GET /v1/status` | Pipeline health, last run, cell count |
| `GET /v1/coverage` | BC boundary and grid metadata |
| `GET /health` | ArSite health spec v1.0 |
| `/mcp` | MCP server for AI agent integration |

---

## Data Sources

INFERNIS uses 21 open data sources with zero proprietary dependencies:

| Category | Sources |
|----------|---------|
| **Weather** | Open-Meteo (ECCC GEM model), ERA5 reanalysis (soil moisture) |
| **Satellite** | Google Earth Engine (MODIS NDVI, snow cover, LAI) |
| **Fire History** | CNFDB, BC Fire Perimeters, BC Fire Incidents |
| **Topography** | Canadian Digital Elevation Model (CDEM) via GEE |
| **Fuel** | CFFDRS fuel type classification by BEC zone |
| **Lightning** | MSC Datamart (Canadian Lightning Detection Network) |
| **Ecology** | BC Biogeoclimatic Ecosystem Classification (14 zones) |

---

## Infrastructure

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| ML Model | XGBoost 2.0+ (gradient boosted trees) |
| Fire Behavior | cffdrs_py (official NRCan CFFDRS) |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| Cache | Redis 7 |
| Deployment | Railway (Docker) |
| CDN | Cloudflare |
| Scheduling | APScheduler (daily 2 PM PT) |

---

## Limitations

- **5km resolution** — risk is per grid cell, not per individual property. Two houses 3km apart get the same score. For property-level assessment, this is area-level screening, not structure-level evaluation.
- **Ignition prediction, not damage** — the model predicts probability of a fire starting, not expected losses. Loss modeling requires structure vulnerability data.
- **BC only** — no coverage outside British Columbia. The engine is province-agnostic but grid generation and data pipelines are BC-configured.
- **Daily updates** — predictions refresh once daily at 2 PM PT. Intra-day conditions are not captured.
- **Historical fire data quality** — older CNFDB records (pre-1960) have less precise coordinates and incomplete cause attribution.
