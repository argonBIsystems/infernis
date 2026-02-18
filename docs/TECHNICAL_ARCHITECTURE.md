# INFERNIS Technical Architecture

> How the system works under the hood.

**Date**: 2026-02-16
**Status**: Living Document
**Companion**: See `docs/plans/2026-02-15-infernis-design.md` for product design and data source inventory.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Deep Dive](#component-deep-dive)
   - [DATA FORGE (Ingestion Pipeline)](#data-forge-ingestion-pipeline)
   - [FIRE CORE (XGBoost Classifier)](#fire-core-xgboost-classifier)
   - [HEATMAP ENGINE (U-Net CNN)](#heatmap-engine-u-net-cnn)
   - [RISK FUSER](#risk-fuser)
4. [BC Grid System](#bc-grid-system)
5. [Database Schema](#database-schema)
6. [Caching Strategy](#caching-strategy)
7. [API Design](#api-design)
8. [Daily Pipeline Flow](#daily-pipeline-flow)
9. [Tech Stack](#tech-stack)
10. [Deployment (Railway)](#deployment-railway)
11. [Monitoring and Reliability](#monitoring-and-reliability)
12. [Security](#security)
13. [Dashboard & Self-Service Auth](#dashboard--self-service-auth)
14. [Scaling Considerations](#scaling-considerations)

---

## System Overview

INFERNIS is a **Python 3.11+ monolith** that predicts wildfire occurrence across British Columbia, Canada. It is deployed as a single Railway service with companion PostgreSQL+PostGIS and Redis instances.

Key architectural characteristics:

- **FastAPI REST API** serves all external traffic. Auto-generated OpenAPI/Swagger docs are available at `/v1/docs`.
- **PostgreSQL + PostGIS** stores the BC spatial grid, daily predictions, pipeline metadata, and API key records. PostGIS enables spatial queries (nearest-cell lookups, bounding-box filters, zone aggregations).
- **Redis** acts as the hot cache layer for the current day's pre-computed predictions. The API reads from Redis first and falls back to PostGIS only on cache miss.
- **Daily batch pipeline** runs at **14:00 PT** (after noon weather observations become available). It fetches weather data, computes fire weather indices, assembles feature matrices, runs ML inference across all grid cells, and writes results to both PostgreSQL and Redis.
- **No real-time inference.** All predictions are pre-computed during the daily pipeline run. The API is a read-through cache over pre-computed results, which keeps response latencies low and operational complexity minimal.

The system is designed to handle 2,113,524 grid cells at 1km resolution (Phase 3, complete) on a single Railway instance, with legacy support for 84,107 cells at 5km.

---

## Architecture Diagram

```
 DATA SOURCES                  INFERNIS ENGINE                          CONSUMERS
 ============                  ===============                          =========

 +-------------+
 | ERA5        |---+
 | (Copernicus)|   |
 +-------------+   |
                   |     +--------------------------------------------------+
 +-------------+   |     |                                                  |
 | GEE         |---+---->|  DATA FORGE                                      |
 | (Satellite) |   |     |  - ERA5 weather fetch (cdsapi)                   |
 +-------------+   |     |  - FWI computation (vectorized NumPy)                   |
                   |     |  - GEE satellite data (NDVI, snow, topo)         |
 +-------------+   |     |  - Lightning density (MSC Datamart)              |
 | MSC Datamart|---+     |  - Static data join (fuel types, BEC, DEM)       |
 | (Lightning) |         |  - Feature matrix assembly [84K x 28]            |
 +-------------+         |                                                  |
                         +------------------+-------------------------------+
 +-------------+                            |
 | Static Data |------- loaded at startup   |
 | (Fuel, BEC, |                            v
 |  DEM)       |         +------------------+-------------------------------+
 +-------------+         |                                                  |
                         |  FIRE CORE (XGBoost)        HEATMAP ENGINE (CNN) |
                         |  - Per-cell binary           - U-Net encoder-    |
                         |    classification               decoder          |
                         |  - 28 features/cell          - Multi-channel     |
                         |  - Calibrated probability      raster input      |
                         |    output (0-1)              - Spatial risk       |
                         |                                heatmap output    |
                         |                                                  |
                         +------------------+-------------------------------+
                                            |
                                            v
                         +------------------+-------------------------------+
                         |                                                  |
                         |  RISK FUSER                                      |
                         |  - Weighted combination of model outputs         |
                         |  - Regional calibration per BEC zone             |
                         |  - 6-level danger classification                 |
                         |                                                  |
                         +------------------+-------------------------------+
                                            |
                         +------------------+-------------------------------+
                         |                  v                               |
                         |  +------------+     +-----------+                |
                         |  | PostgreSQL |     |   Redis   |                |
                         |  | + PostGIS  |     |   Cache   |                |   +---------+
                         |  +-----+------+     +-----+-----+                |   | Web /   |
                         |        |                  |                      |   | Mobile  |
                         |        v                  v                      |-->| Apps    |
                         |  +-------------------------------------------+  |   +---------+
                         |  |           API GATEWAY (FastAPI)            |  |
                         |  |                                           |  |   +---------+
                         |  |  /v1/risk/{lat}/{lon}   /v1/risk/grid     |  |   | Govt /  |
                         |  |  /v1/risk/heatmap       /v1/risk/zones    |  |-->| Fire    |
                         |  |  /v1/fwi/{lat}/{lon}    /v1/conditions    |  |   | Service |
                         |  |  /v1/status             /v1/coverage      |  |   +---------+
                         |  |                                           |  |
                         |  +-------------------------------------------+  |   +---------+
                         |                                                  |   | Insur./ |
                         +--------------------------------------------------+-->| Enterp. |
                                                                                +---------+
         SCHEDULER: APScheduler - daily batch @ 14:00 PT
```

---

## Component Deep Dive

### DATA FORGE (Ingestion Pipeline)

DATA FORGE is the automated data ingestion and feature engineering pipeline. It runs once daily and is responsible for assembling a complete feature matrix for every grid cell in BC.

#### ERA5 Weather Fetch

Weather data is sourced from the **ERA5 Reanalysis** dataset via the Copernicus Climate Data Store API (`cdsapi`). ERA5 provides global gridded weather fields at approximately 31km horizontal resolution with hourly temporal resolution.

Variables fetched:
- **2m temperature** (K, converted to C)
- **2m dewpoint temperature** (K, used to derive relative humidity)
- **10m u-component of wind** and **10m v-component of wind** (m/s, combined into wind speed and direction)
- **Total precipitation** (m, accumulated over 24h, converted to mm)
- **Volumetric soil water content** (all 4 layers: 0-7cm, 7-28cm, 28-100cm, and 100-289cm)
- **Evaporation** (m, converted to mm)

ERA5 data is fetched for the noon LST observation window (12:00 PT / 20:00 UTC) to align with standard FWI computation requirements. The pipeline requests data for both the current day and the previous day to ensure FWI continuity.

ERA5 fields are delivered as NetCDF4 files, opened with `xarray`, and interpolated to the BC grid using `scipy.interpolate.RegularGridInterpolator` for efficient vectorized interpolation at 2M+ cell scale.

#### FWI Computation (Vectorized NumPy)

The **Canadian Forest Fire Weather Index (FWI) System** is the backbone of fire danger assessment in Canada. INFERNIS computes FWI from raw ERA5 weather inputs using a custom vectorized NumPy implementation of the CFFDRS equations (in `services/fwi_service.py`), rather than relying on pre-computed CWFIS grids, because this gives us full control over the spatial resolution and allows computation at every grid cell regardless of weather station proximity. The vectorized implementation processes all 2M+ grid cells simultaneously.

The FWI system has a three-tier structure with six components:

**Tier 1 -- Fuel Moisture Codes** (cumulative, carry forward daily):

| Code | Full Name | Tracks | Time Lag |
|------|-----------|--------|----------|
| FFMC | Fine Fuel Moisture Code | Surface litter moisture (top 1-2 cm of dead fine fuels) | ~2/3 day |
| DMC | Duff Moisture Code | Loosely compacted organic layer (5-10 cm depth) | ~15 days |
| DC | Drought Code | Deep compact organic layer (10-20 cm depth) | ~52 days |

**Tier 2 -- Fire Behavior Indices** (instantaneous, derived daily):

| Index | Full Name | Inputs | Represents |
|-------|-----------|--------|------------|
| ISI | Initial Spread Index | FFMC + wind speed | Expected rate of fire spread |
| BUI | Buildup Index | DMC + DC | Total fuel available for combustion |

**Tier 3 -- Final Output**:

| Index | Full Name | Inputs | Represents |
|-------|-----------|--------|------------|
| FWI | Fire Weather Index | ISI + BUI | Overall fire intensity potential |

The cumulative nature of FWI is critical to understand: FFMC, DMC, and DC are not computed from scratch each day. They carry forward from the previous day's values and are updated based on the current day's weather. This means the pipeline must maintain a running state of moisture codes across the fire season. At season startup (typically April 1 for BC), standard default values are used: FFMC=85.0, DMC=6.0, DC=15.0.

The four weather inputs required at noon local standard time are:
1. Temperature (degrees C)
2. Relative humidity (%)
3. Wind speed (km/h)
4. Precipitation (mm, accumulated over the previous 24 hours)

#### GEE Satellite Data

Google Earth Engine provides server-side processing of satellite imagery, avoiding the need to download raw raster tiles. The pipeline uses the Earth Engine Python API (`ee`) to compute zonal statistics for each grid cell.

Data products fetched via GEE:

| Product | Source | Resolution | Temporal | Use |
|---------|--------|-----------|----------|-----|
| NDVI | MODIS MOD13A1 (or Sentinel-2 in Phase 3) | 500m / 16-day composite | Latest available | Vegetation greenness / dryness |
| Snow cover | MODIS MOD10A1 | 500m / daily | Current day | Binary snow presence per cell |
| Elevation | CDEM via GEE Assets | ~20m | Static | Elevation, slope, aspect, hillshade |
| EVI | MODIS MOD13A1 | 500m / 16-day | Latest available | Enhanced vegetation index (Phase 2) |
| LAI | MODIS MOD15A2H | 500m / 8-day | Latest available | Leaf area index (Phase 2) |

GEE computations are submitted as batch `reduceRegions` calls over the BC grid FeatureCollection. Results are exported as CSV/JSON and parsed into the feature matrix. The pipeline only re-fetches satellite products when new composites are available (e.g., MODIS NDVI updates every 16 days).

#### Lightning Density

Lightning strike data is sourced from the **Meteorological Service of Canada (MSC) Datamart** at `dd.weather.gc.ca`. The Canadian Lightning Detection Network (CLDN) provides flash density grids at 2.5km resolution with 10-minute temporal granularity.

The pipeline aggregates flash counts within each grid cell over the preceding 24-hour and 72-hour windows to produce:
- `lightning_24h`: flash count per cell in the last 24 hours
- `lightning_72h`: flash count per cell in the last 72 hours

Lightning is a key ignition source for BC wildfires and is particularly important for predicting fires in remote interior and northern regions where human-caused ignition is less common.

#### Static Data

Static features are loaded once at application startup (or refreshed annually) and joined to the grid cells:

| Dataset | Source | Derived Features |
|---------|--------|-----------------|
| CDEM (Canadian DEM) | Open Canada / GEE | `elevation`, `slope`, `aspect`, `hillshade` |
| CFFDRS FBP Fuel Types 2024 | Open Canada | `fuel_type` (16 categories + NF/WA) |
| BC BEC Zone Map | BC Data Catalogue | `bec_zone` (14 biogeoclimatic zones) |

These are pre-computed per grid cell during grid initialization and stored in the `grid_cells` table. They do not change between daily pipeline runs.

#### Daily Schedule

The pipeline is triggered at **14:00 PT** (22:00 UTC) by APScheduler. This timing is chosen because:

1. Noon weather observations (the standard for FWI computation) are available by early afternoon.
2. ERA5 near-real-time data typically has a 5-hour latency, so noon UTC data (04:00-05:00 PT) is available by ~10:00 PT.
3. GEE composite products are pre-computed and available for immediate query.
4. Results are published before end-of-business in the Pacific time zone.

---

### FIRE CORE (XGBoost Classifier)

FIRE CORE is the primary prediction model. It is a gradient-boosted decision tree classifier (XGBoost) that answers a binary question for each grid cell: **will a fire ignite here in the next 24 hours?**

#### Feature Vector

Each grid cell is represented by a feature vector of 28 features:

| Category | Features | Count |
|----------|----------|-------|
| FWI Components | FFMC, DMC, DC, ISI, BUI, FWI | 6 |
| Raw Weather | temperature, relative humidity, wind speed, wind direction, precipitation (24h), soil moisture (layers 1-4), evapotranspiration | 10 |
| Vegetation | NDVI, LAI, snow cover (binary) | 3 |
| Topography & Infrastructure | elevation, slope, aspect, hillshade, distance to nearest road | 5 |
| Temporal | day-of-year sine, day-of-year cosine | 2 |
| Lightning | lightning density (24h), lightning density (72h) | 2 |

The feature vector is assembled by DATA FORGE into a NumPy array of shape `[N_cells, N_features]` -- `[2113524, 28]` at 1km resolution (or `[84107, 28]` at 5km).

Day-of-year is encoded as sine and cosine components to capture fire seasonality as a continuous cyclical feature: `doy_sin = sin(2 * pi * doy / 365)`, `doy_cos = cos(2 * pi * doy / 365)`.

#### Training Data

The model is trained on **10 fire seasons of historical data (2015-2024)** at 1km resolution:

- **Positive samples**: Grid cells where a fire was recorded in the Canadian National Fire Database (CNFDB) or BC Fire Perimeters database. Fire point locations are snapped to the nearest grid cell. The training set contains approximately **27,146 positive samples** across the training period.

- **Negative samples**: For each fire-day in the training set, negative samples are drawn from grid cells that did not experience fire. To avoid near-miss contamination (sampling cells that were almost on fire as negatives), a **spatiotemporal buffer** is applied: cells within 60km and 3 days of an observed fire are excluded from the negative pool. Negatives are downsampled at a **10:1 ratio** (271,460 negatives), resulting in approximately **298,606 total training samples**.

- **Historical weather features**: ERA5 reanalysis provides consistent gridded weather data back to 1950. The 2015-2024 window was chosen to align with MODIS LAI coverage and the availability of all 28 features including 4-depth soil moisture.

#### Class Imbalance

Wildfire is a rare event. On any given day in BC, fewer than 0.1% of grid cells experience fire ignition. This extreme class imbalance is handled through multiple strategies:

1. **`scale_pos_weight`**: XGBoost's built-in class weighting parameter is set to the ratio of negative-to-positive samples (10:1 after buffered sampling).
2. **Spatiotemporal-buffered negative sampling**: A 10km/7-day exclusion buffer around fire events prevents the model from learning to distinguish fire-adjacent cells (which are likely on the verge of ignition) as negatives.
4. **Evaluation metrics**: The model is evaluated primarily on **AUC-ROC**, **precision-recall AUC**, and **F1-score** rather than raw accuracy, since a naive all-negative classifier would achieve >99% accuracy.

#### Probability Calibration

Raw XGBoost output probabilities are not well-calibrated (i.e., a predicted probability of 0.3 does not necessarily mean 30% of those predictions are fires). **Platt scaling** (logistic regression on the model's raw outputs) is applied as a post-processing step on a held-out calibration set to produce well-calibrated probability estimates.

Calibration is verified using reliability diagrams (calibration curves) and the Brier score.

#### Measured Performance

The trained 1km XGBoost model achieves the following performance on held-out test data (5-fold cross-validation):

- **AUC-ROC: 0.974**
- **Average Precision: 0.794**
- **Brier Score: 0.036**

Walk-forward temporal backtesting (train on [2015, test_year-1], test on test_year) yields AUC 0.90--0.93 across six held-out fire seasons (2019--2024), confirming temporal generalization.

#### Model Serialization

The trained model is saved in **XGBoost's native JSON format** (`fire_core_v1.json`). JSON format is chosen over pickle for security (no arbitrary code execution on load) and portability. The model file is loaded once at application startup and held in memory for inference.

Model path: `models/fire_core_1km_v1.json` for 1km, `models/fire_core_v1.json` for 5km (configurable via `INFERNIS_MODEL_PATH` / `INFERNIS_MODEL_1KM_PATH` environment variables).

---

### HEATMAP ENGINE (U-Net CNN) [Phase 2]

The HEATMAP ENGINE is a convolutional neural network that captures spatial patterns that a per-cell classifier like XGBoost cannot learn: spatial autocorrelation (adjacent dry zones compound risk), topographic fire corridors (valleys and ridges that channel fire), and neighborhood fuel connectivity.

#### Architecture

The model uses a standard **U-Net encoder-decoder** architecture implemented in PyTorch:

```
Input: [B, 12, 256, 512]    (batch, channels, height, width)
                |
        +-------v-------+
        | Encoder Block 1|  32 filters, 3x3 conv, BN, ReLU
        | MaxPool 2x2    |  -> [B, 32, 128, 256]
        +-------+-------+
                |
        +-------v-------+
        | Encoder Block 2|  64 filters
        | MaxPool 2x2    |  -> [B, 64, 64, 128]
        +-------+-------+
                |
        +-------v-------+
        | Encoder Block 3|  128 filters
        | MaxPool 2x2    |  -> [B, 128, 32, 64]
        +-------+-------+
                |
        +-------v-------+
        | Encoder Block 4|  256 filters
        | MaxPool 2x2    |  -> [B, 256, 16, 32]
        +-------+-------+
                |
        +-------v-------+
        |   Bottleneck   |  512 filters
        +-------+-------+  -> [B, 512, 16, 32]
                |
        +-------v-------+
        | Decoder Block 4|  256 filters + skip connection
        | Upsample 2x2   |  -> [B, 256, 32, 64]
        +-------+-------+
                |
        +-------v-------+
        | Decoder Block 3|  128 filters + skip connection
        | Upsample 2x2   |  -> [B, 128, 64, 128]
        +-------+-------+
                |
        +-------v-------+
        | Decoder Block 2|  64 filters + skip connection
        | Upsample 2x2   |  -> [B, 64, 128, 256]
        +-------+-------+
                |
        +-------v-------+
        | Decoder Block 1|  32 filters + skip connection
        | Upsample 2x2   |  -> [B, 32, 256, 512]
        +-------+-------+
                |
        +-------v-------+
        |  1x1 Conv      |  -> [B, 1, 256, 512]
        |  Sigmoid        |
        +----------------+

Output: Single-channel risk heatmap, values 0.0-1.0
```

#### Input Channels

The CNN receives a multi-channel raster representing the current state of BC, where the grid covers **256 rows x 512 columns at 5km resolution** (covering lat 48.30-60.60, lon -139.10 to -114.00, padded to fit the U-Net's power-of-2 requirements). Each channel is a spatial field:

| Channel | Source | Description |
|---------|--------|-------------|
| 1 | ERA5 | Temperature (C) |
| 2 | ERA5 | Relative humidity (%) |
| 3 | ERA5 | Wind speed (km/h) |
| 4 | ERA5 | Soil moisture (layer 1) |
| 5 | fwi_service | FFMC |
| 6 | fwi_service | BUI |
| 7 | GEE | NDVI |
| 8 | GEE | Snow cover (binary) |
| 9 | Static | Elevation (normalized) |
| 10 | Static | Slope (normalized) |
| 11 | Static | Fuel type (label-encoded, normalized) |
| 12 | Static | BEC zone (label-encoded, normalized) |

Non-BC cells (ocean, out-of-province) are masked with zeros. The mask is also applied to the loss function during training to prevent the model from learning on non-land areas.

#### Training

The CNN is trained on historical daily rasters (2015-2024) where the label is a binary fire/no-fire mask derived from CNFDB and BC Fire Perimeters. Training uses binary cross-entropy loss with the spatial mask applied to prevent the model from learning on non-land areas. The 1km model early-stopped at epoch 24 (patience=10) on Apple MPS, achieving AUC-ROC of 0.815.

#### Inference

At 1km resolution with `base_filters=64`, the FireUNet CNN has approximately **31 million parameters**. At 5km with `base_filters=32`, it has approximately **7.8 million parameters**. Inference processes a single forward pass in under 1 second on CPU. No GPU is required for inference.

---

### RISK FUSER

The RISK FUSER combines outputs from FIRE CORE and HEATMAP ENGINE into a single composite score per grid cell.

#### Fusion Formula

The Risk Fuser operates in **logit space** for numerically stable combination of model outputs:

```
logit(p) = log(p / (1 - p))

fused_logit = w_xgb * logit(xgboost_probability) + w_cnn * logit(cnn_heatmap_value) + bias
infernis_score = sigmoid(fused_logit)
```

Where:
- `xgboost_probability`: calibrated per-cell fire probability from FIRE CORE (0.0-1.0)
- `cnn_heatmap_value`: spatial risk value from HEATMAP ENGINE for the same cell (0.0-1.0)
- `w_xgb`, `w_cnn`, `bias`: per-BEC-zone logistic regression coefficients

#### Weight Learning

Per-BEC-zone logistic regression coefficients are learned on a held-out validation set. For each of BC's 13 calibrated BEC zones, a separate logistic regression is fit on paired (logit(xgb_prob), logit(cnn_value), actual_fire) samples from that zone. This allows the fuser to weight the models differently across ecosystems -- for example, the CNN may contribute more in zones with strong spatial fire patterns, while XGBoost may dominate in zones where point-level weather features are more predictive.

Coefficients are re-learned whenever either model is retrained. In Phase 1 (XGBoost only), the fuser passes through the XGBoost probability directly.

#### Regional Calibration

Fire risk varies dramatically across BC's diverse ecosystems. A 0.30 risk score in the coastal rainforest (CWH zone) may represent an anomalously dangerous day, while the same score in the dry interior (BG or PP zone) may be routine during summer.

The RISK FUSER applies **per-BEC-zone calibration** across BC's biogeoclimatic zones. Of the 14 BEC zones, 13 have sufficient fire history for independent calibration. Calibration is performed by computing historical fire frequency percentiles within each zone and adjusting the fused score so that danger level thresholds correspond to historically meaningful fire probability quantiles within that zone.

The 14 BEC zones:

| Code | Zone Name |
|------|-----------|
| AT | Alpine Tundra |
| BG | Bunch Grass |
| BWBS | Boreal White and Black Spruce |
| CDF | Coastal Douglas-fir |
| CWH | Coastal Western Hemlock |
| ESSF | Engelmann Spruce -- Subalpine Fir |
| ICH | Interior Cedar-Hemlock |
| IDF | Interior Douglas-fir |
| MH | Mountain Hemlock |
| MS | Montane Spruce |
| PP | Ponderosa Pine |
| SBPS | Sub-Boreal Pine-Spruce |
| SBS | Sub-Boreal Spruce |
| SWB | Spruce-Willow-Birch |

#### Danger Classification

The fused and regionally calibrated score maps to a six-level danger classification:

| Level | Score Range | Hex Color | Interpretation |
|-------|-------------|-----------|----------------|
| VERY_LOW | 0.00 -- 0.05 | #22C55E | Negligible fire risk. Conditions are wet or snow-covered. |
| LOW | 0.05 -- 0.15 | #3B82F6 | Minor risk. Fires unlikely under current conditions. |
| MODERATE | 0.15 -- 0.35 | #EAB308 | Elevated risk. Fires possible with ignition source. |
| HIGH | 0.35 -- 0.60 | #F97316 | Significant risk. Fires likely to spread if ignited. |
| VERY_HIGH | 0.60 -- 0.80 | #EF4444 | Severe risk. Aggressive fire behavior expected. |
| EXTREME | 0.80 -- 1.00 | #1A0000 | Critical risk. Explosive fire growth potential. |

Threshold boundaries are the defaults. Per-BEC-zone calibration may shift these boundaries so that, for example, the EXTREME threshold in the CWH (wet coastal) zone triggers at a lower raw score than in the BG (dry grassland) zone, reflecting the relative rarity and severity of fire in each ecosystem.

---

## BC Grid System

### Projection and Cell Layout

BC is covered by an **equal-area grid** in **EPSG:3005 (BC Albers)** projection. This is the standard projected coordinate system for British Columbia, maintaining equal area across the province so that each grid cell represents the same physical area regardless of latitude.

At 1km resolution, the grid covers BC with **2,113,524 land cells** (84,107 at 5km). The grid is generated by:

1. Computing the BC Albers bounding box of the provincial boundary polygon.
2. Dividing the bounding box into 5,000m x 5,000m square cells.
3. Discarding cells that do not intersect BC's land boundary (ocean, neighboring provinces).
4. Assigning each remaining cell a unique identifier.

### Cell Identification

Cell IDs follow the format: **`BC-5K-NNNNNN`**

Where:
- `BC` = province code
- `5K` = resolution indicator (5km); future resolutions use `1K` for 1km
- `NNNNNN` = zero-padded six-digit sequential integer, assigned in row-major order from the northwest corner of the grid

Examples: `BC-5K-000001`, `BC-5K-004921`, `BC-5K-012034`

### Coordinate Storage

Each cell stores two geometric representations:

| Column | Type | CRS | Use |
|--------|------|-----|-----|
| `geom` | POLYGON | EPSG:3005 (BC Albers) | Spatial joins, area calculations, grid queries |
| `centroid` | POINT | EPSG:4326 (WGS84) | API queries by lat/lon, nearest-cell lookups |

The dual-CRS design allows the pipeline to operate in the equal-area projection (where the 5km grid is a perfect square lattice) while the API serves consumers in standard WGS84 latitude/longitude.

### Static Features per Cell

Each cell has the following static attributes pre-computed during grid initialization:

| Attribute | Source | Type |
|-----------|--------|------|
| `elevation` | CDEM zonal mean | float (meters) |
| `slope` | CDEM-derived zonal mean | float (degrees) |
| `aspect` | CDEM-derived zonal mean | float (degrees, 0-360) |
| `hillshade` | CDEM-derived zonal mean | float (0-255) |
| `bec_zone` | BC BEC Map majority zone | enum (14 values) |
| `fuel_type` | CFFDRS FBP 2024 majority type | enum (20 values) |

### Resolution Scaling

| Phase | Resolution | Cell Count | Cell ID Prefix |
|-------|-----------|------------|----------------|
| Phase 1-2 | 5km | 84,107 | `BC-5K-` |
| Phase 3 (current) | 1km | 2,113,524 | `BC-1K-` |

---

## Database Schema

PostgreSQL 16+ with the PostGIS extension. All spatial data uses the geometries described in the grid system section.

### `grid_cells`

Stores the static BC grid. Populated once during initialization and updated only when the grid is regenerated (e.g., resolution change).

```sql
CREATE TABLE grid_cells (
    cell_id       VARCHAR(20) PRIMARY KEY,       -- e.g. 'BC-5K-004921'
    geom          GEOMETRY(POLYGON, 3005),        -- cell boundary in BC Albers
    centroid      GEOMETRY(POINT, 4326),          -- cell center in WGS84
    lat           DOUBLE PRECISION NOT NULL,      -- centroid latitude
    lon           DOUBLE PRECISION NOT NULL,      -- centroid longitude
    bec_zone      VARCHAR(10) NOT NULL,           -- BEC zone code (e.g. 'IDF')
    fuel_type     VARCHAR(5) NOT NULL,            -- FBP fuel type (e.g. 'C7')
    elevation     REAL,                           -- meters above sea level
    slope         REAL,                           -- degrees (0-90)
    aspect        REAL,                           -- degrees (0-360, north=0)
    hillshade     REAL,                           -- 0-255
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_grid_cells_geom ON grid_cells USING GIST (geom);
CREATE INDEX idx_grid_cells_centroid ON grid_cells USING GIST (centroid);
CREATE INDEX idx_grid_cells_bec_zone ON grid_cells (bec_zone);
```

### `predictions`

Stores daily prediction results for every grid cell. One row per cell per day.

```sql
CREATE TABLE predictions (
    id              BIGSERIAL PRIMARY KEY,
    cell_id         VARCHAR(20) NOT NULL REFERENCES grid_cells(cell_id),
    prediction_date DATE NOT NULL,

    -- Model outputs
    score           REAL NOT NULL,                -- fused INFERNIS score (0-1)
    level           VARCHAR(15) NOT NULL,         -- danger level enum
    fire_core_prob  REAL,                         -- raw XGBoost probability
    heatmap_value   REAL,                         -- CNN heatmap value (Phase 2)

    -- FWI components
    ffmc            REAL,
    dmc             REAL,
    dc              REAL,
    isi             REAL,
    bui             REAL,
    fwi             REAL,

    -- Weather conditions
    temperature_c   REAL,
    rh_pct          REAL,
    wind_speed_kmh  REAL,
    wind_dir_deg    REAL,
    precip_24h_mm   REAL,
    soil_moisture_1 REAL,
    soil_moisture_2 REAL,
    evapotrans_mm   REAL,

    -- Vegetation
    ndvi            REAL,
    evi             REAL,
    lai             REAL,
    snow_cover      BOOLEAN,

    -- Extensible feature storage
    features        JSONB,                        -- overflow for additional features

    -- Model metadata
    model_version   VARCHAR(20),

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_predictions_cell_date UNIQUE (cell_id, prediction_date)
);

CREATE INDEX idx_predictions_date ON predictions (prediction_date);
CREATE INDEX idx_predictions_cell_date ON predictions (cell_id, prediction_date);
CREATE INDEX idx_predictions_level ON predictions (prediction_date, level);
```

### `pipeline_runs`

Tracks the execution history and health of the daily pipeline.

```sql
CREATE TABLE pipeline_runs (
    id              SERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL,         -- 'running', 'success', 'failed', 'partial'
    cells_processed INTEGER DEFAULT 0,
    cells_total     INTEGER DEFAULT 0,
    error_message   TEXT,
    model_version   VARCHAR(20),
    era5_timestamp  TIMESTAMPTZ,                  -- timestamp of ERA5 data used
    gee_ndvi_date   DATE,                         -- date of NDVI composite used
    duration_sec    REAL,
    metadata        JSONB,                        -- additional run metadata
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pipeline_runs_date ON pipeline_runs (run_date);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs (status);
```

### `api_keys`

Manages API authentication and rate limiting.

```sql
CREATE TABLE api_keys (
    id              SERIAL PRIMARY KEY,
    key_hash        VARCHAR(128) NOT NULL UNIQUE,  -- SHA-256 hash of the API key
    name            VARCHAR(100) NOT NULL,          -- human-readable key name
    tier            VARCHAR(20) NOT NULL DEFAULT 'free',  -- 'free', 'pro', 'enterprise'
    daily_limit     INTEGER NOT NULL DEFAULT 50,
    requests_today  INTEGER NOT NULL DEFAULT 0,
    last_request_at TIMESTAMPTZ,
    last_reset_at   DATE,                          -- date when requests_today was last reset
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_hash ON api_keys (key_hash);
CREATE INDEX idx_api_keys_tier ON api_keys (tier);
```

### `users`

Stores dashboard user accounts linked to Firebase Authentication and their API keys.

```sql
CREATE TABLE users (
    id                  SERIAL PRIMARY KEY,
    firebase_uid        VARCHAR(128) NOT NULL UNIQUE,   -- Firebase user ID
    email               VARCHAR(255) NOT NULL,
    display_name        VARCHAR(200),
    api_key_id          INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
    tier                VARCHAR(20) NOT NULL DEFAULT 'free',
    billing_cycle_start DATE NOT NULL,                  -- advances every 30 days
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_firebase_uid ON users (firebase_uid);
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_api_key_id ON users (api_key_id);
```

### Schema Migrations

Database schema is managed via **Alembic**. Migration files are stored in `alembic/versions/`. Migrations are run automatically on deployment via the Railway start command or manually via `alembic upgrade head`.

---

## Caching Strategy

Redis serves as the hot cache layer between the daily pipeline and the API. The goal is to serve 100% of normal API requests from Redis without touching PostgreSQL.

### Key Structure

```
pred:{date}:{cell_id}
```

Example: `pred:2026-07-15:BC-5K-004921`

The value is a JSON-serialized prediction record containing all fields that the API may return (score, level, FWI components, weather conditions, vegetation indices, context).

### TTL Policy

All prediction keys are set with a **48-hour TTL**. This ensures:
- The current day's predictions are always available.
- The previous day's predictions remain available as a fallback if the current day's pipeline fails.
- Stale data is automatically evicted without manual cleanup.

### Write Pattern

The pipeline writes predictions to Redis using a **Redis pipeline (batch) command** to minimize round-trip overhead:

```python
pipe = redis_client.pipeline()
for cell_id, prediction_data in predictions.items():
    key = f"pred:{date_str}:{cell_id}"
    pipe.setex(key, ttl_seconds=172800, value=json.dumps(prediction_data))
pipe.execute()
```

A batch of SETEX commands executes in a single network round-trip.

### Read Pattern

The API serves a request for `/v1/risk/{lat}/{lon}` as follows:

1. **Nearest-cell lookup**: The lat/lon is mapped to the nearest grid cell using a PostGIS `ST_DistanceSphere` query on the `grid_cells.centroid` column (with a GIST index). This mapping is itself cached in an LRU cache in the application process since the grid is static.
2. **Redis GET**: Attempt `GET pred:{today}:{cell_id}`.
3. **Cache hit**: Deserialize JSON and return response. This is the fast path (~1-2ms).
4. **Cache miss**: Fall back to a PostgreSQL query on the `predictions` table filtered by `cell_id` and `prediction_date`. Populate the Redis cache with the result for subsequent requests.

### Geographic Index for Nearest-Cell Lookups

The application maintains an in-memory **k-d tree** (via `scipy.spatial.KDTree`) of all grid cell centroids in WGS84. Given an arbitrary lat/lon, the nearest grid cell is found in O(log N) time without a database query. The k-d tree is built once at application startup from the `grid_cells` table and remains static.

---

## API Design

### Base URL

```
https://api.infernis.ca/v1
```

### Authentication

All endpoints require an API key passed in the `X-API-Key` HTTP header:

```
X-API-Key: ifn_live_abc123def456
```

API keys are prefixed with `ifn_live_` (production) or `ifn_test_` (development) for human readability. Keys are hashed with SHA-256 before storage in the `api_keys` table; the raw key is never stored.

### Rate Limiting

Rate limits are enforced per API key based on tier:

| Tier | Daily Limit | Grid Queries | Heatmap Access | History Access |
|------|------------|-------------|---------------|----------------|
| Free | 50 requests/day | No | No | No |
| Pro | 10,000 requests/day | Yes | Yes | Yes |
| Enterprise | Unlimited | Yes | Yes | Yes |

Rate limit state is tracked in the `api_keys.requests_today` column, reset daily at 00:00 UTC. Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1721088000
```

### Endpoints

#### `GET /v1/risk/{lat}/{lon}`

Point query. Returns the complete risk assessment for the nearest grid cell.

**Path Parameters:**
- `lat` (float): Latitude in decimal degrees (WGS84)
- `lon` (float): Longitude in decimal degrees (WGS84)

**Query Parameters:**
- `date` (optional, ISO 8601 date): Prediction date. Defaults to today.

**Response:** JSON object containing location, grid cell ID, risk score and level, model component scores, FWI components, weather conditions, vegetation state, and context (BEC zone, fuel type, elevation).

**Tiers:** Free, Pro, Enterprise

---

#### `GET /v1/risk/grid`

Area query. Returns a GeoJSON FeatureCollection of risk predictions for all cells within a bounding box.

**Query Parameters:**
- `bbox` (required): `west,south,east,north` in decimal degrees
- `resolution` (optional): `5km` (default) or `1km` (Phase 3)
- `level` (optional): Filter by danger level (e.g., `HIGH,VERY_HIGH,EXTREME`)

**Response:** GeoJSON FeatureCollection. Each Feature is a grid cell polygon with risk properties.

**Tiers:** Pro, Enterprise

---

#### `GET /v1/risk/heatmap`

Returns a rendered heatmap image or raster from the CNN HEATMAP ENGINE.

**Query Parameters:**
- `bbox` (required): `west,south,east,north` in decimal degrees
- `format` (optional): `png` (default) or `geotiff`
- `width` (optional): pixel width for PNG output (default: 800)

**Response:** `image/png` or `image/tiff` binary data. GeoTIFF includes georeferencing metadata.

**Tiers:** Pro, Enterprise. Available in Phase 2+.

---

#### `GET /v1/risk/zones`

Returns aggregate risk levels for all BC fire zones / BEC zones.

**Response:** JSON array of zone objects, each containing zone code, zone name, current aggregate danger level, number of cells at each level, and the highest individual cell score.

**Tiers:** Free, Pro, Enterprise

---

#### `GET /v1/fwi/{lat}/{lon}`

Returns raw FWI components for the nearest grid cell.

**Response:** JSON object with `ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi` values.

**Tiers:** Free, Pro, Enterprise

---

#### `GET /v1/conditions/{lat}/{lon}`

Returns current weather and environmental conditions for the nearest grid cell.

**Response:** JSON object with temperature, relative humidity, wind speed/direction, precipitation, soil moisture, NDVI, snow cover.

**Tiers:** Free, Pro, Enterprise

---

#### `GET /v1/status`

Returns pipeline health and system status. No authentication required.

**Response:**
```json
{
  "status": "operational",
  "pipeline": {
    "last_run": "2026-07-15T14:00:00-07:00",
    "last_status": "success",
    "cells_processed": 12034,
    "model_version": "fire_core_v1.2"
  },
  "data_freshness": {
    "era5_timestamp": "2026-07-15T12:00:00Z",
    "ndvi_composite_date": "2026-07-09",
    "prediction_date": "2026-07-15"
  },
  "uptime_seconds": 86412
}
```

**Tiers:** Public (no API key required)

---

#### `GET /v1/coverage`

Returns the BC coverage boundary and grid metadata.

**Response:** GeoJSON Feature representing the BC provincial boundary, plus metadata about the grid (cell count, resolution, CRS).

**Tiers:** Public (no API key required)

---

#### `GET /health`

Lightweight health check for load balancers and Railway.

**Response:** `{"status": "ok", "version": "0.1.0"}`

---

### Response Format

- **Standard endpoints**: JSON (`application/json`)
- **Spatial queries** (`/risk/grid`, `/coverage`): GeoJSON (`application/geo+json`)
- **Heatmap**: PNG (`image/png`) or GeoTIFF (`image/tiff`)

All JSON responses include a top-level `meta` field with request ID, timestamp, and API version. Error responses follow RFC 7807 Problem Details format.

### Pre-Computed Design

A critical architectural decision: **all predictions are pre-computed during the daily pipeline run.** The API performs zero ML inference at request time. This means:

- Response latency is bounded by cache/database lookup time, not model inference time.
- The API can serve thousands of concurrent requests without GPU/CPU inference contention.
- The system can serve predictions even if the model loading or inference code has a bug, as long as the last successful pipeline run is cached.
- The tradeoff is that predictions are at most 24 hours stale. For wildfire risk assessment at a daily cadence, this is acceptable.

---

## Daily Pipeline Flow

The pipeline executes the following sequence every day at 14:00 PT:

```
Step  Description                                          Duration (est.)
----  ---------------------------------------------------  ---------------
 1    Fetch ERA5 weather for yesterday/today                2-5 min
      (noon observations via cdsapi)

 2    Compute FWI components (FFMC, DMC, DC, ISI, BUI,     <30 sec
      FWI) using previous day's codes via fwi_service

 3    Fetch latest NDVI composite from GEE                  1-2 min
      (only if new 16-day composite is available)

 4    Fetch MODIS snow cover from GEE                       1-2 min
      (daily product, always fetched)

 5    Fetch lightning density from MSC Datamart              <30 sec
      (aggregate flash counts per grid cell)

 6    Assemble feature matrix                               <30 sec
      [2.1M cells x 28 features] as NumPy array

 7    Run XGBoost inference -> per-cell probabilities        ~2-3 min
      (batch predict on full feature matrix)

 8    Run CNN inference -> spatial heatmap                   <30 sec
      (single forward pass through U-Net on CPU)

 9    Fuse model outputs with regional calibration           <5 sec
      per BEC zone (14 zones)

10    Write results to PostgreSQL                            30-60 sec
      (batch INSERT INTO predictions)

11    Batch-cache results to Redis                           <5 sec
      (Redis pipeline with ~84,500 SETEX commands)

12    Update pipeline_runs table with status                 <1 sec
      (record timing, cell count, model version, errors)

Total estimated pipeline duration: 5-12 minutes
```

### Error Handling

- **Step 1 failure (ERA5 unavailable)**: The pipeline logs the error and falls back to the previous day's weather data. Predictions are generated with stale weather but current vegetation and static features. The `pipeline_runs` entry is marked `partial` with an error message.
- **Step 3 failure (GEE unavailable)**: The pipeline uses the most recent cached NDVI composite. Since NDVI changes slowly (16-day composites), a few days of staleness is acceptable.
- **Step 7 failure (XGBoost error)**: The pipeline fails entirely. The `pipeline_runs` entry is marked `failed`. The API continues serving the previous day's predictions from Redis (which has a 48-hour TTL).
- **Step 10 failure (PostgreSQL unavailable)**: Results are still written to Redis (step 11). The pipeline retries the PostgreSQL write on the next run.

### FWI State Management

FWI moisture codes (FFMC, DMC, DC) are cumulative and carry forward daily. The pipeline maintains this state by:

1. Querying the most recent `predictions` row for each cell to retrieve yesterday's FFMC, DMC, and DC values.
2. Passing these as initial conditions to `fwi_service` along with today's weather inputs.
3. Storing the updated codes in today's `predictions` row.

At fire season startup (approximately April 1), standard default values are used: FFMC=85.0, DMC=6.0, DC=15.0. During the off-season (November-March), the pipeline may run at reduced frequency or maintain codes at winter defaults.

---

## Tech Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | ML ecosystem maturity, GEE API, fwi_service compatibility |
| API Framework | FastAPI | >=0.115 | Async support, auto-generated OpenAPI docs, Pydantic integration, high performance |
| ML -- Tabular | XGBoost | >=2.0 | Proven top performer for fire occurrence prediction on tabular data |
| ML -- Spatial | PyTorch | >=2.0 | Standard framework for U-Net CNN, ONNX export support |
| FWI Engine | fwi_service (vectorized NumPy) | custom | Vectorized implementation of CFFDRS equations for full-grid computation |
| Satellite Data | earthengine-api | >=1.1 | Server-side satellite processing via Google Earth Engine |
| Weather Data | cdsapi | >=0.7 | Official Copernicus Climate Data Store API client for ERA5 |
| Geospatial -- Vector | GeoPandas | >=1.0 | DataFrame operations on spatial vector data |
| Geospatial -- Raster | Rasterio | >=1.3 | GDAL-backed raster I/O, reprojection, resampling |
| Geometry | Shapely | >=2.0 | Geometric operations (intersections, buffers, containment) |
| Projection | pyproj | >=3.6 | CRS transformations between EPSG:3005 and EPSG:4326 |
| NetCDF / GRIB | xarray + netCDF4 | >=2024.1 / >=1.7 | Reading ERA5 NetCDF4 and GRIB2 weather files |
| Numerics | NumPy | >=1.26 | Array operations, feature matrix assembly |
| Data Frames | pandas | >=2.2 | Tabular data manipulation |
| ORM | SQLAlchemy | >=2.0 | Database abstraction, async session support |
| Spatial ORM | GeoAlchemy2 | >=0.15 | PostGIS geometry type integration for SQLAlchemy |
| DB Driver | psycopg2-binary | >=2.9 | PostgreSQL driver |
| Migrations | Alembic | >=1.13 | Schema versioning and migration management |
| Cache Client | redis-py | >=5.0 | Redis client with pipeline (batch) support |
| Scheduler | APScheduler | >=3.10 | In-process cron-style scheduling for the daily pipeline |
| HTTP Client | httpx | >=0.27 | Async HTTP client for MSC Datamart and external API calls |
| Serialization | joblib | >=1.3 | Efficient serialization for large NumPy arrays (pipeline intermediates) |
| Calibration | scikit-learn | >=1.4 | Platt scaling (CalibratedClassifierCV), evaluation metrics |
| Spatial Index | scipy | >=1.12 | KDTree for nearest-cell lookups |
| ASGI Server | uvicorn | >=0.30 | High-performance ASGI server for FastAPI |
| Configuration | pydantic-settings | >=2.0 | Typed environment variable parsing |
| Error Tracking | sentry-sdk | >=2.0 | Exception tracking and alerting |
| Hosting | Railway | -- | Managed platform with PostgreSQL, Redis, and Docker support |

### System-Level Dependencies (Dockerfile)

The following C/C++ libraries are required and installed in the Docker image:

- **GDAL** (libgdal-dev): Geospatial Data Abstraction Library, required by Rasterio and GeoPandas
- **GEOS** (libgeos-dev): Geometry Engine Open Source, required by Shapely
- **PROJ** (libproj-dev): Cartographic projection library, required by pyproj

---

## Deployment (Railway)

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system geospatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/
COPY models/ models/
COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000
CMD ["uvicorn", "infernis.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway Services

The Railway project consists of three services:

| Service | Type | Configuration |
|---------|------|---------------|
| **web** | Docker (Dockerfile) | Runs the FastAPI application via uvicorn. Port auto-detected by Railway. Start command overridden by Procfile: `uvicorn infernis.main:app --host 0.0.0.0 --port $PORT` |
| **PostgreSQL + PostGIS** | Railway Plugin | Managed PostgreSQL instance with PostGIS extension enabled. Connection string injected as `DATABASE_URL`. |
| **Redis** | Railway Plugin | Managed Redis instance. Connection string injected as `REDIS_URL`. |

### Environment Variables

All configuration is passed via environment variables with the `INFERNIS_` prefix:

| Variable | Required | Description |
|----------|----------|-------------|
| `INFERNIS_DATABASE_URL` | Yes | PostgreSQL connection string (with PostGIS) |
| `INFERNIS_REDIS_URL` | Yes | Redis connection string |
| `INFERNIS_CDS_KEY` | Yes | Copernicus CDS API key for ERA5 data |
| `INFERNIS_GEE_PROJECT` | Yes | Google Earth Engine project ID |
| `INFERNIS_GEE_SERVICE_ACCOUNT_KEY` | Yes | GEE service account JSON key (base64-encoded or file path) |
| `INFERNIS_FIRMS_MAP_KEY` | No | NASA FIRMS API key (for active fire validation) |
| `INFERNIS_MODEL_PATH` | No | Path to XGBoost model file (default: `models/fire_core_v1.json`) |
| `INFERNIS_DEBUG` | No | Enable debug logging (default: `false`) |
| `INFERNIS_SENTRY_DSN` | No | Sentry DSN for error tracking |

### Health Check

Railway is configured to probe the `/health` endpoint:

```json
{
  "healthcheckPath": "/health",
  "healthcheckTimeout": 10
}
```

The health check returns HTTP 200 with `{"status": "ok"}` when the application is ready to serve requests. The application reports unhealthy if it cannot connect to PostgreSQL or Redis on startup.

### Railway.json

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn infernis.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 10,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

---

## Monitoring and Reliability

### Pipeline Health Tracking

Every pipeline execution is recorded in the `pipeline_runs` table with:
- Start/end timestamps and duration
- Status (`running`, `success`, `failed`, `partial`)
- Number of cells processed vs. total
- Error messages (if any)
- Model version and data timestamps

The `/v1/status` endpoint exposes the most recent pipeline run information, allowing consumers to verify data freshness.

### Alerting

**Sentry** is integrated for exception tracking across both the API and the pipeline. Critical alerts are configured for:
- Pipeline failure (status = `failed`)
- Pipeline partial completion with fewer than 90% of cells processed
- API error rate exceeding 1% over a 5-minute window
- Redis connection failures

### Fallback Strategy

The system is designed to degrade gracefully when upstream data sources are unavailable:

| Failure Scenario | Fallback Behavior |
|-----------------|-------------------|
| ERA5 data delayed or unavailable | Use previous day's weather data. Mark pipeline as `partial`. |
| GEE unavailable | Use most recent cached NDVI/snow composite. Acceptable for 2-3 days. |
| MSC Datamart unavailable | Set lightning density features to 0. Most fires are not lightning-caused. |
| Pipeline fails entirely | API continues serving previous day's predictions from Redis (48h TTL). |
| Redis down | API falls back to direct PostgreSQL queries (slower but functional). |
| PostgreSQL down | API serves from Redis only. No historical queries available. |

### Model Drift Monitoring

Fire prediction accuracy can degrade over time due to:
- Climate change shifting fire weather patterns
- Land use changes altering fuel loads
- Changes in fire suppression practices

Drift is monitored by comparing predicted vs. actual fire occurrences each fire season:

1. During fire season (May-October), actual fire reports from CWFIS are ingested weekly.
2. Predicted probabilities are compared against observed fires using AUC-ROC, calibration curves, and spatial hit/miss analysis.
3. If AUC-ROC drops below 0.85 or calibration error exceeds 0.05, a model retraining cycle is triggered.
4. Annual retraining is performed regardless, incorporating the most recent fire season's data.

---

## Security

### API Key Management

- API keys are generated as cryptographically random tokens (32 bytes, base64-encoded).
- Keys are prefixed for human readability: `ifn_live_` for production, `ifn_test_` for development.
- Only the **SHA-256 hash** of the key is stored in the `api_keys` table. The raw key is shown to the user exactly once at creation time and never stored.
- Key validation: on each request, the provided key is hashed and compared against the `key_hash` column.

### Rate Limiting

Rate limits are enforced at the application level using the `api_keys.requests_today` counter. The counter is reset daily at 00:00 UTC. Requests exceeding the daily limit receive HTTP 429 (Too Many Requests) with a `Retry-After` header.

### CORS

Cross-Origin Resource Sharing is configured to allow requests from approved frontend domains. In development, `*` is permitted. In production, CORS origins are restricted to known consumer applications.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # e.g., ["https://api.infernis.ca"]
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type"],
)
```

`GET` is used for data queries. `POST` is used for dashboard management endpoints (register, key regeneration). The `Authorization` header carries Firebase Bearer tokens for dashboard auth.

### Data Privacy

- The `users` table stores email addresses and display names provided during Firebase sign-up. No other PII is stored.
- API keys are associated with user accounts via the `users.api_key_id` foreign key.
- Request logs contain API key hashes (not raw keys), timestamps, and requested coordinates. No client IP addresses are logged.
- Firebase Authentication handles password storage, email verification, and OAuth tokens. No credentials are stored in INFERNIS databases.

### Data Sources

All input data sources are open, publicly funded datasets:
- ERA5: Copernicus Climate Change Service (free with registration)
- GEE: Google Earth Engine (currently under non-commercial license; commercial license planned once revenue is generated)
- CNFDB/CWFIS: Natural Resources Canada (open data)
- BC Data Catalogue: Province of BC (open data)
- MSC Datamart: Environment and Climate Change Canada (open data)

No proprietary or restricted data sources are used in the prediction pipeline.

---

## Dashboard & Self-Service Auth

> **Note:** The dashboard and Firebase authentication are part of the hosted service at `api.infernis.ca` only. They are not included in the open-source repository. Self-hosted deployments use direct API key management via the database.

### Overview

INFERNIS provides a self-service dashboard at `https://api.infernis.ca/static/index.html` where users sign up, receive an API key, and monitor usage. The dashboard runs alongside the data API as part of the same FastAPI application.

### Dual Auth Architecture

Two independent authentication layers run side-by-side:

| Layer | Scope | Mechanism | Purpose |
|-------|-------|-----------|---------|
| API Key | `/v1/*` data endpoints | `X-API-Key` header (SHA-256 hash lookup) | Rate-limited data access |
| Firebase | `/api/dashboard/*` management endpoints | `Authorization: Bearer <firebase-id-token>` | User account management |

The API key middleware (`APIKeyMiddleware`) bypasses `/api/dashboard` and `/static` paths entirely. Dashboard routes use a FastAPI `Depends()` on `verify_firebase_token` which validates the Firebase ID token server-side.

### Firebase Integration

**Project**: `infernis-55fc3`
**Auth Providers**: Email/Password, Google Sign-In
**Admin SDK**: `firebase-admin` Python package, initialized from `INFERNIS_FIREBASE_SA_JSON` env var (service account JSON as a single-line string)

The Firebase client SDK is loaded in the browser from the CDN (`firebase-app-compat.js`, `firebase-auth-compat.js`). Client-side config (apiKey, projectId, authDomain) is served dynamically by the FastAPI endpoint `GET /static/js/firebase-config.js` which reads values from environment variables — no config is hardcoded in static files.

### Self-Service Key Provisioning Flow

```
User                Browser              FastAPI               Firebase Admin SDK
 |                    |                     |                        |
 |  Sign up/Sign in   |                     |                        |
 |  (email or Google) |                     |                        |
 | -----------------> |                     |                        |
 |                    |  Firebase Auth      |                        |
 |                    | ------------------> |                        |
 |                    |  ID Token           |                        |
 |                    | <------------------ |                        |
 |                    |                     |                        |
 |                    |  POST /api/dashboard/register                |
 |                    |  Authorization: Bearer <token>               |
 |                    | ------------------> |                        |
 |                    |                     |  verify_id_token()     |
 |                    |                     | ---------------------> |
 |                    |                     |  {uid, email, name}    |
 |                    |                     | <--------------------- |
 |                    |                     |                        |
 |                    |                     |  Create user record    |
 |                    |                     |  Generate API key      |
 |                    |                     |  (secrets.token_hex)   |
 |                    |                     |  Store SHA-256 hash    |
 |                    |                     |                        |
 |                    |  {api_key: "abc..", |                        |
 |                    |   tier: "free"}     |                        |
 |                    | <------------------ |                        |
 |                    |                     |                        |
 |  Dashboard shows   |                     |                        |
 |  key (once only)   |                     |                        |
 | <----------------- |                     |                        |
```

The plaintext API key is returned **only** at registration and key regeneration. It is never stored server-side. The dashboard shows a masked preview derived from the key hash (e.g., `a3f8****...****d2e1`).

### Dashboard API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/dashboard/register` | POST | Firebase Bearer | Idempotent. First call: creates user + free API key, returns plaintext key. Subsequent: returns profile. |
| `/api/dashboard/profile` | GET | Firebase Bearer | User profile with masked key preview, tier, billing cycle. |
| `/api/dashboard/usage` | GET | Firebase Bearer | Requests today, daily limit, cycle dates, days remaining. |
| `/api/dashboard/key/regenerate` | POST | Firebase Bearer | Deactivates old key, creates new one, returns plaintext (one-time). |

### Frontend Stack

The dashboard is vanilla HTML/CSS/JS with no build step:

- **CSS**: Custom dark theme ("Dark Forge") with Syne, Outfit, and JetBrains Mono fonts. Ember orange (#f05e23) accent color.
- **JS**: Firebase SDK (CDN, compat), `fetch()` for API calls, `navigator.clipboard` for copy.
- **Serving**: FastAPI `StaticFiles` mount at `/static`.
- **Auth guard**: `onAuthStateChanged` redirects unauthenticated users to the login page.

### Billing Cycle

Each user has a 30-day rolling billing cycle starting from their signup date. The cycle resets lazily — when a profile or usage request is made and `today >= billing_cycle_start + 30`, the start date advances by 30 days and the request counter on the API key is reset.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `INFERNIS_FIREBASE_PROJECT_ID` | Firebase project ID (`infernis-55fc3`) |
| `INFERNIS_FIREBASE_API_KEY` | Firebase web API key (public, used in client config) |
| `INFERNIS_FIREBASE_SA_JSON` | Service account JSON (single-line string, used by Admin SDK) |

---

## Scaling Considerations

### Current: 1km Grid (2,113,524 cells)

At 1km resolution (Phase 3, complete), the system is deployed on a single Railway instance:

| Resource | Estimated Usage | Railway Limit |
|----------|----------------|---------------|
| CPU (pipeline) | ~10-15 min per day | Sufficient on standard plan |
| CPU (API) | Negligible (cache lookups) | Sufficient on standard plan |
| RAM (application) | ~500 MB - 1 GB (model + grid + k-d tree) | 512 MB - 8 GB available |
| Redis memory | ~760 MB for 2.1M predictions (compressed) | 1 GB recommended |
| PostgreSQL storage | ~8 GB/year of prediction history | 10 GB available |
| XGBoost inference | ~2-3 min for 2.1M cells (batch) | Well within pipeline window |
| CNN inference | <30 sec for 256x512x12 raster on CPU | Well within pipeline window |
| Feature storage | ~546 GB for full training data (float16 parquets) | Local/cloud storage |

### Horizontal Scaling Path

If the single-instance architecture becomes insufficient:

1. **Separate worker**: Move the pipeline to a dedicated Railway service that writes to the shared PostgreSQL and Redis. The API service becomes stateless and read-only.
2. **Multiple API replicas**: FastAPI instances are stateless (all state is in PostgreSQL/Redis), so horizontal scaling via Railway's replica count is straightforward.
3. **Redis Cluster**: If Redis memory becomes a bottleneck, switch to Redis Cluster or use a larger managed instance.
4. **PostgreSQL read replicas**: If query load exceeds what a single PostgreSQL instance can handle, add read replicas for the API while the pipeline writes to the primary.

### Cost Estimation (Railway)

| Service | Phase 1-2 (5km) | Phase 3 (1km) |
|---------|-----------------|---------------|
| Web (FastAPI) | ~$5-10/month | ~$10-20/month |
| PostgreSQL | ~$5-10/month | ~$15-30/month |
| Redis | ~$5/month | ~$10-20/month |
| **Total** | **~$15-25/month** | **~$35-70/month** |

These estimates are for Railway's usage-based pricing and will vary with actual traffic.
