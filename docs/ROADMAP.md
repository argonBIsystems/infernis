# INFERNIS Roadmap

> Last updated: March 2026

This roadmap outlines planned features for INFERNIS. Priorities may shift based on community feedback and contributor interest. If you want to work on any of these, [open an issue](https://github.com/argonBIsystems/infernis/issues) or [start a discussion](https://github.com/argonBIsystems/infernis/discussions).

---

## Shipped (v0.2.0 — March 2026)

- [x] Map tile overlays `/v1/tiles/{z}/{x}/{y}.png` for Google Maps, Leaflet, Mapbox
- [x] Batch risk queries `POST /v1/risk/batch` (up to 50 locations)
- [x] Historical risk `/v1/risk/history/{lat}/{lon}` (90-day retention)
- [x] Nearby active fires `/v1/fires/near/{lat}/{lon}` from BC Wildfire Service
- [x] Webhook alerts with threshold triggers `POST /v1/alerts`
- [x] `change_24h` field in risk response
- [x] Weather data in forecast response (temp, RH, wind, precip per day)
- [x] Color hex in grid GeoJSON properties
- [x] Coordinate-based demo endpoints mirroring real API
- [x] Zero-downtime deploys (Redis startup cache)
- [x] Real soil moisture in forecasts (ERA5 carry-forward)
- [x] Pre-trained models included in repo (Git LFS)
- [x] Swagger docs with use cases and examples per endpoint
- [x] CONTRIBUTING.md, issue templates, PR template, brand kit
- [x] Community showcase: Fire Forecast BC
- [x] Single-tier API with configurable per-key rate limits
- [x] Webhook hardening (auto-disable after failures, HTTPS-only, cooldown, stale cleanup)

## Shipped (v0.3.0 — March 2026)

### Core Engine Upgrades
- [x] **FWI validation against cffdrs_py** — cross-validated INFERNIS FWI against NRCan's canonical implementation; found and fixed 3 bugs (DMC rain formula, DMC/DC temperature floor thresholds)
- [x] **Diurnal FFMC adjustment** — sub-daily fire weather correction using Red Book (3rd ed.) lookup tables; afternoon FFMC at 14:00 PT reflects peak drying conditions (+5-10 FFMC points)
- [x] **C-Haines atmospheric instability index** — continuous Haines Index computed from Open-Meteo pressure-level data (850/500 hPa); added to `/v1/risk` and `/v1/conditions` responses; C-Haines > 10 = blow-up fire potential
- [x] **Prediction confidence intervals** — XGBoost quantile regression (`reg:quantileerror`) producing 90% bounds on every risk score; `confidence_interval: {lower, upper, level}` in all risk endpoints
- [x] **Explainability API (SHAP)** — per-cell, per-feature TreeSHAP contributions computed daily; `GET /v1/explain/{lat}/{lon}` returns top 5 drivers with human-readable descriptions + composed summary; `GET /v1/explain/zones` for province-wide driver aggregation by BEC zone
- [x] **FBP fire behavior prediction** — rate of spread, head fire intensity, crown fraction burned, fire type, flame length via cffdrs_py FBP system; `fire_behaviour` field in risk responses for all 14 combustible fuel types

### Location Risk Profile
- [x] **Location fire risk profile** — `GET /v1/risk/profile/{lat}/{lon}` combining historical fire exposure (10yr/30yr/all-time within 10km), structural susceptibility (empirical fire rate with hierarchical cell→BEC+fuel→BEC fallback), fire regime characteristics, and composite risk rating blending static + live data
- [x] `fire_statistics` table with pre-computed fire statistics per grid cell
- [x] `compute_fire_stats` admin CLI command for offline spatial computation
- [x] BEC zone full name mapping (14 zones)
- [x] Alembic migration 007: fire_statistics table

### Anomaly Detection & MCP Server
- [x] **Anomaly & trend detection** — `GET /v1/trends/{lat}/{lon}` comparing current risk vs historical susceptibility baseline; departure %, anomaly status (NEAR_NORMAL through RECORD_HIGH), 3-day and 7-day velocity
- [x] **MCP Server** — auto-generated from OpenAPI spec via fastapi-mcp; all INFERNIS endpoints exposed as tools for Claude, ChatGPT, Cursor, Copilot; mounted at `/mcp`

### Infrastructure
- [x] NaN/Inf sanitization across all API endpoints (fixed Victoria 500 errors for coastal edge cells)
- [x] `--quantile` flag in training script for confidence interval model generation
- [x] Alembic migration 006: `shap_values` JSONB column on predictions table
- [x] Coarse weather grid (1,512 points at ~50km) with KDTree interpolation to 84K prediction cells — eliminates Open-Meteo rate limiting (6 batches vs 282)
- [x] XGBoost native `pred_contribs` for SHAP (replaces shap library C-extension crash)

---

## Next Up

### Insurance Vertical
**Status:** Planned
**Impact:** Direct path to underwriting integration

Purpose-built endpoints for wildfire exposure assessment:

```
POST /v1/insurance/portfolio   — Bulk risk assessment for up to 1,000 properties
GET  /v1/insurance/property/{lat}/{lon}  — Deep-dive single property report
```

Portfolio endpoint returns per-property risk scores, BEC zone fire regime classification, historical fire proximity (CNFDB), forecast outlook, top risk drivers (SHAP), and aggregate portfolio metrics (`value_at_risk_high`, risk distribution). The property endpoint adds 90-day risk history, 10-year fire proximity timeline, and seasonal risk profile.

No API-accessible, BC-specific wildfire underwriting tool exists. ZestyAI and CAPE Analytics are enterprise-only, proprietary, and US-focused.

### Utility Corridor Risk
**Status:** Planned
**Impact:** Vegetation management and PSPS decision support

```
POST /v1/utility/corridor      — Segment-level risk along transmission line geometry
GET  /v1/utility/psps-advisory — System-wide de-energization advisory
```

Accept a GeoJSON LineString (transmission corridor), return segment-level fire risk with wind overlay, PSPS score (fire risk + wind + fuel type composite), and vegetation contact risk classification. BC Hydro, FortisBC, and independent power producers have no API for this today.

### Embeddable Risk Widget
**Status:** Planned
**Impact:** Opens INFERNIS to non-developers

Zero-code integration for any website:

```html
<iframe src="https://infernis.ca/embed/risk?lat=50.67&lon=-120.33"
        width="400" height="300" frameborder="0"></iframe>
```

A self-contained card showing current risk score, danger level, FWI, weather, and a mini forecast chart. Targets real estate listings, municipal websites, news articles, tourism operators, and community dashboards.

---

## Near-Term

### Multi-Scale Risk Aggregation
**Status:** Planned

```
GET /v1/risk/summary?scale=fire_centre
```

Risk aggregated at configurable spatial scales: province, fire centre (6), BEC zone (14), regional district (27), or watershed. Each region includes cell count, mean/max risk, cells above HIGH, dominant SHAP driver, and escalation velocity. Enterprise dashboards need rolled-up views, not 84K individual cells.

Boundary data from BC Data Catalogue via [`bcdata_py`](https://github.com/bcgov/bcdata_py) — BC Gov's official Python WFS client for all provincial geodata layers (fire centres, regional districts, watersheds, terrain, admin boundaries). Pip-installable, geopandas-compatible, supports direct PostGIS loading.

### Post-Fire Recovery Monitoring
**Status:** Planned
**Data sources:** [ORNL DAAC](https://daac.ornl.gov/) burn severity datasets + Sentinel-2 NDVI
**Reference:** [bcgov/burn-severity](https://github.com/bcgov/burn-severity) dNBR methodology, [bcgov/wps-fire-perimeter](https://github.com/bcgov/wps-fire-perimeter) GEE-based detection

```
GET /v1/recovery/{lat}/{lon}
```

Track vegetation recovery after wildfires using satellite-derived burn severity (dNBR) and ongoing NDVI time series. The ORNL DAAC provides Landsat-derived burn scar data covering Alaska and Canada (1985–2015, 4.5 GB) and circumpolar fire polygons with NBR/dNBR/RdNBR metrics (1986–2020, 1.6 GB).

Returns: burn severity classification at the location, months since fire, vegetation recovery trajectory (NDVI time series post-burn), estimated recovery percentage, and reburn risk assessment (recently burned areas with regrowing fuel). Serves insurance (claim validation, recovery monitoring) and forestry (reforestation tracking) verticals directly.

### SSE Streaming for Live Alerts
**Status:** Planned

```
GET /v1/alerts/stream
```

Server-Sent Events connection that pushes risk changes in real-time instead of requiring polling. Subscribe to specific cells, BEC zones, or danger thresholds.

### Auto-Generated SDKs
**Status:** Planned

Official client libraries generated from the OpenAPI spec:

```python
# Python
pip install infernis
from infernis import Infernis
client = Infernis(api_key="your_key")
risk = client.risk.get(50.67, -120.33)
```

```javascript
// TypeScript
import { Infernis } from 'infernis';
const client = new Infernis({ apiKey: 'your_key' });
const risk = await client.risk.get(50.67, -120.33);
```

Python, TypeScript, and Go initially. Auto-published to PyPI, npm, and Go modules. Generated using Fern or Stainless from the OpenAPI spec.

### PMTiles for Serverless Map Tiles
**Status:** Planned

Pre-compute the daily risk surface as a PMTiles archive served from Cloudflare R2 or S3. One file, no tile server, HTTP range requests fetch only needed tiles. 70% smaller than individual PNGs. Eliminates per-request tile rendering compute.

### GeoParquet Bulk Exports
**Status:** Planned

```
GET /v1/export/risk?format=geoparquet&date=2026-03-14
```

Daily full-BC risk surface as a GeoParquet download. Works directly with DuckDB, Pandas, Spark, QGIS, Snowflake. Data scientists don't want to hit an API 84,535 times — give them the whole dataset in one cloud-native file.

---

## Medium-Term

### Government / Fire Services Vertical
**Status:** Planned
**Impact:** Operational decision support for BC Wildfire Service

```
GET  /v1/ops/escalation    — Zones with fastest-rising risk (3-day, 7-day velocity)
POST /v1/ops/preposition   — Crew distribution optimizer across fire centre bases
```

Escalation endpoint ranks BEC zones and fire centres by risk velocity — where conditions are deteriorating fastest. Pre-positioning endpoint accepts base locations and a shared crew pool, recommends optimal distribution weighted by current and forecast risk surface.

### Forestry Vertical
**Status:** Planned
**Impact:** Harvest scheduling and fire guard prioritization

```
POST /v1/forestry/harvest-risk               — Risk assessment for harvest unit polygons
GET  /v1/forestry/fire-guard/{lat}/{lon}     — Fire guard maintenance priority ranking
```

Harvest risk endpoint accepts cutblock geometries, returns per-unit risk with 7-day forecast and operational advisory (GO / CAUTION / STAND DOWN). Fire guard endpoint ranks nearby cells by combined risk, fuel load, and proximity to high-value timber.

### Climate Scenario Projections
**Status:** Research
**Impact:** 10–30 year risk outlooks for insurance and government planning
**Data sources:** [bcgov/climr](https://github.com/bcgov/climr) (CMIP6 downscaled to 2.5km), [ClimateData.ca](https://climatedata.ca/) (BCCAQv2)

```
GET /v1/climate/{lat}/{lon}?scenario=ssp245&horizon=2050
```

Forward-project fire risk under CMIP6 SSP scenarios. `climr` (BC Gov's climate downscaling R package, 18 stars) provides 13-model CMIP6 ensemble data downscaled to 2.5km resolution across all of North America — 1901-2023 historical + 2015-2100 projections. Uses change-factor downscaling with elevation adjustment, remote PostGIS backend with local caching to minimize storage. BCCAQv2 from ClimateData.ca is the alternative data path.

Returns baseline vs. projected annual risk, fire season length projections, and return period estimates. Supports SSP1-2.6, SSP2-4.5, SSP5-8.5 at horizons 2030–2080.

Note: projections reflect how future climate conditions would score under the model trained on 2015–2024 data. They do not account for ecosystem adaptation, land use changes, or fire management evolution.

### Fire Spread Simulation API
**Status:** Research
**Impact:** Unprecedented as a public API
**Foundation:** FBP integration (shipped in v0.3.0) provides rate-of-spread and fire intensity per cell

```
POST /v1/simulate/spread
{
  "ignition_lat": 50.23,
  "ignition_lon": -121.58,
  "hours": [2, 6, 24],
  "conditions": "current"
}
```

Given an ignition point, use FBP rate-of-spread + wind direction + slope + fuel type to simulate fire perimeter growth. Return GeoJSON polygons for each time step. Uses FWI, FBP, and spatial data INFERNIS already computes daily. Reference: [bcgov/fbp-go](https://github.com/bcgov/fbp-go) fire shape ellipse modeling.

No public API offers this. Technosylva does it internally for fire agencies. NASA has a prototype digital twin. This would make INFERNIS the most technically ambitious wildfire API in existence.

### Forest Drought Index Integration
**Status:** Research
**Reference:** [bcgov/forestDroughtTool](https://github.com/bcgov/forestDroughtTool) (12 stars), [bcgov/forDRAT](https://github.com/bcgov/forDRAT)

BC Gov's forestDroughtTool computes stand-level drought hazard using ASMR (Actual-to-Potential Evapotranspiration Ratio) water-balance modeling across BEC zones and soil moisture regimes. Outputs drought risk codes (L/M/H/VH) by species and biogeoclimatic unit with climate projection overlays (RCP 4.5, 2020/2050/2080).

ASMR serves as an antecedent fuel moisture proxy — drier conditions = higher fire risk — complementing the FWI's moisture codes (FFMC/DMC/DC) with a physically-based soil water balance. Could add a `drought_index` field to risk responses and improve forestry vertical accuracy.

### Automated Fire Perimeter Detection
**Status:** Research
**Reference:** [bcgov/wps-fire-perimeter](https://github.com/bcgov/wps-fire-perimeter) — GEE + Sentinel-2 NBR classification

Automated fire perimeter generation from Sentinel-2 satellite imagery (10m resolution) via Google Earth Engine. Uses Normalized Burn Ratio classification `(B12-B8)/(B12+B8)` with cloud masking. BC Gov's implementation targets fires >90ha with daily scheduled processing → PostGIS storage → WFS serving.

Could improve INFERNIS's active fire tracking beyond point-based BCWS ArcGIS data — generating actual fire perimeter polygons for the `/v1/fires/near` endpoint and feeding into the fire spread simulation as ground-truth validation.

### H3 Hexagonal Grid
**Status:** Research

Uber's H3 indexing as an alternative to the current square grid. Equal-area hexagons with 6 equidistant neighbors are better for fire spread modeling. Resolution 8 (~0.74 km^2) maps closely to the 1km grid. Enables hierarchical drill-down from province to neighborhood. Endpoints would accept H3 cell IDs alongside lat/lon.

### Cloud Optimized GeoTIFF (COG) Endpoints
**Status:** Planned

Serve daily risk surfaces as COG files so GIS professionals can load them directly into QGIS or ArcGIS via HTTP range requests — no full download needed. Standard raster format for the geospatial industry.

### Air Quality & Smoke Forecasts
**Status:** Planned
**Data source:** [FireSmoke Canada](https://firesmoke.ca/) (UBC / NRCan / ECCC)

```
GET /v1/smoke/{lat}/{lon}
```

Wildfire smoke is the #1 public health impact. FireSmoke Canada provides free, government-backed PM2.5 smoke concentration forecasts at 12km resolution, updated multiple times daily (08:00, 14:00, 20:00, 02:00 UTC) in NetCDF format covering all of Canada. No commercial API needed.

Returns current and forecast PM2.5 concentrations, AQI classification, smoke plume direction, and correlation with nearby active fires and INFERNIS risk levels. Endpoints:
- `GET /v1/smoke/{lat}/{lon}` — Current PM2.5 + 48h smoke forecast
- `GET /v1/smoke/forecast?hours=48` — Province-wide smoke surface as GeoJSON

No wildfire API combines fire risk prediction with smoke forecasts. This gives INFERNIS a unique "fire + air quality" story for public health, municipal, and tourism use cases.

### Interactive API Playground
**Status:** Planned

Browser-based tool where developers enter coordinates and see fire risk on a live map. Generates copy-paste code in Python, JavaScript, and curl. Personalized with the user's API key. Modeled after Mapbox's API playgrounds.

---

## Long-Term / Research

### IoT Sensor Ingestion (MQTT)
Accept real-time ground-truth data from IoT weather stations and fire detection sensors (Dryad Networks Silvanet, WIFIRE Edge). MQTT ingestion layer feeds into the prediction pipeline, improving accuracy in instrumented areas.

### Multi-Province Expansion
Extend coverage beyond BC to Alberta, Ontario, and other Canadian provinces. The engine, FWI system, and data pipelines are province-agnostic — the main work is grid generation and data source configuration per province.

### Property-Level Risk Scoring
Move from 1-5km grid cells to individual structure assessment using aerial/satellite imagery. Evaluate defensible space, roof materials, vegetation proximity. This is what ZestyAI and CAPE Analytics charge enterprise pricing for.

### Stripe-Style Developer Portal
Three-column documentation layout with nav, explanation, and live code. Language switching across Python, JavaScript, Go, curl. Personalized code samples with API key injected. Request logs showing exactly what the server received. Friction-logged onboarding flow optimized for time-to-first-API-call under 5 minutes.

### Date-Based API Versioning
Migrate from `/v1/` to date-based versions (e.g., `/v2026-04/`) with quarterly releases, 12-month support windows, and 9-month migration overlap. Following Shopify's model. Webhook versions independent of API versions.

---

## How to Contribute

Interested in working on any of these? Here's how:

1. **Check the issue tracker** — some roadmap items have associated issues with more detail
2. **Open a discussion** — propose your approach before starting
3. **Start with "Near-Term" items** — they have the clearest specs and smallest scope
4. **Read [CONTRIBUTING.md](CONTRIBUTING.md)** for setup and PR guidelines

We especially welcome contributions in:
- Explainability and SHAP integration
- FBP (Fire Behavior Prediction) integration via `cffdrs_py`
- Fire spread simulation algorithms
- Smoke forecast integration (FireSmoke Canada NetCDF pipeline)
- Post-fire recovery monitoring (ORNL DAAC dNBR + Sentinel-2)
- H3 grid integration
- SDK generation and testing
- Climate projection data pipelines (BCCAQv2 / ClimateData.ca)
- IoT sensor protocol integration
- Multi-province data pipeline configuration

---

## Open Data & Research Sources

Key external data sources and tools referenced in this roadmap:

| Resource | Provider | Use in INFERNIS |
|----------|----------|-----------------|
| [FireSmoke Canada](https://firesmoke.ca/) | UBC / NRCan / ECCC | PM2.5 smoke forecasts (12km, NetCDF, multi-daily) |
| [cffdrs_py](https://github.com/cffdrs/cffdrs_py) | NRCan (official) | FWI validation + FBP fire behavior prediction |
| [ORNL DAAC Fire Products](https://daac.ornl.gov/) | NASA / ORNL | Burn severity (dNBR), post-fire recovery, carbon emissions |
| [bcgov/wps](https://github.com/bcgov/wps) | BC Government | Reference architecture, C-Haines, HFI, diurnal FFMC (64 stars) |
| [bcgov/fbp-go](https://github.com/bcgov/fbp-go) | BC Government | FBP field calculations, fire shape ellipse, slope-corrected wind (9 stars) |
| [bcgov/wps-fire-perimeter](https://github.com/bcgov/wps-fire-perimeter) | BC Government | Automated fire perimeter generation from Sentinel-2 via GEE |
| [bcgov/burn-severity](https://github.com/bcgov/burn-severity) | BC Government | dNBR burn severity classification methodology |
| [bcgov/climr](https://github.com/bcgov/climr) | BC Government | CMIP6 downscaled to 2.5km for North America (18 stars) |
| [bcgov/forestDroughtTool](https://github.com/bcgov/forestDroughtTool) | BC Government | ASMR water-balance drought hazard by BEC zone (12 stars) |
| [bcgov/bcdata_py](https://github.com/bcgov/bcdata_py) | BC Government | Python WFS client for all BC open geodata layers (33 stars) |
| [bcgov/castor](https://github.com/bcgov/castor) | BC Government | Forest harvest simulation model at 1ha resolution (20 stars) |
| [ClimateData.ca](https://climatedata.ca/) | ECCC / PCIC | BCCAQv2 downscaled CMIP6 projections for climate scenarios |
| [CNFDB](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb) | NRCan | Historical fire perimeters (1950–present) |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | NASA | Near-real-time active fire detection (MODIS/VIIRS) |
| [NIFC Open Data](https://data-nifc.opendata.arcgis.com/) | US NIFC | US fire data for cross-border expansion |
| [awesome-wildfire](https://github.com/ubc-lib-geo/awesome-wildfire) | UBC Library | Curated list of wildfire data, tools, and research |
