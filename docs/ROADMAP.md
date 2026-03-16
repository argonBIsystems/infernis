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

---

## Next Up

### Explainability API — "Why Is Risk High?"
**Status:** Planned
**Impact:** First wildfire API that explains its predictions

Every wildfire system returns a number. None tell you *why*. INFERNIS will be the first to answer: "Risk is HIGH at Kamloops because DMC has been climbing for 8 days, NDVI dropped below seasonal baseline, and forecast winds exceed 30 km/h."

```
GET /v1/explain/{lat}/{lon}
```

Returns per-feature SHAP contributions ranked by impact, with human-readable descriptions and a `baseline_comparison` showing today vs. 10-year seasonal average. Powered by TreeSHAP computed during the daily pipeline — deterministic, auditable, no LLM dependency.

```
GET /v1/explain/{lat}/{lon}/history?days=30  — Driver trends over time
GET /v1/explain/zones                        — Province-wide: which BEC zones are elevated and why
```

This is the foundation for everything below. Insurance underwriters need it for decision justification. Fire chiefs need it for crew briefings. Utility operators need it for PSPS decisions. Developers need it for trust and debugging.

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

### Prediction Confidence Intervals
**Status:** Planned
**Impact:** Quantified uncertainty — no competitor offers this

Every risk score gains `lower_bound` and `upper_bound` at 90% confidence. Built with XGBoost quantile regression (`reg:quantileerror`). "Risk is 0.42 (0.31–0.55)" is far more actionable than a point estimate alone. Flows into every endpoint: risk, forecast, explain, and all verticals.

### MCP Server — AI Agent Integration
**Status:** Planned
**Impact:** First wildfire API with native AI agent support

Auto-generate an MCP (Model Context Protocol) server from the INFERNIS OpenAPI spec so AI assistants (Claude, ChatGPT, Cursor, Copilot) can query fire risk in natural language:

> *"What's the fire risk near my cabin in Kamloops this weekend?"*
> *"Show me all areas in BC where risk exceeds HIGH"*
> *"Is it safe to have a campfire at Shuswap Lake on Saturday?"*

The MCP server exposes INFERNIS endpoints as tools that LLMs call autonomously. This is the emerging standard for AI-to-API integration (Mapbox, Stripe, Cloudflare already ship MCP servers).

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

### Anomaly & Trend Detection
**Status:** Planned

```
GET /v1/trends/{lat}/{lon}
```

Statistical comparison of current conditions vs. 10-year seasonal baseline. Returns departure percentages, anomaly status (`NEAR_NORMAL`, `ABOVE_NORMAL`, `WELL_ABOVE_NORMAL`, `RECORD_HIGH_FOR_DATE`) per feature, and risk escalation velocity (3-day, 7-day rate of change). Answers the question: "Is this unusual?"

### Multi-Scale Risk Aggregation
**Status:** Planned

```
GET /v1/risk/summary?scale=fire_centre
```

Risk aggregated at configurable spatial scales: province, fire centre (6), BEC zone (14), regional district (27), or watershed. Each region includes cell count, mean/max risk, cells above HIGH, dominant SHAP driver, and escalation velocity. Enterprise dashboards need rolled-up views, not 84K individual cells.

Boundary data from BC Data Catalogue (fire centres, regional districts) and BC Freshwater Atlas (watersheds) — same open data pipeline as BEC zones.

### Post-Fire Recovery Monitoring
**Status:** Planned
**Data source:** [ORNL DAAC](https://daac.ornl.gov/) burn severity datasets + Sentinel-2 NDVI

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

```
GET /v1/climate/{lat}/{lon}?scenario=ssp245&horizon=2050
```

Forward-project fire risk under CMIP6 SSP scenarios using BCCAQv2 downscaled climate data from ClimateData.ca. Returns baseline vs. projected annual risk, fire season length projections, and return period estimates. Supports SSP1-2.6, SSP2-4.5, SSP5-8.5 at horizons 2030–2080.

Note: projections reflect how future climate conditions would score under the model trained on 2015–2024 data. They do not account for ecosystem adaptation, land use changes, or fire management evolution.

### Fire Behavior Prediction (FBP) & Spread Simulation
**Status:** Research
**Impact:** Unprecedented as a public API
**Foundation:** [cffdrs_py](https://github.com/cffdrs/cffdrs_py) — official Python CFFDRS module (FWI + FBP)

INFERNIS currently implements only the FWI (Fire Weather Index) system. The Canadian CFFDRS also includes FBP (Fire Behavior Prediction) — which calculates rate of spread, head fire intensity, crown fraction burned, and fire type (surface/intermittent crown/active crown). An official Python module (`cffdrs_py`) now implements both FWI and FBP with 25+ parameters per cell.

**Phase 1 — FBP integration:**
- Integrate `cffdrs_py` FBP calculations into the daily pipeline
- Add fire behavior fields to risk responses: `rate_of_spread_mpm`, `head_fire_intensity_kwm`, `fire_type`, `crown_fraction_burned`
- Validate INFERNIS's custom FWI implementation against `cffdrs_py` canonical output
- Cross-reference with BC Gov's open-source [Wildfire Predictive Services](https://github.com/bcgov/wps) (FastAPI + PostGIS, 64 stars, 2,147 commits) for architectural patterns

**Phase 2 — Spread simulation API:**

```
POST /v1/simulate/spread
{
  "ignition_lat": 50.23,
  "ignition_lon": -121.58,
  "hours": [2, 6, 24],
  "conditions": "current"
}
```

Given an ignition point, use FBP rate-of-spread + wind direction + slope + fuel type to simulate fire perimeter growth. Return GeoJSON polygons for each time step. Uses FWI, FBP, and spatial data INFERNIS already computes daily.

No public API offers this. Technosylva does it internally for fire agencies. NASA has a prototype digital twin. This would make INFERNIS the most technically ambitious wildfire API in existence.

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
| [BC Wildfire Predictive Services](https://github.com/bcgov/wps) | BC Government | Reference architecture (FastAPI + PostGIS, open source) |
| [ClimateData.ca](https://climatedata.ca/) | ECCC / PCIC | BCCAQv2 downscaled CMIP6 projections for climate scenarios |
| [CNFDB](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb) | NRCan | Historical fire perimeters (1950–present) |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | NASA | Near-real-time active fire detection (MODIS/VIIRS) |
| [NIFC Open Data](https://data-nifc.opendata.arcgis.com/) | US NIFC | US fire data for cross-border expansion |
| [awesome-wildfire](https://github.com/ubc-lib-geo/awesome-wildfire) | UBC Library | Curated list of wildfire data, tools, and research |
