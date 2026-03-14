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

---

## Next Up

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

A self-contained card showing current risk score, danger level, FWI, weather, and a mini forecast chart. Targets real estate listings, municipal websites, news articles, tourism operators, and community dashboards — audiences that will never write API code but need fire risk data.

### Fire Spread Simulation API
**Status:** Research
**Impact:** Unprecedented as a public API

```
POST /v1/simulate/spread
{
  "ignition_lat": 50.23,
  "ignition_lon": -121.58,
  "hours": [2, 6, 24],
  "conditions": "current"
}
```

Given an ignition point and current conditions (FWI, wind speed/direction, fuel type, slope), simulate fire spread using simplified Huygens wavelet propagation and return GeoJSON polygons for each time step. Uses data INFERNIS already computes daily.

No public API offers this. Technosylva does it internally for fire agencies. NASA has a prototype digital twin. This would make INFERNIS the most technically ambitious wildfire API in existence.

---

## Near-Term

### SSE Streaming for Live Alerts
**Status:** Planned

```
GET /v1/alerts/stream
```

Server-Sent Events connection that pushes risk changes in real-time instead of requiring polling. Fire agencies monitoring multiple locations get pushed updates as conditions change. Subscribe to specific cells, BEC zones, or danger thresholds.

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

### H3 Hexagonal Grid
**Status:** Research

Uber's H3 indexing as an alternative to the current square grid. Equal-area hexagons with 6 equidistant neighbors are better for fire spread modeling. Resolution 8 (~0.74 km²) maps closely to the 1km grid. Enables hierarchical drill-down from province to neighborhood. Endpoints would accept H3 cell IDs alongside lat/lon.

### Cloud Optimized GeoTIFF (COG) Endpoints
**Status:** Planned

Serve daily risk surfaces as COG files so GIS professionals can load them directly into QGIS or ArcGIS via HTTP range requests — no full download needed. Standard raster format for the geospatial industry.

### Air Quality Integration
**Status:** Research

```
GET /v1/air-quality/{lat}/{lon}
```

Wildfire smoke is the #1 public health impact. Correlate smoke data (PurpleAir, IQAir, Copernicus CAMS) with active fires and risk levels. Show AQI, PM2.5, and smoke forecast alongside fire risk. No wildfire API does this.

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
- Fire spread modeling algorithms
- H3 grid integration
- SDK generation and testing
- IoT sensor protocol integration
- Air quality data source integration
- Multi-province data pipeline configuration
