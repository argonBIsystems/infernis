# INFERNIS API Reference

> BC Forest Fire Prediction ML Engine -- Developer API Documentation

---

## Base URL

```
https://api.infernis.ca/v1
```

All endpoints are served over HTTPS. HTTP requests will be rejected. The API follows RESTful conventions and returns JSON responses unless otherwise specified.

---

## Authentication

Every request must include a valid API key in the `X-API-Key` header.

```
X-API-Key: your_api_key_here
```

### Obtaining an API Key

API keys are provisioned automatically through the INFERNIS dashboard at `https://api.infernis.ca/static/index.html`. Sign up with email/password or Google, and a free-tier API key is generated immediately. The plaintext key is shown once at signup — copy it and store it securely. You can view a masked preview and regenerate your key from the dashboard at any time.

### Rate Limit Tiers

| Tier       | Requests per Day | Grid/Heatmap Access | Support            |
|------------|------------------|---------------------|--------------------|
| Free       | 50               | No                  | GitHub Issues      |
| Pro        | 10,000           | Grid access         | Email (48h SLA)    |
| Enterprise | 100,000          | Grid + Heatmap      | Dedicated (4h SLA) |

Enterprise customers may also request custom SLAs, dedicated endpoints, and webhook integrations. Contact `hello@argonbi.com` for details.

---

## Endpoints

### GET /v1/risk/{lat}/{lon}

**Point risk query.** Returns the fire risk score and all supporting data for the nearest grid cell to the specified coordinates. This is the primary endpoint for single-location risk assessment.

**Tier:** Free, Pro, Enterprise

#### Path Parameters

| Parameter | Type  | Required | Description                                                        |
|-----------|-------|----------|--------------------------------------------------------------------|
| `lat`     | float | Yes      | Latitude in decimal degrees. Range: -90 to 90. Must fall within British Columbia boundaries (approximately 48.0 to 60.0). |
| `lon`     | float | Yes      | Longitude in decimal degrees. Range: -180 to 180. Must fall within British Columbia boundaries (approximately -140.0 to -114.0). |

#### Query Parameters

None.

#### Response 200 -- Success

Returns the complete risk assessment for the nearest grid cell.

```json
{
  "location": {
    "lat": 49.25,
    "lon": -121.77
  },
  "grid_cell_id": "BC-5K-004921",
  "timestamp": "2026-07-15T21:00:00+00:00",
  "risk": {
    "score": 0.72,
    "level": "VERY_HIGH",
    "color": "#EF4444"
  },
  "fwi": {
    "ffmc": 91.2,
    "dmc": 68.4,
    "dc": 412.0,
    "isi": 12.8,
    "bui": 89.1,
    "fwi": 34.6
  },
  "conditions": {
    "temperature_c": 32.1,
    "rh_pct": 18.0,
    "wind_kmh": 24.5,
    "precip_24h_mm": 0.0,
    "soil_moisture": 0.12,
    "ndvi": 0.45,
    "snow_cover": false
  },
  "context": {
    "bec_zone": "IDF",
    "fuel_type": "C7",
    "elevation_m": 845
  },
  "forecast_horizon": "24h",
  "next_update": "2026-07-16T21:00:00+00:00"
}
```

#### Error Responses

| Status | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| 400    | Malformed request. Latitude or longitude is not a valid number.             |
| 404    | The specified coordinates do not fall within the BC coverage area.          |
| 422    | Coordinates are valid numbers but outside the acceptable range for BC.      |
| 429    | Rate limit exceeded. See [Rate Limiting](#rate-limiting) for details.       |
| 503    | Service temporarily unavailable. The prediction pipeline may be updating.   |

---

### GET /v1/risk/grid

**Area risk query.** Returns a GeoJSON FeatureCollection containing risk scores for all grid cells within the specified bounding box. Each Feature includes the same risk data as the point query endpoint.

**Tier:** Pro, Enterprise

#### Query Parameters

| Parameter    | Type   | Required | Default | Description                                                                                   |
|--------------|--------|----------|---------|-----------------------------------------------------------------------------------------------|
| `bbox`       | string | Yes      | --      | Bounding box as `south,west,north,east` in decimal degrees. All four values must fall within BC boundaries. |
| `level`      | string | No       | --      | Filter by danger level (e.g., `VERY_HIGH`, `EXTREME`).                                        |

#### Response 200 -- Success

Returns a GeoJSON FeatureCollection. Each Feature represents a single grid cell.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-121.80, 49.23],
            [-121.75, 49.23],
            [-121.75, 49.27],
            [-121.80, 49.27],
            [-121.80, 49.23]
          ]
        ]
      },
      "properties": {
        "cell_id": "BC-5K-004921",
        "score": 0.72,
        "level": "VERY_HIGH",
        "bec_zone": "IDF",
        "fuel_type": "C7",
        "fwi": 34.6,
        "temperature_c": 32.1
      }
    }
  ],
  "metadata": {
    "bbox": [49.0, -123.5, 50.0, -122.0],
    "cell_count": 42,
    "timestamp": "2026-07-15T21:00:00+00:00"
  }
}
```

#### Error Responses

| Status | Description                                                                    |
|--------|--------------------------------------------------------------------------------|
| 400    | Malformed bounding box. Must be four comma-separated decimal values.           |
| 403    | Tier does not have access to this endpoint. Upgrade to Pro or Enterprise.      |
| 422    | Bounding box is outside BC boundaries or exceeds maximum area for your tier.   |
| 429    | Rate limit exceeded.                                                           |
| 503    | Service temporarily unavailable.                                               |

---

### GET /v1/risk/heatmap

**Visual risk heatmap.** Returns a rendered PNG image of fire risk for the specified bounding box. Useful for embedding risk maps in dashboards or GIS applications.

**Tier:** Enterprise

#### Query Parameters

| Parameter  | Type   | Required | Default | Description                                                                                              |
|------------|--------|----------|---------|----------------------------------------------------------------------------------------------------------|
| `bbox`     | string | Yes      | --      | Bounding box as `south,west,north,east` in decimal degrees. Must fall within BC boundaries.              |
| `width`    | int    | No       | 256     | Image width in pixels. Range: 64 to 2048.                                                                |
| `height`   | int    | No       | 256     | Image height in pixels. Range: 64 to 2048.                                                               |
| `colormap` | string | No       | `risk`  | Color map. `risk` uses the danger level palette; `grayscale` uses linear grayscale.                       |

#### Response 200 -- Success

Returns `image/png` with a color-mapped risk overlay. The color scale follows the standard danger level palette (see [Danger Levels](#danger-levels)).

Response headers include:

```
Content-Type: image/png
X-Bbox: 49.0,-123.5,50.0,-122.0
X-Timestamp: 2026-07-15T21:00:00+00:00
```

#### Error Responses

| Status | Description                                                               |
|--------|---------------------------------------------------------------------------|
| 400    | Malformed bounding box or invalid format parameter.                       |
| 403    | Tier does not have access to this endpoint.                               |
| 422    | Bounding box is outside BC boundaries.                                    |
| 429    | Rate limit exceeded.                                                      |
| 503    | Service temporarily unavailable.                                          |

---

### GET /v1/risk/zones

**Zone-level risk summary.** Returns aggregate fire risk information for all BC biogeoclimatic (BEC) zones. Each zone includes its average and maximum risk score, danger level, cell count, and number of high-risk cells.

**Tier:** Free, Pro, Enterprise

#### Query Parameters

None.

#### Response 200 -- Success

```json
{
  "zones": [
    {
      "bec_zone": "IDF",
      "avg_risk_score": 0.542,
      "max_risk_score": 0.891,
      "level": "HIGH",
      "cell_count": 487,
      "high_risk_cells": 15
    },
    {
      "bec_zone": "BWBS",
      "avg_risk_score": 0.234,
      "max_risk_score": 0.612,
      "level": "MODERATE",
      "cell_count": 1203,
      "high_risk_cells": 3
    }
  ],
  "timestamp": "2026-07-15T21:00:00+00:00"
}
```

---

### GET /v1/fwi/{lat}/{lon}

**Raw FWI components.** Returns the Canadian Forest Fire Weather Index System components for the specified location without the INFERNIS risk model overlay. Useful when you need the underlying fire weather data independent of the ML predictions.

**Tier:** Free, Pro, Enterprise

#### Path Parameters

| Parameter | Type  | Required | Description                                      |
|-----------|-------|----------|--------------------------------------------------|
| `lat`     | float | Yes      | Latitude in decimal degrees. Must be within BC.  |
| `lon`     | float | Yes      | Longitude in decimal degrees. Must be within BC. |

#### Response 200 -- Success

```json
{
  "location": {
    "lat": 49.25,
    "lon": -121.77
  },
  "grid_cell_id": "BC-5K-004921",
  "timestamp": "2026-07-15T21:00:00+00:00",
  "fwi": {
    "ffmc": 91.2,
    "dmc": 68.4,
    "dc": 412.0,
    "isi": 12.8,
    "bui": 89.1,
    "fwi": 34.6
  }
}
```

---

### GET /v1/conditions/{lat}/{lon}

**Current conditions.** Returns the latest weather observations and environmental data for the specified location. This includes the inputs used by the INFERNIS prediction models.

**Tier:** Free, Pro, Enterprise

#### Path Parameters

| Parameter | Type  | Required | Description                                      |
|-----------|-------|----------|--------------------------------------------------|
| `lat`     | float | Yes      | Latitude in decimal degrees. Must be within BC.  |
| `lon`     | float | Yes      | Longitude in decimal degrees. Must be within BC. |

#### Response 200 -- Success

```json
{
  "location": {
    "lat": 49.25,
    "lon": -121.77
  },
  "grid_cell_id": "BC-5K-004921",
  "timestamp": "2026-07-15T21:00:00+00:00",
  "conditions": {
    "temperature_c": 32.1,
    "rh_pct": 18.0,
    "wind_kmh": 24.5,
    "precip_24h_mm": 0.0,
    "soil_moisture": 0.12,
    "ndvi": 0.45,
    "snow_cover": false
  }
}
```

---

### GET /v1/forecast/{lat}/{lon}

**Multi-day fire risk forecast.** Returns up to 10 days of forecast fire risk trajectories for the nearest grid cell. Days 1--2 use high-resolution HRDPS weather forecasts (2.5km); days 3--10 use GDPS global forecasts (15km). FWI moisture codes are rolled forward day-by-day using forecast weather. A confidence decay factor (0.95 per lead day) attenuates predictions at longer lead times to reflect increasing forecast uncertainty.

**Tier:** Free, Pro, Enterprise

#### Path Parameters

| Parameter | Type  | Required | Description                                      |
|-----------|-------|----------|--------------------------------------------------|
| `lat`     | float | Yes      | Latitude in decimal degrees. Must be within BC.  |
| `lon`     | float | Yes      | Longitude in decimal degrees. Must be within BC. |

#### Query Parameters

| Parameter | Type | Required | Default | Description                                                |
|-----------|------|----------|---------|------------------------------------------------------------|
| `days`    | int  | No       | 10      | Number of forecast days to return. Range: 1 to 10.         |

#### Response 200 -- Success

```json
{
  "latitude": 49.25,
  "longitude": -121.77,
  "cell_id": "BC-5K-004921",
  "base_date": "2026-07-15",
  "forecast": [
    {
      "valid_date": "2026-07-16",
      "lead_day": 1,
      "risk_score": 0.68,
      "danger_level": 4,
      "danger_label": "VERY_HIGH",
      "confidence": 0.95,
      "fwi": {
        "ffmc": 90.5,
        "dmc": 70.2,
        "dc": 418.0,
        "isi": 11.9,
        "bui": 91.3,
        "fwi": 33.1
      },
      "data_source": "HRDPS"
    },
    {
      "valid_date": "2026-07-17",
      "lead_day": 2,
      "risk_score": 0.71,
      "danger_level": 4,
      "danger_label": "VERY_HIGH",
      "confidence": 0.90,
      "fwi": { "ffmc": 91.0, "dmc": 72.1, "dc": 422.0, "isi": 12.3, "bui": 93.0, "fwi": 34.2 },
      "data_source": "HRDPS"
    },
    {
      "valid_date": "2026-07-18",
      "lead_day": 3,
      "risk_score": 0.62,
      "danger_level": 4,
      "danger_label": "VERY_HIGH",
      "confidence": 0.86,
      "fwi": { "ffmc": 89.8, "dmc": 73.5, "dc": 426.0, "isi": 11.2, "bui": 94.1, "fwi": 32.0 },
      "data_source": "GDPS"
    }
  ],
  "generated_at": "2026-07-15T21:00:00+00:00"
}
```

---

### GET /v1/history/{lat}/{lon}

**Historical fire events.** Returns a list of past fire events near the specified location, drawn from the BC Wildfire Service historical database and satellite-detected hotspot archives.

**Tier:** Free, Pro, Enterprise

#### Path Parameters

| Parameter | Type  | Required | Description                                      |
|-----------|-------|----------|--------------------------------------------------|
| `lat`     | float | Yes      | Latitude in decimal degrees. Must be within BC.  |
| `lon`     | float | Yes      | Longitude in decimal degrees. Must be within BC. |

#### Query Parameters

| Parameter | Type | Required | Default | Description                                                |
|-----------|------|----------|---------|------------------------------------------------------------|
| `years`   | int  | No       | 5       | Number of years to look back from the current date. Range: 1 to 50. |
| `radius_km` | float | No    | 25      | Search radius in kilometers from the specified point. Range: 1 to 100. |

#### Response 200 -- Success

```json
{
  "location": {
    "lat": 49.25,
    "lon": -121.77
  },
  "search_radius_km": 25.0,
  "years_back": 5,
  "fires": [
    {
      "fire_id": "K50837",
      "fire_name": "Lytton Creek",
      "year": 2021,
      "start_date": "2021-06-30",
      "end_date": "2021-10-15",
      "cause": "LIGHTNING",
      "size_ha": 83816.0,
      "distance_km": 12.4,
      "lat": 50.23,
      "lon": -121.58,
      "source": "CNFDB"
    }
  ],
  "total_fires": 7
}
```

---

### GET /v1/status

**System health.** Returns the current operational status of the INFERNIS API and its underlying data pipelines. Use this endpoint for monitoring and integration health checks.

**Tier:** Free, Pro, Enterprise (no rate limit applied)

#### Query Parameters

None.

#### Response 200 -- Success

```json
{
  "status": "operational",
  "version": "0.1.0",
  "last_pipeline_run": "2026-07-15T21:00:00+00:00",
  "model_version": "fire_core_v1",
  "grid_cells": 84535,
  "pipeline_healthy": true
}
```

---

### GET /v1/coverage

**Coverage metadata.** Returns the BC boundary polygon and grid metadata, including total cell count, resolution, and coordinate reference system information. Useful for initializing map views or validating coordinate inputs before making risk queries.

**Tier:** Free, Pro, Enterprise

#### Query Parameters

None.

#### Response 200 -- Success

```json
{
  "province": "British Columbia",
  "crs": "EPSG:4326",
  "grid": {
    "resolution_km": 5.0,
    "total_cells": 84535,
    "lat_range": [48.22, 60.00],
    "lon_range": [-139.06, -114.03]
  },
  "bec_zones_count": 14,
  "fuel_types_count": 16
}
```

---

## Field Descriptions

### Risk Response Fields

| Field                | Type    | Description                                                                                                                                                   |
|----------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `location.lat`       | float   | Latitude of the query point in decimal degrees.                                                                                                               |
| `location.lon`       | float   | Longitude of the query point in decimal degrees.                                                                                                              |
| `grid_cell_id`       | string  | Unique identifier for the 5km grid cell. Format: `BC-{resolution}-{cell_number}`, e.g., `BC-5K-004921`.                                                      |
| `timestamp`          | string  | ISO 8601 timestamp indicating when this prediction was generated. Includes Pacific Time offset.                                                               |
| `risk.score`         | float   | Composite fire risk score from 0.0 (no risk) to 1.0 (maximum risk). This is the weighted output of both ML models.                                           |
| `risk.level`         | string  | Danger level derived from the risk score. One of: `VERY_LOW`, `LOW`, `MODERATE`, `HIGH`, `VERY_HIGH`, `EXTREME`.                                              |
| `risk.color`         | string  | Hex color code associated with the danger level, suitable for map rendering.                                                                                  |
| `fwi.ffmc`           | float   | Fine Fuel Moisture Code. See [FWI Components](#fwi-components).                                                                                               |
| `fwi.dmc`            | float   | Duff Moisture Code. See [FWI Components](#fwi-components).                                                                                                    |
| `fwi.dc`             | float   | Drought Code. See [FWI Components](#fwi-components).                                                                                                          |
| `fwi.isi`            | float   | Initial Spread Index. See [FWI Components](#fwi-components).                                                                                                  |
| `fwi.bui`            | float   | Buildup Index. See [FWI Components](#fwi-components).                                                                                                         |
| `fwi.fwi`            | float   | Fire Weather Index (composite). See [FWI Components](#fwi-components).                                                                                        |
| `conditions.temperature_c` | float | Air temperature in degrees Celsius at the nearest weather station or interpolated grid point.                                                            |
| `conditions.rh_pct`  | float   | Relative humidity as a percentage (0--100).                                                                                                                   |
| `conditions.wind_kmh` | float  | Wind speed in kilometers per hour at 10-meter height.                                                                                                         |
| `conditions.precip_24h_mm` | float | Accumulated precipitation over the past 24 hours in millimeters.                                                                                         |
| `conditions.soil_moisture` | float | Volumetric soil moisture content (0.0--1.0). Derived from remote sensing and station interpolation.                                                      |
| `conditions.ndvi`    | float   | Normalized Difference Vegetation Index (-1.0 to 1.0). Indicates vegetation greenness and fuel moisture. Values below 0.3 in forested areas suggest dry, fire-prone vegetation. |
| `conditions.snow_cover` | boolean | Whether snow cover is present at the location. When `true`, fire risk is typically negligible.                                                             |
| `context.bec_zone`   | string  | Biogeoclimatic Ecosystem Classification zone code. See [BEC Zone Codes](#bec-zone-codes).                                                                     |
| `context.fuel_type`  | string  | Canadian Forest Fire Behaviour Prediction System fuel type code. See [Fuel Type Codes](#fuel-type-codes).                                                     |
| `context.elevation_m` | int    | Elevation above sea level in meters.                                                                                                                          |
| `forecast_horizon`   | string  | The time window the prediction covers. Currently `24h` (next 24 hours from the timestamp).                                                                    |
| `next_update`        | string  | ISO 8601 timestamp of the next scheduled prediction update.                                                                                                   |

---

### FWI Components

The Canadian Forest Fire Weather Index (FWI) System is a set of six components that rate fire danger based on weather observations. INFERNIS computes FWI values from ERA5 reanalysis weather data, interpolated to the prediction grid.

#### Moisture Codes (Fuel Moisture Tracking)

| Code | Name                    | Range    | Description                                                                                                                                                   |
|------|-------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FFMC | Fine Fuel Moisture Code | 0--101   | Tracks moisture content of surface litter and fine fuels (leaves, needles, grass). Responds quickly to weather changes. Values above 85 indicate dry fine fuels; values above 90 indicate very dry conditions where ignition is likely. |
| DMC  | Duff Moisture Code      | 0--300+  | Tracks moisture in loosely compacted organic material (duff layers 5--10 cm deep). Responds more slowly than FFMC; a rain event may lower FFMC immediately but take days to affect DMC. Values above 40 are considered high. |
| DC   | Drought Code            | 0--800+  | Tracks deep, compact organic layer moisture and seasonal drought. Very slow to respond to rainfall. Values above 300 indicate significant drought; values above 500 indicate severe drought with deep-seated fire potential. |

#### Behaviour Indices (Fire Behaviour Rating)

| Index | Name                  | Range    | Description                                                                                                                                                   |
|-------|-----------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISI   | Initial Spread Index  | 0--70+   | Combines FFMC and wind speed to rate the expected rate of fire spread. Higher values indicate faster-spreading fires. Values above 10 are considered high.    |
| BUI   | Buildup Index         | 0--300+  | Combines DMC and DC to represent the total fuel available for combustion. Higher values mean more fuel can burn, producing more intense fires. Values above 60 are considered high. |
| FWI   | Fire Weather Index    | 0--100+  | Combines ISI and BUI into a single fire intensity rating. This is the most commonly referenced single value from the FWI System. Values above 20 are high; above 30 are very high; above 40 are extreme. |

---

### Danger Levels

Risk scores from the INFERNIS models are mapped to six danger levels.

| Level       | Score Range     | Color Code | Hex       | Description                                                                                          |
|-------------|-----------------|------------|-----------|------------------------------------------------------------------------------------------------------|
| VERY_LOW    | 0.00 -- 0.05    | Green      | `#22C55E` | Negligible risk. Wet or snow-covered conditions.                                                     |
| LOW         | 0.05 -- 0.15    | Blue       | `#3B82F6` | Minor risk. Fires unlikely under current conditions.                                                 |
| MODERATE    | 0.15 -- 0.35    | Yellow     | `#EAB308` | Elevated. Fires possible with an ignition source.                                                    |
| HIGH        | 0.35 -- 0.60    | Orange     | `#F97316` | Significant. Fires likely to spread if ignited.                                                      |
| VERY_HIGH   | 0.60 -- 0.80    | Red        | `#EF4444` | Severe. Aggressive fire behavior expected.                                                           |
| EXTREME     | 0.80 -- 1.00    | Dark Red   | `#1A0000` | Critical. Explosive fire growth potential.                                                           |

---

### BEC Zone Codes

British Columbia uses the Biogeoclimatic Ecosystem Classification (BEC) system to classify ecological zones. The `bec_zone` field in the API response uses the following standard abbreviations.

| Code  | Full Name                                  | Typical Elevation (m) | Fire Relevance                                                    |
|-------|--------------------------------------------|-----------------------|-------------------------------------------------------------------|
| AT    | Alpine Tundra                              | >1800                 | Low fire risk due to sparse vegetation and cool temperatures.     |
| BG    | Bunchgrass                                 | 350--900              | Grass fires can spread quickly in dry conditions.                 |
| BWBS  | Boreal White and Black Spruce              | 400--1200             | Significant fire zone; spruce is highly flammable.                |
| CDF   | Coastal Douglas-fir                        | 0--450                | Moderate risk; drought conditions in summer increase danger.      |
| CWH   | Coastal Western Hemlock                    | 0--1050               | Generally wet, but extreme drought years create high risk.        |
| ESSF  | Engelmann Spruce -- Subalpine Fir          | 1200--2000            | Subalpine forests prone to stand-replacing fires.                 |
| ICH   | Interior Cedar -- Hemlock                  | 400--1500             | Mixed fire regime; moderate to high risk in dry summers.          |
| IDF   | Interior Douglas-fir                       | 300--1450             | Frequent fire zone; historically fire-maintained ecosystems.      |
| MH    | Mountain Hemlock                           | 900--1800             | Moderate risk; snow limits fire season duration.                  |
| MS    | Montane Spruce                             | 1100--1700            | Moderate risk; dry years produce significant fire activity.       |
| PP    | Ponderosa Pine                             | 300--900              | High fire frequency zone; adapted to frequent low-intensity fire. |
| SBPS  | Sub-Boreal Pine -- Spruce                  | 900--1400             | Significant fire zone; large fires common in dry years.           |
| SBS   | Sub-Boreal Spruce                          | 500--1300             | High fire activity; large fires common in the BC interior.        |
| SWB   | Spruce -- Willow -- Birch                  | 400--1300             | Northern boreal zone; fire is the dominant disturbance agent.     |

---

### Fuel Type Codes

Fuel types follow the Canadian Forest Fire Behaviour Prediction (FBP) System classification. The `fuel_type` field indicates the dominant fuel type in the grid cell.

#### Coniferous (C) Types

| Code | Name                  | Description                                                                                              |
|------|-----------------------|----------------------------------------------------------------------------------------------------------|
| C1   | Spruce -- Lichen Woodland | Open spruce stands with lichen ground cover. Very high spread rates and crowning potential.           |
| C2   | Boreal Spruce         | Dense boreal spruce with feather moss and lichen. High crowning potential and intense fire behaviour.    |
| C3   | Mature Jack/Lodgepole Pine | Fully stocked pine stands with closed canopy. Moderate to high fire intensity.                       |
| C4   | Immature Jack/Lodgepole Pine | Dense immature pine with high crown closure. Very high fire intensity and crown fire potential.    |
| C5   | Red/White Pine        | Mature red or white pine stands. Moderate fire intensity; less prone to crowning than C3/C4.             |
| C6   | Conifer Plantation     | Planted conifer stands, typically young and dense. High fire intensity if spacing is tight.              |
| C7   | Ponderosa Pine / Douglas-fir | Open stands of ponderosa pine or Douglas-fir. Surface fire dominant; low to moderate crowning.    |

#### Deciduous (D) Types

| Code | Name                  | Description                                                                                              |
|------|-----------------------|----------------------------------------------------------------------------------------------------------|
| D1   | Leafless Aspen        | Aspen stands without foliage (spring/fall). Higher fire risk than leafed-out stands.                     |
| D2   | Green Aspen (with BUI >= 80) | Aspen stands with full foliage. Fire only occurs under extreme drought (high BUI).                |

#### Mixedwood (M) Types

| Code | Name                               | Description                                                                              |
|------|-------------------------------------|------------------------------------------------------------------------------------------|
| M1   | Boreal Mixedwood -- Leafless        | Mixed boreal conifer and deciduous, no foliage. Fire behaviour intermediate.             |
| M2   | Boreal Mixedwood -- Green           | Mixed boreal conifer and deciduous, with foliage. Reduced spread compared to M1.         |
| M3   | Dead Balsam Fir Mixedwood -- Leafless | Stands with dead balsam fir component. Highly flammable due to standing dead trees.    |
| M4   | Dead Balsam Fir Mixedwood -- Green  | Same as M3 but with green deciduous foliage. Somewhat reduced fire behaviour.            |

#### Open (O) Types

| Code | Name              | Description                                                                                          |
|------|-------------------|------------------------------------------------------------------------------------------------------|
| O1a  | Matted Grass      | Continuous matted grass fuel. Very high spread rate under wind; low intensity.                        |
| O1b  | Standing Grass    | Standing dead grass. Extremely high spread rate; the fastest-spreading fuel type in the FBP System.  |

#### Slash (S) Types

| Code | Name                  | Description                                                                                          |
|------|-----------------------|------------------------------------------------------------------------------------------------------|
| S1   | Jack/Lodgepole Pine Slash | Logging slash from pine harvest. High fire intensity with heavy fuel loads.                       |
| S2   | White Spruce / Balsam Slash | Logging slash from spruce/balsam harvest. Very high intensity and difficult suppression.         |
| S3   | Coastal Cedar / Hemlock / Douglas-fir Slash | Coastal logging slash. Extremely high fuel loads and intense fire behaviour.      |

---

## Rate Limiting

All API requests (except `/v1/status`) are subject to rate limiting based on your tier.

### Rate Limit Headers

Every response includes the following headers to help you manage request pacing.

| Header                  | Type   | Description                                                                                     |
|-------------------------|--------|-------------------------------------------------------------------------------------------------|
| `X-RateLimit-Limit`     | int    | Maximum number of requests allowed per day for your tier.                                       |
| `X-RateLimit-Remaining` | int    | Number of requests remaining in the current daily window.                                       |
| `X-RateLimit-Reset`     | string | Reset time indicator (currently returns `"midnight PST"`).                                      |

### 429 Too Many Requests

When the rate limit is exceeded, the API returns a `429` status code with the following body:

```json
{
  "detail": "Rate limit exceeded. Daily limit of 50 requests reached. Resets at 2026-07-16T00:00:00-07:00.",
  "retry_after_seconds": 3600
}
```

The response also includes a `Retry-After` header with the number of seconds until additional capacity is available.

**Best practices:**

- Cache responses locally when the data has not changed (check `next_update` field).
- Use the `/v1/status` endpoint for health checks, as it is not rate-limited.
- For bulk operations, prefer `/v1/risk/grid` over multiple point queries, as a single grid request counts as one API call.

---

## Error Responses

All errors follow a consistent JSON format.

### Standard Error Body

```json
{
  "detail": "Error message here"
}
```

### Error Codes

| Status | Meaning                  | Common Causes                                                                                              |
|--------|--------------------------|------------------------------------------------------------------------------------------------------------|
| 400    | Bad Request              | Malformed parameters, missing required fields, non-numeric coordinate values.                              |
| 401    | Unauthorized             | Missing or invalid `X-API-Key` header.                                                                     |
| 403    | Forbidden                | Valid key but insufficient tier for the requested endpoint (e.g., Free tier accessing `/v1/risk/grid`).    |
| 404    | Not Found                | Coordinates outside BC coverage area, or endpoint does not exist.                                          |
| 422    | Unprocessable Entity     | Parameters are syntactically valid but semantically incorrect (e.g., latitude of 75.0, which is outside BC). |
| 429    | Too Many Requests        | Rate limit exceeded. See [Rate Limiting](#rate-limiting).                                                  |
| 500    | Internal Server Error    | Unexpected server failure. Contact support if persistent.                                                  |
| 503    | Service Unavailable      | Prediction pipeline is updating or undergoing maintenance. Typically resolves within minutes.              |

### Error Examples

**Invalid coordinates (400):**

```json
{
  "detail": "Invalid latitude value: 'abc' is not a valid number."
}
```

**Outside BC boundaries (422):**

```json
{
  "detail": "Coordinates (45.00, -121.00) are outside the British Columbia coverage area. Latitude must be between 48.0 and 60.0; longitude must be between -140.0 and -114.0."
}
```

**Unauthorized (401):**

```json
{
  "detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."
}
```

**Tier restriction (403):**

```json
{
  "detail": "Your current tier (Free) does not have access to the /v1/risk/grid endpoint. Upgrade to Pro or Enterprise at https://api.infernis.ca/static/dashboard.html."
}
```

---

## Code Examples

### Python (requests)

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.infernis.ca/v1"

headers = {
    "X-API-Key": API_KEY,
}

# Point risk query
response = requests.get(
    f"{BASE_URL}/risk/49.25/-121.77",
    headers=headers,
)
response.raise_for_status()
data = response.json()

print(f"Risk Score: {data['risk']['score']}")
print(f"Risk Level: {data['risk']['level']}")
print(f"FWI: {data['fwi']['fwi']}")
print(f"Temperature: {data['conditions']['temperature_c']}C")
print(f"Fuel Type: {data['context']['fuel_type']}")

# Check rate limit status
remaining = response.headers.get("X-RateLimit-Remaining")
print(f"Requests remaining today: {remaining}")
```

### Python -- Grid Query (Pro tier)

```python
import requests

API_KEY = "your_pro_api_key_here"
BASE_URL = "https://api.infernis.ca/v1"

headers = {
    "X-API-Key": API_KEY,
}

# Area risk query with bounding box (south,west,north,east)
params = {
    "bbox": "49.0,-122.5,50.0,-121.0",
}

response = requests.get(
    f"{BASE_URL}/risk/grid",
    headers=headers,
    params=params,
)
response.raise_for_status()
geojson = response.json()

print(f"Cells returned: {geojson['metadata']['cell_count']}")

for feature in geojson["features"]:
    props = feature["properties"]
    if props["level"] in ("VERY_HIGH", "EXTREME"):
        print(f"  {props['cell_id']}: {props['score']:.2f} ({props['level']})")
```

### curl

```bash
# Point risk query
curl -H "X-API-Key: your_api_key_here" \
  "https://api.infernis.ca/v1/risk/49.25/-121.77"

# FWI components
curl -H "X-API-Key: your_api_key_here" \
  "https://api.infernis.ca/v1/fwi/49.25/-121.77"

# Current conditions
curl -H "X-API-Key: your_api_key_here" \
  "https://api.infernis.ca/v1/conditions/49.25/-121.77"

# Grid query (Pro tier) — bbox is south,west,north,east
curl -H "X-API-Key: your_pro_api_key_here" \
  "https://api.infernis.ca/v1/risk/grid?bbox=49.0,-122.5,50.0,-121.0"

# Historical fires within 50km over the past 10 years
curl -H "X-API-Key: your_api_key_here" \
  "https://api.infernis.ca/v1/history/49.25/-121.77?years=10&radius_km=50"

# System status (no API key required for basic check)
curl "https://api.infernis.ca/v1/status"

# Download heatmap as PNG (Enterprise tier)
curl -H "X-API-Key: your_enterprise_api_key_here" \
  -o heatmap.png \
  "https://api.infernis.ca/v1/risk/heatmap?bbox=49.0,-122.5,50.0,-121.0"
```

### JavaScript (fetch)

```javascript
const API_KEY = "your_api_key_here";
const BASE_URL = "https://api.infernis.ca/v1";

async function getFireRisk(lat, lon) {
  const response = await fetch(`${BASE_URL}/risk/${lat}/${lon}`, {
    headers: {
      "X-API-Key": API_KEY,
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`API error ${response.status}: ${error.detail}`);
  }

  const data = await response.json();

  console.log(`Risk Score: ${data.risk.score}`);
  console.log(`Risk Level: ${data.risk.level}`);
  console.log(`FWI: ${data.fwi.fwi}`);
  console.log(`BEC Zone: ${data.context.bec_zone}`);
  console.log(`Fuel Type: ${data.context.fuel_type}`);

  // Check rate limit
  const remaining = response.headers.get("X-RateLimit-Remaining");
  console.log(`Requests remaining: ${remaining}`);

  return data;
}

async function getGridRisk(bbox) {
  const params = new URLSearchParams({
    bbox: bbox,
  });

  const response = await fetch(`${BASE_URL}/risk/grid?${params}`, {
    headers: {
      "X-API-Key": API_KEY,
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`API error ${response.status}: ${error.detail}`);
  }

  return response.json();
}

// Usage
getFireRisk(49.25, -121.77)
  .then((data) => {
    if (data.risk.score >= 0.6) {
      console.warn(`WARNING: ${data.risk.level} fire risk at this location.`);
    }
  })
  .catch(console.error);
```

---

## Data Update Schedule

INFERNIS integrates multiple data sources, each with its own update cadence. The following table summarizes when data becomes available.

| Data Source                   | Update Frequency              | Typical Availability         | Notes                                                               |
|-------------------------------|-------------------------------|------------------------------|---------------------------------------------------------------------|
| Risk predictions              | Daily                         | ~14:00 PT                    | Composite ML model output. Depends on all upstream data being current. |
| Weather observations          | Hourly                        | ~15 minutes past the hour    | From BC Wildfire Service weather stations and Environment Canada.   |
| FWI calculations              | Daily                         | ~13:50 PT                    | Computed after the noon weather observation, per CWFIS standard.    |
| NDVI (vegetation index)       | Every 16 days                 | 2--3 days after satellite pass | MODIS 16-day composite cycle. Updated as new composites are processed. |
| Soil moisture                 | Daily                         | ~06:00 PT                    | Derived from remote sensing products and station interpolation.     |
| Topography / DEM              | Annually                      | January                      | Canadian Digital Elevation Model. Updated if significant revisions occur. |
| Fuel type classification      | Annually                      | March                        | BC Wildfire Service fuel type maps. Updated to reflect logging, growth, and disturbance. |
| BEC zone boundaries           | As needed                     | Varies                       | BC Ministry of Forests classification. Rarely changes.              |
| Historical fire database      | Annually                      | After fire season (November) | BC Wildfire Service historical records. Updated after each season closes. |

### Pipeline Dependency Chain

The daily prediction cycle follows this order:

1. **Weather ingest** (~12:30 PT) -- Station observations are collected and quality-checked.
2. **FWI calculation** (~13:00 PT) -- FWI components are computed from noon weather data.
3. **Condition assembly** (~13:30 PT) -- All condition layers (weather, soil, NDVI) are assembled into the grid.
4. **Prediction run** (~13:45 PT) -- Both ML models (FireCore and HeatmapEngine) generate scores.
5. **API update** (~14:00 PT) -- New predictions are published and the API begins serving updated data.

If any upstream pipeline fails, the API will continue to serve the most recent successful prediction and the `/v1/status` endpoint will reflect the degraded pipeline status.

---

## Dashboard API

The dashboard endpoints manage user accounts and API key provisioning. They use **Firebase Authentication** (not API keys). Include a valid Firebase ID token in the `Authorization` header:

```
Authorization: Bearer <firebase-id-token>
```

### POST /api/dashboard/register

**Idempotent registration.** On first call, creates a user account, generates a free-tier API key, and returns the plaintext key. On subsequent calls, returns the user profile without the key.

**First-time response (200):**
```json
{
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "tier": "free",
  "api_key": "a3f8c9e1d2b4...64 hex characters"
}
```

**Returning user response (200):**
```json
{
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "tier": "free",
  "key_preview": "a3f8****...****d2e1",
  "daily_limit": 50,
  "billing_cycle_start": "2026-02-15"
}
```

### GET /api/dashboard/profile

Returns the authenticated user's profile with a masked key preview.

**Response (200):**
```json
{
  "email": "user@example.com",
  "display_name": "Jane Doe",
  "tier": "free",
  "key_preview": "a3f8****...****d2e1",
  "daily_limit": 50,
  "billing_cycle_start": "2026-02-15"
}
```

**Errors:** 404 if the user has not registered.

### GET /api/dashboard/usage

Returns current billing cycle usage.

**Response (200):**
```json
{
  "requests_today": 42,
  "daily_limit": 50,
  "tier": "free",
  "billing_cycle_start": "2026-02-15",
  "billing_cycle_end": "2026-03-17",
  "days_remaining": 30
}
```

### POST /api/dashboard/key/regenerate

Deactivates the current API key and generates a new one. The new plaintext key is returned once.

**Response (200):**
```json
{
  "api_key": "b7d1e4f2a8c6...64 hex characters",
  "message": "New key generated. Your old key has been deactivated."
}
```

**Errors:** 404 if the user has not registered.

---

## Versioning

The API is versioned via the URL path (`/v1/`). Breaking changes will only be introduced in new major versions (e.g., `/v2/`). Non-breaking additions (new fields in responses, new optional query parameters) may be added to the current version without notice.

When a new major version is released, the previous version will remain available for at least 12 months with a deprecation notice in the response headers:

```
X-API-Deprecated: true
X-API-Sunset: 2028-01-01
```

---

## Support

| Channel                          | Availability           | Tier                  |
|----------------------------------|------------------------|-----------------------|
| Dashboard: api.infernis.ca/static/index.html | Always       | All                   |
| Email: hello@argonbi.com          | 48-hour SLA            | Pro, Enterprise       |
| Dedicated support                | 4-hour SLA             | Enterprise            |

For bug reports, feature requests, or tier upgrades, email hello@argonbi.com.
