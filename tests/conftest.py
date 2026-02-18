"""Shared test fixtures."""

import os

# Set debug mode BEFORE any infernis imports so Settings picks it up
os.environ["INFERNIS_DEBUG"] = "true"

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_weather():
    """Typical summer fire-weather conditions in BC interior."""
    return {
        "temp": 28.0,
        "rh": 25.0,
        "wind": 15.0,
        "precip": 0.0,
        "month": 7,
    }


@pytest.fixture
def sample_weather_arrays():
    """Weather arrays for 3 grid cells."""
    return {
        "temperature_c": np.array([25.0, 30.0, 20.0]),
        "rh_pct": np.array([30.0, 20.0, 50.0]),
        "wind_kmh": np.array([10.0, 20.0, 5.0]),
        "wind_dir_deg": np.array([180.0, 270.0, 90.0]),
        "precip_24h_mm": np.array([0.0, 0.0, 2.0]),
        "soil_moisture_1": np.array([0.2, 0.15, 0.35]),
        "soil_moisture_2": np.array([0.25, 0.2, 0.35]),
        "soil_moisture_3": np.array([0.28, 0.22, 0.38]),
        "soil_moisture_4": np.array([0.30, 0.25, 0.40]),
        "evapotrans_mm": np.array([3.0, 4.0, 2.0]),
    }


@pytest.fixture
def sample_grid_df():
    """Small 3-cell grid DataFrame for testing."""
    return pd.DataFrame(
        {
            "cell_id": ["BC-5K-000000", "BC-5K-000001", "BC-5K-000002"],
            "lat": [50.0, 51.0, 52.0],
            "lon": [-122.0, -123.0, -124.0],
            "elevation_m": [500.0, 1200.0, 800.0],
            "slope_deg": [10.0, 25.0, 5.0],
            "aspect_deg": [180.0, 90.0, 270.0],
            "hillshade": [128.0, 100.0, 150.0],
            "distance_to_road_km": [5.0, 50.0, 20.0],
            "bec_zone": ["IDF", "ESSF", "SBS"],
            "fuel_type": ["C3", "C5", "D1"],
        }
    )
