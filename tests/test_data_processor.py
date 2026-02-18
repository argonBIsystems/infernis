"""Tests for the data processing ETL pipeline."""

import numpy as np
import pytest

from infernis.pipelines.data_processor import FEATURE_NAMES, DataProcessor


class TestFeatureNames:
    def test_feature_count(self):
        # 6 FWI + 10 Weather + 3 Vegetation + 5 Topo/Infra + 2 Temporal + 2 Lightning
        assert len(FEATURE_NAMES) == 28

    def test_fwi_features(self):
        fwi = FEATURE_NAMES[:6]
        assert fwi == ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]

    def test_weather_features(self):
        weather = FEATURE_NAMES[6:16]
        assert "temperature_c" in weather
        assert "soil_moisture_1" in weather
        assert "soil_moisture_3" in weather
        assert "soil_moisture_4" in weather

    def test_vegetation_features(self):
        assert "ndvi" in FEATURE_NAMES
        assert "lai" in FEATURE_NAMES
        assert "snow_cover" in FEATURE_NAMES

    def test_temporal_features(self):
        doy_sin_idx = FEATURE_NAMES.index("doy_sin")
        doy_cos_idx = FEATURE_NAMES.index("doy_cos")
        assert doy_cos_idx == doy_sin_idx + 1

    def test_lightning_features(self):
        assert FEATURE_NAMES[-2] == "lightning_24h"
        assert FEATURE_NAMES[-1] == "lightning_72h"

    def test_infrastructure_features(self):
        assert "distance_to_road_km" in FEATURE_NAMES


class TestBuildDailyFeatures:
    def test_output_shape(self, sample_grid_df):
        from datetime import date

        processor = DataProcessor()

        n = len(sample_grid_df)
        weather = {
            "temperature_c": np.full(n, 25.0),
            "rh_pct": np.full(n, 30.0),
            "wind_kmh": np.full(n, 15.0),
            "wind_dir_deg": np.full(n, 180.0),
            "precip_24h_mm": np.zeros(n),
            "soil_moisture_1": np.full(n, 0.2),
            "soil_moisture_2": np.full(n, 0.25),
            "evapotrans_mm": np.full(n, 3.0),
        }
        fwi_day = {
            "ffmc": np.full(n, 85.0),
            "dmc": np.full(n, 6.0),
            "dc": np.full(n, 15.0),
            "isi": np.full(n, 3.0),
            "bui": np.full(n, 8.0),
            "fwi": np.full(n, 5.0),
        }
        satellite = {"ndvi": np.full(n, 0.5), "snow_cover": np.zeros(n)}
        static = {"elevation_m": np.full(n, 500.0)}

        features = processor.build_daily_features(
            target_date=date(2024, 7, 15),
            weather=weather,
            fwi_day=fwi_day,
            satellite=satellite,
            static=static,
            grid_df=sample_grid_df,
        )

        assert features.shape == (n, len(FEATURE_NAMES))
        assert features.dtype == np.float32

    def test_doy_encoding(self, sample_grid_df):
        """Day-of-year sin/cos should be cyclic."""
        from datetime import date

        processor = DataProcessor()

        n = len(sample_grid_df)
        dummy_weather = {
            k: np.zeros(n)
            for k in [
                "temperature_c",
                "rh_pct",
                "wind_kmh",
                "wind_dir_deg",
                "precip_24h_mm",
                "soil_moisture_1",
                "soil_moisture_2",
                "evapotrans_mm",
            ]
        }
        dummy_fwi = {k: np.zeros(n) for k in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]}

        # Summer solstice vs winter
        feat_summer = processor.build_daily_features(
            date(2024, 6, 21),
            dummy_weather,
            dummy_fwi,
            {"ndvi": np.zeros(n), "snow_cover": np.zeros(n)},
            {"elevation_m": np.zeros(n)},
            sample_grid_df,
        )
        feat_winter = processor.build_daily_features(
            date(2024, 12, 21),
            dummy_weather,
            dummy_fwi,
            {"ndvi": np.zeros(n), "snow_cover": np.zeros(n)},
            {"elevation_m": np.zeros(n)},
            sample_grid_df,
        )

        # doy_sin (index 20) should differ between summer and winter
        doy_sin_idx = FEATURE_NAMES.index("doy_sin")
        doy_cos_idx = FEATURE_NAMES.index("doy_cos")
        assert feat_summer[0, doy_sin_idx] != feat_winter[0, doy_sin_idx]

        # Both should be in [-1, 1]
        assert -1 <= feat_summer[0, doy_sin_idx] <= 1
        assert -1 <= feat_summer[0, doy_cos_idx] <= 1


class TestVectorizedRasterSampling:
    """Test that vectorized sample_raster_at_grid produces correct results."""

    def test_sample_known_values(self, tmp_path):
        """Create a simple raster and verify sampling at known coordinates."""
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        # Create 10x10 raster covering BC-like bbox
        height, width = 10, 10
        west, south, east, north = -130.0, 49.0, -120.0, 55.0
        transform = from_bounds(west, south, east, north, width, height)

        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        tif_path = tmp_path / "test.tif"

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        processor = DataProcessor()
        lats = np.array([54.0, 52.0, 50.0])
        lons = np.array([-128.0, -125.0, -122.0])

        values = processor.sample_raster_at_grid(tif_path, lats, lons, band=1)

        assert values.shape == (3,)
        assert not np.any(np.isnan(values))

    def test_out_of_bounds_returns_nan(self, tmp_path):
        """Points outside raster bounds should return NaN."""
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        height, width = 5, 5
        transform = from_bounds(-130, 49, -120, 55, width, height)
        data = np.ones((5, 5), dtype=np.float32) * 42.0
        tif_path = tmp_path / "bounded.tif"

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        processor = DataProcessor()
        lats = np.array([52.0, 70.0])  # second is way out of bounds
        lons = np.array([-125.0, -100.0])

        values = processor.sample_raster_at_grid(tif_path, lats, lons)
        assert values[0] == 42.0
        assert np.isnan(values[1])

    def test_nodata_handled(self, tmp_path):
        """Nodata values in raster should become NaN."""
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        height, width = 5, 5
        transform = from_bounds(-130, 49, -120, 55, width, height)
        data = np.ones((5, 5), dtype=np.float32) * 99.0
        data[2, 2] = -9999.0
        tif_path = tmp_path / "nodata.tif"

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        processor = DataProcessor()
        # Point that falls on the nodata pixel
        lats = np.array([52.0])
        lons = np.array([-125.0])
        values = processor.sample_raster_at_grid(tif_path, lats, lons)
        assert np.isnan(values[0])

    def test_missing_file_returns_nan(self, tmp_path):
        """Missing file should return all NaN."""
        processor = DataProcessor()
        lats = np.array([50.0, 52.0])
        lons = np.array([-125.0, -123.0])
        values = processor.sample_raster_at_grid(tmp_path / "nope.tif", lats, lons)
        assert values.shape == (2,)
        assert np.all(np.isnan(values))


class TestChunkedOutput:
    """Test chunked parquet output in process_training_period."""

    def test_chunk_days_parameter(self):
        """Verify the chunk_days parameter is accepted."""
        import inspect

        sig = inspect.signature(DataProcessor.process_training_period)
        assert "chunk_days" in sig.parameters
        assert sig.parameters["chunk_days"].default == 0
