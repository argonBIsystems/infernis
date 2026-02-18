"""Tests for the training data feature builder."""

import pandas as pd

from infernis.training.feature_builder import FeatureBuilder


class TestFireToGridAssignment:
    def test_assigns_nearest_cell(self, sample_grid_df):
        builder = FeatureBuilder()
        fires = pd.DataFrame(
            {
                "lat": [50.01, 51.01],
                "lon": [-122.01, -123.01],
                "date": ["2024-07-15", "2024-08-01"],
                "size_ha": [100, 500],
                "source": ["test", "test"],
            }
        )

        result = builder.assign_fires_to_grid(fires, sample_grid_df)
        assert len(result) == 2
        assert "BC-5K-000000" in result["cell_id"].values
        assert "BC-5K-000001" in result["cell_id"].values

    def test_deduplicates_same_cell_same_day(self, sample_grid_df):
        builder = FeatureBuilder()
        fires = pd.DataFrame(
            {
                "lat": [50.0, 50.01],  # Both near cell 0
                "lon": [-122.0, -121.99],
                "date": ["2024-07-15", "2024-07-15"],  # Same day
                "size_ha": [10, 20],
                "source": ["a", "b"],
            }
        )

        result = builder.assign_fires_to_grid(fires, sample_grid_df)
        # Should be deduplicated to 1 fire for cell 0 on July 15
        cell0_fires = result[result["cell_id"] == "BC-5K-000000"]
        assert len(cell0_fires) == 1

    def test_empty_fires(self, sample_grid_df):
        builder = FeatureBuilder()
        fires = pd.DataFrame(columns=["lat", "lon", "date", "size_ha", "source"])
        result = builder.assign_fires_to_grid(fires, sample_grid_df)
        assert len(result) == 0

    def test_fire_too_far_from_grid(self, sample_grid_df):
        builder = FeatureBuilder()
        fires = pd.DataFrame(
            {
                "lat": [30.0],  # Far from BC
                "lon": [-100.0],
                "date": ["2024-07-15"],
                "size_ha": [100],
                "source": ["test"],
            }
        )

        result = builder.assign_fires_to_grid(fires, sample_grid_df)
        assert len(result) == 0


class TestNegativeSampling:
    def test_respects_exclusion(self, sample_grid_df, tmp_path):
        """Negatives should not overlap with fire cell-days."""
        builder = FeatureBuilder()

        fire_cells = pd.DataFrame(
            {
                "cell_id": ["BC-5K-000000"],
                "date": [pd.Timestamp("2024-07-15")],
                "fire": [True],
            }
        )

        # Create a tiny feature file
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        feat_df = pd.DataFrame(
            {
                "cell_id": ["BC-5K-000000", "BC-5K-000001", "BC-5K-000002"] * 30,
                "date": [f"2024-07-{d:02d}" for d in range(1, 31)] * 3,
            }
        )
        feat_df.to_parquet(features_dir / "features_2024_07.parquet", index=False)

        negatives = builder.sample_negatives(fire_cells, sample_grid_df, features_dir, ratio=5)
        assert len(negatives) > 0
        assert not negatives["fire"].any()

        # The exact fire cell-day should not be in negatives
        fire_match = negatives[
            (negatives["cell_id"] == "BC-5K-000000")
            & (negatives["date"] == pd.Timestamp("2024-07-15"))
        ]
        assert len(fire_match) == 0
