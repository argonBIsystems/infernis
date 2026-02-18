"""Tests for heatmap model and API endpoint."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from infernis.training.heatmap_model import (  # noqa: E402
    ConvBlock,
    FireUNet,
    build_raster_from_grid,
)


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(12, 64)
        x = torch.randn(1, 12, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)


class TestFireUNet:
    def test_forward_shape(self):
        model = FireUNet(in_channels=12, base_filters=16)
        x = torch.randn(1, 12, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_output_range(self):
        """Output should be in [0, 1] due to sigmoid."""
        model = FireUNet(in_channels=12, base_filters=16)
        x = torch.randn(2, 12, 32, 32)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_odd_dimensions(self):
        """Should handle non-power-of-2 dimensions."""
        model = FireUNet(in_channels=12, base_filters=16)
        x = torch.randn(1, 12, 33, 45)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1


class TestBuildRaster:
    def test_basic_raster(self):
        predictions = {
            "cell1": {"score": 0.8},
            "cell2": {"score": 0.2},
        }
        grid_cells = {
            "cell1": {"lat": 55.0, "lon": -130.0},
            "cell2": {"lat": 50.0, "lon": -120.0},
        }
        raster = build_raster_from_grid(
            predictions,
            grid_cells,
            "score",
            h=10,
            w=10,
            bbox=(48.3, -139.06, 60.0, -114.03),
        )
        assert raster.shape == (10, 10)
        # At least one non-NaN value
        assert not np.all(np.isnan(raster))
