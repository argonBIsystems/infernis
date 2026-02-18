"""Tests for the Risk Fuser module."""

import numpy as np

from infernis.training.risk_fuser import BEC_ZONES, RiskFuser


class TestDefaultWeights:
    def test_all_zones_initialized(self):
        fuser = RiskFuser()
        assert len(fuser.zone_params) == len(BEC_ZONES)

    def test_weights_sum_to_one(self):
        fuser = RiskFuser()
        for zone, params in fuser.zone_params.items():
            total = params["xgb_weight"] + params["cnn_weight"]
            assert abs(total - 1.0) < 1e-6, f"Zone {zone} weights don't sum to 1"


class TestFusion:
    def test_fuse_basic(self):
        fuser = RiskFuser()
        xgb = np.array([0.5, 0.8, 0.1])
        cnn = np.array([0.6, 0.7, 0.2])
        zones = np.array(["IDF", "ESSF", "CWH"])

        result = fuser.fuse(xgb, cnn, zones)
        assert result.shape == (3,)
        assert all(0 <= r <= 1 for r in result)

    def test_fuse_xgb_only(self):
        fuser = RiskFuser()
        xgb = np.array([0.3, 0.7])
        zones = np.array(["IDF", "ESSF"])

        result = fuser.fuse_xgb_only(xgb, zones)
        assert result.shape == (2,)
        # With default zero bias, should be close to original
        assert abs(result[0] - 0.3) < 0.01
        assert abs(result[1] - 0.7) < 0.01

    def test_output_bounded(self):
        fuser = RiskFuser()
        xgb = np.array([0.0, 1.0, 0.5])
        cnn = np.array([0.0, 1.0, 0.5])
        zones = np.array(["IDF", "IDF", "IDF"])

        result = fuser.fuse(xgb, cnn, zones)
        assert all(0 <= r <= 1 for r in result)

    def test_unknown_zone_uses_defaults(self):
        fuser = RiskFuser()
        xgb = np.array([0.5])
        cnn = np.array([0.5])
        zones = np.array(["UNKNOWN_ZONE"])

        result = fuser.fuse(xgb, cnn, zones)
        assert 0 <= result[0] <= 1


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        fuser = RiskFuser()
        fuser.zone_params["IDF"]["bias"] = 0.05

        path = tmp_path / "fuser_weights.json"
        fuser.save_weights(path)

        loaded = RiskFuser(weights_path=path)
        assert loaded.zone_params["IDF"]["bias"] == 0.05
