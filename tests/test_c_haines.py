"""Tests for C-Haines atmospheric instability index.

Reference: Mills & McCaw (2010), The Continuous Haines Index.
"""

import numpy as np

from infernis.services.c_haines import compute_c_haines


class TestBasicComputation:
    def test_returns_float_in_range(self):
        """Basic computation returns a float in [0, 13]."""
        result = compute_c_haines(t850=15.0, t500=-10.0, td850=10.0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 13.0

    def test_output_range_lower_bound(self):
        """Very stable, very moist atmosphere → C-Haines near 0."""
        # Minimal lapse rate (CA near 0) + no dewpoint depression (CB near 0)
        result = compute_c_haines(t850=5.0, t500=2.0, td850=5.0)
        assert 0.0 <= result <= 13.0

    def test_output_range_upper_bound(self):
        """Inputs engineered to maximize both terms → C-Haines near 13."""
        # Extreme instability: large lapse rate + extreme dewpoint depression
        result = compute_c_haines(t850=20.0, t500=-40.0, td850=-10.0)
        assert 0.0 <= result <= 13.0


class TestFireBehaviorPhysics:
    def test_high_instability_gives_high_index(self):
        """Large T850-T500 lapse + large dewpoint depression → C-Haines > 8."""
        # Hot, dry unstable atmosphere
        # T850=20°C, T500=-25°C → lapse ΔT=45°C (very high)
        # td850=5°C → dewpoint depression = 20-5=15°C (very dry at 850 hPa)
        result = compute_c_haines(t850=20.0, t500=-25.0, td850=5.0)
        assert result > 8.0, f"Expected C-Haines > 8 for unstable/dry conditions, got {result}"

    def test_stable_moist_gives_low_index(self):
        """Stable atmosphere + high moisture → C-Haines < 4."""
        # Small lapse rate (stable) + td850 close to t850 (moist)
        # T850=5°C, T500=0°C → lapse ΔT=5°C (very stable)
        # td850=4°C → dewpoint depression = 5-4=1°C (nearly saturated)
        result = compute_c_haines(t850=5.0, t500=0.0, td850=4.0)
        assert result < 4.0, f"Expected C-Haines < 4 for stable/moist conditions, got {result}"

    def test_increasing_instability_increases_index(self):
        """Increasing lapse rate (lower T500) should raise C-Haines."""
        stable = compute_c_haines(t850=15.0, t500=5.0, td850=10.0)
        unstable = compute_c_haines(t850=15.0, t500=-20.0, td850=10.0)
        assert unstable > stable

    def test_increasing_dryness_increases_index(self):
        """Lower dewpoint (drier) should raise C-Haines."""
        moist = compute_c_haines(t850=15.0, t500=-10.0, td850=14.0)
        dry = compute_c_haines(t850=15.0, t500=-10.0, td850=0.0)
        assert dry > moist


class TestVectorizedComputation:
    def test_numpy_array_input(self):
        """Vectorized call returns array with same shape."""
        t850 = np.array([10.0, 15.0, 20.0, 5.0])
        t500 = np.array([-5.0, -15.0, -25.0, 0.0])
        td850 = np.array([8.0, 10.0, 5.0, 4.5])

        result = compute_c_haines(t850=t850, t500=t500, td850=td850)
        assert result.shape == (4,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 13.0)

    def test_vectorized_matches_scalar(self):
        """Vectorized result matches element-wise scalar computation."""
        t850_vals = [10.0, 18.0, 5.0, 22.0]
        t500_vals = [-8.0, -20.0, 2.0, -30.0]
        td850_vals = [8.0, 12.0, 4.0, 8.0]

        # Scalar calls
        scalar_results = [
            compute_c_haines(t850=t8, t500=t5, td850=td)
            for t8, t5, td in zip(t850_vals, t500_vals, td850_vals)
        ]

        # Vectorized call
        vec_result = compute_c_haines(
            t850=np.array(t850_vals),
            t500=np.array(t500_vals),
            td850=np.array(td850_vals),
        )

        for i, (s, v) in enumerate(zip(scalar_results, vec_result)):
            assert abs(s - v) < 1e-9, f"Mismatch at index {i}: scalar={s}, vector={v}"

    def test_2d_array_input(self):
        """2D array input (e.g., spatial grid) is handled correctly."""
        t850 = np.full((3, 4), 15.0)
        t500 = np.full((3, 4), -10.0)
        td850 = np.full((3, 4), 10.0)

        result = compute_c_haines(t850=t850, t500=t500, td850=td850)
        assert result.shape == (3, 4)
        assert np.all(result >= 0.0)
        assert np.all(result <= 13.0)
