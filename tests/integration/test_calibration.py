"""Smoke test for the calibration harness — runs a tiny version end-to-end."""

import pytest

from experiments.calibration import run_calibration


@pytest.mark.slow
def test_calibration_produces_monotonic_curve():
    result = run_calibration(
        elementary_fractions=[0.0, 0.5, 1.0],
        n_datasets_per_fraction=1,
        n=150,
        seed=0,
        max_rounds=10,
        n_restarts=3,
        depth_eml=2,
        depth_dt=2,
    )
    assert len(result.eml_win_rates) == 3
    # Loose monotonicity check — 3-point series should trend upward.
    assert result.eml_win_rates[-1] >= result.eml_win_rates[0] - 0.25
