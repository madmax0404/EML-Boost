import numpy as np

from eml_boost.metrics import exact_recovery_flag, graceful_degradation_curve


def test_exact_recovery_flag_true_when_tiny_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = y + 1e-10
    assert exact_recovery_flag(y, pred)


def test_exact_recovery_flag_false_when_large_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = y + 1.0
    assert not exact_recovery_flag(y, pred)


def test_graceful_degradation_curve_monotonic():
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    eml_win_rates = [0.05, 0.25, 0.55, 0.8, 0.97]
    assert graceful_degradation_curve.monotonic(eml_win_rates)


def test_graceful_degradation_curve_not_monotonic():
    assert not graceful_degradation_curve.monotonic([0.5, 0.3, 0.6])
