"""Evaluation metrics: exact-recovery flag, graceful-degradation curve."""

from __future__ import annotations

import numpy as np

_EXACT_REL_TOL = 1e-6


def exact_recovery_flag(y: np.ndarray, pred: np.ndarray, tol: float = _EXACT_REL_TOL) -> bool:
    """True iff pred reproduces y within relative tolerance."""
    denom = max(float(np.linalg.norm(y)), 1e-12)
    return float(np.linalg.norm(pred - y)) / denom < tol


class graceful_degradation_curve:
    """Utilities for the calibration plot from spec 9.3."""

    @staticmethod
    def monotonic(values: list[float]) -> bool:
        return all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
