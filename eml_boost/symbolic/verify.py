"""Numerical verification that a snapped-and-simplified sympy expression
reproduces the same outputs as the underlying parameterized tree on a
batch of samples.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


def reproduces_numerically(
    expr: sp.Expr,
    feature_names: tuple[str, ...],
    samples: np.ndarray,  # (n, k)
    reference_outputs: np.ndarray,  # (n,)
    tol: float,
) -> bool:
    """Check that expr evaluates close to reference_outputs on samples."""
    symbols = [sp.Symbol(name) for name in feature_names]
    f = sp.lambdify(symbols, expr, modules=["numpy"])
    try:
        predicted = np.asarray(f(*[samples[:, i] for i in range(samples.shape[1])]))
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    if predicted.shape != reference_outputs.shape:
        # Scalar expression broadcast — normalize
        if predicted.ndim == 0:
            predicted = np.full_like(reference_outputs, float(predicted))
        else:
            return False
    diff = np.asarray(predicted) - reference_outputs
    denom = max(np.linalg.norm(reference_outputs), 1e-12)
    return bool(np.linalg.norm(diff) / denom < tol)
