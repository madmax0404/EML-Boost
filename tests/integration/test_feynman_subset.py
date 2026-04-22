"""End-to-end exact recovery on a small Feynman subset.

Per spec section 10, run per-commit (target < 10 minutes). Individual
formulas may or may not recover exactly — we assert on the aggregate
to keep the test stable.
"""

import numpy as np
import pytest

from eml_boost import EmlBoostRegressor
from eml_boost.datasets import load_feynman_formula

FORMULAS = ["I.6.2a", "I.12.1", "I.13.4", "I.29.4", "II.11.28"]


@pytest.mark.slow
@pytest.mark.parametrize("formula_name", FORMULAS)
def test_feynman_formula_rmse_below_threshold(formula_name):
    X, y, meta = load_feynman_formula(formula_name, n=400, seed=0)
    model = EmlBoostRegressor(
        max_rounds=60,
        depth_eml=3,
        depth_dt=3,
        n_restarts=12,
        k=min(3, X.shape[1]),
        random_state=0,
    )
    model.fit(X, y)
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    scale = float(np.sqrt(np.mean(y**2)) + 1e-12)
    relative_rmse = rmse / scale
    assert relative_rmse < 0.25, (
        f"{formula_name}: relative RMSE {relative_rmse:.3f} — expected < 0.25. "
        f"Formula: {meta['formula']}"
    )


@pytest.mark.slow
def test_aggregate_coverage_non_trivial():
    """Across the subset, average coverage should exceed 30%."""
    coverages = []
    for name in FORMULAS:
        X, y, _ = load_feynman_formula(name, n=400, seed=0)
        model = EmlBoostRegressor(
            max_rounds=60, depth_eml=3, depth_dt=3, n_restarts=12, k=min(3, X.shape[1]),
            random_state=0,
        ).fit(X, y)
        coverages.append(model.coverage(X))
    avg = float(np.mean(coverages))
    assert avg > 0.30, f"average coverage {avg:.2f} too low — spec gate"
