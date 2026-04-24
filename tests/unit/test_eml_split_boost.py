"""Phase 3 tests for EmlSplitBoostRegressor (single-family boosting)."""

from __future__ import annotations

import numpy as np
import pytest

from eml_boost.tree_split import EmlSplitBoostRegressor


def _rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def test_fit_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    m = EmlSplitBoostRegressor(
        max_rounds=10, max_depth=3, n_eml_candidates=0,
        patience=None, random_state=0,
    ).fit(X, y)
    assert m.predict(X).shape == y.shape
    assert m.n_rounds == 10


def test_boosting_reduces_residual():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 3))
    y = X[:, 0] ** 2 + 0.1 * rng.normal(size=300)
    m = EmlSplitBoostRegressor(
        max_rounds=30, max_depth=4, learning_rate=0.1,
        n_eml_candidates=10, k_eml=3,
        patience=None, random_state=0,
    ).fit(X, y)
    baseline = _rmse(np.full_like(y, y.mean()), y)
    fit_rmse = _rmse(m.predict(X), y)
    # 30 rounds × lr 0.1 × depth 4 should comfortably cut RMSE by ~30% on
    # a noisy quadratic; allow generous slack to avoid flaky thresholds.
    assert fit_rmse < 0.7 * baseline


def test_early_stopping_triggers():
    """With tight patience, boosting should stop well before max_rounds
    once the held-out val MSE stops improving.

    k_leaf_eml=0 disables EML leaves so the test is about early-stopping
    behavior on a linear signal, not leaf-fit behavior. With the default
    min_samples_leaf_eml=30 and depth-3 trees on 240 train samples,
    leaves land right at the EML threshold and would otherwise stay
    productive for all 200 rounds."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = X[:, 0]  # linear, trivially fit in a few rounds
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=3, learning_rate=0.1,
        n_eml_candidates=0, k_leaf_eml=0, patience=5,
        val_fraction=0.2, random_state=0,
    ).fit(X, y)
    assert m.n_rounds < 200
    # Should have logged at least one val_mse entry
    assert any("val_mse" in h for h in m.history)


@pytest.mark.parametrize("n_eml", [0, 10])
def test_raw_only_and_eml_mode_both_work(n_eml):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = np.exp(X[:, 0]) + 0.05 * rng.normal(size=200)
    m = EmlSplitBoostRegressor(
        max_rounds=20, max_depth=3, n_eml_candidates=n_eml, k_eml=2,
        patience=None, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    baseline = _rmse(np.full_like(y, y.mean()), y)
    assert _rmse(pred, y) < 0.5 * baseline
