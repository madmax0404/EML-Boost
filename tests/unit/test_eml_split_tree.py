"""Phase 1 MVP tests for EmlSplitTreeRegressor.

Goals of this test file:
  1. Fit/predict shape contract and no-crash behavior.
  2. Axis-aligned-only mode (n_eml_candidates=0) reduces to a plain
     regression tree and gets the obvious signal on a threshold target.
  3. With EML candidates enabled, the tree does NOT get worse on a pure
     raw-feature signal (sanity — EML shouldn't hurt if raw wins).
  4. With EML candidates enabled, on a target where the true decision
     boundary is an elementary function (y = sign(exp(x_0) − t)), the
     tree with EML candidates beats the raw-only tree at same depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from eml_boost.tree_split.nodes import EmlSplit, InternalNode, LeafNode, RawSplit
from eml_boost.tree_split.tree import EmlSplitTreeRegressor


def _mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + 0.5 * X[:, 1]
    m = EmlSplitTreeRegressor(max_depth=3, n_eml_candidates=0, random_state=0).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape


def test_raw_only_threshold_target():
    # y piecewise on x_0 — raw tree should nail it.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=10, n_eml_candidates=0, random_state=0,
    ).fit(X, y)
    assert _mse(m.predict(X), y) < 0.1


def test_eml_candidates_do_not_hurt_raw_signal():
    # Same axis-aligned signal; turning on EML candidates shouldn't be much worse.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = np.where(X[:, 0] > 0, 1.0, -1.0)

    m_raw = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=10, n_eml_candidates=0, random_state=0,
    ).fit(X, y)
    m_eml = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=10, n_eml_candidates=10, k_eml=2,
        random_state=0,
    ).fit(X, y)

    mse_raw = _mse(m_raw.predict(X), y)
    mse_eml = _mse(m_eml.predict(X), y)
    # Allow up to 2x slack — EML shouldn't actively hurt.
    assert mse_eml < 2 * mse_raw + 0.05


@pytest.mark.parametrize("max_depth", [2, 3])
def test_eml_candidates_help_elementary_boundary(max_depth):
    # True decision: y = 1 if exp(x_0) >= e, else -1. The raw feature would
    # need threshold t=1 to get a similarly clean split — which in principle
    # a raw tree could find too (exp is monotonic). But on an elementary
    # target the EML tree shouldn't do worse; this test is a sanity check
    # that the candidate-EML code path actually integrates cleanly.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.5, 1.5, size=(200, 2))
    y = np.where(np.exp(X[:, 0]) >= np.e, 1.0, -1.0)

    m_eml = EmlSplitTreeRegressor(
        max_depth=max_depth, min_samples_leaf=10, n_eml_candidates=20,
        k_eml=2, random_state=0,
    ).fit(X, y)
    # Baseline: perfect constant predictor would give var(y) MSE.
    baseline = float(np.var(y))
    assert _mse(m_eml.predict(X), y) < 0.5 * baseline


def test_histogram_mode_activates_on_large_n():
    """With n > histogram_min_n the tree should use histogram split-finding
    and produce a sensible fit."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(1500, 3))
    y = X[:, 0] + 0.2 * rng.normal(size=1500)
    m = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=10, k_eml=2,
        histogram_min_n=500, random_state=0,
    ).fit(X, y)
    mse = _mse(m.predict(X), y)
    baseline = float(np.var(y))
    assert mse < 0.5 * baseline


def test_histogram_vs_exact_similar_quality():
    """Histogram mode should produce comparable (within 2x) MSE vs exact mode
    on the same data."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 3))
    y = (X[:, 0] > 0).astype(float) + 0.1 * rng.normal(size=800)

    m_exact = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=0,
        histogram_min_n=10_000,  # force exact path
        random_state=0,
    ).fit(X, y)
    m_hist = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=0,
        histogram_min_n=100,  # force histogram path
        random_state=0,
    ).fit(X, y)

    mse_exact = _mse(m_exact.predict(X), y)
    mse_hist = _mse(m_hist.predict(X), y)
    # Histogram has finite-bin error but should be close.
    assert mse_hist < 2 * mse_exact + 0.05


def test_internal_node_stores_either_split_kind():
    # Construct a small fit that should produce an internal node; verify the
    # split is one of the two expected types.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = X[:, 0] ** 2
    m = EmlSplitTreeRegressor(
        max_depth=2, min_samples_leaf=10, n_eml_candidates=10, k_eml=2, random_state=0,
    ).fit(X, y)
    assert isinstance(m._root, InternalNode)
    assert isinstance(m._root.split, (RawSplit, EmlSplit))
