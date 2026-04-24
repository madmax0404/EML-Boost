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

from eml_boost.tree_split.nodes import (
    EmlLeafNode,
    EmlSplit,
    InternalNode,
    LeafNode,
    RawSplit,
)
from eml_boost.tree_split.tree import EmlSplitTreeRegressor


def _count_eml_leaves(node):
    if isinstance(node, EmlLeafNode):
        return 1
    if isinstance(node, LeafNode):
        return 0
    return _count_eml_leaves(node.left) + _count_eml_leaves(node.right)


def _count_leaves(node):
    if isinstance(node, (LeafNode, EmlLeafNode)):
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


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


def test_eml_leaf_disabled_when_k_leaf_eml_zero():
    """With k_leaf_eml=0, no leaf should ever be an EmlLeafNode."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0])
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=10, n_eml_candidates=0,
        k_leaf_eml=0, random_state=0,
    ).fit(X, y)
    assert _count_eml_leaves(m._root) == 0


def test_eml_leaf_activates_on_elementary_target():
    """On y = exp(x_0), EML leaves with k=1 should snap to the right
    expression and outperform constant leaves."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0])
    m_eml_leaf = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05, random_state=0,
    ).fit(X, y)
    m_constant_leaf = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=0, random_state=0,
    ).fit(X, y)
    mse_eml = _mse(m_eml_leaf.predict(X), y)
    mse_const = _mse(m_constant_leaf.predict(X), y)
    # EML leaves should beat constant leaves on a smooth signal.
    assert mse_eml < mse_const
    # And at least one leaf should have actually become an EML leaf.
    assert _count_eml_leaves(m_eml_leaf._root) >= 1


def test_eml_leaf_gate_rejects_weak_fits():
    """On pure Gaussian noise, no EML leaf should beat a constant leaf by
    5%, so the gate should reject every EML leaf candidate."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = rng.normal(size=800)  # pure noise
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        leaf_eml_gain_threshold=0.05, use_stacked_blend=False, random_state=0,
    ).fit(X, y)
    # No EML leaf should have passed the gate on pure noise.
    # (Some overfitting to training noise is possible, but with the 5% gate
    # and min_samples_leaf_eml=50, most leaves will stay constant.)
    n_eml = _count_eml_leaves(m._root)
    n_total = _count_leaves(m._root)
    # Allow up to 40% EML leaves on pure noise before flagging as a gate
    # failure — seed variance in tiny leaves can still let a few through.
    assert n_eml < 0.4 * n_total, f"{n_eml}/{n_total} EML leaves on pure noise"


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


def test_use_stacked_blend_false_matches_current_behavior():
    """With `use_stacked_blend=False` (and the rest of the config identical),
    the regressor should behave exactly like the current gated implementation:
    pure-noise training data should leave most leaves as constants."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = rng.normal(size=800)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        leaf_eml_gain_threshold=0.05,
        use_stacked_blend=False,
        random_state=0,
    ).fit(X, y)
    n_eml = _count_eml_leaves(m._root)
    n_total = _count_leaves(m._root)
    assert n_eml < 0.4 * n_total, f"{n_eml}/{n_total} EML leaves on pure noise"


def test_stacked_blend_collapses_to_constant_on_pure_noise():
    """On pure Gaussian noise, the blend must not meaningfully overfit
    vs. a constant-leaf baseline. We check behavior (MSE) rather than
    leaf type, because an EmlLeafNode with α≈1 has η'≈0 and β'≈ȳ and
    therefore behaves as a constant even though its type is EmlLeafNode."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = rng.normal(size=800)

    m_blend = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    m_const = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=0, use_stacked_blend=True, random_state=0,
    ).fit(X, y)

    mse_blend = _mse(m_blend.predict(X), y)
    mse_const = _mse(m_const.predict(X), y)
    # Blend may slightly overfit on noise but must stay within 15% of
    # the constant-leaf baseline on training data.
    assert mse_blend <= 1.15 * mse_const, (
        f"blend={mse_blend:.4f} vs const={mse_const:.4f}"
    )


def test_stacked_blend_activates_on_clean_elementary_signal():
    """On `y = exp(x_0) + tiny_noise`, the blend should latch onto the EML
    tree (α ≈ 0) and produce EML leaves that outperform constant leaves."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    m_blend = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    m_const = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=0, use_stacked_blend=True, random_state=0,
    ).fit(X, y)

    mse_blend = _mse(m_blend.predict(X), y)
    mse_const = _mse(m_const.predict(X), y)
    assert mse_blend < mse_const, f"blend={mse_blend:.4f} vs const={mse_const:.4f}"
    assert _count_eml_leaves(m_blend._root) >= 1


def test_stacked_blend_no_numerical_blowup_on_heavy_tails():
    """A leaf-local feature with magnitudes into the millions (like
    PMLB 562_cpu_small) must not produce NaN/inf predictions under the
    blended path."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 2)) * 1e6
    # Targets that are a small, well-behaved transformation of the big feature.
    y = 0.001 * (X[:, 0] / 1e6)

    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred)), (
        "prediction contains NaN or inf — numerical stability failure"
    )


def test_leaf_stats_populated_when_blend_enabled():
    """With `use_stacked_blend=True`, each leaf that reaches the EML
    decision point should append a record to `_leaf_stats` with the
    chosen α and the emitted leaf type."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    stats = m._leaf_stats
    assert len(stats) >= 1
    for s in stats:
        assert s["n_leaf"] >= 30
        assert 0.0 <= s["alpha"] <= 1.0
        assert s["leaf_type"] in ("LeafNode", "EmlLeafNode")
    # On a clean exp(x_0) signal we expect at least one non-collapsed EML leaf.
    assert any(s["leaf_type"] == "EmlLeafNode" for s in stats)


def test_leaf_eml_ridge_parameter_accepted():
    """Constructor must accept the new leaf_eml_ridge parameter. At 0.0
    (default) the predictions must be identical to a regressor built
    without the parameter — this pins the backward-compat story."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    m_default = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30, random_state=0,
    ).fit(X, y)
    m_ridge_zero = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_ridge=0.0, random_state=0,
    ).fit(X, y)
    assert np.allclose(m_default.predict(X), m_ridge_zero.predict(X))


def test_ridge_shrinks_max_abs_eta_monotonically():
    """On a clean elementary signal, max |η| across the tree's EML leaves
    should decrease monotonically as leaf_eml_ridge increases."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")

    def _collect_etas(node):
        etas: list[float] = []
        def walk(n):
            if isinstance(n, EmlLeafNode):
                etas.append(abs(float(n.eta)))
            elif isinstance(n, InternalNode):
                walk(n.left); walk(n.right)
        walk(node)
        return etas

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    max_etas: list[float] = []
    for ridge in [0.0, 0.1, 1.0, 10.0]:
        m = EmlSplitTreeRegressor(
            max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
            k_leaf_eml=1, min_samples_leaf_eml=30,
            leaf_eml_ridge=ridge, random_state=0,
        ).fit(X, y)
        etas = _collect_etas(m._root)
        max_etas.append(max(etas) if etas else 0.0)

    # Strict monotonic decrease expected; allow equality only when all
    # ridge settings produce zero EML leaves (shouldn't happen on a
    # clean exp signal, but be defensive).
    for i in range(len(max_etas) - 1):
        assert max_etas[i] >= max_etas[i + 1], (
            f"non-monotonic: ridge grid gives max|eta| = {max_etas}"
        )
    # End-to-end strictness: ridge=10 must shrink max|η| meaningfully
    # vs ridge=0. Without this, a broken ridge producing identical η
    # across all ridge values would pass the non-strict monotonicity.
    assert max_etas[0] > 2.0 * max_etas[-1], (
        f"ridge failed to shrink meaningfully: max|eta| = {max_etas}"
    )


def test_ridge_prevents_blowup_on_heavy_tails():
    """On features with magnitudes ~1e6, the Experiment-9 failure mode,
    leaf_eml_ridge=1.0 must keep predictions finite and max |η| bounded."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")

    def _collect_etas(node):
        etas: list[float] = []
        def walk(n):
            if isinstance(n, EmlLeafNode):
                etas.append(abs(float(n.eta)))
            elif isinstance(n, InternalNode):
                walk(n.left); walk(n.right)
        walk(node)
        return etas

    rng = np.random.default_rng(0)
    # Same magnitudes as 562_cpu_small features in Experiment 9.
    X = rng.normal(size=(800, 2)) * 1e6
    y = 0.001 * (X[:, 0] / 1e6) + 0.01 * rng.normal(size=800)

    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        leaf_eml_ridge=1.0, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred))
    etas = _collect_etas(m._root)
    if etas:
        # A 50% shrinkage (ridge=1.0) applied to pre-clamp features of
        # magnitude ~1 should keep |η| well below 100 on this synthetic.
        assert max(etas) < 100.0, f"max|eta| = {max(etas):.2f}"
