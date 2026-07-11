# tests/unit/test_multinode_hist.py
"""Multi-node fixed-point histogram split vs the single-node torch oracle,
plus the run-to-run determinism property the old float-atomic path lacked."""
import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make(n=20_000, c=6, n_seg=5, seed=0):
    rng = np.random.default_rng(seed)
    vals = torch.tensor(rng.standard_normal((n, c)), dtype=torch.float32, device="cuda")
    seg = torch.tensor(rng.integers(0, n_seg, size=n), dtype=torch.long, device="cuda")
    y = (vals[:, 0] > 0.3).float() * 2.0 + vals[:, 1] * 0.5
    y = y + torch.tensor(rng.standard_normal(n) * 0.1, dtype=torch.float32, device="cuda")
    return vals, y, seg, n_seg


@requires_cuda
@pytest.mark.parametrize("leaf_l2", [0.0, 1.0])
@pytest.mark.parametrize("min_leaf", [1, 20])
def test_multinode_matches_single_node_oracle(leaf_l2, min_leaf):
    from eml_boost.tree_split._gpu_split import gpu_histogram_split_torch
    from eml_boost.tree_split._multinode_hist import multinode_histogram_split

    vals, y, seg, n_seg = _make()
    B = 64
    col, thr, gain = multinode_histogram_split(
        vals, y, seg, n_seg, B, min_leaf, leaf_l2
    )
    for s in range(n_seg):
        m = seg == s
        f_ref, t_ref, g_ref = gpu_histogram_split_torch(
            vals[m], y[m], B, min_leaf_count=min_leaf, leaf_l2=leaf_l2
        )
        if g_ref <= 0:
            assert float(gain[s]) <= 0
            continue
        assert int(col[s]) == f_ref
        np.testing.assert_allclose(float(thr[s]), t_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(float(gain[s]), g_ref, rtol=5e-3)


@requires_cuda
def test_multinode_deterministic_under_row_shuffle():
    """Fixed-point integer accumulation => bit-identical results regardless
    of row order (the property float atomics cannot give)."""
    from eml_boost.tree_split._multinode_hist import multinode_histogram_split

    vals, y, seg, n_seg = _make(seed=1)
    col1, thr1, gain1 = multinode_histogram_split(vals, y, seg, n_seg, 256, 1, 1.0)
    perm = torch.randperm(vals.shape[0], device="cuda")
    col2, thr2, gain2 = multinode_histogram_split(
        vals[perm], y[perm], seg[perm], n_seg, 256, 1, 1.0
    )
    assert torch.equal(col1, col2)
    assert torch.equal(thr1, thr2)
    assert torch.equal(gain1, gain2)


@requires_cuda
def test_multinode_col_valid_mask_and_empty_segment():
    from eml_boost.tree_split._multinode_hist import multinode_histogram_split

    vals, y, seg, n_seg = _make(seed=2)
    col_valid = torch.ones(n_seg + 2, vals.shape[1], dtype=torch.bool, device="cuda")
    col_valid[:, 0] = False  # mask out the strongest column everywhere
    col, thr, gain = multinode_histogram_split(
        vals, y, seg, n_seg + 2, 64, 1, 1.0, col_valid=col_valid
    )
    assert (col[:n_seg] != 0).all(), "masked column must never win"
    assert (gain[n_seg:] <= 0).all(), "empty segments must report no split"


@requires_cuda
def test_multinode_constant_column_never_wins():
    from eml_boost.tree_split._multinode_hist import multinode_histogram_split

    vals, y, seg, n_seg = _make(seed=3)
    vals = vals.clone()
    vals[:, 2] = 7.5  # constant column: no legal split on it
    col, _thr, gain = multinode_histogram_split(vals, y, seg, n_seg, 64, 1, 0.0)
    assert (col != 2).all()


@requires_cuda
def test_gpu_histogram_split_dispatch_matches_torch_oracle():
    """The rewired dispatcher (single-segment multinode core) keeps the old
    contract and agrees with the float oracle within fixed-point tolerance."""
    from eml_boost.tree_split._gpu_split import (
        gpu_histogram_split,
        gpu_histogram_split_torch,
    )

    vals, y, _seg, _ = _make(seed=4)
    f, t, g = gpu_histogram_split(vals, y, 256, min_leaf_count=1, leaf_l2=1.0)
    f_ref, t_ref, g_ref = gpu_histogram_split_torch(
        vals, y, 256, min_leaf_count=1, leaf_l2=1.0
    )
    assert isinstance(f, int) and isinstance(t, float) and isinstance(g, float)
    assert f == f_ref
    np.testing.assert_allclose(t, t_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(g, g_ref, rtol=5e-3)


@requires_cuda
def test_nodewise_fit_run_to_run_deterministic():
    """THE unblocking property: two same-seed GPU boost fits must now be
    byte-identical (was 419/500 predictions differing before the rewire).
    Do not weaken this to allclose."""
    from eml_boost.tree_split import EmlSplitBoostRegressor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 8))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + rng.standard_normal(3000)
    )

    def _fit_predict():
        m = EmlSplitBoostRegressor(
            max_rounds=8, max_depth=6, patience=0, use_gpu=True, random_state=0
        )
        m.fit(X, y)
        return m.predict(X[:500])

    p1, p2 = _fit_predict(), _fit_predict()
    np.testing.assert_array_equal(p1, p2)
