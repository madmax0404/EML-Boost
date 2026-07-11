# tests/unit/test_leaf_batch.py
"""Stage-1 (batched leaf fitting) tests: deferral bit-exactness + batched A/B."""
import numpy as np
import pytest
import torch

from eml_boost.tree_split import EmlSplitBoostRegressor

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

SNAPSHOT = "tests/unit/fixtures/leaf_deferral_snapshot.npy"


def _friedman(n=3000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + rng.standard_normal(n)
    )
    return X, y


@requires_cuda
def test_leaf_deferral_matches_snapshot():
    """Deferring leaf fits to post-growth must not change a single bit.

    Snapshot captured pre-refactor (commit of Task 2 Step 2). Reference
    (per-leaf) path pinned via _batched_leaves=False on every tree — done
    here by patching the class default attribute.
    """
    import eml_boost.tree_split.tree as tree_mod

    X, y = _friedman()
    model = EmlSplitBoostRegressor(
        max_rounds=8, max_depth=6, patience=0, use_gpu=True, random_state=0
    )
    # Force reference per-leaf finalize on the trees this boost fit creates
    # (attribute exists only post-refactor; pre-refactor this is a no-op).
    orig_init = tree_mod.EmlSplitTreeRegressor.__init__

    def patched(self, **kw):
        orig_init(self, **kw)
        self._batched_leaves = False

    tree_mod.EmlSplitTreeRegressor.__init__ = patched
    try:
        model.fit(X, y)
        pred = model.predict(X[:500])
    finally:
        tree_mod.EmlSplitTreeRegressor.__init__ = orig_init

    import os
    if not os.path.exists(SNAPSHOT):
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        np.save(SNAPSHOT, pred)
        pytest.skip("snapshot captured; rerun to compare")
    want = np.load(SNAPSHOT)
    np.testing.assert_array_equal(pred, want)


def _manual_tree(X, y, **hyper):
    """Instantiate a tree with GPU internals populated, without growing.

    Lets tests call _fit_leaf / fit_leaves_batched on hand-built leaves.
    """
    from eml_boost.tree_split.tree import EmlSplitTreeRegressor

    t = EmlSplitTreeRegressor(**hyper)
    device = torch.device("cuda")
    t._device = device
    t._X_cpu = X
    t._X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    t._y_gpu = torch.tensor(y, dtype=torch.float32, device=device)
    t._global_mean = X.mean(axis=0)
    t._global_std = np.maximum(X.std(axis=0), 1e-6)
    t._global_mean_gpu = torch.tensor(t._global_mean, dtype=torch.float32, device=device)
    t._global_std_gpu = torch.tensor(t._global_std, dtype=torch.float32, device=device)
    return t


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_leaf_fit_matches_reference(seed):
    """fit_leaves_batched vs per-leaf _fit_leaf on identical leaf partitions:
    identical node types, identical descriptor/feature choices, params
    within float32 reduction-order tolerance."""
    from eml_boost.tree_split._leaf_batch import fit_leaves_batched
    from eml_boost.tree_split.nodes import EmlLeafNode, LeafNode
    from eml_boost.tree_split.tree import _PendingLeaf

    rng = np.random.default_rng(seed)
    X, y = _friedman(n=4000, seed=seed)
    t = _manual_tree(X, y)  # library defaults: k_leaf_eml=1, gated

    # Hand-build a mix of leaf sizes: below-eligibility, boundary, large.
    order = rng.permutation(len(X))
    sizes = [3, 12, 29, 30, 31, 60, 200, 800, len(X) - 1165]
    pending, start = [], 0
    for sz in sizes:
        idx = torch.tensor(order[start : start + sz], dtype=torch.long, device="cuda")
        pending.append(_PendingLeaf(indices=idx))
        start += sz

    ref = [t._fit_leaf(p.indices) for p in pending]
    got = fit_leaves_batched(t, pending)

    assert len(got) == len(ref)
    for r, g in zip(ref, got, strict=True):
        assert type(r) is type(g)
        if isinstance(r, LeafNode):
            np.testing.assert_allclose(g.value, r.value, rtol=1e-4, atol=1e-6)
        else:
            assert isinstance(r, EmlLeafNode)
            assert g.snapped.terminal_choices == r.snapped.terminal_choices
            assert g.feature_subset == r.feature_subset
            np.testing.assert_allclose(g.eta, r.eta, rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(g.bias, r.bias, rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(g.cap, r.cap, rtol=1e-4)
            np.testing.assert_allclose(g.feature_mean, r.feature_mean, rtol=1e-5)
            np.testing.assert_allclose(g.feature_std, r.feature_std, rtol=1e-5)


@requires_cuda
def test_batched_leaf_fit_ridge_and_capless_variants():
    """leaf_eml_ridge>0 and leaf_eml_cap_k=0 branches match reference."""
    from eml_boost.tree_split._leaf_batch import fit_leaves_batched
    from eml_boost.tree_split.tree import _PendingLeaf

    X, y = _friedman(n=2000, seed=3)
    for hyper in (dict(leaf_eml_ridge=0.5), dict(leaf_eml_cap_k=0.0)):
        t = _manual_tree(X, y, **hyper)
        idx = torch.arange(0, 900, dtype=torch.long, device="cuda")
        pending = [_PendingLeaf(indices=idx)]
        (ref,), (got,) = [t._fit_leaf(idx)], fit_leaves_batched(t, pending)
        assert type(ref) is type(got)
        if hasattr(ref, "eta"):
            np.testing.assert_allclose(got.eta, ref.eta, rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(got.bias, ref.bias, rtol=1e-3, atol=1e-6)


@requires_cuda
def test_batched_leaf_fit_empty_and_zero_row_leaf():
    from eml_boost.tree_split._leaf_batch import fit_leaves_batched
    from eml_boost.tree_split.nodes import LeafNode
    from eml_boost.tree_split.tree import _PendingLeaf

    X, y = _friedman(n=200, seed=4)
    t = _manual_tree(X, y)
    assert fit_leaves_batched(t, []) == []
    empty = _PendingLeaf(indices=torch.empty(0, dtype=torch.long, device="cuda"))
    (got,) = fit_leaves_batched(t, [empty])
    assert isinstance(got, LeafNode) and got.value == 0.0
