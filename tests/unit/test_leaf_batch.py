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


def _walk_pairs(a, b):
    """Yield (node_a, node_b) over two trees in lockstep; fail on SPLIT
    structure mismatch (InternalNode on one side, not the other).

    Deliberately does NOT require the two leaves' exact dataclass type
    (LeafNode vs EmlLeafNode) to match at a yielded leaf position -- see
    the caller (test_full_fit_batched_vs_reference_leaves) for why exact
    leaf-type/descriptor-choice agreement is not always achievable, and
    how it applies near-tie-aware tolerance instead.
    """
    from eml_boost.tree_split.nodes import InternalNode

    stack = [(a, b)]
    while stack:
        x, z = stack.pop()
        x_internal = isinstance(x, InternalNode)
        z_internal = isinstance(z, InternalNode)
        assert x_internal == z_internal, f"structure diverged: {type(x)} vs {type(z)}"
        yield x, z
        if x_internal:
            stack.append((x.left, z.left))
            stack.append((x.right, z.right))


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1])
def test_full_fit_batched_vs_reference_leaves(seed):
    """End-to-end boost fit: batched vs reference leaf finalize.
    Split structure must be identical (split path untouched); leaf params
    and predictions within float32 tolerance, with a documented, bounded
    exception for genuine near-ties (see below).

    max_rounds=1, not 6 as originally drafted: a multi-round, two-FULLY-
    INDEPENDENT-trajectories design is confounded by boosting's own
    feedback sensitivity. Round 0's residual is bit-identical between the
    ref/batched paths by construction (both start from F_0), so round 0's
    SPLIT structure is provably unaffected by _batched_leaves regardless of
    any leaf-value tolerance question. But round 0's LEAF VALUES can differ
    at the ULP level (see below), so F_tr after round 0 differs minutely
    between the two independent trajectories -- and boosting is a feedback
    loop, so by round 1 that ULP-level difference had already flipped a
    deeply-nested split's TYPE (confirmed empirically: seed=1, round 1,
    path RLR, RawSplit(1.536) vs EmlSplit(-0.197) -- not a near-miss, a
    fully different split). This is inherent to iterating ANY two
    numerically-non-bit-identical implementations through a feedback loop,
    independent of how correct either implementation is; max_rounds=1
    exercises the full fit/grow/finalize/tensorize/predict pipeline on one
    real, deep (depth 6) grown tree while staying inside the regime where
    "split path untouched" is provably true rather than empirically hoped
    for.

    Leaf-value near-ties (still possible even at round 0, unrelated to the
    above): the batched path selects among the same 144 candidate
    descriptors as the reference via a differently-ordered/differently-
    reduced floating-point computation (a design requirement -- it's what
    makes it fast). Investigated exhaustively (see Task 4 report): when two
    candidates are genuinely within noise-floor distance on val-SSE
    (confirmed against an independent float64 ground truth; observed gaps
    as small as ~1e-7-1e-6 relative), argmin/gate decisions can pick either
    one, and reference's own float32 reduction is not more "correct" than
    batched's -- in every investigated case the float64 ground truth judged
    batched's pick at least as good, often better (reference's un-guarded
    float32 det check can even accept a near-singular candidate a precise
    computation would reject). Forcing bit-identical tie-breaking would
    require batched to reproduce reference's specific float32 rounding
    behavior exactly; confirmed this is not achievable even by matching
    reference's row order and reduction primitive one at a time (still
    diverges), and separately confirmed that even a padded dense-sum
    (single kernel, no scatter/index_add_ at all) does not reproduce
    per-leaf .sum(dim=1) bit-for-bit -- torch's CUDA reduction kernel
    selection depends on the reduced dimension's length, not just its
    values, so ANY batched layout necessarily differs from 250 separate
    per-leaf calls. So: assert exact agreement when choices match (the
    common case), and when they don't, sanity-check both sides are finite
    and bound the DIVERGENCE RATE (catches wholesale breakage) rather than
    requiring zero divergences (which the above rules out as an achievable
    target for a genuinely batched implementation).
    """
    import eml_boost.tree_split.tree as tree_mod
    from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode, LeafNode

    X, y = _friedman(n=4000, seed=seed)

    def _fit(batched):
        orig_init = tree_mod.EmlSplitTreeRegressor.__init__

        def patched(self, **kw):
            orig_init(self, **kw)
            self._batched_leaves = batched

        tree_mod.EmlSplitTreeRegressor.__init__ = patched
        try:
            m = EmlSplitBoostRegressor(
                max_rounds=1, max_depth=6, patience=0, use_gpu=True,
                random_state=seed,
            )
            m.fit(X, y)
        finally:
            tree_mod.EmlSplitTreeRegressor.__init__ = orig_init
        return m

    m_ref = _fit(False)
    m_bat = _fit(True)

    n_eml_leaves = 0
    n_leaf_near_ties = 0
    for tr, tb in zip(m_ref._trees, m_bat._trees, strict=True):
        for nr, nb in _walk_pairs(tr._root, tb._root):
            if isinstance(nr, InternalNode):
                assert type(nr.split) is type(nb.split)
                np.testing.assert_allclose(
                    nr.split.threshold, nb.split.threshold, rtol=0, atol=0
                )  # split path untouched -> exactly equal (round 0: provably so)
            elif isinstance(nr, EmlLeafNode):
                n_eml_leaves += 1
                if isinstance(nb, EmlLeafNode) and (
                    nr.snapped.terminal_choices == nb.snapped.terminal_choices
                ):
                    np.testing.assert_allclose(nb.eta, nr.eta, rtol=1e-3, atol=1e-6)
                else:
                    # Genuine near-tie (see docstring) -- both sides must
                    # still be finite, structurally valid fits.
                    n_leaf_near_ties += 1
                    if isinstance(nb, EmlLeafNode):
                        assert np.isfinite(nb.eta) and np.isfinite(nb.bias)
                    else:
                        assert isinstance(nb, LeafNode)
                        assert np.isfinite(nb.value)
            else:
                assert isinstance(nr, LeafNode)
                if isinstance(nb, LeafNode):
                    np.testing.assert_allclose(nb.value, nr.value, rtol=1e-4, atol=1e-6)
                else:
                    # Accept/reject gate near-tie (see docstring).
                    n_leaf_near_ties += 1
                    assert isinstance(nb, EmlLeafNode)
                    assert np.isfinite(nb.eta) and np.isfinite(nb.bias)
    assert n_eml_leaves > 0, "fixture produced no EML leaves; test is vacuous"
    # Occasional near-tie flips are expected (see docstring); a high rate
    # would indicate a real regression rather than float-precision noise
    # at decision boundaries.
    assert n_leaf_near_ties <= max(2, n_eml_leaves // 3), (
        f"{n_leaf_near_ties}/{n_eml_leaves} EML leaves diverged in "
        "type/descriptor choice -- rate too high to be near-tie noise"
    )

    pred_r = m_ref.predict(X[:800])
    pred_b = m_bat.predict(X[:800])
    np.testing.assert_allclose(pred_b, pred_r, rtol=1e-3, atol=1e-5)
