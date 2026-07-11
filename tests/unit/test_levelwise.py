"""Level-wise growth engine: structural oracle, invariants, determinism, speed."""
import numpy as np
import pytest
import torch

from eml_boost.tree_split import EmlSplitBoostRegressor
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode, RawSplit

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _friedman(n=6000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + rng.standard_normal(n)
    )
    return X, y


def _friedman_corr(n=8000, seed=0):
    """Friedman with a duplicated column (X[:,4]=X[:,0]) and a near-duplicate
    (X[:,5]=X[:,1]+tiny noise): forces exact/near correlation ties in
    segment_topk_corr so the split-time float index_add_ reduction is
    exercised at ties — the path grow_levelwise's deterministic scope now
    covers (Exp-19 run-2 evidence)."""
    X, y = _friedman(n=n, seed=seed)
    X = X.copy()
    rng = np.random.default_rng(seed + 100)
    X[:, 4] = X[:, 0]                                   # exact-duplicate -> corr tie
    X[:, 5] = X[:, 1] + 1e-6 * rng.standard_normal(n)   # near-duplicate -> near-tie
    return X, y


def _tree_signature(node, out, path="r"):
    """Flatten a tree into comparable (path, kind, payload) rows."""
    if isinstance(node, InternalNode):
        s = node.split
        if isinstance(s, RawSplit):
            out.append((path, "raw", s.feature_idx, s.threshold))
        else:
            out.append(
                (path, "eml", s.feature_subset, s.snapped.terminal_choices, s.threshold)
            )
        _tree_signature(node.left, out, path + "L")
        _tree_signature(node.right, out, path + "R")
    elif isinstance(node, EmlLeafNode):
        out.append((path, "emlleaf", node.snapped.terminal_choices))
    else:
        out.append((path, "leaf", node.value))
    return out


def _fit_single_tree(X, y, growth, **hyper):
    from eml_boost.tree_split.tree import EmlSplitTreeRegressor

    kwargs = dict(max_depth=8, use_gpu=True, random_state=0)
    kwargs.update(hyper)
    t = EmlSplitTreeRegressor(tree_growth=growth, **kwargs)
    t.fit(X, y)
    return t


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("min_leaf", [1, 20])
def test_no_eml_levelwise_matches_nodewise_structure(seed, min_leaf):
    """With the RNG never consumed (no EML anywhere), level-wise growth must
    reproduce node-wise trees: identical shape and split features, with
    thresholds/values within float32+fixed-point tolerance."""
    X, y = _friedman(seed=seed)
    common = dict(
        n_eml_candidates=0, k_leaf_eml=0, min_samples_leaf=min_leaf,
        random_state=seed,
    )
    t_node = _fit_single_tree(X, y, "nodewise", **common)
    t_lvl = _fit_single_tree(X, y, "levelwise", **common)

    sig_n = _tree_signature(t_node._root, [])
    sig_l = _tree_signature(t_lvl._root, [])
    assert len(sig_n) == len(sig_l)
    for rn, rl in zip(sig_n, sig_l, strict=True):
        assert rn[0] == rl[0], f"shape diverged at {rn[0]} vs {rl[0]}"
        assert rn[1] == rl[1]
        if rn[1] == "raw":
            assert rn[2] == rl[2], f"split feature diverged at {rn[0]}"
            # Exact equality, not tolerance: both engines compute this
            # threshold from the SAME shared integer-domain histogram core
            # (multinode_histogram_split, S=1 vs S=A) — Plan Amendment 1
            # made per-node quantized sums (and therefore argmax decisions
            # and bin-edge thresholds) bit-identical by construction. A
            # nonzero delta here would mean the two dispatch paths have
            # silently diverged, not float32 noise to tolerate.
            assert rl[3] == rn[3], f"threshold diverged at {rn[0]}: {rl[3]} vs {rn[3]}"
        elif rn[1] == "leaf":
            np.testing.assert_allclose(rl[2], rn[2], rtol=1e-4, atol=1e-5)


def _walk(node):
    yield node
    if isinstance(node, InternalNode):
        yield from _walk(node.left)
        yield from _walk(node.right)


@requires_cuda
def test_levelwise_boost_eml_invariants():
    """EML-enabled level-wise boost fit: structural invariants + learning."""
    from eml_boost._triton_exhaustive import get_valid_descriptors_np
    from eml_boost.tree_split.nodes import EmlSplit

    X, y = _friedman(n=6000, seed=0)
    m = EmlSplitBoostRegressor(
        max_rounds=10, max_depth=6, patience=0, use_gpu=True,
        random_state=0, tree_growth="levelwise",
    )
    m.fit(X, y)

    d = X.shape[1]
    n_eml_splits = 0
    valid = {tuple(int(v) for v in row) for row in get_valid_descriptors_np(2, 3)}
    for t in m._trees:
        for node in _walk(t._root):
            if isinstance(node, InternalNode):
                assert np.isfinite(node.split.threshold)
                if isinstance(node.split, EmlSplit):
                    n_eml_splits += 1
                    assert node.split.snapped.terminal_choices in valid
                    assert all(0 <= f < d for f in node.split.feature_subset)
    assert n_eml_splits > 0, "levelwise engine never chose an EML split"

    pred = m.predict(X)
    assert np.isfinite(pred).all()
    base = float(np.mean((y - y.mean()) ** 2))
    fit_mse = float(np.mean((y - pred) ** 2))
    assert fit_mse < 0.5 * base, f"did not learn: {fit_mse} vs baseline {base}"


@requires_cuda
def test_levelwise_rmse_parity_with_nodewise():
    """Statistical sanity: same data, both engines, held-out RMSE within a
    generous band (RNG orders differ; exact match is impossible)."""
    X, y = _friedman(n=8000, seed=1)
    Xtr, Xte, ytr, yte = X[:6000], X[6000:], y[:6000], y[6000:]

    def _rmse(growth):
        m = EmlSplitBoostRegressor(
            max_rounds=25, max_depth=6, patience=0, use_gpu=True,
            random_state=1, tree_growth=growth,
        )
        m.fit(Xtr, ytr)
        return float(np.sqrt(np.mean((yte - m.predict(Xte)) ** 2)))

    r_node = _rmse("nodewise")
    r_lvl = _rmse("levelwise")
    assert r_lvl < r_node * 1.15, f"levelwise {r_lvl} vs nodewise {r_node}"


def test_tree_growth_param_validation_and_sklearn_roundtrip():
    from sklearn.base import clone

    with pytest.raises(ValueError, match="tree_growth"):
        from eml_boost.tree_split.tree import EmlSplitTreeRegressor

        EmlSplitTreeRegressor(tree_growth="diagonal")
    m = EmlSplitBoostRegressor(tree_growth="levelwise")
    m2 = clone(m)
    assert m2.tree_growth == "levelwise"


@requires_cuda
@pytest.mark.parametrize("seed,corr", [(0, False), (7, False), (0, True)])
def test_levelwise_same_seed_bitwise_deterministic(seed, corr):
    """Spec acceptance: two same-seed fits -> byte-identical predictions.

    The corr=True case duplicates/near-duplicates feature columns so
    segment_topk_corr hits exact/near correlation ties; without the
    deterministic scope now wrapping grow_levelwise's float index_add_ corr
    reduction, those ties can flip run-to-run (Exp-19 run-2 evidence)."""
    X, y = _friedman_corr(n=8000, seed=seed) if corr else _friedman(n=8000, seed=seed)

    def _fit_predict():
        m = EmlSplitBoostRegressor(
            max_rounds=12, max_depth=8, patience=0, use_gpu=True,
            random_state=seed, tree_growth="levelwise",
        )
        m.fit(X, y)
        return m.predict(X[:2000])

    p1, p2 = _fit_predict(), _fit_predict()
    np.testing.assert_array_equal(p1, p2)


@requires_cuda
def test_levelwise_speedup_over_nodewise():
    """Conservative CI gate (expected ~5-10x; assert 3x to absorb noise)."""
    import time

    X, y = _friedman(n=32_000, d=10, seed=0)

    def _time(growth):
        m = EmlSplitBoostRegressor(
            max_rounds=3, max_depth=8, patience=0, use_gpu=True,
            random_state=0, tree_growth=growth,
        )
        m.fit(X, y)  # warmup (JIT, caches)
        t0 = time.perf_counter()
        m = EmlSplitBoostRegressor(
            max_rounds=10, max_depth=8, patience=0, use_gpu=True,
            random_state=0, tree_growth=growth,
        )
        m.fit(X, y)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    t_node = _time("nodewise")
    t_lvl = _time("levelwise")
    assert t_lvl * 3.0 < t_node, f"levelwise {t_lvl:.2f}s vs nodewise {t_node:.2f}s"
