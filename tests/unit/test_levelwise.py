"""Level-wise growth engine: structural oracle, invariants, determinism, speed."""
import numpy as np
import pytest
import torch

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
            np.testing.assert_allclose(rl[3], rn[3], rtol=1e-4, atol=1e-5)
        elif rn[1] == "leaf":
            np.testing.assert_allclose(rl[2], rn[2], rtol=1e-4, atol=1e-5)
