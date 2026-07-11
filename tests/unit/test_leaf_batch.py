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
