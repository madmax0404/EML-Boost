# Level-wise Growth Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove SplitBoost's per-node Python/GPU-sync overhead (60× slower than XGB on CTR23) via Stage 1 batched leaf fitting (structure-bit-exact, ~2×) and Stage 2 level-wise batched split growth (statistically equivalent, target ≤10× XGB suite-total), per `docs/superpowers/specs/2026-07-11-levelwise-growth-design.md`.

**Architecture:** Stage 1: growth emits `_PendingLeaf` placeholders; a post-growth `_finalize_leaves()` fits all leaves in one batched pass (`_leaf_batch.py`), with the existing per-leaf `_fit_leaf` retained as reference oracle. Stage 2: a new `_levelwise.py` engine grows the tree breadth-first behind `tree_growth="levelwise"`, using shared segmented-statistics helpers (`_segmented.py`), a deterministic fixed-point multi-node histogram (`_multinode_hist.py`), and a new row-wise multi-descriptor Triton evaluator in `_triton_exhaustive.py`. Default stays `nodewise` until the CTR23 parity run (Task 13) passes.

**Tech Stack:** Python 3.12, numpy, torch (CUDA), Triton, pytest, openml/xgboost/lightgbm for the parity run. Everything runs via `uv run`. RTX 3090 assumed present.

## Global Constraints

- Commit directly on `master` after each task (project convention — no worktrees, no branches).
- All commands via `uv run`; GPU paths require CUDA (skip-guard tests with `torch.cuda.is_available()` like the existing suite).
- No new pip dependencies.
- `tests/unit/test_eml_weak_learner.py::test_fit_recovers_simple_formula` is a pre-existing unrelated failure — leave it alone; "suite passes" means "no NEW failures".
- No split-math changes: gain formula (incl. `leaf_l2`), per-node uniform binning, `min_samples_leaf` legality, leaf OLS/gate/cap semantics preserved exactly.
- CPU pipeline (`_grow`, `_find_best_split_cpu`, `_fit_cpu_loop`) untouched.
- Stage-1 bit-compat guarantee binds at library defaults (`k_leaf_eml=1`); `k_leaf_eml>1` batched path uses corr-sorted feature order (documented deviation, statistically neutral).
- End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Plan Amendment 1 (2026-07-11, during execution)

**Finding (Task 2 implementer, verified):** the CURRENT node-wise GPU engine is
run-to-run nondeterministic — `gpu_histogram_split`'s Triton kernel
(`tl.atomic_add`) and torch fallback (`scatter_add_`) accumulate float32 in
nondeterministic order; 1-ULP gain wobble flips near-tied split argmaxes and
cascades (two same-seed fits of unmodified code: 419/500 predictions differ,
max 5%). This breaks every cross-fit exactness gate in this plan (Task 2
snapshot, Task 4 A/B, Task 8 structural oracle) and means same-seed
experiment fits were never re-runnable.

**Decision (user-approved):** build the deterministic fixed-point histogram
core FIRST and route BOTH engines through it.

- **NEW Task 2A** (brief: `.superpowers/sdd/task-2a-brief.md`): implements
  `segment_minmax` (old Task 5 content), `_multinode_hist.py` (old Task 6
  content, with tensor-scale quantization instead of a `.item()` sync), AND
  rewires `gpu_histogram_split` in `_gpu_split.py` to call
  `multinode_histogram_split` with one segment — the Triton float-atomic hist
  kernel leaves the dispatch path (file and kernel tests remain). Adds a
  run-to-run nodewise-fit determinism test. Accepted behavior change: node-wise
  thresholds shift within fixed-point quantization tolerance (~2^-20 relative)
  and node-wise loses the Triton hist speedup (~7% e2e on big data) — it is
  becoming the test oracle.
- **Tasks 5 and 6 are folded into Task 2A** — when reached, skip them (verify
  their tests exist and pass, nothing more).
- Task 8's no-EML structural oracle now compares two engines sharing ONE
  histogram backend — the near-tie flakiness caveat in Task 8 Step 5 becomes
  a hard bug signal instead of a triage case: identical quantized sums must
  yield identical argmax decisions.
- Task 2's snapshot test is unchanged but now valid (fits are reproducible).
  It additionally serves as the determinism regression canary for the
  node-wise engine.

---

## File Structure

| file | role | tasks |
|---|---|---|
| `eml_boost/tree_split/_segmented.py` | NEW — per-segment GPU statistics (corr top-k, counts, min/max) | 1, 5 |
| `eml_boost/tree_split/tree.py` | `_PendingLeaf` + deferred finalize; `tree_growth` param + dispatch | 2, 4, 9, 13 |
| `eml_boost/tree_split/_leaf_batch.py` | NEW — batched leaf finalize | 3 |
| `eml_boost/tree_split/_multinode_hist.py` | NEW — fixed-point multi-node histogram + split decision | 6 |
| `eml_boost/_triton_exhaustive.py` | ADD row-wise multi-descriptor evaluator (torch ref + Triton) | 7 |
| `eml_boost/tree_split/_levelwise.py` | NEW — level-wise growth engine | 8, 9 |
| `eml_boost/tree_split/ensemble.py` | thread `tree_growth` through | 9, 13 |
| `tests/unit/test_segmented.py` | NEW | 1, 5 |
| `tests/unit/test_leaf_batch.py` | NEW | 2, 3, 4 |
| `tests/unit/test_multinode_hist.py` | NEW | 6 |
| `tests/unit/test_triton_exhaustive.py` | ADD row-wise evaluator oracle tests | 7 |
| `tests/unit/test_levelwise.py` | NEW — structural oracle, invariants, determinism, speed gate | 8, 9, 10, 11 |
| `experiments/run_experiment19_levelwise_parity.py` | NEW — parity runner (Exp-18 clone + `tree_growth`) | 12 |
| `experiments/experiment19/` | NEW — parity outputs + report | 13 |

Prediction paths, CPU pipeline, and prior experiment artifacts are NOT touched.

---

## Task 1: Segmented correlation top-k (`_segmented.py`)

**Files:**
- Create: `eml_boost/tree_split/_segmented.py`
- Test: `tests/unit/test_segmented.py`

**Interfaces:**
- Produces: `segment_counts(seg_id: torch.Tensor, n_segments: int) -> torch.Tensor` — (S,) float32 row counts per segment.
- Produces: `segment_corr(X: torch.Tensor, y: torch.Tensor, seg_id: torch.Tensor, n_segments: int) -> torch.Tensor` — (S, D) float32 |Pearson corr| per (segment, feature), math mirroring `EmlSplitTreeRegressor._top_features_by_corr_gpu` (two-pass centered, `+1e-12` denom guard).
- Produces: `segment_topk_corr(X, y, seg_id, n_segments, k) -> torch.Tensor` — (S, k) long, corr-descending order (`sorted=True`).
- Consumes: nothing project-internal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_segmented.py
"""Oracle tests for per-segment GPU statistics against per-node reference ops."""
import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_segments(n=3000, d=8, n_seg=7, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32, device="cuda")
    # y correlates with different features per segment so top-k is nontrivial
    seg = torch.tensor(rng.integers(0, n_seg, size=n), dtype=torch.long, device="cuda")
    y = torch.zeros(n, device="cuda")
    for s in range(n_seg):
        m = seg == s
        j = s % d
        y[m] = 3.0 * X[m, j] + 0.5 * X[m, (j + 1) % d]
    y = y + 0.01 * torch.tensor(rng.standard_normal(n), dtype=torch.float32, device="cuda")
    return X, y, seg, n_seg


@requires_cuda
def test_segment_counts_matches_bincount():
    from eml_boost.tree_split._segmented import segment_counts
    _X, _y, seg, n_seg = _make_segments()
    got = segment_counts(seg, n_seg)
    want = torch.bincount(seg, minlength=n_seg).float()
    assert torch.equal(got, want)


@requires_cuda
def test_segment_corr_matches_per_node_reference():
    from eml_boost.tree_split._segmented import segment_corr
    from eml_boost.tree_split.tree import EmlSplitTreeRegressor
    X, y, seg, n_seg = _make_segments()
    got = segment_corr(X, y, seg, n_seg)  # (S, D)
    for s in range(n_seg):
        m = seg == s
        Xs, ys = X[m], y[m]
        Xc = Xs - Xs.mean(dim=0, keepdim=True)
        yc = ys - ys.mean()
        num = (Xc * yc.unsqueeze(1)).sum(dim=0)
        denom = Xc.norm(dim=0) * yc.norm() + 1e-12
        want = (num / denom).abs()
        np.testing.assert_allclose(
            got[s].cpu().numpy(), want.cpu().numpy(), rtol=1e-4, atol=1e-6
        )


@requires_cuda
def test_segment_topk_corr_picks_reference_features():
    from eml_boost.tree_split._segmented import segment_topk_corr
    X, y, seg, n_seg = _make_segments()
    idx = segment_topk_corr(X, y, seg, n_seg, k=2)  # (S, 2)
    for s in range(n_seg):
        j = s % 8
        # segment s was built as 3*x_j + 0.5*x_{j+1}: top-1 must be j
        assert int(idx[s, 0]) == j
        assert int(idx[s, 1]) == (j + 1) % 8


@requires_cuda
def test_segment_corr_empty_segment_is_zero():
    from eml_boost.tree_split._segmented import segment_corr
    X, y, seg, n_seg = _make_segments()
    got = segment_corr(X, y, seg, n_seg + 3)  # 3 segments with no rows
    assert torch.isfinite(got).all()
    assert torch.equal(got[n_seg:], torch.zeros_like(got[n_seg:]))
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_segmented.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eml_boost.tree_split._segmented'`

- [ ] **Step 3: Implement `_segmented.py`**

```python
# eml_boost/tree_split/_segmented.py
"""Per-segment (per-node / per-leaf) GPU statistics for batched tree growth.

All helpers take a `seg_id` long tensor mapping each row to its segment and
compute one statistic per segment in O(1) kernel launches via `index_add_`
/ `scatter_reduce_`, replacing per-node Python loops.

Determinism note: `index_add_` on float32 CUDA uses atomic adds whose
accumulation order is nondeterministic; resulting corr values can wobble at
float32-ulp scale between runs. This only affects top-k selection on EXACT
corr ties (measure-zero on continuous data). Histogram statistics — the
load-bearing split decision — do NOT use this module's float path; they use
the fixed-point deterministic path in `_multinode_hist.py`.
"""

from __future__ import annotations

import torch


def segment_counts(seg_id: torch.Tensor, n_segments: int) -> torch.Tensor:
    """(S,) float32 row count per segment."""
    return torch.bincount(seg_id, minlength=n_segments).float()


def segment_corr(
    X: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    """|Pearson corr| per (segment, feature): (S, D) float32.

    Mirrors `EmlSplitTreeRegressor._top_features_by_corr_gpu` per segment:
    two-pass centered computation, denominator guarded with +1e-12.
    Segments with no rows return 0 for every feature.
    """
    n, d = X.shape
    device = X.device
    cnt = segment_counts(seg_id, n_segments).clamp(min=1.0)  # (S,)

    sum_x = torch.zeros(n_segments, d, device=device).index_add_(0, seg_id, X)
    mean_x = sum_x / cnt.unsqueeze(1)
    sum_y = torch.zeros(n_segments, device=device).index_add_(0, seg_id, y)
    mean_y = sum_y / cnt

    xc = X - mean_x[seg_id]
    yc = y - mean_y[seg_id]

    num = torch.zeros(n_segments, d, device=device).index_add_(
        0, seg_id, xc * yc.unsqueeze(1)
    )
    sq_x = torch.zeros(n_segments, d, device=device).index_add_(0, seg_id, xc * xc)
    sq_y = torch.zeros(n_segments, device=device).index_add_(0, seg_id, yc * yc)

    denom = sq_x.sqrt() * sq_y.sqrt().unsqueeze(1) + 1e-12
    return (num / denom).abs()


def segment_topk_corr(
    X: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
    k: int,
) -> torch.Tensor:
    """Top-k feature indices per segment by |corr|, corr-descending: (S, k) long.

    NOTE: returns `sorted=True` order (corr descending). The per-node
    reference uses `sorted=False`; order only matters for k > 1 where it
    changes descriptor semantics — an accepted statistically-neutral
    deviation (see plan Global Constraints).
    """
    corr = segment_corr(X, y, seg_id, n_segments)
    k_clipped = min(k, X.shape[1])
    return torch.topk(corr, k_clipped, dim=1, sorted=True).indices.to(torch.long)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/unit/test_segmented.py -v`
Expected: 4 PASS

- [ ] **Step 5: Full suite sanity + commit**

Run: `uv run pytest tests/unit/ -q` — expected: no new failures (1 pre-existing).

```bash
git add eml_boost/tree_split/_segmented.py tests/unit/test_segmented.py
git commit -m "$(cat <<'EOF'
feat: segmented per-node GPU statistics module (Task 1 of levelwise plan)

segment_counts / segment_corr / segment_topk_corr compute per-segment
statistics in O(1) kernel launches via index_add_, oracle-tested against
_top_features_by_corr_gpu math per segment. Foundation for batched leaf
fitting (Stage 1) and level-wise growth (Stage 2).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Deferred leaf fitting — bit-exact refactor (`_PendingLeaf`)

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (imports, `__init__`, `fit`, `_fit_xy_gpu`, `_grow_gpu`)
- Test: `tests/unit/test_leaf_batch.py`

**Interfaces:**
- Produces: `_PendingLeaf` dataclass in `tree.py` with fields `indices: torch.Tensor` (GPU long) and `resolved: Node | None = None`.
- Produces: `EmlSplitTreeRegressor._finalize_leaves(root: Node) -> Node` — fits every `_PendingLeaf` in `self._pending_leaves`, patches the tree.
- Produces: instance attr `self._batched_leaves: bool` (set in `__init__`, NOT a constructor param; default `False` in this task, flipped `True` in Task 4). Tests toggle it directly.
- Consumes: existing `_fit_leaf(indices)`.

- [ ] **Step 1: Write the pre-change snapshot test (BEFORE touching tree.py)**

This pins today's exact predictions so the refactor can prove bit-exactness. It must PASS against the current code before the refactor, and keep passing after.

```python
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
```

- [ ] **Step 2: Capture the snapshot against CURRENT code**

Run: `uv run pytest tests/unit/test_leaf_batch.py -v` (twice)
Expected: first run SKIP ("snapshot captured"), second run PASS.

```bash
git add tests/unit/test_leaf_batch.py tests/unit/fixtures/leaf_deferral_snapshot.npy
git commit -m "$(cat <<'EOF'
test: capture pre-refactor leaf-fit snapshot (Task 2 of levelwise plan)

Pins exact predictions of an 8-round GPU boost fit before the leaf-fit
deferral refactor, so the refactor can prove bit-exactness.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Implement the deferral in `tree.py`**

3a. Add near the top of `tree.py` (after imports):

```python
from dataclasses import dataclass


@dataclass
class _PendingLeaf:
    """Placeholder emitted during growth; resolved by _finalize_leaves.

    Never escapes fit(): _finalize_leaves replaces every instance before
    _tensorize_tree / predict can observe it.
    """

    indices: "torch.Tensor"  # GPU long tensor of the leaf's row indices
    resolved: "Node | None" = None
```

3b. In `EmlSplitTreeRegressor.__init__`, at the end, add:

```python
        # Stage-1 (levelwise plan): leaf fits are deferred to _finalize_leaves.
        # False = per-leaf reference path (_fit_leaf); True = batched path
        # (_leaf_batch.fit_leaves_batched, Task 3). Not a constructor param:
        # tests toggle the instance attribute directly.
        self._batched_leaves = False
        self._pending_leaves: list[_PendingLeaf] = []
```

3c. In `_grow_gpu`, replace all three `return self._fit_leaf(indices)` call sites (the depth/min-samples early-out, the `best is None` case, and the child-count re-check) with:

```python
            return self._make_pending_leaf(indices)
```

and add the helper method:

```python
    def _make_pending_leaf(self, indices: "torch.Tensor") -> "_PendingLeaf":
        p = _PendingLeaf(indices=indices)
        self._pending_leaves.append(p)
        return p
```

3d. Add the finalize + patch methods:

```python
    def _finalize_leaves(self, root: "Node") -> "Node":
        """Fit every _PendingLeaf collected during growth, then patch the
        tree. Reference path fits leaves one at a time via _fit_leaf in
        creation (DFS) order — identical calls to the pre-deferral code,
        so results are bit-exact. use_stacked_blend always takes the
        reference path (batched covers the gated policy only)."""
        pending = self._pending_leaves
        if not pending:
            return root
        use_batched = self._batched_leaves and not self.use_stacked_blend
        if use_batched:
            from eml_boost.tree_split._leaf_batch import fit_leaves_batched

            fitted = fit_leaves_batched(self, pending)
        else:
            fitted = [self._fit_leaf(p.indices) for p in pending]
        for p, node in zip(pending, fitted):
            p.resolved = node
        self._pending_leaves = []
        return self._replace_pending(root)

    @classmethod
    def _replace_pending(cls, node: "Node") -> "Node":
        if isinstance(node, _PendingLeaf):
            assert node.resolved is not None
            return node.resolved
        if isinstance(node, InternalNode):
            node.left = cls._replace_pending(node.left)
            node.right = cls._replace_pending(node.right)
        return node
```

3e. In `fit()`, change the GPU branch:

```python
            self._root: Node = self._grow_gpu(indices_gpu, depth=0, rng=rng)
            self._root = self._finalize_leaves(self._root)
            self._gpu_tree = self._tensorize_tree(self._root)
```

3f. In `_fit_xy_gpu()`, apply the same change:

```python
        self._root: Node = self._grow_gpu(indices_gpu, depth=0, rng=rng)
        self._root = self._finalize_leaves(self._root)
        self._gpu_tree = self._tensorize_tree(self._root)
```

(In both methods `_finalize_leaves` runs BEFORE the `self._X_gpu = None` release block — `_fit_leaf` needs the GPU handles. Also reset `self._pending_leaves = []` at the top of both `fit` and `_fit_xy_gpu` alongside `self._leaf_stats = []`.)

- [ ] **Step 4: Verify bit-exactness + suite**

Run: `uv run pytest tests/unit/test_leaf_batch.py -v`
Expected: PASS (`assert_array_equal` against the pre-refactor snapshot — exact).

Run: `uv run pytest tests/unit/ -q`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_leaf_batch.py
git commit -m "$(cat <<'EOF'
refactor: defer leaf fitting to post-growth _finalize_leaves (Task 2)

_grow_gpu now emits _PendingLeaf placeholders; _finalize_leaves fits them
after the skeleton completes (per-leaf reference path, DFS order) and
patches the tree. Bit-exact vs pre-refactor snapshot: leaf fits never
consumed the shared tree RNG (seed = leaf's first row index), so the
timing move changes nothing. Prepares Stage-1 batched finalize.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Batched leaf finalize (`_leaf_batch.py`)

**Files:**
- Create: `eml_boost/tree_split/_leaf_batch.py`
- Test: `tests/unit/test_leaf_batch.py` (extend)

**Interfaces:**
- Produces: `fit_leaves_batched(tree: "EmlSplitTreeRegressor", pending: list["_PendingLeaf"]) -> list["Node"]` — returns fitted nodes aligned with `pending` order.
- Consumes: `segment_counts`, `segment_topk_corr` (Task 1); `evaluate_trees_triton`, `get_descriptor_np/gpu`, `get_feature_mask_gpu` (`_triton_exhaustive.py`); tree attrs `_X_gpu, _y_gpu, _device, _global_mean_gpu, _global_std_gpu, _X_cpu`, hyperparams `k_leaf_eml, min_samples_leaf_eml, leaf_eml_gain_threshold, leaf_eml_ridge, leaf_eml_cap_k, leaf_l2`; `LeafNode, EmlLeafNode, SnappedTree`.

- [ ] **Step 1: Write the failing component A/B test**

Append to `tests/unit/test_leaf_batch.py`:

```python
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
    from eml_boost.tree_split.tree import _PendingLeaf
    from eml_boost.tree_split.nodes import EmlLeafNode, LeafNode

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
    for r, g in zip(ref, got):
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
    from eml_boost.tree_split.tree import _PendingLeaf
    from eml_boost.tree_split.nodes import LeafNode

    X, y = _friedman(n=200, seed=4)
    t = _manual_tree(X, y)
    assert fit_leaves_batched(t, []) == []
    empty = _PendingLeaf(indices=torch.empty(0, dtype=torch.long, device="cuda"))
    (got,) = fit_leaves_batched(t, [empty])
    assert isinstance(got, LeafNode) and got.value == 0.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_leaf_batch.py -v -k batched`
Expected: FAIL with `ModuleNotFoundError: No module named 'eml_boost.tree_split._leaf_batch'`

- [ ] **Step 3: Implement `_leaf_batch.py`**

```python
# eml_boost/tree_split/_leaf_batch.py
"""Batched leaf finalization for Stage 1 of the level-wise growth plan.

Replaces ~250 sequential `_fit_leaf` calls per tree (each ~30 kernel
launches + 2-3 device syncs) with one batched pass over all pending
leaves: segmented constant values, segmented top-k correlation, one
Triton evaluation of the shared descriptor set over all leaf rows,
segmented OLS + vectorized gate, and a single readback.

Semantics mirror `EmlSplitTreeRegressor._fit_leaf` (gated policy) exactly;
see that method for the reference formulas. The stacked-blend policy is
NOT handled here — `_finalize_leaves` routes it to the per-leaf path.
"""

from __future__ import annotations

import numpy as np
import torch

from eml_boost._triton_exhaustive import (
    evaluate_trees_triton,
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split._segmented import segment_topk_corr
from eml_boost.tree_split.nodes import EmlLeafNode, LeafNode, Node

# Chunk the (n_trees × total_rows) prediction matrix past this many rows
# to bound peak memory (~144 trees × 500k rows × 4B ≈ 288 MB per matrix).
_CHUNK_ROWS = 500_000


def fit_leaves_batched(tree, pending) -> list[Node]:
    if len(pending) == 0:
        return []
    device = tree._device
    assert device is not None and tree._X_gpu is not None
    y_gpu = tree._y_gpu
    leaf_l2 = float(tree.leaf_l2)
    L = len(pending)

    sizes = [int(p.indices.shape[0]) for p in pending]  # shape metadata, no sync
    n_raw = tree._X_cpu.shape[1] if tree._X_cpu is not None else 0

    # ---- constant values for ALL leaves (one segmented pass) ----
    nonempty = [i for i, sz in enumerate(sizes) if sz > 0]
    const_vals = np.zeros(L, dtype=np.float64)
    if nonempty:
        rows_all = torch.cat([pending[i].indices for i in nonempty])
        rep = torch.tensor(
            [sizes[i] for i in nonempty], dtype=torch.long, device=device
        )
        seg_all = torch.repeat_interleave(
            torch.arange(len(nonempty), dtype=torch.long, device=device), rep
        )
        sum_y = torch.zeros(len(nonempty), device=device).index_add_(
            0, seg_all, y_gpu[rows_all]
        )
        n_t = rep.float()
        cv = (sum_y / (n_t + leaf_l2)).cpu().numpy()
        for slot, i in enumerate(nonempty):
            const_vals[i] = float(cv[slot])

    # ---- eligibility (pure CPU metadata; mirrors _fit_leaf's early-outs) ----
    def _eligible(sz: int) -> bool:
        if tree.k_leaf_eml <= 0 or sz < tree.min_samples_leaf_eml or n_raw == 0:
            return False
        val_sz = max(sz // 4, 5)
        return sz - val_sz >= tree.min_samples_leaf_eml // 2

    e_ids = [i for i, sz in enumerate(sizes) if sz > 0 and _eligible(sz)]
    e_id_set = set(e_ids)
    out: list[Node | None] = [None] * L
    for i in range(L):
        if i not in e_id_set:
            out[i] = LeafNode(value=float(const_vals[i]))
    if not e_ids:
        return [n for n in out]  # type: ignore[list-item]

    # ---- eligible sub-batch ----
    E = len(e_ids)
    k = min(tree.k_leaf_eml, n_raw)
    rows_e = torch.cat([pending[i].indices for i in e_ids])
    sizes_e = [sizes[i] for i in e_ids]
    rep_e = torch.tensor(sizes_e, dtype=torch.long, device=device)
    seg_e = torch.repeat_interleave(
        torch.arange(E, dtype=torch.long, device=device), rep_e
    )
    Ne = int(rows_e.shape[0])
    X_e = tree._X_gpu[rows_e]  # (Ne, d)
    y_e = y_gpu[rows_e]  # (Ne,)

    # Per-leaf top-k features (corr-descending; identical to reference at k=1).
    top_feats = segment_topk_corr(X_e, y_e, seg_e, E, k)  # (E, k)
    mean_x = tree._global_mean_gpu[top_feats]  # (E, k)
    std_x = tree._global_std_gpu[top_feats]  # (E, k)

    Xsub = torch.clamp(
        (X_e.gather(1, top_feats[seg_e]) - mean_x[seg_e]) / std_x[seg_e], -3.0, 3.0
    )  # (Ne, k)

    # Per-leaf deterministic train/val masks (seed = first row index — the
    # exact rule _fit_leaf uses; independent of the shared tree RNG).
    firsts = (
        torch.stack([pending[i].indices[0] for i in e_ids]).cpu().numpy()
    )  # one small D2H
    fit_mask_np = np.empty(Ne, dtype=bool)
    pos = 0
    for slot, sz in enumerate(sizes_e):
        rng_leaf = np.random.default_rng(int(firsts[slot]))
        perm = rng_leaf.permutation(sz)
        val_sz = max(sz // 4, 5)
        m = np.ones(sz, dtype=bool)
        m[perm[:val_sz]] = False  # False = val row
        fit_mask_np[pos : pos + sz] = m
        pos += sz
    fit_f = torch.tensor(fit_mask_np, device=device).float()  # (Ne,)
    val_f = 1.0 - fit_f

    # cap per leaf: leaf_eml_cap_k * max|y_leaf| (order-independent amax).
    cap_k = float(tree.leaf_eml_cap_k)
    if cap_k > 0.0:
        cap = torch.full((E,), -float("inf"), device=device).scatter_reduce_(
            0, seg_e, y_e.abs(), reduce="amax", include_self=True
        ) * cap_k
    else:
        cap = torch.full((E,), float("inf"), device=device)

    # ---- evaluate the shared descriptor set over all leaf rows ----
    desc_gpu = get_descriptor_gpu(depth=2, k=k, device=device)  # (T, 6)
    feature_mask = get_feature_mask_gpu(depth=2, k=k, device=device)  # (T,)
    T = int(desc_gpu.shape[0])

    if Ne <= _CHUNK_ROWS:
        preds = evaluate_trees_triton(desc_gpu, Xsub, k)  # (T, Ne)
        preds_list = [(preds, seg_e, y_e, fit_f, val_f, 0)]
    else:
        preds_list = []
        for s in range(0, Ne, _CHUNK_ROWS):
            e = min(s + _CHUNK_ROWS, Ne)
            preds_list.append(
                (
                    evaluate_trees_triton(desc_gpu, Xsub[s:e], k),
                    seg_e[s:e],
                    y_e[s:e],
                    fit_f[s:e],
                    val_f[s:e],
                    s,
                )
            )

    def _acc(fn):
        acc = torch.zeros(T, E, device=device)
        for preds, seg, yy, ff, vf, _s in preds_list:
            acc.index_add_(1, seg, fn(preds, yy, ff, vf))
        return acc

    n_fit = torch.zeros(E, device=device).index_add_(0, seg_e, fit_f)  # (E,)
    sum_p = _acc(lambda p, yy, ff, vf: p * ff)
    sum_p2 = _acc(lambda p, yy, ff, vf: p * p * ff)
    sum_py = _acc(lambda p, yy, ff, vf: p * (yy * ff).unsqueeze(0))
    sum_y_f = torch.zeros(E, device=device).index_add_(0, seg_e, y_e * fit_f)
    bad = _acc(lambda p, yy, ff, vf: (~torch.isfinite(p)).float())  # any row (fit+val)

    # ---- closed-form OLS per (tree, leaf) — formulas verbatim from _fit_leaf ----
    n_fit_reg = n_fit + leaf_l2
    det = sum_p2 * n_fit - sum_p * sum_p
    lam = float(tree.leaf_eml_ridge)
    det_ridged = det + n_fit * lam
    det_safe = torch.where(
        det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
    )
    eta = (n_fit * sum_py - sum_p * sum_y_f) / det_safe
    if lam == 0.0:
        bias = (sum_p2 * sum_y_f - sum_p * sum_py) / det_safe
        if leaf_l2 > 0.0:
            bias = bias * n_fit / n_fit_reg
    else:
        bias = (sum_y_f - eta * sum_p) / n_fit_reg

    finite_coefs = torch.isfinite(eta) & torch.isfinite(bias)
    valid = (
        feature_mask.unsqueeze(1)
        & (bad == 0)
        & finite_coefs
        & (det.abs() > 1e-6)
    )  # (T, E)

    # ---- val SSE per (tree, leaf) with cap; constant val SSE per leaf ----
    val_sse = torch.zeros(T, E, device=device)
    const_sse = torch.zeros(E, device=device)
    mean_full = (
        torch.zeros(E, device=device).index_add_(0, seg_e, y_e)
        / torch.zeros(E, device=device).index_add_(0, seg_e, torch.ones_like(y_e))
    )
    for preds, seg, yy, ff, vf, _s in preds_list:
        vp = eta[:, seg] * preds + bias[:, seg]
        vp = torch.clamp(vp, min=-cap[seg], max=cap[seg])
        res = (yy.unsqueeze(0) - vp) * vf
        val_sse.index_add_(1, seg, res * res)
        cres = (yy - mean_full[seg]) * vf
        const_sse.index_add_(0, seg, cres * cres)
    val_sse = torch.where(valid, val_sse, torch.full_like(val_sse, float("inf")))

    # ---- gate + select + ONE readback ----
    best_idx = val_sse.argmin(dim=0)  # (E,)
    ar = torch.arange(E, device=device)
    best_sse = val_sse[best_idx, ar]
    valid_b = valid[best_idx, ar]
    eta_b = eta[best_idx, ar]
    bias_b = bias[best_idx, ar]
    thr = float(tree.leaf_eml_gain_threshold)
    accept = valid_b & (best_sse < const_sse * (1.0 - thr))

    best_idx_np = best_idx.cpu().numpy()
    accept_np = accept.cpu().numpy()
    eta_np = eta_b.cpu().numpy()
    bias_np = bias_b.cpu().numpy()
    cap_np = cap.cpu().numpy()
    feats_np = top_feats.cpu().numpy()
    mean_np = mean_x.cpu().numpy()
    std_np = std_x.cpu().numpy()

    desc_np = get_descriptor_np(2, k)
    for slot, i in enumerate(e_ids):
        if not bool(accept_np[slot]):
            out[i] = LeafNode(value=float(const_vals[i]))
            continue
        drow = desc_np[int(best_idx_np[slot])]
        out[i] = EmlLeafNode(
            snapped=SnappedTree(
                depth=2,
                k=k,
                internal_input_count=2,
                leaf_input_count=4,
                terminal_choices=tuple(int(v) for v in drow),
            ),
            feature_subset=tuple(int(v) for v in feats_np[slot]),
            feature_mean=tuple(float(v) for v in mean_np[slot]),
            feature_std=tuple(float(v) for v in std_np[slot]),
            eta=float(eta_np[slot]),
            bias=float(bias_np[slot]),
            cap=float(cap_np[slot]),
        )
    return [n for n in out]  # type: ignore[list-item]
```

Implementation notes for the engineer:
- `val_sse >= const_sse * (1 - thr)` rejects in the reference; here `accept` uses the strict complement `<` — keep it strict-`<` to match `_fit_leaf` exactly.
- The reference's `constant_val_sse` uses the UNSHRUNK leaf mean (`y_full.mean()`), while the stored constant value is the l2-shrunk `sum/(n+l2)` — both reproduced here (`mean_full` vs `const_vals`). Do not "fix" this asymmetry; it is the reference behavior.
- `pending[i].indices[0]` per leaf and `torch.stack` produce one small D2H read for all seeds; int64 end-to-end (the reference's float32 round-trip + `n < 2**24` assert is not needed on this path).

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/unit/test_leaf_batch.py -v`
Expected: all PASS (snapshot test still pinned to reference path).

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest tests/unit/ -q` — no new failures.

```bash
git add eml_boost/tree_split/_leaf_batch.py tests/unit/test_leaf_batch.py
git commit -m "$(cat <<'EOF'
feat: batched leaf finalize (Task 3 of levelwise plan)

fit_leaves_batched fits every pending leaf in one pass: segmented
constants, segmented top-k corr, one Triton eval of the shared 144-
descriptor set over all leaf rows (chunked past 500k), segmented OLS +
vectorized gate, single readback. A/B-tested against _fit_leaf on mixed
leaf sizes incl. ridge/capless variants: identical types and choices,
params within float32 reduction tolerance.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Enable batched leaves by default + integration A/B + Stage-1 bench

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (one line: `self._batched_leaves = True`)
- Test: `tests/unit/test_leaf_batch.py` (extend)

**Interfaces:**
- Produces: batched finalize is the default for every GPU fit (`_batched_leaves = True`).
- Consumes: Tasks 2-3.

- [ ] **Step 1: Write the failing integration A/B test**

Append to `tests/unit/test_leaf_batch.py`:

```python
def _walk_pairs(a, b):
    """Yield (node_a, node_b) over two trees in lockstep; fail on shape mismatch."""
    from eml_boost.tree_split.nodes import InternalNode

    stack = [(a, b)]
    while stack:
        x, z = stack.pop()
        assert type(x) is type(z), f"structure diverged: {type(x)} vs {type(z)}"
        yield x, z
        if isinstance(x, InternalNode):
            stack.append((x.left, z.left))
            stack.append((x.right, z.right))


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1])
def test_full_fit_batched_vs_reference_leaves(seed):
    """End-to-end boost fit: batched vs reference leaf finalize.
    Split structure must be identical (split path untouched); leaf types
    identical; leaf params and predictions within float32 tolerance."""
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
                max_rounds=6, max_depth=6, patience=0, use_gpu=True,
                random_state=seed,
            )
            m.fit(X, y)
        finally:
            tree_mod.EmlSplitTreeRegressor.__init__ = orig_init
        return m

    m_ref = _fit(False)
    m_bat = _fit(True)

    n_eml_leaves = 0
    for tr, tb in zip(m_ref._trees, m_bat._trees):
        for nr, nb in _walk_pairs(tr._root, tb._root):
            if isinstance(nr, InternalNode):
                assert type(nr.split) is type(nb.split)
                np.testing.assert_allclose(
                    nr.split.threshold, nb.split.threshold, rtol=0, atol=0
                )  # split path untouched -> exactly equal
            elif isinstance(nr, EmlLeafNode):
                n_eml_leaves += 1
                assert nr.snapped.terminal_choices == nb.snapped.terminal_choices
                np.testing.assert_allclose(nb.eta, nr.eta, rtol=1e-3, atol=1e-6)
            else:
                assert isinstance(nr, LeafNode)
                np.testing.assert_allclose(nb.value, nr.value, rtol=1e-4, atol=1e-6)
    assert n_eml_leaves > 0, "fixture produced no EML leaves; test is vacuous"

    pred_r = m_ref.predict(X[:800])
    pred_b = m_bat.predict(X[:800])
    np.testing.assert_allclose(pred_b, pred_r, rtol=1e-3, atol=1e-5)
```

- [ ] **Step 2: Run test — verify it currently passes only with default False (it fits both paths explicitly), then flip the default**

Run: `uv run pytest tests/unit/test_leaf_batch.py -v -k full_fit`
Expected: PASS already (the test pins both paths itself). Now flip the default in `tree.py` `__init__`:

```python
        self._batched_leaves = True
```

- [ ] **Step 3: Full suite**

Run: `uv run pytest tests/unit/ -q`
Expected: no new failures. NOTE: `test_leaf_deferral_matches_snapshot` (Task 2) pins `_batched_leaves = False` inside the test, so it keeps validating the reference path exactly — confirm it still passes.

- [ ] **Step 4: Stage-1 benchmark checkpoint**

Write and run `/tmp/stage1_bench.py` (throwaway, do not commit):

```python
import time
import numpy as np
import torch
from eml_boost.tree_split import EmlSplitBoostRegressor

rng = np.random.default_rng(0)
X = rng.standard_normal((32_000, 10))
y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + rng.standard_normal(32_000)
EmlSplitBoostRegressor(max_rounds=2, patience=0, random_state=0).fit(X, y)  # warmup
for _ in range(2):
    t0 = time.perf_counter()
    EmlSplitBoostRegressor(max_rounds=20, patience=0, random_state=0).fit(X, y)
    torch.cuda.synchronize()
    print(f"{(time.perf_counter() - t0) / 20:.4f} s/round")
```

Run: `uv run python /tmp/stage1_bench.py`
Expected: ≤ ~0.25 s/round (pre-change baseline: ~0.40 s/round at n=32k). Record the number for the commit message. If the win is < 1.4×, STOP and profile before proceeding (something in the batch path is not batched).

- [ ] **Step 5: Commit**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_leaf_batch.py
git commit -m "$(cat <<'EOF'
feat: enable batched leaf finalize by default (Task 4, Stage 1 complete)

Full-fit A/B test: identical split structure (exact), identical leaf
types/choices, params and predictions within float32 tolerance.
Stage-1 bench on 32k x 10 synthetic, 20 rounds, depth 8:
<X.XX> s/round (was ~0.40 pre-change) = <N.N>x.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Segmented min/max (`_segmented.py` additions)

**Files:**
- Modify: `eml_boost/tree_split/_segmented.py`
- Test: `tests/unit/test_segmented.py` (extend)

**Interfaces:**
- Produces: `segment_minmax(values: torch.Tensor, seg_id: torch.Tensor, n_segments: int) -> tuple[torch.Tensor, torch.Tensor]` — ((S, C) min, (S, C) max) float32 per (segment, column); empty segments get (+inf, −inf).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_segmented.py`:

```python
@requires_cuda
def test_segment_minmax_matches_loop():
    from eml_boost.tree_split._segmented import segment_minmax
    rng = np.random.default_rng(1)
    vals = torch.tensor(rng.standard_normal((5000, 4)), dtype=torch.float32, device="cuda")
    seg = torch.tensor(rng.integers(0, 6, size=5000), dtype=torch.long, device="cuda")
    mn, mx = segment_minmax(vals, seg, 8)  # segments 6, 7 empty
    for s in range(6):
        m = seg == s
        assert torch.equal(mn[s], vals[m].min(dim=0).values)
        assert torch.equal(mx[s], vals[m].max(dim=0).values)
    assert bool(torch.isinf(mn[6:]).all()) and bool(torch.isinf(mx[6:]).all())
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/unit/test_segmented.py -v -k minmax`
Expected: FAIL with `ImportError` (name `segment_minmax` not defined).

- [ ] **Step 3: Implement**

Append to `_segmented.py`:

```python
def segment_minmax(
    values: torch.Tensor, seg_id: torch.Tensor, n_segments: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(segment, column) min and max: ((S, C), (S, C)) float32.

    min/max are associative+commutative, so scatter_reduce_ is bitwise
    deterministic regardless of atomic ordering. Empty segments return
    (+inf, -inf) — callers must mask them (their bin width degenerates).
    """
    n, c = values.shape
    idx = seg_id.unsqueeze(1).expand(n, c)
    mn = torch.full(
        (n_segments, c), float("inf"), device=values.device
    ).scatter_reduce_(0, idx, values, reduce="amin", include_self=True)
    mx = torch.full(
        (n_segments, c), -float("inf"), device=values.device
    ).scatter_reduce_(0, idx, values, reduce="amax", include_self=True)
    return mn, mx
```

- [ ] **Step 4: Run tests, verify pass, commit**

Run: `uv run pytest tests/unit/test_segmented.py -v` — all PASS.

```bash
git add eml_boost/tree_split/_segmented.py tests/unit/test_segmented.py
git commit -m "$(cat <<'EOF'
feat: segment_minmax for per-node bin ranges (Task 5 of levelwise plan)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Deterministic multi-node histogram split (`_multinode_hist.py`)

**Files:**
- Create: `eml_boost/tree_split/_multinode_hist.py`
- Test: `tests/unit/test_multinode_hist.py`

**Interfaces:**
- Produces:
```python
def multinode_histogram_split(
    values: torch.Tensor,     # (N, C) float32 CUDA — candidate columns for all active rows
    y: torch.Tensor,          # (N,) float32
    seg_id: torch.Tensor,     # (N,) long — frontier slot per row
    n_segments: int,
    n_bins: int,
    min_leaf_count: int,
    leaf_l2: float,
    col_valid: torch.Tensor | None = None,  # (S, C) bool; False masks a column out
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (S,) long best_col, (S,) float32 best_threshold, (S,) float32 best_gain
    # gain <= 0 means "no legal split" for that segment.
```
- Consumes: `segment_minmax` (Task 5). Math mirrors `gpu_histogram_split_torch` (`_gpu_split.py:40`) exactly, generalized over segments, with fixed-point deterministic accumulation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_multinode_hist.py
"""Multi-node fixed-point histogram split vs the single-node torch oracle."""
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
        # Same winning column when the oracle's win margin is decisive;
        # thresholds/gains within fixed-point + float32 tolerance.
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_multinode_hist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eml_boost.tree_split._multinode_hist'`

- [ ] **Step 3: Implement `_multinode_hist.py`**

```python
# eml_boost/tree_split/_multinode_hist.py
"""Deterministic multi-node histogram split-finding.

Generalizes `gpu_histogram_split_torch` over a batch of frontier nodes:
one scatter pass builds (n_segments, n_cols, n_bins) histograms for every
node at a tree level, one reduction finds each node's best (column, bin).

Determinism: float atomic adds are accumulation-order nondeterministic,
which would break same-seed reproducibility of fitted trees. y and y**2
are therefore quantized to int64 fixed-point before scatter_add_ —
integer addition is associative, so histogram totals are bit-identical
regardless of row order. Quantization scale is 2**20 relative to the
global max|y| (max is order-independent), giving ~1e-6 relative error on
bin sums — far below the O(1/n_bins) threshold-placement error inherent
to histogram splitting.

Gain math (incl. leaf_l2 and min-leaf legality) mirrors
`gpu_histogram_split_torch` line for line; see _gpu_split.py.
"""

from __future__ import annotations

import torch

from eml_boost.tree_split._segmented import segment_minmax

_FP_BITS = 20  # fixed-point fractional bits for y-sum quantization


def multinode_histogram_split(
    values: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
    n_bins: int,
    min_leaf_count: int,
    leaf_l2: float,
    col_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = values.shape
    device = values.device
    S, B = n_segments, n_bins

    no_split = (
        torch.zeros(S, dtype=torch.long, device=device),
        torch.zeros(S, device=device),
        torch.zeros(S, device=device),
    )
    if n == 0:
        return no_split

    # Per-(segment, column) uniform bins — same rule as the single-node path.
    vmin, vmax = segment_minmax(values, seg_id, S)  # (S, C) each
    empty = torch.isinf(vmin)  # empty segment marker
    vmin = torch.where(empty, torch.zeros_like(vmin), vmin)
    vmax = torch.where(empty, torch.zeros_like(vmax), vmax)
    rng_ = (vmax - vmin).clamp(min=1e-12)
    bin_width = rng_ / B  # (S, C)

    bin_idx = torch.clamp(
        ((values - vmin[seg_id]) / bin_width[seg_id]).long(), min=0, max=B - 1
    )  # (N, C)

    # Fixed-point quantization of y and y^2 (global scale, order-independent).
    y_absmax = y.abs().max().clamp(min=1e-12)
    scale = float(1 << _FP_BITS) / float(y_absmax.item())
    y_q = torch.round(y * scale).to(torch.int64)  # |y_q| <= 2^20
    y2_q = torch.round((y * y) * scale).to(torch.int64)  # <= 2^20 * max|y|

    col_offs = torch.arange(c, device=device) * B  # (C,)
    flat = (seg_id.unsqueeze(1) * (c * B) + col_offs + bin_idx).reshape(-1)  # (N*C,)

    slots = S * c * B
    hist_sum = torch.zeros(slots, dtype=torch.int64, device=device).scatter_add_(
        0, flat, y_q.unsqueeze(1).expand(n, c).reshape(-1)
    )
    hist_sq = torch.zeros(slots, dtype=torch.int64, device=device).scatter_add_(
        0, flat, y2_q.unsqueeze(1).expand(n, c).reshape(-1)
    )
    hist_cnt = torch.zeros(slots, dtype=torch.int64, device=device).scatter_add_(
        0, flat, torch.ones(n * c, dtype=torch.int64, device=device)
    )

    hist_sum = hist_sum.view(S, c, B).float() / scale
    hist_sq = hist_sq.view(S, c, B).float() / scale
    hist_cnt = hist_cnt.view(S, c, B).float()

    c_sum = torch.cumsum(hist_sum, dim=2)
    c_sq = torch.cumsum(hist_sq, dim=2)
    c_cnt = torch.cumsum(hist_cnt, dim=2)

    total_sum = c_sum[:, :, -1:]
    total_sq = c_sq[:, :, -1:]
    total_cnt = c_cnt[:, :, -1:]
    total_sse = total_sq - total_sum**2 / (total_cnt.clamp(min=1.0) + leaf_l2)

    left_cnt = c_cnt[:, :, :-1]
    left_sum = c_sum[:, :, :-1]
    left_sq = c_sq[:, :, :-1]
    right_cnt = total_cnt - left_cnt
    right_sum = total_sum - left_sum
    right_sq = total_sq - left_sq

    legal = (left_cnt >= min_leaf_count) & (right_cnt >= min_leaf_count)
    if col_valid is not None:
        legal = legal & col_valid.unsqueeze(2)
    left_sse = left_sq - left_sum**2 / (left_cnt.clamp(min=1.0) + leaf_l2)
    right_sse = right_sq - right_sum**2 / (right_cnt.clamp(min=1.0) + leaf_l2)
    gain = total_sse - left_sse - right_sse
    gain = torch.where(legal, gain, torch.full_like(gain, float("-inf")))

    flat_gain = gain.reshape(S, c * (B - 1))
    flat_best = torch.argmax(flat_gain, dim=1)  # (S,)
    best_gain = flat_gain.gather(1, flat_best.unsqueeze(1)).squeeze(1)
    best_col = flat_best // (B - 1)
    best_bin = flat_best % (B - 1)

    ar = torch.arange(S, device=device)
    thr = vmin[ar, best_col] + bin_width[ar, best_col] * (best_bin + 1).float()

    found = torch.isfinite(best_gain) & (best_gain > 0)
    best_gain = torch.where(found, best_gain, torch.zeros_like(best_gain))
    thr = torch.where(found, thr, torch.zeros_like(thr))
    best_col = torch.where(found, best_col, torch.zeros_like(best_col))
    return best_col, thr, best_gain
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/unit/test_multinode_hist.py -v`
Expected: all PASS. If `test_multinode_matches_single_node_oracle` fails on the winning column: check whether the oracle's top-2 gains are within `5e-3` relative of each other (a genuine near-tie the fixed-point shift can legally flip). If so, regenerate the fixture seed to avoid the tie and note it; if the gains differ decisively, it is a real bug — debug before proceeding.

- [ ] **Step 5: Commit**

```bash
git add eml_boost/tree_split/_multinode_hist.py tests/unit/test_multinode_hist.py
git commit -m "$(cat <<'EOF'
feat: deterministic multi-node histogram split (Task 6 of levelwise plan)

One scatter pass builds (nodes x cols x bins) histograms for a whole
tree level; int64 fixed-point accumulation (2^20 scale vs global max|y|)
makes totals bit-identical under any row order. Gain math mirrors
gpu_histogram_split_torch incl. leaf_l2 + min-leaf legality. Oracle-
tested per segment; shuffle-determinism and mask/edge tests included.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Row-wise multi-descriptor evaluator (`_triton_exhaustive.py`)

**Files:**
- Modify: `eml_boost/_triton_exhaustive.py`
- Test: `tests/unit/test_triton_exhaustive.py` (extend)

**Interfaces:**
- Produces:
```python
def evaluate_trees_torch_nodewise(
    desc_nodes: torch.Tensor,  # (L, C, 6) int32 — C descriptors per frontier node
    node_of: torch.Tensor,     # (N,) long — frontier slot per row
    X: torch.Tensor,           # (N, k) float32 — per-row feature values
    k: int,
) -> torch.Tensor:             # (C, N) float32: out[c, i] = eval(desc_nodes[node_of[i], c], X[i])

def evaluate_trees_triton_nodewise(desc_nodes, node_of, X, k) -> torch.Tensor
    # same contract; Triton kernel with fallback to the torch version when
    # Triton unavailable, X not on CUDA, or k > _MAX_K.
```
- Consumes: existing `evaluate_trees_torch_per_sample` (reference), kernel idioms from `_eval_depth2_kernel`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_triton_exhaustive.py`:

```python
def test_evaluate_trees_torch_nodewise_matches_per_sample():
    import numpy as np
    import torch
    from eml_boost._triton_exhaustive import (
        evaluate_trees_torch_nodewise,
        evaluate_trees_torch_per_sample,
        get_valid_descriptors_np,
    )

    rng = np.random.default_rng(0)
    L, C, k, N = 7, 10, 3, 4000
    valid = get_valid_descriptors_np(2, k)
    desc_nodes = torch.tensor(
        valid[rng.integers(0, len(valid), size=(L, C))], dtype=torch.int32
    )  # (L, C, 6)
    node_of = torch.tensor(rng.integers(0, L, size=N), dtype=torch.long)
    X = torch.tensor(rng.standard_normal((N, k)), dtype=torch.float32)

    got = evaluate_trees_torch_nodewise(desc_nodes, node_of, X, k)  # (C, N)
    assert got.shape == (C, N)
    for c in range(C):
        want = evaluate_trees_torch_per_sample(desc_nodes[node_of, c, :], X, k)
        np.testing.assert_allclose(
            got[c].numpy(), want.numpy(), rtol=1e-5, atol=1e-6
        )


def test_evaluate_trees_triton_nodewise_matches_torch():
    import numpy as np
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from eml_boost._triton_exhaustive import (
        evaluate_trees_torch_nodewise,
        evaluate_trees_triton_nodewise,
        get_valid_descriptors_np,
    )

    rng = np.random.default_rng(1)
    L, C, k, N = 12, 10, 3, 50_000
    valid = get_valid_descriptors_np(2, k)
    desc_nodes = torch.tensor(
        valid[rng.integers(0, len(valid), size=(L, C))],
        dtype=torch.int32, device="cuda",
    )
    node_of = torch.tensor(rng.integers(0, L, size=N), dtype=torch.long, device="cuda")
    X = torch.tensor(
        rng.standard_normal((N, k)), dtype=torch.float32, device="cuda"
    )

    got = evaluate_trees_triton_nodewise(desc_nodes, node_of, X, k)
    want = evaluate_trees_torch_nodewise(desc_nodes, node_of, X, k)
    # exp(50)-scale values possible; rtol comparison handles magnitude.
    np.testing.assert_allclose(
        got.cpu().numpy(), want.cpu().numpy(), rtol=1e-3, atol=1e-4
    )
    # Guard against silent fallback: outputs must come from the kernel path.
    assert got.is_cuda
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_triton_exhaustive.py -v -k nodewise`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement (append to `_triton_exhaustive.py`)**

```python
def evaluate_trees_torch_nodewise(
    desc_nodes: torch.Tensor,
    node_of: torch.Tensor,
    X: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Torch reference: per-row evaluation of that row's node's C descriptors.

    desc_nodes: (L, C, 6) int32; node_of: (N,) long; X: (N, k).
    Returns (C, N): out[c, i] = eval(desc_nodes[node_of[i], c], X[i]).
    """
    C = desc_nodes.shape[1]
    n = X.shape[0]
    out = torch.empty(C, n, dtype=X.dtype, device=X.device)
    per_row = desc_nodes[node_of]  # (N, C, 6)
    for c in range(C):
        out[c] = evaluate_trees_torch_per_sample(per_row[:, c, :], X, k)
    return out


if _TRITON_AVAILABLE:

    @triton.jit
    def _eval_depth2_nodewise_kernel(
        X_ptr,
        desc_ptr,      # (L, C, 6) int32 contiguous
        node_ptr,      # (N,) int32
        out_ptr,       # (C, N) float32
        n_samples,
        C: tl.constexpr,
        X_stride_sample,
        K: tl.constexpr,
        EXP_CLAMP: tl.constexpr,
        LOG_EPS: tl.constexpr,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        """One program per (candidate c, sample block). Each row loads its
        node id, then its node's c-th descriptor (6 ints), selects terminals
        via the same constexpr-unrolled mask arithmetic as
        _eval_depth2_kernel, and writes out[c, i]."""
        pid_c = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs = pid_s * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        mask = offs < n_samples

        node = tl.load(node_ptr + offs, mask=mask, other=0)
        dbase = (node * C + pid_c) * 6
        c0 = tl.load(desc_ptr + dbase + 0, mask=mask, other=0)
        c1 = tl.load(desc_ptr + dbase + 1, mask=mask, other=0)
        c2 = tl.load(desc_ptr + dbase + 2, mask=mask, other=0)
        c3 = tl.load(desc_ptr + dbase + 3, mask=mask, other=0)
        c4 = tl.load(desc_ptr + dbase + 4, mask=mask, other=0)
        c5 = tl.load(desc_ptr + dbase + 5, mask=mask, other=0)

        one = tl.full((BLOCK_SAMPLES,), 1.0, dtype=tl.float32)
        feat_base = X_ptr + offs * X_stride_sample

        v_c2 = one * (c2 == 0).to(tl.float32)
        v_c3 = one * (c3 == 0).to(tl.float32)
        v_c4 = one * (c4 == 0).to(tl.float32)
        v_c5 = one * (c5 == 0).to(tl.float32)
        left = one * (c0 == 0).to(tl.float32)
        right = one * (c1 == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=mask, other=0.0)
            v_c2 = v_c2 + (c2 == (j + 1)).to(tl.float32) * x_j
            v_c3 = v_c3 + (c3 == (j + 1)).to(tl.float32) * x_j
            v_c4 = v_c4 + (c4 == (j + 1)).to(tl.float32) * x_j
            v_c5 = v_c5 + (c5 == (j + 1)).to(tl.float32) * x_j
            left = left + (c0 == (j + 1)).to(tl.float32) * x_j
            right = right + (c1 == (j + 1)).to(tl.float32) * x_j

        node_0 = tl.exp(tl.minimum(tl.maximum(v_c2, -EXP_CLAMP), EXP_CLAMP)) - tl.log(
            tl.maximum(v_c3, LOG_EPS)
        )
        node_1 = tl.exp(tl.minimum(tl.maximum(v_c4, -EXP_CLAMP), EXP_CLAMP)) - tl.log(
            tl.maximum(v_c5, LOG_EPS)
        )
        left = left + (c0 == (K + 1)).to(tl.float32) * node_0
        right = right + (c1 == (K + 1)).to(tl.float32) * node_1

        out = tl.exp(tl.minimum(tl.maximum(left, -EXP_CLAMP), EXP_CLAMP)) - tl.log(
            tl.maximum(right, LOG_EPS)
        )
        tl.store(out_ptr + pid_c * n_samples + offs, out, mask=mask)


def evaluate_trees_triton_nodewise(
    desc_nodes: torch.Tensor,
    node_of: torch.Tensor,
    X: torch.Tensor,
    k: int,
    block_samples: int = 256,
) -> torch.Tensor:
    """Triton row-wise multi-descriptor evaluator; falls back to torch."""
    if not _TRITON_AVAILABLE or k > _MAX_K or not X.is_cuda:
        return evaluate_trees_torch_nodewise(desc_nodes, node_of, X, k)
    desc_nodes = desc_nodes.to(torch.int32).contiguous()
    node_of32 = node_of.to(torch.int32).contiguous()
    X = X.contiguous().to(torch.float32)
    L, C, _ = desc_nodes.shape
    n = X.shape[0]
    out = torch.empty(C, n, device=X.device, dtype=torch.float32)
    if n == 0:
        return out
    grid = (C, triton.cdiv(n, block_samples))
    _eval_depth2_nodewise_kernel[grid](
        X,
        desc_nodes,
        node_of32,
        out,
        n,
        C=C,
        X_stride_sample=X.stride(0),
        K=k,
        EXP_CLAMP=_EXP_CLAMP,
        LOG_EPS=_LOG_EPS,
        BLOCK_SAMPLES=block_samples,
    )
    return out
```

- [ ] **Step 4: Run tests, verify pass, full suite, commit**

Run: `uv run pytest tests/unit/test_triton_exhaustive.py -v` then `uv run pytest tests/unit/ -q`

```bash
git add eml_boost/_triton_exhaustive.py tests/unit/test_triton_exhaustive.py
git commit -m "$(cat <<'EOF'
feat: row-wise multi-descriptor EML evaluator (Task 7 of levelwise plan)

evaluate_trees_{torch,triton}_nodewise evaluate, for every active row,
its frontier node's C candidate descriptors in one launch: the level-
wise engine's step-4 primitive. Kernel reuses _eval_depth2_kernel's
constexpr-unrolled terminal selection with per-row descriptor
indirection via node_of. Oracle-tested against per-sample evaluator.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Level-wise engine, no-EML mode (`_levelwise.py`)

**Files:**
- Create: `eml_boost/tree_split/_levelwise.py`
- Test: `tests/unit/test_levelwise.py`

**Interfaces:**
- Produces: `grow_levelwise(tree: "EmlSplitTreeRegressor", indices: torch.Tensor, rng: np.random.Generator) -> "Node"` — full tree via breadth-first batched growth; leaves emitted through `tree._make_pending_leaf` (Stage-1 finalize handles them).
- Consumes: `segment_counts` (T1), `multinode_histogram_split` (T6), `evaluate_trees_triton_nodewise` (T7, used from Task 9 onward), `segment_topk_corr` (T1), `get_valid_descriptors_np`, node classes, `tree._make_pending_leaf`.

This task implements the FULL engine but exercises/validates only the no-EML path (`n_eml_candidates=0`), where the RNG is never consumed and structure must match the node-wise engine exactly. Task 9 turns on the EML path.

- [ ] **Step 1: Write the failing structural-oracle test**

```python
# tests/unit/test_levelwise.py
"""Level-wise growth engine: structural oracle, invariants, determinism, speed."""
import numpy as np
import pytest
import torch

from eml_boost.tree_split import EmlSplitBoostRegressor
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode, LeafNode, RawSplit

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
    for rn, rl in zip(sig_n, sig_l):
        assert rn[0] == rl[0], f"shape diverged at {rn[0]} vs {rl[0]}"
        assert rn[1] == rl[1]
        if rn[1] == "raw":
            assert rn[2] == rl[2], f"split feature diverged at {rn[0]}"
            np.testing.assert_allclose(rl[3], rn[3], rtol=1e-4, atol=1e-5)
        elif rn[1] == "leaf":
            np.testing.assert_allclose(rl[2], rn[2], rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/unit/test_levelwise.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'tree_growth'`

- [ ] **Step 3: Add the `tree_growth` param to `EmlSplitTreeRegressor`**

In `tree.py` `__init__` signature, after `use_stacked_blend: bool = False,` add:

```python
        tree_growth: str = "nodewise",
```

and in the body:

```python
        if tree_growth not in ("nodewise", "levelwise"):
            raise ValueError(
                f"tree_growth must be 'nodewise' or 'levelwise', got {tree_growth!r}"
            )
        self.tree_growth = tree_growth
```

In BOTH `fit()` and `_fit_xy_gpu()`, replace the grow call:

```python
        if self.tree_growth == "levelwise":
            from eml_boost.tree_split._levelwise import grow_levelwise

            self._root: Node = grow_levelwise(self, indices_gpu, rng)
        else:
            self._root: Node = self._grow_gpu(indices_gpu, depth=0, rng=rng)
        self._root = self._finalize_leaves(self._root)
```

(`fit()`'s CPU branch is untouched — `tree_growth` is silently ignored without CUDA, matching how `use_gpu` already degrades.)

- [ ] **Step 4: Implement `_levelwise.py`**

```python
# eml_boost/tree_split/_levelwise.py
"""Level-wise (breadth-first) batched tree growth — Stage 2 of the plan.

Grows all nodes of a depth level in one batched pass (~20 kernel launches
+ 2 small syncs per LEVEL) instead of visiting ~2^depth nodes sequentially
(~25 launches + 3 syncs per NODE). Split semantics mirror
`_find_best_split_gpu` + `_grow_gpu`: per-node uniform-bin histograms,
identical gain formula, identical partition rule (val <= thr), identical
child-count re-check. Leaves are emitted as _PendingLeaf via
tree._make_pending_leaf and fitted by the Stage-1 batched finalize.

Semantic deviations from the node-wise engine (accepted by the spec):
  * descriptor RNG draws happen in BFS order, not DFS;
  * per-node top-k features are corr-descending (topk sorted=True);
  * histogram sums accumulate in fixed-point (deterministic).
"""

from __future__ import annotations

import numpy as np
import torch

from eml_boost._triton_exhaustive import (
    evaluate_trees_triton_nodewise,
    get_valid_descriptors_np,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split._multinode_hist import multinode_histogram_split
from eml_boost.tree_split._segmented import segment_topk_corr
from eml_boost.tree_split.nodes import EmlSplit, InternalNode, Node, RawSplit


class _Slot:
    """Python-side record for one frontier node."""

    __slots__ = ("attach",)

    def __init__(self, attach):
        # attach(node) grafts the finished Node into the parent (or root).
        self.attach = attach


def grow_levelwise(tree, indices: torch.Tensor, rng: np.random.Generator) -> Node:
    device = tree._device
    X = tree._X_gpu
    y = tree._y_gpu
    assert device is not None and X is not None and y is not None
    d = X.shape[1]
    msl = int(tree.min_samples_leaf)
    C = int(tree.n_eml_candidates)
    B = int(tree.n_bins)

    root_box: list[Node | None] = [None]

    def _attach_root(node):
        root_box[0] = node

    rows = indices  # active rows, any order
    seg = torch.zeros(rows.shape[0], dtype=torch.long, device=device)
    slots = [_Slot(_attach_root)]

    for depth in range(tree.max_depth + 1):
        L = len(slots)
        if L == 0:
            break
        # NOTE: do NOT break on rows.shape[0] == 0 — with min_samples_leaf=0
        # a slot can legitimately own zero rows and still needs its leaf
        # attached (bincount/argsort/slicing below all handle empty tensors).

        # ---- sync 1: per-slot row counts ----
        counts = torch.bincount(seg, minlength=L)
        counts_np = counts.cpu().numpy()

        # Sort rows by slot so per-slot extraction is slicing (stable →
        # row order within a slot is preserved deterministically, which
        # keeps each leaf's indices[0] seed rule reproducible).
        order = torch.argsort(seg, stable=True)
        rows = rows[order]
        seg = seg[order]
        offsets = np.zeros(L + 1, dtype=np.int64)
        np.cumsum(counts_np, out=offsets[1:])

        # Leaf-vs-attempt decision per slot (pure CPU metadata).
        attempt = []
        for s in range(L):
            n_s = int(counts_np[s])
            if depth >= tree.max_depth or n_s <= 2 * msl:
                tree_node = tree._make_pending_leaf(rows[offsets[s] : offsets[s + 1]])
                slots[s].attach(tree_node)
            else:
                attempt.append(s)
        if not attempt:
            break

        # Compact the attempting slots to 0..A-1.
        A = len(attempt)
        remap = torch.full((L,), -1, dtype=torch.long, device=device)
        remap[torch.tensor(attempt, dtype=torch.long, device=device)] = torch.arange(
            A, dtype=torch.long, device=device
        )
        keep = remap[seg] >= 0
        rows_a = rows[keep]
        seg_a = remap[seg[keep]]
        y_a = y[rows_a]
        X_a = X[rows_a]

        # ---- EML candidate columns ----
        eml_vals = None
        col_valid = None
        desc_a = None
        topk_np = None
        k_used = 0
        if C > 0 and d > 0:
            k_used = min(tree.k_eml, d)
            topk = segment_topk_corr(X_a, y_a, seg_a, A, k_used)  # (A, k)
            valid_desc = get_valid_descriptors_np(2, k_used)
            if len(valid_desc) > 0:
                # BFS-order RNG consumption: one block draw per level.
                draw = rng.integers(0, len(valid_desc), size=(A, C))
                desc_a = valid_desc[draw]  # (A, C, 6) int32 numpy
                desc_gpu = torch.tensor(desc_a, dtype=torch.int32, device=device)
                Xk = X_a.gather(1, topk[seg_a])  # (Na, k)
                eml = evaluate_trees_triton_nodewise(desc_gpu, seg_a, Xk, k_used)
                # per-(slot, candidate) finiteness -> col_valid for EML cols
                nonfinite = torch.zeros(A, C, device=device).index_add_(
                    0, seg_a, (~torch.isfinite(eml)).float().T
                )
                eml_ok = nonfinite == 0
                eml_vals = torch.nan_to_num(
                    eml.T, nan=0.0, posinf=0.0, neginf=0.0
                )  # neutralized; invalid cols are masked out of the gain argmax
                col_valid = torch.cat(
                    [torch.ones(A, d, dtype=torch.bool, device=device), eml_ok],
                    dim=1,
                )
                topk_np = topk.cpu().numpy()

        values = X_a if eml_vals is None else torch.cat([X_a, eml_vals], dim=1)

        # ---- batched split decision ----
        best_col, best_thr, best_gain = multinode_histogram_split(
            values, y_a, seg_a, A, B,
            min_leaf_count=msl, leaf_l2=tree.leaf_l2, col_valid=col_valid,
        )

        # ---- partition + child-count re-check (mirrors _grow_gpu) ----
        row_col = best_col[seg_a]
        row_thr = best_thr[seg_a]
        go_left = values.gather(1, row_col.unsqueeze(1)).squeeze(1) <= row_thr
        left_cnt = torch.zeros(A, device=device).index_add_(
            0, seg_a, go_left.float()
        )
        right_cnt = counts[torch.tensor(attempt, device=device)].float() - left_cnt

        # ---- sync 2: level decisions ----
        dec = torch.cat(
            [
                best_col.float(), best_thr, best_gain, left_cnt, right_cnt,
            ]
        ).cpu().numpy()
        col_np = dec[0 * A : 1 * A].astype(np.int64)
        thr_np = dec[1 * A : 2 * A]
        gain_np = dec[2 * A : 3 * A]
        lcnt_np = dec[3 * A : 4 * A].astype(np.int64)
        rcnt_np = dec[4 * A : 5 * A].astype(np.int64)

        # Build next frontier.
        new_slots: list[_Slot] = []
        child_of = torch.full((A,), -1, dtype=torch.long, device=device)
        for a, s in enumerate(attempt):
            n_s = int(counts_np[s])
            sl = rows[offsets[s] : offsets[s + 1]]
            if gain_np[a] <= 0 or lcnt_np[a] < msl or rcnt_np[a] < msl:
                slots[s].attach(tree._make_pending_leaf(sl))
                continue
            if col_np[a] < d:
                split: RawSplit | EmlSplit = RawSplit(
                    feature_idx=int(col_np[a]), threshold=float(thr_np[a])
                )
            else:
                c_idx = int(col_np[a]) - d
                split = EmlSplit(
                    snapped=SnappedTree(
                        depth=2, k=k_used,
                        internal_input_count=2, leaf_input_count=4,
                        terminal_choices=tuple(int(v) for v in desc_a[a, c_idx]),
                    ),
                    feature_subset=tuple(int(v) for v in topk_np[a]),
                    threshold=float(thr_np[a]),
                )
            node = InternalNode(split=split, left=None, right=None)  # type: ignore[arg-type]
            slots[s].attach(node)
            base = len(new_slots)
            new_slots.append(_Slot(lambda n_, nd=node: setattr(nd, "left", n_)))
            new_slots.append(_Slot(lambda n_, nd=node: setattr(nd, "right", n_)))
            child_of[a] = base

        if not new_slots:
            break

        # Advance active rows to child slots; drop rows of nodes that leafed.
        row_child = child_of[seg_a]
        alive = row_child >= 0
        seg = row_child[alive] + (~go_left[alive]).long()
        rows = rows_a[alive]
        slots = new_slots

    root = root_box[0]
    assert root is not None
    return root
```

Implementation notes:
- `InternalNode(split, left=None, right=None)` briefly holds `None` children; both are attached in the SAME level iteration (children slots created immediately), and every slot's `attach` is called exactly once — the loop's last level attaches leaves for every remaining slot because `depth >= tree.max_depth` triggers first. Add `assert root is not None` at the end (already present).
- `rows_a`/`seg_a` are the post-argsort compacted views — leaf slices (`rows[offsets[s]:offsets[s+1]]`) use the SORTED `rows`, so a leaf's `indices[0]` is its first row in stable-sorted order. The node-wise engine's leaf indices arrive in original-partition order instead. Both are deterministic, but they can differ → leaf-permutation seeds can differ between engines. This is within the statistical-equivalence bar (leaf STRUCTURE is unaffected; only which rows land in val/fit inside a leaf shifts). The no-EML structural test (Step 1) compares structure + thresholds + constant-leaf values: constant values don't depend on the permutation, so the test stays exact. Document this in the module docstring if it surprises you — do NOT try to replicate node-wise row order.
- Neutralizing non-finite EML values with `nan_to_num` is safe: their columns are masked via `col_valid`, so they can never win the argmax; the substitute values never influence a taken split.

- [ ] **Step 5: Run the structural oracle**

Run: `uv run pytest tests/unit/test_levelwise.py -v`
Expected: 6 parametrized PASS. If a shape divergence appears at exactly one node: print both engines' (gain, runner-up gain) for that node — a near-tie within fixed-point tolerance means adjust the fixture seed (document in test); a decisive gap means a real bug (most likely the child-count re-check or bin-edge rule) — debug before proceeding.

- [ ] **Step 6: Full suite + commit**

Run: `uv run pytest tests/unit/ -q` — no new failures (existing tests run node-wise default).

```bash
git add eml_boost/tree_split/_levelwise.py eml_boost/tree_split/tree.py tests/unit/test_levelwise.py
git commit -m "$(cat <<'EOF'
feat: level-wise growth engine, no-EML structural parity (Task 8)

grow_levelwise grows all frontier nodes per depth in one batched pass:
segmented counts, batched per-node histograms (fixed-point), one level
readback, vectorized partition with the nodewise child-count re-check.
tree_growth={nodewise,levelwise} param added (default nodewise).
Structural oracle: no-EML levelwise == nodewise trees across seeds and
min_samples_leaf in {1,20}, thresholds within float32+fixedpoint tol.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: EML splits in level-wise mode + ensemble wiring + invariants

**Files:**
- Modify: `eml_boost/tree_split/ensemble.py` (param + both tree constructions)
- Test: `tests/unit/test_levelwise.py` (extend)

**Interfaces:**
- Produces: `EmlSplitBoostRegressor(tree_growth="levelwise")` end-to-end; both boost loops thread `tree_growth=self.tree_growth` into `EmlSplitTreeRegressor(...)`.
- Consumes: Task 8's engine (EML path already implemented there).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_levelwise.py`:

```python
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


@requires_cuda
def test_tree_growth_param_validation_and_sklearn_roundtrip():
    from sklearn.base import clone

    with pytest.raises(ValueError, match="tree_growth"):
        from eml_boost.tree_split.tree import EmlSplitTreeRegressor

        EmlSplitTreeRegressor(tree_growth="diagonal")
    m = EmlSplitBoostRegressor(tree_growth="levelwise")
    m2 = clone(m)
    assert m2.tree_growth == "levelwise"
```

- [ ] **Step 2: Run tests, verify failures**

Run: `uv run pytest tests/unit/test_levelwise.py -v -k "invariants or parity or roundtrip"`
Expected: FAIL — `EmlSplitBoostRegressor.__init__() got an unexpected keyword argument 'tree_growth'`.

- [ ] **Step 3: Wire `tree_growth` through `ensemble.py`**

In `EmlSplitBoostRegressor.__init__` signature after `use_stacked_blend: bool = False,`:

```python
        tree_growth: str = "nodewise",
```

body (after the `leaf_l2` validation, mirroring its style):

```python
        if tree_growth not in ("nodewise", "levelwise"):
            raise ValueError(
                f"tree_growth must be 'nodewise' or 'levelwise', got {tree_growth!r}"
            )
        self.tree_growth = tree_growth
```

In BOTH `_fit_cpu_loop` and `_fit_gpu_loop`, add to the `EmlSplitTreeRegressor(...)` construction, after `use_stacked_blend=self.use_stacked_blend,`:

```python
                tree_growth=self.tree_growth,
```

Also add the parameter to the class docstring's parameter list:

```python
    tree_growth : str
        Growth engine: "nodewise" (recursive, the historical engine) or
        "levelwise" (breadth-first batched — much faster on GPU;
        statistically equivalent, not bit-identical: descriptor RNG is
        consumed in BFS order). Default "nodewise" until the CTR23 parity
        run promotes "levelwise".
```

- [ ] **Step 4: Run tests, verify pass; full suite; commit**

Run: `uv run pytest tests/unit/test_levelwise.py -v` then `uv run pytest tests/unit/ -q`
Expected: all PASS, no new failures elsewhere.

```bash
git add eml_boost/tree_split/ensemble.py tests/unit/test_levelwise.py
git commit -m "$(cat <<'EOF'
feat: thread tree_growth through boost ensemble + EML invariant tests (Task 9)

EmlSplitBoostRegressor(tree_growth="levelwise") now runs the batched
engine end-to-end. Invariant tests: EML splits reference valid
descriptors and in-range features, fits learn (2x MSE reduction), and
levelwise held-out RMSE lands within 15% of nodewise on the same data.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Same-seed determinism test

**Files:**
- Test: `tests/unit/test_levelwise.py` (extend)

**Interfaces:**
- Consumes: everything landed so far. No new production code expected; this is the spec's determinism acceptance gate. If it fails, the fix goes into `_segmented.segment_corr` (fixed-point corr) — do NOT weaken the test.

- [ ] **Step 1: Write the test**

```python
@requires_cuda
@pytest.mark.parametrize("seed", [0, 7])
def test_levelwise_same_seed_bitwise_deterministic(seed):
    """Spec acceptance: two same-seed fits -> byte-identical predictions."""
    X, y = _friedman(n=8000, seed=seed)

    def _fit_predict():
        m = EmlSplitBoostRegressor(
            max_rounds=12, max_depth=8, patience=0, use_gpu=True,
            random_state=seed, tree_growth="levelwise",
        )
        m.fit(X, y)
        return m.predict(X[:2000])

    p1, p2 = _fit_predict(), _fit_predict()
    np.testing.assert_array_equal(p1, p2)
```

- [ ] **Step 2: Run it 5 times to catch flakes**

Run: `for i in 1 2 3 4 5; do uv run pytest tests/unit/test_levelwise.py -k deterministic -q || break; done`
Expected: 5× PASS.

If it FAILS on any run: the nondeterminism source is float `index_add_` in `segment_corr` flipping a top-k near-tie or the leaf-OLS reductions flipping a gate near-tie. Diagnose by fitting twice and diffing tree signatures (`_tree_signature`) to locate the first divergent node, then: if top-k related, convert `segment_corr`'s three `index_add_` accumulations to the same int64 fixed-point pattern as `_multinode_hist` (scale `2**20` against per-call `max|operand|`); if leaf-OLS related, do the same in `_leaf_batch._acc`. Re-run the 5× loop until stable.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_levelwise.py
git commit -m "$(cat <<'EOF'
test: same-seed bitwise determinism gate for levelwise engine (Task 10)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Speed gate + benchmark checkpoint

**Files:**
- Test: `tests/unit/test_levelwise.py` (extend)

**Interfaces:**
- Consumes: complete engine. Records the Stage-2 numbers the parity run's timing claim will cite.

- [ ] **Step 1: Write the speed-gate test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/unit/test_levelwise.py -k speedup -v`
Expected: PASS with levelwise ≥3× (record actual ratio).

If the ratio is < 3×: profile before touching thresholds —

```bash
uv run python - <<'EOF'
import cProfile, pstats
import numpy as np
from eml_boost.tree_split import EmlSplitBoostRegressor
rng = np.random.default_rng(0)
X = rng.standard_normal((32_000, 10))
y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + rng.standard_normal(32_000)
m = EmlSplitBoostRegressor(max_rounds=2, patience=0, random_state=0, tree_growth="levelwise")
m.fit(X, y)  # warmup
p = cProfile.Profile(); p.enable()
EmlSplitBoostRegressor(max_rounds=10, patience=0, random_state=0, tree_growth="levelwise").fit(X, y)
p.disable(); pstats.Stats(p).sort_stats("tottime").print_stats(25)
EOF
```

The expected hot spots at this stage are genuinely batched kernels. If a per-slot Python loop dominates (e.g., the leaf-permutation loop or the level-decision loop), the batch sizes are wrong — fix the loop, don't lower the gate. Promote the multi-node histogram to a Triton kernel ONLY if the profile shows `multinode_histogram_split` dominating after the Python loops are clean (spec's torch-first policy).

- [ ] **Step 3: Benchmark checkpoint (record, don't commit scripts)**

Re-run the scaling probe and 344_mv equivalents:

```bash
uv run python - <<'EOF'
import time
import numpy as np, torch
from eml_boost.tree_split import EmlSplitBoostRegressor

def friedman(n, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + rng.standard_normal(n)
    return X, y

X, y = friedman(4096)
EmlSplitBoostRegressor(max_rounds=2, patience=0, random_state=0, tree_growth="levelwise").fit(X, y)
for n in [2_000, 8_000, 32_000, 128_000]:
    X, y = friedman(n)
    t0 = time.perf_counter()
    EmlSplitBoostRegressor(max_rounds=10, max_depth=8, patience=0, random_state=0, tree_growth="levelwise").fit(X, y)
    torch.cuda.synchronize()
    print(f"n={n}: {(time.perf_counter()-t0)/10:.4f} s/round  (baseline was 0.195/0.220/0.397/0.595)")
EOF
```

Expected: ≤ ~0.06 s/round at n=32k (spec target). Record all four numbers.

Optionally (network + pmlb required): `uv run python profile_344mv/run_profile.py` after temporarily adding `tree_growth="levelwise"` to its constructor — expect ≤ ~3 s for 50 rounds vs the historical 26.5 s. Revert the temporary edit; do not commit it.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_levelwise.py
git commit -m "$(cat <<'EOF'
test: levelwise speed gate 3x + Stage-2 bench numbers (Task 11)

Measured on RTX 3090, synthetic 32k x 10, depth 8, 10 rounds:
nodewise <X.XX> s/round -> levelwise <Y.YY> s/round (<N.N>x).
Scaling probe s/round: 2k <A>, 8k <B>, 32k <C>, 128k <D>
(baseline 0.195 / 0.220 / 0.397 / 0.595).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Parity runner (`experiments/run_experiment19_levelwise_parity.py`)

**Files:**
- Create: `experiments/run_experiment19_levelwise_parity.py` (copy of `experiments/run_experiment18_openml_ctr23.py` with three edits)

**Interfaces:**
- Produces: runner writing `experiments/experiment19/{summary.csv, summary.json, openml_rmse.png, failures.json}` with identical schema to Exp 18 (`dataset,seed,config,rmse,fit_time,n_rounds`).
- Consumes: `EmlSplitBoostRegressor(tree_growth="levelwise")`.

- [ ] **Step 1: Copy the Exp-18 runner**

```bash
cp experiments/run_experiment18_openml_ctr23.py experiments/run_experiment19_levelwise_parity.py
```

- [ ] **Step 2: Make exactly three kinds of edits**

1. Module docstring: retitle to "Experiment 19: level-wise engine parity run on OpenML-CTR23" and note: identical protocol to Exp 18 (same seeds, same matched XGB/LGB configs, same datasets), with SB running `tree_growth="levelwise"` — the spec's default-flip gate.
2. Output directory: every occurrence of `experiments/experiment18` → `experiments/experiment19` (check the OUT_DIR / plot-path constants at the top of the file).
3. The SB constructor call: add `tree_growth="levelwise",` to the `EmlSplitBoostRegressor(...)` kwargs. Touch nothing else — XGB/LGB configs, seeds, dataset list, early stopping all stay verbatim.

- [ ] **Step 3: Smoke test on 2 datasets**

The Exp-18 runner exposes a `--datasets` / limit mechanism — check its argparse block; if present run:

`uv run python experiments/run_experiment19_levelwise_parity.py --datasets abalone,cars`

If no such flag exists, temporarily set the dataset list to `["abalone", "cars"]`, run, then restore. Expected: `experiments/experiment19/summary.csv` gains 30 rows (2 datasets × 5 seeds × 3 configs), no failures, SB fit_time visibly below Exp-18's for the same datasets (abalone was 6.9 s mean — expect ≲ 1.5 s).

- [ ] **Step 4: Clean smoke outputs + commit runner**

```bash
rm -rf experiments/experiment19
git add experiments/run_experiment19_levelwise_parity.py
git commit -m "$(cat <<'EOF'
add: Experiment 19 runner — levelwise parity on OpenML-CTR23 (Task 12)

Exp-18 protocol verbatim (datasets, seeds, matched XGB/LGB configs);
SB runs tree_growth="levelwise". Gates: suite-total SB <= 10x XGB,
RMSE headline within Exp-18 seed-noise envelope.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: CHECKPOINT — ask the user before the full run**

Per project memory ("pause and confirm before kicking off multi-hour experiments") and the spec: report Stage-1/2 bench numbers and ask to launch the full 34-dataset × 5-seed run (expected ~15-40 min post-speedup). Do not launch without confirmation.

---

## Task 13: Parity run, report, default flip

**Files:**
- Create: `experiments/experiment19/{summary.csv, summary.json, openml_rmse.png, failures.json}` (generated), `experiments/experiment19/report.md`
- Modify: `eml_boost/tree_split/tree.py`, `eml_boost/tree_split/ensemble.py` (default `tree_growth="levelwise"`), `tests/unit/test_levelwise.py` (structural-oracle tests pin `tree_growth` explicitly, so they keep testing both engines — verify, adjust only if a test relied on the old default)

- [ ] **Step 1: Launch the full run in background (after Task 12 checkpoint approval)**

`uv run python experiments/run_experiment19_levelwise_parity.py 2>&1 | tee experiments/experiment19/run.log`

- [ ] **Step 2: Gate analysis**

Compute from `experiments/experiment19/summary.csv` and `experiments/experiment18/summary.csv`:
- suite-total SB fit_time (sum of per-dataset seed-means) and the ratio to same-run XGB total — **gate: ≤ 10×**;
- win rate (mean-ratio < 1.00 vs XGB), median ratio, catastrophic count — **gate: within ±5 pp / ±0.01 / still 0** of Exp-18's 76.5% / 0.987 / 0;
- per-dataset |Δratio| Exp-19 vs Exp-18 — list any beyond the per-dataset 5-seed std; investigate each before proceeding (re-read spec's statistical-equivalence bar; `brazilian_houses`-class high-variance datasets are expected to move).

If a gate fails: STOP, write findings into `experiments/experiment19/report.md` as a negative/partial result (project convention — Exp 16 precedent), and bring options to the user. Do NOT flip the default.

- [ ] **Step 3: Write `experiments/experiment19/report.md`**

Mirror `experiments/experiment18/report.md` structure: what/config/coverage/headline table (Exp-19 vs Exp-18 columns)/timing table (suite totals + per-dataset worst cases)/what-it-shows/caveats/action-taken. The headline claim this report exists for: fit-time ratio before → after, at unchanged RMSE headline.

- [ ] **Step 4: Flip the default (only if gates passed)**

In `tree.py` and `ensemble.py`:

```python
        tree_growth: str = "levelwise",       # was "nodewise"; flipped after Exp-19 parity gates passed
```

Run: `uv run pytest tests/unit/ -q` — triage: tests that implicitly assumed node-wise default (the Task-2 snapshot test constructs trees through the patched `__init__`, which still pins `_batched_leaves` but now inherits levelwise default — pin `tree_growth="nodewise"` inside that test, since its snapshot is a node-wise artifact). Expected: no other changes needed; anything else that breaks gets `tree_growth="nodewise"` pinned only if its INTENT is node-wise-specific, else adjusted per Exp-18 Task-1 triage rules.

- [ ] **Step 5: Commit results, report, and flip separately**

```bash
git add experiments/experiment19/
git commit -m "results: Experiment 19 levelwise parity run (34 datasets, 5 seeds)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add experiments/experiment19/report.md
git commit -m "exp 19 report: levelwise engine parity + timing on OpenML-CTR23

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/
git commit -m "feat: flip tree_growth default nodewise -> levelwise (Exp-19 gates passed)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review (completed)

- **Spec coverage:** Stage-1 mechanics → Tasks 1-4; Stage-2 pipeline → Tasks 5-9; determinism requirement → Task 6 (histograms) + Task 10 (acceptance); RNG/equivalence policy → Task 8 no-EML oracle + Task 9 parity band + Task 13 gates; benchmarks → Tasks 4, 11; validation run & rollout (incl. user checkpoint + default flip) → Tasks 12-13; nodewise retained as oracle → never removed. Memory-ceiling note → `_CHUNK_ROWS` in Task 3, small-tensor sizes documented in Task 6.
- **Placeholder scan:** commit messages in Tasks 4 and 11 contain `<X.XX>`-style slots to be filled with MEASURED numbers at execution time — intentional (numbers cannot exist before the run); everything else is concrete.
- **Type consistency:** `_PendingLeaf(indices, resolved)` (T2) consumed by T3/T8; `fit_leaves_batched(tree, pending) -> list[Node]` (T3) called in T2's `_finalize_leaves`; `segment_topk_corr` signature identical at T1 definition and T3/T8 call sites; `multinode_histogram_split` returns `(best_col, best_thr, best_gain)` consumed in that order in T8; `evaluate_trees_triton_nodewise(desc_nodes, node_of, X, k)` matches T7 definition at the T8 call site; `tree_growth` spelled identically in tree.py, ensemble.py, and tests.

---

## Plan Amendment 2 (2026-07-12, during execution)

Exp-19 first run (idle box, user-confirmed): RMSE gates ALL PASS (win rate 73.5% in band;
median 0.980 in band; 0 catastrophic; 0/34 datasets beyond noise; within-10% improved to
94.1%). Timing gate FAIL: SB 173s / XGB 12s = 14.6x vs <=10x (SB suite speedup 4.0x,
691->173s). Spec contingency invoked ("~15x => promote per-level hot spots"). Profile at
n=3400: grow_levelwise Python 24%, hist 28%, leaf batch 17%, corr 11%, 37 syncs/round.

- **NEW Task 14** (brief: `.superpowers/sdd/task-14-brief.md`): launch-consolidation pass
  (fuse hist scatters, fuse corr scatters, trim per-slot Python, consolidate syncs) under
  hard bit-exactness/determinism gates. Target <=18 ms/round on the profile workload.
- Task 13 then resumes: fresh full Exp-19 re-run (delete experiment19 outputs; determinism
  reproduces RMSEs, timings decide), gate re-check, report, flip-if-pass.
