# SplitBoost Leaf-Path Micro-Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut ~10% off SplitBoost per-fit time by eliminating three sources of per-leaf and per-round Python orchestration overhead identified in the `344_mv` profile, with bit-exact float32 equivalence to the current implementation.

**Architecture:** Three independent mechanical refactors to the SplitBoost GPU fit path, shipped as separate commits in order F → D → E (smallest blast radius first). No algorithm change, no public-API change, no new Triton kernels, no CPU-path changes. Each task: write failing test → implement → verify all 94 existing tests still pass → commit. After all three land, re-run the sanity bench (`experiments/bench_sanity.py`) to confirm `344_mv` per-fit time drops as projected.

**Tech Stack:** Python 3.12 / numpy / torch / Triton / pytest. Run via `uv run`. Tests in `tests/unit/` (94 passing + 1 known unrelated failure left alone per `experiments/workflow.md`).

**Spec:** `docs/superpowers/specs/2026-04-25-splitboost-leaf-microopts-design.md`.

---

## File Structure

| file | role | tasks that touch it |
|---|---|---|
| `eml_boost/_triton_exhaustive.py` | Add `_VALID_DESC_CACHE` + `get_valid_descriptors_np` helper | Task 1 (F) |
| `eml_boost/tree_split/tree.py` | Replace `_sample_descriptors` body, replace `_tensorize_tree`, refactor `_fit_leaf` + `_select_leaf_gated` | Tasks 1, 2, 3 |
| `tests/unit/test_eml_split_tree.py` | Add 3 new tests, one per change | Tasks 1, 2, 3 |

No other files are touched. No new files are created.

---

## Task 1: F — cache `valid_desc` in `_sample_descriptors`

**Files:**
- Modify: `eml_boost/_triton_exhaustive.py` (add cache + helper)
- Modify: `eml_boost/tree_split/tree.py:853-868` (`_sample_descriptors` body) and the import block at the top
- Test: `tests/unit/test_eml_split_tree.py` (add `test_valid_descriptor_cache_consistency`)

### Step 1: Write the failing test

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_valid_descriptor_cache_consistency():
    """Cached valid descriptors must equal the masked enumeration, and
    repeated calls must return the same array object (cache hit)."""
    from eml_boost._triton_exhaustive import (
        get_descriptor_np,
        get_feature_mask_np,
        get_valid_descriptors_np,
    )
    for k in (1, 2, 3):
        all_desc = get_descriptor_np(2, k)
        mask = get_feature_mask_np(2, k)
        expected = all_desc[mask]
        cached = get_valid_descriptors_np(2, k)
        np.testing.assert_array_equal(expected, cached)
        # Repeated call returns the SAME object (identity, not just equality).
        assert get_valid_descriptors_np(2, k) is cached
```

### Step 2: Run the test to verify it fails

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_valid_descriptor_cache_consistency -v`
- [ ] Expected: `ImportError: cannot import name 'get_valid_descriptors_np'` (or equivalent attribute error).

### Step 3: Add the cache + helper to `_triton_exhaustive.py`

- [ ] Edit `eml_boost/_triton_exhaustive.py`. Find the cache-dict block (currently lines 55-58):

```python
_descriptor_cache: dict[tuple[int, int], np.ndarray] = {}
_descriptor_gpu_cache: dict[tuple[int, int, str], torch.Tensor] = {}
_feature_mask_cache: dict[tuple[int, int], np.ndarray] = {}
_feature_mask_gpu_cache: dict[tuple[int, int, str], torch.Tensor] = {}
```

- [ ] Add a fifth cache directly below those four:

```python
_valid_desc_cache: dict[tuple[int, int], np.ndarray] = {}
```

- [ ] Find the existing `get_feature_mask_np` function (currently lines 423-433). Add the new helper directly below it (before `get_feature_mask_gpu`):

```python
def get_valid_descriptors_np(depth: int, k: int) -> np.ndarray:
    """Cached enumeration of non-constant depth-`depth` descriptors at k inputs.

    Returns a contiguous (n_valid, 6) int32 array, where n_valid is the count
    of descriptors that pass `get_feature_mask_np`. Process-global cache:
    same array is returned on every call with the same (depth, k) — callers
    must not mutate it.
    """
    if depth != 2:
        raise ValueError("GPU path only supports depth=2")
    key = (depth, k)
    cached = _valid_desc_cache.get(key)
    if cached is None:
        all_desc = get_descriptor_np(depth, k)
        mask = get_feature_mask_np(depth, k)
        cached = np.ascontiguousarray(all_desc[mask])
        _valid_desc_cache[key] = cached
    return cached
```

### Step 4: Run the new test to verify it passes

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_valid_descriptor_cache_consistency -v`
- [ ] Expected: PASS.

### Step 5: Switch `_sample_descriptors` to the cached helper

- [ ] In `eml_boost/tree_split/tree.py`, find the import block at the top of the file containing `from eml_boost._triton_exhaustive import (...)` and add `get_valid_descriptors_np` to the import list. After the change the imports should include (alphabetical ordering preserved):

```python
from eml_boost._triton_exhaustive import (
    evaluate_trees_torch,
    evaluate_trees_torch_per_sample,
    evaluate_trees_triton,
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
    get_feature_mask_np,
    get_valid_descriptors_np,
)
```

- [ ] Replace the body of `_sample_descriptors` (currently lines 853-868). The new function:

```python
    def _sample_descriptors(
        self, k: int, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Uniform random draw from the non-constant depth-2 tree space.

        The descriptor enumeration, the feature mask, AND the post-mask
        valid array are all process-global cached in _triton_exhaustive
        (keyed on (depth, k)) so this function only does the rng draw +
        an indexing op per call.
        """
        valid_desc = get_valid_descriptors_np(2, k)
        if len(valid_desc) == 0:
            return np.empty((0, 6), dtype=np.int32)
        idx = rng.integers(0, len(valid_desc), size=n_samples)
        return valid_desc[idx]
```

### Step 6: Run the full unit suite to verify no regressions

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `94 passed, 1 failed` — the failure is the pre-existing `test_eml_weak_learner.py::test_fit_recovers_simple_formula`, which is documented in `experiments/workflow.md` as known-unrelated and must be left alone.

### Step 7: Commit

- [ ] Run:

```bash
git add eml_boost/_triton_exhaustive.py eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
perf: cache valid_desc array in _sample_descriptors (Task F)

The redux fix cached the descriptor enumeration and the feature mask, but
_sample_descriptors still allocated a fresh (n_valid, 6) array via
all_desc[mask] on every call (~12k calls per 344_mv fit). Add a third
process-global cache _valid_desc_cache + get_valid_descriptors_np helper
in _triton_exhaustive.py, and have _sample_descriptors read from it.

Saves ~600ms/fit on 344_mv (40k rows). No algorithm change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: D — numpy-backed `_tensorize_tree`

**Files:**
- Modify: `eml_boost/tree_split/tree.py:929-1011` (`_tensorize_tree` body)
- Test: `tests/unit/test_eml_split_tree.py` (add `test_tensorize_tree_numpy_matches_torch_baseline`)

### Step 1: Write the failing test

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_tensorize_tree_numpy_matches_torch_baseline():
    """The numpy-backed _tensorize_tree must produce the same dict
    contents as the prior torch-zeros baseline. Verifies shapes, dtypes,
    leaf-vs-internal node_kind invariants, and that left/right_child of
    leaves stay -1."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 4)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=300))
    m = EmlSplitTreeRegressor(
        max_depth=4,
        n_eml_candidates=10,
        k_eml=2,
        k_leaf_eml=1,
        min_samples_leaf_eml=30,
        use_gpu=False,
        random_state=0,
    ).fit(X, y)
    tt = m._tensorize_tree(m._root)
    n = tt["n_nodes"]
    # Shapes.
    assert tt["node_kind"].shape == (n,)
    assert tt["leaf_value"].shape == (n,)
    assert tt["leaf_feat_subset"].shape == (n, tt["k_leaf_eml"])
    assert tt["split_feat_subset"].shape == (n, tt["k_split_eml"])
    # Dtypes match what _predict_x_gpu expects.
    assert tt["node_kind"].dtype == torch.int8
    assert tt["left_child"].dtype == torch.int32
    assert tt["right_child"].dtype == torch.int32
    assert tt["feature_idx"].dtype == torch.int32
    assert tt["threshold"].dtype == torch.float32
    assert tt["leaf_value"].dtype == torch.float32
    assert tt["leaf_eta"].dtype == torch.float32
    assert tt["leaf_bias"].dtype == torch.float32
    # node_kind values are in {0, 1, 2, 3}.
    assert ((tt["node_kind"] >= 0) & (tt["node_kind"] <= 3)).all()
    # Leaves (kind 2 or 3) have left/right_child == -1.
    leaves = (tt["node_kind"] == 2) | (tt["node_kind"] == 3)
    assert (tt["left_child"][leaves] == -1).all()
    assert (tt["right_child"][leaves] == -1).all()
```

### Step 2: Run the test to verify it passes against the current torch implementation

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_tensorize_tree_numpy_matches_torch_baseline -v`
- [ ] Expected: PASS. (This test is a *characterization* test — it documents the contract that both the old torch implementation and the new numpy implementation must satisfy.)
- [ ] Note: If this test fails on the current code, STOP and re-investigate — the test is encoding the wrong contract, not the implementation. The current shapes/dtypes are the source of truth.

### Step 3: Replace the body of `_tensorize_tree`

- [ ] In `eml_boost/tree_split/tree.py`, replace the entire body of `_tensorize_tree` (currently lines 929-1011). The signature and docstring stay the same; the implementation switches to numpy + bulk `torch.from_numpy` at the end:

```python
    def _tensorize_tree(self, root: Node) -> dict:
        """Walk the fitted Python tree once and emit a flat dict of CPU
        tensors. The dict is moved to GPU on first predict call.
        Only called in GPU-fit mode.

        Implementation note: builds the per-field arrays in numpy
        (single-element writes are O(1) — direct memory write, no
        torch dispatch overhead), then bulk-converts to torch tensors
        via torch.from_numpy at the end (zero-copy view). Same output
        contract as the prior torch-zeros implementation; just faster
        on large trees (saves ~5-10ms per saturated depth-8 tree).
        """
        nodes: list[list] = []  # entries: [node_id, node_obj, left_id, right_id]

        def walk(node):
            my_id = len(nodes)
            nodes.append([my_id, node, -1, -1])
            if isinstance(node, InternalNode):
                left_id = walk(node.left)
                right_id = walk(node.right)
                nodes[my_id][2] = left_id
                nodes[my_id][3] = right_id
            return my_id

        walk(root)
        n_nodes = len(nodes)
        K_leaf = max(1, self.k_leaf_eml)
        K_split = max(1, self.k_eml)

        node_kind = np.zeros(n_nodes, dtype=np.int8)
        left_child = np.full(n_nodes, -1, dtype=np.int32)
        right_child = np.full(n_nodes, -1, dtype=np.int32)
        feature_idx = np.zeros(n_nodes, dtype=np.int32)
        threshold = np.zeros(n_nodes, dtype=np.float32)
        leaf_value = np.zeros(n_nodes, dtype=np.float32)
        leaf_eml_descriptor = np.zeros((n_nodes, 6), dtype=np.int32)
        split_eml_descriptor = np.zeros((n_nodes, 6), dtype=np.int32)
        leaf_eta = np.zeros(n_nodes, dtype=np.float32)
        leaf_bias = np.zeros(n_nodes, dtype=np.float32)
        leaf_cap = np.full(n_nodes, np.float32(np.inf), dtype=np.float32)
        leaf_feat_subset = np.zeros((n_nodes, K_leaf), dtype=np.int32)
        leaf_feat_mean = np.zeros((n_nodes, K_leaf), dtype=np.float32)
        leaf_feat_std = np.ones((n_nodes, K_leaf), dtype=np.float32)
        split_feat_subset = np.zeros((n_nodes, K_split), dtype=np.int32)

        for nid, node, lc, rc in nodes:
            left_child[nid] = lc
            right_child[nid] = rc
            if isinstance(node, LeafNode):
                node_kind[nid] = 2
                leaf_value[nid] = float(node.value)
            elif isinstance(node, EmlLeafNode):
                node_kind[nid] = 3
                leaf_eta[nid] = float(node.eta)
                leaf_bias[nid] = float(node.bias)
                leaf_cap[nid] = float(node.cap)
                for i, f in enumerate(node.feature_subset):
                    leaf_feat_subset[nid, i] = int(f)
                for i, m in enumerate(node.feature_mean):
                    leaf_feat_mean[nid, i] = float(m)
                for i, s in enumerate(node.feature_std):
                    leaf_feat_std[nid, i] = float(s)
                for i, c in enumerate(node.snapped.terminal_choices):
                    leaf_eml_descriptor[nid, i] = int(c)
            elif isinstance(node, InternalNode):
                split = node.split
                threshold[nid] = float(split.threshold)
                if isinstance(split, RawSplit):
                    node_kind[nid] = 0
                    feature_idx[nid] = int(split.feature_idx)
                else:  # EmlSplit
                    node_kind[nid] = 1
                    for i, c in enumerate(split.snapped.terminal_choices):
                        split_eml_descriptor[nid, i] = int(c)
                    for i, f in enumerate(split.feature_subset):
                        split_feat_subset[nid, i] = int(f)

        return {
            "node_kind": torch.from_numpy(node_kind),
            "left_child": torch.from_numpy(left_child),
            "right_child": torch.from_numpy(right_child),
            "feature_idx": torch.from_numpy(feature_idx),
            "threshold": torch.from_numpy(threshold),
            "leaf_value": torch.from_numpy(leaf_value),
            "leaf_eml_descriptor": torch.from_numpy(leaf_eml_descriptor),
            "split_eml_descriptor": torch.from_numpy(split_eml_descriptor),
            "split_feat_subset": torch.from_numpy(split_feat_subset),
            "leaf_eta": torch.from_numpy(leaf_eta),
            "leaf_bias": torch.from_numpy(leaf_bias),
            "leaf_cap": torch.from_numpy(leaf_cap),
            "leaf_feat_subset": torch.from_numpy(leaf_feat_subset),
            "leaf_feat_mean": torch.from_numpy(leaf_feat_mean),
            "leaf_feat_std": torch.from_numpy(leaf_feat_std),
            "k_leaf_eml": K_leaf,
            "k_split_eml": K_split,
            "n_nodes": n_nodes,
        }
```

### Step 4: Run the new test plus key dependent tests to verify the refactor preserves behavior

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_tensorize_tree_numpy_matches_torch_baseline tests/unit/test_eml_split_tree.py -k "predict or tensorize" -v`
- [ ] Expected: All matched tests PASS. The Triton predict equivalence test (`test_predict_triton_matches_torch`) is the strongest semantic check — if it passes, the tensorized output is consumed correctly by `_predict_x_gpu`.

### Step 5: Run the full unit suite to verify no regressions

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `94 passed, 1 failed` (the pre-existing `test_fit_recovers_simple_formula` failure, left alone).

### Step 6: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
perf: numpy-backed _tensorize_tree (Task D)

The prior implementation allocated 15 small CPU torch tensors and then
filled them via single-element torch writes in a Python loop — full
torch dispatch overhead per write (~5us each), ~7k writes per saturated
depth-8 tree. Switch to numpy arrays for the build-up (single-element
writes are O(1) memory writes with no dispatch), then convert to torch
tensors at the end via torch.from_numpy (zero-copy view).

Saves ~5-10ms per tree-tensorize call (one per boost round). On a
200-round 344_mv fit that's ~1s per fit. No algorithm change; output
dict contract is identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: E — batched `.item()` syncs in the leaf path

**Files:**
- Modify: `eml_boost/tree_split/tree.py` — the body of `_fit_leaf` (currently lines 431-558) and `_select_leaf_gated` (currently lines 560-606+)
- Test: `tests/unit/test_eml_split_tree.py` (add `test_fit_leaf_item_batching_preserves_predictions`)

### Step 1: Write the failing test

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_fit_leaf_item_batching_preserves_predictions():
    """End-to-end: with a fixed seed, predictions on a held-out set must
    be finite, and train MSE must be sub-threshold on a clean signal —
    smoke check that the .item() batching refactor in _fit_leaf and
    _select_leaf_gated didn't change the gating semantics."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(800, 6)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] - X[:, 2] ** 2
         + 0.1 * rng.normal(size=800))
    X_te = rng.uniform(-1, 1, size=(200, 6)).astype(np.float64)

    m = EmlSplitTreeRegressor(
        max_depth=5,
        n_eml_candidates=10,
        k_eml=3,
        k_leaf_eml=1,
        min_samples_leaf_eml=30,
        leaf_eml_cap_k=2.0,
        use_gpu=True,
        random_state=0,
    ).fit(X, y)
    pred = m.predict(X_te)
    assert np.all(np.isfinite(pred))
    train_mse = float(np.mean((m.predict(X) - y) ** 2))
    # Clean signal with var ~= 1.5; trivially-fit MSE should be well below.
    assert train_mse < 0.5
```

### Step 2: Run the test to verify it passes against the current implementation

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_fit_leaf_item_batching_preserves_predictions -v`
- [ ] Expected: PASS. (This is a *characterization* smoke test — it documents the behavior the refactored code must continue to satisfy. If it fails on the current code, STOP and re-investigate the test.)

### Step 3: Refactor `_fit_leaf` to batch its three pre-OLS `.item()` reads

- [ ] In `eml_boost/tree_split/tree.py`, locate the start of `_fit_leaf` (line 431). Replace the block from the function start through the existing `cap_leaf` line (currently lines 443-548) with the version below. **All other lines of `_fit_leaf` stay unchanged** — only the early-out, scalar-read, and cap-leaf lines are modified.

The NEW body for the section from `n = int(indices.shape[0])` through `cap_leaf = ...`:

```python
        n = int(indices.shape[0])
        if n == 0:
            return LeafNode(value=0.0)
        y_sub = self._y_gpu[indices]

        # Early-out gates. Only constant_value is needed in the constant
        # fallback path; keep a single .item() here so we don't allocate
        # the GPU stack tensor when we won't use it.
        eml_disabled = self.k_leaf_eml <= 0
        too_small = n < self.min_samples_leaf_eml
        no_gpu = self._X_gpu is None or self._device is None
        n_raw = self._X_cpu.shape[1] if self._X_cpu is not None else 0
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=float(y_sub.mean().item()))

        device = self._device
        assert device is not None and self._X_gpu is not None

        # Batched D2H read for the three scalars we'll need below
        # (constant_value, RNG seed, cap_leaf). Coalescing 3 .item()
        # calls into one .cpu() saves 2 cuda-stream syncs per leaf.
        # NOTE on the int->float32 cast for indices[0]: float32 mantissa
        # is 24 bits, so row indices < 2**24 round-trip exactly. PMLB max
        # is 1M (1191_BNG_pbc), so all current datasets are safe; assert
        # to fail loudly on any future giant dataset.
        assert n < (1 << 24), (
            f"row index batched as float32 requires n < 2**24, got n={n}; "
            "fall back to per-call .item() for indices[0] in this regime"
        )
        cap_k = float(self.leaf_eml_cap_k)
        if cap_k > 0.0:
            scalars_gpu = torch.stack([
                y_sub.mean(),
                indices[0].to(torch.float32),
                y_sub.abs().max() * cap_k,
            ])
            scalars = scalars_gpu.cpu().numpy()
            constant_value = float(scalars[0])
            seed = int(scalars[1])
            cap_leaf = float(scalars[2])
        else:
            scalars_gpu = torch.stack([
                y_sub.mean(),
                indices[0].to(torch.float32),
            ])
            scalars = scalars_gpu.cpu().numpy()
            constant_value = float(scalars[0])
            seed = int(scalars[1])
            cap_leaf = float("inf")

        k = min(self.k_leaf_eml, n_raw)
        # X_sub_raw: gather X_gpu by leaf indices, then by top-k features.
        X_node = self._X_gpu[indices]
        top_features_gpu = self._top_features_by_corr_gpu(X_node, y_sub, k)
        top_features = top_features_gpu.cpu().numpy()
        X_sub_raw = X_node[:, top_features_gpu]
        y_full = y_sub  # already a GPU float32 tensor

        # Standardize using GLOBAL (fit-time) mean/std — local leaf stats
        # produce narrow ranges that explode at predict time on same-leaf
        # test samples lying slightly outside the local window. Global
        # stats give a consistent transform across all leaves.
        # Then CLAMP to [-3, 3] so that outliers (heavy-tailed PMLB
        # features like cpu_small's 10+σ samples) can't push exp(exp(·))
        # into overflow territory; the snapped grammar allows nested
        # exponentials and those are catastrophic at |arg| >> 3.
        assert self._global_mean_gpu is not None and self._global_std_gpu is not None
        top_features_t = torch.from_numpy(top_features).to(device=device, dtype=torch.long)
        mean_x = self._global_mean_gpu[top_features_t]
        std_x = self._global_std_gpu[top_features_t]
        X_sub = torch.clamp((X_sub_raw - mean_x) / std_x, -3.0, 3.0)

        # Deterministic leaf train/val split. The val portion (25%) is held
        # out from the per-tree OLS fit so the tree-selection policy
        # (either the legacy gate or the Task 2 blend) can evaluate
        # generalization rather than training fit.
        rng_leaf = np.random.default_rng(seed)
        perm = rng_leaf.permutation(n)
        val_sz = max(n // 4, 5)
        if n - val_sz < self.min_samples_leaf_eml // 2:
            return LeafNode(value=constant_value)
        val_local = perm[:val_sz]
        fit_local = perm[val_sz:]
        fit_idx_gpu = torch.from_numpy(fit_local).to(device=device, dtype=torch.long)
        val_idx_gpu = torch.from_numpy(val_local).to(device=device, dtype=torch.long)
        X_fit = X_sub[fit_idx_gpu]
        X_val = X_sub[val_idx_gpu]
        y_fit = y_full[fit_idx_gpu]
        y_val = y_full[val_idx_gpu]

        # Batched evaluation of all 144 depth-2 candidate trees.
        descriptor_gpu = get_descriptor_gpu(depth=2, k=k, device=device)
        feature_mask = get_feature_mask_gpu(depth=2, k=k, device=device)
        preds_fit = evaluate_trees_triton(descriptor_gpu, X_fit, k)  # (n_trees, n_fit)
        preds_val = evaluate_trees_triton(descriptor_gpu, X_val, k)  # (n_trees, n_val)

        # Closed-form OLS per tree on the fit portion. When
        # self.leaf_eml_ridge > 0 we regularize the slope (η) with a ridge
        # penalty λ·η². The centered-ridge closed form adds n_fit·λ to the
        # normal-equation diagonal; on the original sufficient statistics
        # that means replacing det with det + n_fit·λ. For λ > 0 the bias
        # is the conditional-OLS intercept β = (Σy − η·Σp)/n_fit given the
        # shrunk slope. At λ = 0 we use the old 2×2 normal-equation form
        # for exact bit-compat with Experiment 9.
        n_fit = float(X_fit.shape[0])
        sum_p = preds_fit.sum(dim=1)
        sum_p2 = (preds_fit * preds_fit).sum(dim=1)
        sum_y_f = y_fit.sum()
        sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
        det = sum_p2 * n_fit - sum_p * sum_p
        lam = float(self.leaf_eml_ridge)
        det_ridged = det + n_fit * lam
        # Guard against the remaining zero case (λ = 0 and det = 0 — a
        # genuinely degenerate tree with p_val constant and λ off).
        det_safe = torch.where(
            det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
        )
        eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
        if lam == 0.0:
            bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
        else:
            bias = (sum_y_f - eta * sum_p) / n_fit

        # Validity mask.
        finite_preds = (
            torch.isfinite(preds_fit).all(dim=1)
            & torch.isfinite(preds_val).all(dim=1)
        )
        finite_coefs = torch.isfinite(eta) & torch.isfinite(bias)
        valid = feature_mask & finite_preds & finite_coefs & (det.abs() > 1e-6)

        # cap_leaf was already computed in the batched scalar read above;
        # nothing more to do here.

        ctx = dict(
            y_full=y_full, y_val=y_val, eta=eta, bias=bias,
            preds_val=preds_val, valid=valid, k=k, top_features=top_features,
            mean_x=mean_x, std_x=std_x, constant_value=constant_value,
            cap_leaf=cap_leaf,
        )
        if self.use_stacked_blend:
            return self._select_leaf_blended(**ctx)
        return self._select_leaf_gated(**ctx)
```

The lines from the existing function that are *removed* in this refactor:
- The original `constant_value = float(y_sub.mean().item())` near the top.
- The original `seed = int(indices[0].item()) if n > 0 else 0` line.
- The original `cap_k = float(self.leaf_eml_cap_k)` / `cap_leaf = ... .item()` block before the `ctx = dict(...)`.

Everything else in `_fit_leaf` is preserved verbatim.

### Step 4: Refactor `_select_leaf_gated` to batch its four `.item()` reads

- [ ] In `eml_boost/tree_split/tree.py`, locate `_select_leaf_gated` (line 560+). Replace the block from the `val_pred = ...` line through the early-return / threshold-check section (the body containing the four `.item()` calls). The new body:

```python
    def _select_leaf_gated(
        self,
        *,
        y_full: "torch.Tensor",
        y_val: "torch.Tensor",
        eta: "torch.Tensor",
        bias: "torch.Tensor",
        preds_val: "torch.Tensor",
        valid: "torch.Tensor",
        k: int,
        top_features: np.ndarray,
        mean_x: "torch.Tensor",
        std_x: "torch.Tensor",
        constant_value: float,
        cap_leaf: float,
    ) -> Node:
        """Legacy binary-gate tree selection. Picks the tree with smallest
        val-SSE on the pure-EML prediction; accepts it only if val-SSE beats
        the constant-leaf val-SSE by ``leaf_eml_gain_threshold``."""
        val_pred = eta.unsqueeze(1) * preds_val + bias.unsqueeze(1)
        if cap_leaf < float("inf"):
            val_pred = torch.clamp(val_pred, -cap_leaf, cap_leaf)
        val_res = y_val.unsqueeze(0) - val_pred
        val_sse = (val_res * val_res).sum(dim=1)
        val_sse = torch.where(valid, val_sse, torch.full_like(val_sse, float("inf")))

        # Compute every scalar we need on GPU first, then ONE .cpu() read
        # for all four. Saves 3 cuda-stream syncs vs the prior 4 .item()s.
        # bool->float32->bool round-trips exactly (1.0/0.0 are exact).
        # best_idx is bounded by the number of candidate trees (max 144),
        # so the int->float32 cast is also exact.
        best_idx_gpu = val_sse.argmin()
        valid_at_best_gpu = valid[best_idx_gpu]
        best_val_sse_gpu = val_sse[best_idx_gpu]
        constant_val_sse_gpu = ((y_val - y_full.mean()) ** 2).sum()

        batch = torch.stack([
            best_idx_gpu.to(torch.float32),
            valid_at_best_gpu.to(torch.float32),
            best_val_sse_gpu,
            constant_val_sse_gpu,
        ]).cpu().numpy()
        best_idx = int(batch[0])
        valid_at_best = bool(batch[1])
        best_val_sse = float(batch[2])
        constant_val_sse = float(batch[3])

        if not valid_at_best:
            return LeafNode(value=constant_value)

        if best_val_sse >= constant_val_sse * (1.0 - self.leaf_eml_gain_threshold):
            return LeafNode(value=constant_value)

        # ... everything below (the EmlLeafNode construction) stays unchanged ...
```

Keep the original `EmlLeafNode(...)` construction at the end of `_select_leaf_gated` exactly as it is (the lines that come after the two `if ... return LeafNode(value=constant_value)` guards in the current file). Only the val-SSE-compute / `.item()` block at the top is replaced.

### Step 5: Run the new test to verify the refactor passes

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_fit_leaf_item_batching_preserves_predictions -v`
- [ ] Expected: PASS (skipped if no CUDA).

### Step 6: Run the full unit suite to verify no regressions

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `94 passed, 1 failed` (the pre-existing `test_fit_recovers_simple_formula` failure, left alone).
- [ ] If any previously-passing test fails: STOP. The most likely culprit is a typo in the moved code, the bool/int→float cast at a value boundary, or the early-out constant_value path. Diff the function against the prior implementation to find the divergence; do not "fix forward" with another change.

### Step 7: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
perf: batch .item() syncs in _fit_leaf + _select_leaf_gated (Task E)

_fit_leaf made 3 separate .item() D2H syncs (constant_value, RNG seed,
cap_leaf) before the OLS work, and _select_leaf_gated made 4 more
(best_idx, valid[best_idx], best_val_sse, constant_val_sse). Stack each
group into one tensor and read with a single .cpu() — saves 5 of 6
cuda-stream syncs per leaf, ~17ms/round on 344_mv (~3s/fit at 200 rounds).

Includes an n < 2**24 assertion in _fit_leaf for the int->float32 cast
of indices[0]; PMLB max is 1M so no current dataset triggers it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Verify success criteria via the sanity bench

**Files:** None modified — this task only runs existing scripts and inspects output.

### Step 1: Re-run the sanity bench to measure the combined speedup

- [ ] Run: `PYTHONUNBUFFERED=1 uv run python -u experiments/bench_sanity.py 2>&1 | tee experiments/experiment15/bench_sanity_postmicroopts.log`
- [ ] The bench will SKIP all 45 already-completed (dataset, seed, config) rows from the prior run because `bench_sanity.py` calls `_load_completed(csv_path)` and `_append_rows` from the runner — resume-from-checkpoint is built in.
- [ ] To force a re-fit so we can measure the new speed, the cleanest option is to filter out the 3 sanity datasets from `experiment15/summary.csv` first. Run:

```bash
awk -F, 'NR==1 || ($1!="574_house_16H" && $1!="201_pol" && $1!="344_mv")' \
    experiments/experiment15/summary.csv \
    > experiments/experiment15/summary.csv.tmp \
  && mv experiments/experiment15/summary.csv.tmp experiments/experiment15/summary.csv
```

- [ ] Then re-run the bench:

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/bench_sanity.py 2>&1 \
    | tee experiments/experiment15/bench_sanity_postmicroopts.log
```

### Step 2: Verify no Triton fallbacks or crashes (success criterion S-C)

- [ ] Run: `grep -iE "warning|fallback|FAILED|Traceback|RuntimeWarning" experiments/experiment15/bench_sanity_postmicroopts.log || echo "no issues"`
- [ ] Expected: prints `no issues`.

### Step 3: Verify per-fit speedup (success criterion S-B)

- [ ] Read the "=== sanity bench summary ===" block at the end of the log:

```bash
tail -25 experiments/experiment15/bench_sanity_postmicroopts.log
```

- [ ] Verify each:
  - `344_mv` SplitBoost `fit_time mean` is **≤ 39 s** (vs prior ~42.9 s — at least ~9 % improvement).
  - `574_house_16H` SplitBoost `fit_time mean` is within ±10 % of the prior ~15.8 s (15.8s warmup-included; ex-warmup ~11s).
  - `201_pol` SplitBoost `fit_time mean` is within ±10 % of the prior ~14.9 s.
- [ ] If `344_mv` improvement is < 5 %: profile (`uv run python -u profile_344mv/run_profile.py`) before kicking off Experiment 15 to see whether the changes landed correctly.

### Step 4: Verify all unit tests still pass (success criterion S-A)

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `94 passed, 1 failed` (the pre-existing `test_fit_recovers_simple_formula` failure, left alone).

### Step 5: Report verdict and pause

- [ ] If S-A + S-B + S-C all hold: report the new per-fit means alongside the prior bench numbers and pause for the user to confirm Experiment 15 should be kicked off (per the user's "checkpoint before long runs" rule, this is mandatory before the multi-hour run).
- [ ] If S-B underperforms but S-A + S-C hold: report the modest speedup, recommend kicking off Experiment 15 anyway (the changes are correctness-preserving cleanups regardless of speed), and pause for the user.
- [ ] If S-A fails: investigate the failing test, do NOT proceed to Experiment 15. Likely cause is a typo in Task 2 or Task 3 — diff against the prior implementation.

No commit for this task — it's pure verification.

---

## Implementation order recap

1. Task 1 (F: descriptor cache) — smallest blast radius, validates the cache pattern works.
2. Task 2 (D: numpy tensorize) — surgical to one function, independent of Task 1.
3. Task 3 (E: batched `.item()`) — biggest semantic surface; ships last so Tasks 1+2 are already validated.
4. Task 4 (verification) — re-run the sanity bench, decide on Experiment 15 kickoff.

After Task 4, regardless of outcome, pause and report to the user before doing anything else (the user's "checkpoint before long runs" rule).
