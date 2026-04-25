# SplitBoost GPU Port

**Date:** 2026-04-25
**Context:** Experiment 15 was launched but profiling against XGBoost/LightGBM on the BNG_* synthetic datasets revealed `EmlSplitBoostRegressor` is Python-bound (8 cores at 800% CPU, GPU at 11% utilization). On the 1M-row `1191_BNG_pbc` dataset, SplitBoost takes ~690 s/fit while XGBoost CUDA takes 1.1 s — a 600× gap. The Triton EML kernel and torch-based histogram split-finding (Experiments 7-8) handle the inner ops, but the outer orchestration (`_grow` recursion, `_predict_vec`, per-node host↔device transfers, `_top_features_by_corr` numpy correlation) is pure Python and dominates wall-clock on large datasets. This document specifies a focused GPU port to close that gap.

## Goal

Port the GPU pipeline of `EmlSplitTreeRegressor` so that *all* hot-path operations during fit and predict are on-device. Specifically: (1) eliminate per-node host↔device round-trips by keeping `indices`, `y`, and `left_mask` on GPU throughout `_grow`; (2) port `_top_features_by_corr` to torch GPU ops; (3) replace the recursive Python `_predict_vec` with a tensorized GPU traversal. The CPU fallback path stays untouched. Target: 5-20× speedup on datasets with n > 100k, GPU utilization above 50% during fit.

## Non-goals

- **No Triton kernels.** The existing EML evaluation Triton kernel stays. New code uses pure torch GPU ops.
- **No wide-tree (parallel-sibling) growing.** Each tree still grows depth-first via Python recursion; only the inner work moves to GPU.
- **No CPU pipeline changes.** `use_gpu=False` and no-CUDA paths keep numpy code untouched.
- **No new hyperparameters.** Pure performance refactor; behavior must match the GPU pipeline pre-port (within float32 tolerance).
- **No changes to `EmlSplitBoostRegressor`'s boost loop.** It already calls `tree.predict(X_tr)`; that will benefit automatically once predict is GPU.

## Design

### Storage changes in `EmlSplitTreeRegressor.fit()`

Add one new attribute and keep the existing GPU storage:

```python
self._device = torch.device("cuda")  # already present when use_gpu and CUDA available
self._X_gpu = torch.tensor(X, dtype=torch.float32, device=self._device)  # already
self._y_gpu = torch.tensor(y, dtype=torch.float32, device=self._device)  # NEW
self._global_mean_gpu, self._global_std_gpu = ...  # already
```

`self._y_cpu` is not needed — the GPU pipeline never touches numpy `y` after the initial allocation.

### Dispatch in `fit()`

```python
if self._device is not None:
    indices_gpu = torch.arange(len(X), device=self._device, dtype=torch.long)
    self._root = self._grow_gpu(indices_gpu, depth=0, rng=rng)
    self._gpu_tree = self._tensorize_tree(self._root)  # for fast predict
else:
    indices_np = np.arange(len(X))
    self._root = self._grow(indices_np, y, depth=0, rng=rng)
    # No _gpu_tree; predict() uses numpy fallback
```

GPU storage handles (`_X_gpu`, `_y_gpu`, etc.) are released after fit completes; the persistent state for predict is `self._root` (Python tree) and `self._gpu_tree` (CPU-side tensors that move to GPU on demand).

### `_grow_gpu(indices: torch.Tensor, depth: int, rng) -> Node`

Mirror the structure of the existing `_grow` but with torch tensors:

```python
def _grow_gpu(self, indices: torch.Tensor, depth: int, rng):
    n = int(indices.shape[0])
    if depth >= self.max_depth or n <= 2 * self.min_samples_leaf:
        return self._fit_leaf(indices)
    best = self._find_best_split_gpu(indices, rng)
    if best is None:
        return self._fit_leaf(indices)
    split, _gain, left_mask = best   # left_mask is now a GPU bool tensor
    if int(left_mask.sum().item()) < self.min_samples_leaf or \
       int((~left_mask).sum().item()) < self.min_samples_leaf:
        return self._fit_leaf(indices)
    return InternalNode(
        split=split,
        left=self._grow_gpu(indices[left_mask], depth + 1, rng),
        right=self._grow_gpu(indices[~left_mask], depth + 1, rng),
    )
```

Key changes vs. existing `_grow`:
- `indices` is a torch tensor, not numpy
- No `y_sub` parameter — the split-finder gathers `y` from `self._y_gpu` itself
- `left_mask` is a GPU bool tensor; `indices[left_mask]` stays on GPU

### `_find_best_split_gpu(indices: torch.Tensor, rng)` refactor

Three round-trips eliminated; one numpy correlation moved to GPU:

```python
def _find_best_split_gpu(self, indices: torch.Tensor, rng):
    device = self._device
    X_node = self._X_gpu[indices]          # already GPU op
    y_node = self._y_gpu[indices]          # NEW: GPU gather instead of host→device copy
    n_raw = X_node.shape[1]
    feat_cols = [X_node]
    valid_candidates = None
    top_features = None
    k_used = 0

    if self.n_eml_candidates > 0 and n_raw > 0:
        k_used = min(self.k_eml, n_raw)
        top_features = self._top_features_by_corr_gpu(X_node, y_node, k_used)  # NEW: GPU
        candidates = self._sample_descriptors(k_used, self.n_eml_candidates, rng)
        if len(candidates) > 0:
            X_sub = X_node[:, top_features]
            desc_gpu = torch.tensor(candidates, dtype=torch.int32, device=device)
            eml_values = evaluate_trees_triton(desc_gpu, X_sub, k_used)
            finite = torch.isfinite(eml_values).all(dim=1)
            if finite.any():
                # `top_features` returned by GPU corr — keep it as a GPU long tensor for slicing,
                # but materialize a numpy view for snapshot when we emit an EmlSplit.
                valid_candidates = candidates[finite.cpu().numpy()]
                feat_cols.append(eml_values[finite].T.contiguous())
            else:
                valid_candidates = candidates[:0]
        else:
            valid_candidates = candidates[:0]

    all_feats = torch.cat(feat_cols, dim=1) if len(feat_cols) > 1 else feat_cols[0]
    best_idx, best_t, best_gain = gpu_histogram_split(
        all_feats, y_node, self.n_bins, min_leaf_count=self.min_samples_leaf,
    )
    if best_gain <= 0:
        return None

    if best_idx < n_raw:
        split = RawSplit(feature_idx=int(best_idx), threshold=float(best_t))
    else:
        c_idx = int(best_idx) - n_raw
        desc = valid_candidates[c_idx]
        # top_features is GPU; copy the small (k_used,) tensor once for the dataclass
        top_features_np = top_features.cpu().numpy()
        split = EmlSplit(
            snapped=SnappedTree(
                depth=2, k=k_used, internal_input_count=2, leaf_input_count=4,
                terminal_choices=tuple(int(v) for v in desc),
            ),
            feature_subset=tuple(int(v) for v in top_features_np),
            threshold=float(best_t),
        )

    left_mask = all_feats[:, best_idx] <= best_t   # GPU bool tensor; NO .cpu().numpy()
    return split, float(best_gain), left_mask
```

The `valid_candidates` numpy view is unavoidable: `candidates` is generated by numpy `rng` and stored as numpy ints; the `[finite.cpu().numpy()]` view gives us a small `(n_finite,)` array used later to construct the `EmlSplit` dataclass — that one-time tiny transfer is fine.

`top_features.cpu().numpy()` on the chosen split is also unavoidable but happens at most once per node, on a `(k_used,)` tensor (3 elements typically) — irrelevant.

### `_top_features_by_corr_gpu(X: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor`

GPU-native version of the existing numpy correlation helper. Returns a long tensor on `self._device`:

```python
def _top_features_by_corr_gpu(self, X, y, k):
    # X: (n, d), y: (n,)
    n = X.shape[0]
    if n == 0 or k == 0:
        return torch.empty(0, dtype=torch.long, device=X.device)
    X_centered = X - X.mean(dim=0, keepdim=True)
    y_centered = y - y.mean()
    num = (X_centered * y_centered.unsqueeze(1)).sum(dim=0)
    denom = X_centered.norm(dim=0) * y_centered.norm() + 1e-12
    corr = (num / denom).abs()
    k_clipped = min(k, X.shape[1])
    return torch.topk(corr, k_clipped, sorted=False).indices.to(torch.long)
```

Numerical equivalence with the existing numpy version is within float32 tolerance — different cores compute the centered-moment slightly differently, but the top-k selection is robust.

### `_fit_leaf(indices: torch.Tensor)` refactor

The existing implementation already does GPU work but accepts numpy `indices`. Remove the conversion and accept a torch tensor directly:

```python
def _fit_leaf(self, indices: torch.Tensor) -> Node:
    n = int(indices.shape[0])
    y_sub = self._y_gpu[indices]   # GPU gather (was: torch.tensor(y_sub, dtype=float32, device=device) host copy)
    constant_value = float(y_sub.mean().item()) if n > 0 else 0.0
    # ... rest of the function unchanged ...
```

Internally the existing code path already uses `idx_gpu = torch.from_numpy(indices).to(device=device, dtype=torch.long)` — that line goes away. Everything else stays.

### `_tensorize_tree(self, root: Node) -> dict`

Walk the fitted Python tree once and emit a dict of CPU-side tensors. Tensors are sized to the total node count (internal + leaf) and indexed by node-id.

```python
def _tensorize_tree(self, root):
    # First pass: assign node ids and collect node info
    nodes = []  # list of (node_id, node_obj, left_id, right_id)
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
    # Allocate flat tensors
    node_kind = torch.zeros(n_nodes, dtype=torch.int8)
    left_child = torch.full((n_nodes,), -1, dtype=torch.int32)
    right_child = torch.full((n_nodes,), -1, dtype=torch.int32)
    feature_idx = torch.zeros(n_nodes, dtype=torch.int32)
    threshold = torch.zeros(n_nodes, dtype=torch.float32)
    leaf_value = torch.zeros(n_nodes, dtype=torch.float32)
    eml_descriptor = torch.zeros((n_nodes, 6), dtype=torch.int32)
    # k_leaf_eml is a known constant for this fit; size leaf-eml tensors accordingly
    K = max(1, self.k_leaf_eml)
    leaf_eta = torch.zeros(n_nodes, dtype=torch.float32)
    leaf_bias = torch.zeros(n_nodes, dtype=torch.float32)
    leaf_cap = torch.full((n_nodes,), float("inf"), dtype=torch.float32)
    leaf_feat_subset = torch.zeros((n_nodes, K), dtype=torch.int32)
    leaf_feat_mean = torch.zeros((n_nodes, K), dtype=torch.float32)
    leaf_feat_std = torch.ones((n_nodes, K), dtype=torch.float32)
    # For EmlSplit internal nodes: store feat_subset alongside descriptor
    split_feat_subset = torch.zeros((n_nodes, K), dtype=torch.int32)

    # Second pass: fill tensors based on node type
    for nid, node, lc, rc in nodes:
        left_child[nid] = lc; right_child[nid] = rc
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
                eml_descriptor[nid, i] = int(c)
        elif isinstance(node, InternalNode):
            split = node.split
            threshold[nid] = float(split.threshold)
            if isinstance(split, RawSplit):
                node_kind[nid] = 0
                feature_idx[nid] = int(split.feature_idx)
            else:  # EmlSplit
                node_kind[nid] = 1
                for i, c in enumerate(split.snapped.terminal_choices):
                    eml_descriptor[nid, i] = int(c)
                for i, f in enumerate(split.feature_subset):
                    split_feat_subset[nid, i] = int(f)

    return {
        "node_kind": node_kind, "left_child": left_child, "right_child": right_child,
        "feature_idx": feature_idx, "threshold": threshold,
        "leaf_value": leaf_value,
        "eml_descriptor": eml_descriptor,
        "split_feat_subset": split_feat_subset,
        "leaf_eta": leaf_eta, "leaf_bias": leaf_bias, "leaf_cap": leaf_cap,
        "leaf_feat_subset": leaf_feat_subset,
        "leaf_feat_mean": leaf_feat_mean, "leaf_feat_std": leaf_feat_std,
        "k_leaf_eml": K,
    }
```

The dict is stored as `self._gpu_tree` after fit. It's small (a few KB to maybe 50 KB per tree) and lives on CPU until first predict call, when it's moved to GPU.

### GPU `predict()` algorithm

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if self._gpu_tree is None or not torch.cuda.is_available():
        return self._predict_cpu_fallback(X)

    device = torch.device("cuda")
    if not hasattr(self, "_gpu_tree_on_device") or self._gpu_tree_on_device != device:
        # Lazy move to GPU on first predict
        self._gpu_tree = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in self._gpu_tree.items()}
        self._gpu_tree_on_device = device

    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    n_samples = X_gpu.shape[0]

    t = self._gpu_tree
    current = torch.zeros(n_samples, dtype=torch.long, device=device)  # all start at root (id=0)

    # Iterate up to max_depth times. Each iteration advances the samples that are
    # still at internal nodes; samples that have already reached a leaf stay put.
    for _ in range(self.max_depth + 1):
        kind_now = t["node_kind"][current]
        is_internal = kind_now < 2
        if not is_internal.any():
            break

        # Compute go-left for every sample (will be ignored at leaves).
        feat = t["feature_idx"][current]                    # (n,)
        thr = t["threshold"][current]                       # (n,)
        # Raw split: gather X[i, feat[i]]
        raw_vals = X_gpu.gather(1, feat.unsqueeze(1).long()).squeeze(1)
        is_raw = (kind_now == 0)
        is_eml = (kind_now == 1)

        # EML split values: only compute for samples whose current node is an EML internal
        eml_vals = torch.zeros(n_samples, dtype=torch.float32, device=device)
        if is_eml.any():
            eml_indices = is_eml.nonzero(as_tuple=True)[0]
            eml_node_ids = current[eml_indices]
            descs = t["eml_descriptor"][eml_node_ids]                           # (m, 6)
            feat_subs = t["split_feat_subset"][eml_node_ids]                    # (m, K)
            X_eml_input = X_gpu[eml_indices.unsqueeze(1).expand(-1, t["k_leaf_eml"]),
                                feat_subs.long()]                               # (m, K)
            # Standardize+clamp via predict-time global stats — reuse standardization
            # only if EML LEAVES need it; for EML SPLITS the descriptor was fit on raw
            # X, so no standardization here.
            eml_out = evaluate_trees_torch_per_sample(descs, X_eml_input, t["k_leaf_eml"])
            eml_vals[eml_indices] = eml_out

        split_vals = torch.where(is_raw, raw_vals, eml_vals)
        go_left = split_vals <= thr

        # Advance only internal-node samples; leaves stay at their current node id
        next_node = torch.where(go_left, t["left_child"][current].long(), t["right_child"][current].long())
        current = torch.where(is_internal, next_node, current)

    # All samples have now reached a leaf. Compute predictions per leaf type.
    kind_final = t["node_kind"][current]
    out_gpu = torch.zeros(n_samples, dtype=torch.float32, device=device)

    is_leaf_const = (kind_final == 2)
    if is_leaf_const.any():
        out_gpu[is_leaf_const] = t["leaf_value"][current[is_leaf_const]]

    is_leaf_eml = (kind_final == 3)
    if is_leaf_eml.any():
        leaf_idx = is_leaf_eml.nonzero(as_tuple=True)[0]
        leaf_node_ids = current[leaf_idx]
        # Compute eta * eml(standardized_X) + bias, clip to cap
        feat_subs = t["leaf_feat_subset"][leaf_node_ids]                        # (m, K)
        means = t["leaf_feat_mean"][leaf_node_ids]                              # (m, K)
        stds = t["leaf_feat_std"][leaf_node_ids]                                # (m, K)
        descs = t["eml_descriptor"][leaf_node_ids]                              # (m, 6)
        eta = t["leaf_eta"][leaf_node_ids]                                      # (m,)
        bias = t["leaf_bias"][leaf_node_ids]                                    # (m,)
        cap = t["leaf_cap"][leaf_node_ids]                                      # (m,)

        K = t["k_leaf_eml"]
        X_leaf_raw = X_gpu[leaf_idx.unsqueeze(1).expand(-1, K),
                            feat_subs.long()]                                   # (m, K)
        X_leaf_std = torch.clamp((X_leaf_raw - means) / stds, -3.0, 3.0)
        eml_pred = evaluate_trees_torch_per_sample(descs, X_leaf_std, K)        # (m,)
        pred = eta * eml_pred + bias
        # Apply cap where finite
        capped = torch.where(cap < float("inf"), torch.clamp(pred, -cap, cap), pred)
        out_gpu[leaf_idx] = capped

    return out_gpu.cpu().numpy().astype(np.float64)
```

`evaluate_trees_torch_per_sample` is a new helper: given `(m, 6)` descriptors and `(m, K)` per-sample feature inputs, evaluate one tree per sample (not the same tree for all samples — different leaves use different snapped expressions). Implementable with the existing depth-2 grammar evaluator structure but vectorized differently. If the existing `evaluate_trees_torch` doesn't support this shape directly, add a small wrapper.

### CPU fallback

The existing `_predict_vec` stays as `_predict_cpu_fallback`. Triggered only when `_gpu_tree is None` (fit was on CPU) or CUDA isn't available at predict time.

## Testing

Three new tests in `tests/unit/test_eml_split_tree.py`:

1. **`test_gpu_grow_matches_cpu_grow`** — fit two regressors on the same `y = exp(x_0)` synthetic with identical config except `use_gpu=True/False`. Assert the leaf count and structure match (same `max_depth`, similar leaf assignments) and `predict()` outputs agree to `np.allclose(rtol=1e-3, atol=1e-4)`. Tests that the GPU `_grow` produces equivalent trees.

2. **`test_gpu_predict_matches_cpu_predict`** — fit one regressor with `use_gpu=True`. Predict via the new GPU path. Then temporarily set `self._gpu_tree = None` and predict again (which falls back to `_predict_cpu_fallback`). Assert the two outputs agree to `np.allclose(rtol=1e-4, atol=1e-5)`. Tests numerical equivalence of the two predict paths.

3. **`test_gpu_speedup_on_synthetic_large`** — `n=100_000, k=10` synthetic. Fit at `max_depth=8`. Assert wall-clock < 30 seconds (loose sanity bound — pre-port this would take 60-120 s based on the cpu_small extrapolation). Skipped on non-CUDA.

All 28 existing tests must still pass.

## Risks

- **Float32 vs float64 numerical drift.** GPU pipeline runs in float32; existing CPU path mixes float64 (numpy) and float32 (GPU when used). The `test_gpu_grow_matches_cpu_grow` test uses `rtol=1e-3` to accommodate this. If two paths produce structurally different trees on the same fit data, the test will fail and we'll need to investigate (most likely cause: a borderline split where float32 vs float64 picks different thresholds).
- **GPU memory.** `_X_gpu` for a 1M-row × 100-feature dataset is 400 MB float32. We already hold this in current code — no change. `_gpu_tree` is small per tree but boost loops keep 200 trees; on disk these are Python objects but at predict time they get tensorized. If memory becomes a concern, predict can re-tensorize per-tree per call. For Experiment 15 we expect to fit one ensemble at a time, so this isn't an issue in practice.
- **`evaluate_trees_torch` per-sample variant.** The existing kernel evaluates one descriptor on many samples; the predict path needs many descriptors on many samples (one per leaf). Likely needs a small wrapper that loops descriptors or batches them. Will validate during implementation.
- **Lazy GPU-tree move on first predict.** If the regressor is unpickled or used after CUDA context loss, `_gpu_tree_on_device` may be stale. Protect with try/except and rebuild. Probably YAGNI for now.

## Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — add `_y_gpu`, `_grow_gpu`, `_top_features_by_corr_gpu`, `_tensorize_tree`, GPU `predict()`, `_predict_cpu_fallback` (renamed from `_predict_vec`); refactor `_find_best_split_gpu` and `_fit_leaf` to accept GPU tensor `indices`.
- `tests/unit/test_eml_split_tree.py` — add 3 new tests.

**Possibly modified:**
- `eml_boost/_triton_exhaustive.py` — may need a per-sample variant of `evaluate_trees_torch` for the predict path. Add only if the existing kernel can't be adapted with a wrapper.

**Unchanged:**
- `eml_boost/tree_split/ensemble.py` — boost loop benefits transparently from faster `tree.predict`.
- `eml_boost/tree_split/_gpu_split.py`, `eml_boost/tree_split/nodes.py`.

## Success criteria

- **S-A (correctness):** all 28 existing unit tests pass; the 3 new equivalence tests pass.
- **S-B (speed):** SplitBoost fit on `1191_BNG_pbc` (1M rows, 18 features) completes in **< 60 s per seed** (down from ~690 s). That's a 10× speedup.
- **S-C (GPU utilization):** during `1191_BNG_pbc` fit, `nvidia-smi` reports GPU utilization > 50% averaged over a 30-second window. Indicates Python overhead is no longer dominant.

If S-B and S-C are met, restart Experiment 15. Estimated full-suite runtime drops from 13-15 hours to ~2-3 hours.

## Action on verdict

- **All success criteria met:** ship the GPU port as the new default GPU pipeline; restart Experiment 15.
- **Correctness OK but speed underwhelming:** identify the new bottleneck via profiler and iterate.
- **Correctness fails:** revert; the 7-dataset story still ships under the existing implementation. Document the float32 drift as a known limitation.
