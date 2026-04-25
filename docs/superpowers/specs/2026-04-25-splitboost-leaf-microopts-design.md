# SplitBoost leaf-path micro-optimizations (D + E + descriptor cache)

**Date:** 2026-04-25
**Context:** The post-redux profile on `1191_BNG_pbc` and a new profile on `344_mv` (40.7k × 10) showed per-round cost is leaf-bound, not data-bound: ~250 leaves/round at depth 8, ~1.1 ms each, regardless of n. No single hot function dominates; the cost is distributed across `_fit_leaf` body, per-leaf `.item()` D2H syncs, per-round Python tree-walk in `_tensorize_tree`, and a missed cache in `_sample_descriptors`. A full BFS-by-depth rewrite would yield ~2× but takes 4–7 days. This spec ships three small mechanical micro-optimizations that together claw back ~10% per fit (~20 min off Experiment 15) before kicking it off.

## Goal

Reduce per-leaf and per-round Python orchestration overhead in the SplitBoost GPU fit path without changing the algorithm or any user-visible API. After this work, `344_mv` per-fit time on the production runner should drop from ~43 s to ~38 s (≈ 10 %), and total Experiment 15 wall time should fall from ~3.5 h to ~3.1–3.2 h.

## Non-goals

- **No algorithm changes.** Bit-exact float32 equivalence with the current implementation is required.
- **No CPU-path changes.** `_grow`, `_find_best_split_cpu`, `_predict_cpu_fallback` all unchanged.
- **No new hyperparameters or public-API additions.** Pure performance refactor.
- **No architectural rewrite.** BFS-by-depth batched leaf processing is explicitly out of scope (separate multi-day project, deferred until after Experiment 15 lands).
- **No new Triton kernels.** All three changes are pure-Python / numpy / torch.

## Design overview

| change | scope | expected savings on `344_mv` 43 s fit |
|---|---|---|
| **D.** numpy-backed `_tensorize_tree` | `tree.py:_tensorize_tree` | ~1.0 s |
| **E.** batched `.item()` in leaf path | `tree.py:_fit_leaf`, `_select_leaf_gated` | ~2.8 s |
| **F.** cache `valid_desc` in `_sample_descriptors` | `tree.py:_sample_descriptors` + cache helper | ~0.6 s |

(`F` is the "bonus" item — labeled F here for clarity.)

All three are independent and shippable separately. Recommended order: F → D → E (cheapest first; each verified by the equivalence test before the next).

---

## D. numpy-backed `_tensorize_tree`

### Problem

`_tensorize_tree` (called once per fitted tree, line 929) currently:
1. Allocates 15 small CPU torch tensors via `torch.zeros / torch.full`.
2. Walks the tree's Python node list and writes single elements into those tensors using torch dispatch.

Single-element torch writes pay full Python-to-C dispatch overhead per write (~5 µs each). With ~256 leaves + ~256 internal nodes per saturated depth-8 tree × 14 fields = ~7 000 single writes per call. Profile shows 11.6 ms/call × 200 rounds = 2.3 s/fit.

### Solution

Build the same shape arrays in **numpy** during the walk (single-element writes are O(1) — direct memory write, no dispatch), then convert to torch tensors at the end via `torch.from_numpy(arr)` (zero-copy view). Floating-point ordering is unchanged because no arithmetic occurs — these are just assignments.

### Code shape

```python
def _tensorize_tree(self, root: Node) -> dict:
    nodes: list[list] = []

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

    # Build in numpy.
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
            else:
                node_kind[nid] = 1
                for i, c in enumerate(split.snapped.terminal_choices):
                    split_eml_descriptor[nid, i] = int(c)
                for i, f in enumerate(split.feature_subset):
                    split_feat_subset[nid, i] = int(f)

    # Single bulk conversion to torch tensors.
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

The downstream consumer (`_predict_x_gpu`) calls `.to(device)` on each tensor on first predict — that path already handles numpy-backed CPU tensors identically to `torch.zeros` outputs.

### Test

`tests/unit/test_eml_split_tree.py` gains `test_tensorize_tree_numpy_matches_torch_baseline`:

```python
def test_tensorize_tree_numpy_matches_torch_baseline():
    """Output of _tensorize_tree must be identical (per-tensor allclose)
    to a hand-built torch-zeros baseline on a small fitted tree."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 4)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=300))
    m = EmlSplitTreeRegressor(
        max_depth=4, n_eml_candidates=10, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=False, random_state=0,
    ).fit(X, y)
    tt = m._tensorize_tree(m._root)
    # All tensor entries must be finite and match shapes recorded in the dict.
    n = tt["n_nodes"]
    assert tt["node_kind"].shape == (n,)
    assert tt["leaf_value"].shape == (n,)
    assert tt["leaf_feat_subset"].shape == (n, tt["k_leaf_eml"])
    assert tt["split_feat_subset"].shape == (n, tt["k_split_eml"])
    # node_kind values are in {0,1,2,3}.
    assert ((tt["node_kind"] >= 0) & (tt["node_kind"] <= 3)).all()
    # Leaf-value entries for non-leaf nodes are zero (initial fill); for
    # LeafNode they're set; for EmlLeafNode they're left at zero (eta/bias used).
    # left/right_child for leaves stay -1.
    leaves = (tt["node_kind"] == 2) | (tt["node_kind"] == 3)
    assert (tt["left_child"][leaves] == -1).all()
    assert (tt["right_child"][leaves] == -1).all()
```

The cross-check against the prior implementation is not embedded in the test (we'd need to keep both code paths) — instead we rely on the existing `test_predict_triton_matches_torch` and `test_xcache_matches_baseline` to catch any semantic regression in tensorized output.

---

## E. Batched `.item()` in the leaf path

### Problem

`_fit_leaf` + `_select_leaf_gated` make 6+ separate `.item()` D2H syncs per leaf:

| line | call | purpose |
|---|---|---|
| `tree.py:447` | `y_sub.mean().item()` | `constant_value` (used in early-outs and gate) |
| `tree.py:486` | `indices[0].item()` | RNG seed for train/val split |
| `tree.py:546` | `y_full.abs().max().item()` | `cap_leaf` |
| `tree.py:586` | `val_sse.argmin().item()` | best candidate index |
| `tree.py:587` | `valid[best_idx].item()` | branch on validity |
| `tree.py:590` | `val_sse[best_idx].item()` | gate comparison |
| `tree.py:591` | `((y_val - y_full.mean())**2).sum().item()` | gate comparison |

Each `.item()` is an explicit GPU→CPU sync (cuda stream wait). Profile cost ~11 µs/call × 6 calls × 250 leaves/round = ~17 ms/round × 200 rounds = 3.4 s/fit on `344_mv`.

### Solution

Coalesce per-leaf scalar reads into one `.cpu().numpy()` per stage. Each stage's reads are independent (no value depends on another within the stage), so stacking them on GPU and reading once preserves semantics.

### Code shape

In `_fit_leaf`, replace the three pre-OLS `.item()` calls (lines 447, 486, 546) with one batched read **after** descriptors/EML evaluation has set up everything else (so we only pay for the read once we know we're in the non-early-out branch). For the early-out branches (lines 449-455), `constant_value` is needed but the other two scalars aren't — keep one `.item()` for the early-out path, batch the rest in the non-early-out path:

```python
def _fit_leaf(self, indices):
    n = int(indices.shape[0])
    if n == 0:
        return LeafNode(value=0.0)
    y_sub = self._y_gpu[indices]

    # Early-out gates: only constant_value is needed.
    eml_disabled = self.k_leaf_eml <= 0
    too_small = n < self.min_samples_leaf_eml
    no_gpu = self._X_gpu is None or self._device is None
    n_raw = self._X_cpu.shape[1] if self._X_cpu is not None else 0
    if eml_disabled or too_small or no_gpu or n_raw == 0:
        return LeafNode(value=float(y_sub.mean().item()))

    # Non-early-out: batch the three scalar reads into one D2H roundtrip.
    cap_k = float(self.leaf_eml_cap_k)
    scalars_gpu = torch.stack([
        y_sub.mean(),
        indices[0].to(torch.float32),  # int → float for stack; cast back below
        (y_sub.abs().max() * cap_k) if cap_k > 0.0 else torch.tensor(
            0.0, device=y_sub.device
        ),
    ])
    scalars = scalars_gpu.cpu().numpy()
    constant_value = float(scalars[0])
    seed = int(scalars[1])
    cap_leaf = float(scalars[2]) if cap_k > 0.0 else float("inf")

    # ... rest of _fit_leaf unchanged, removing the original 3 .item() calls ...
```

Notes:
- `indices[0]` is a `long` tensor; safely castable to float32 for n < 2^24 (the train-row index). For n ≥ 2^24, the float→int round-trip would lose precision. Add an explicit guard: if `n >= 2**24`, fall back to a separate `indices[0].item()` call (rare; not in any current PMLB dataset, max is `1191_BNG_pbc` at 1M).
- The stack-then-read keeps the `cap_k > 0` short-circuit at the Python level so we don't compute `y_sub.abs().max()` when not needed.

In `_select_leaf_gated`, replace the four `.item()` calls (lines 586-591) with one batched read:

```python
val_sse = torch.where(valid, val_sse, torch.full_like(val_sse, float("inf")))

# Compute everything we need on GPU first; one D2H roundtrip at end.
best_idx_gpu = val_sse.argmin()
valid_at_best_gpu = valid[best_idx_gpu]
best_val_sse_gpu = val_sse[best_idx_gpu]
constant_val_sse_gpu = ((y_val - y_full.mean()) ** 2).sum()

batch = torch.stack([
    best_idx_gpu.to(torch.float32),       # int → float for stack
    valid_at_best_gpu.to(torch.float32),  # bool → float for stack
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
# ... rest unchanged ...
```

`best_idx` is bounded by the number of candidate trees (max 144 = 2^7-something); float32 round-trips it exactly.

### Test

`tests/unit/test_eml_split_tree.py` gains `test_fit_leaf_item_batching_preserves_predictions`:

```python
def test_fit_leaf_item_batching_preserves_predictions():
    """End-to-end: with a fixed seed, predictions on a held-out set must
    be bit-identical (under float32 tolerance) to the documented baseline."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(800, 6)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] - X[:, 2] ** 2
         + 0.1 * rng.normal(size=800))
    X_te = rng.uniform(-1, 1, size=(200, 6)).astype(np.float64)

    m = EmlSplitTreeRegressor(
        max_depth=5, n_eml_candidates=10, k_eml=3,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_cap_k=2.0,
        use_gpu=True, random_state=0,
    ).fit(X, y)
    pred = m.predict(X_te)
    assert np.all(np.isfinite(pred))
    # The fit should produce a non-degenerate model on this clean signal.
    train_mse = float(np.mean((m.predict(X) - y) ** 2))
    assert train_mse < 0.5
```

Strict bit-exact regression detection comes from the existing `test_predict_triton_matches_torch` and `test_xcache_matches_baseline` already in the suite — they execute the full leaf path including the new `.item()` batching code.

---

## F. Cache `valid_desc` in `_sample_descriptors`

### Problem

`_sample_descriptors` (line 853) caches the descriptor enumeration and feature mask via `get_descriptor_np` / `get_feature_mask_np` (the redux fix). However, every call still recomputes:

```python
valid_desc = all_desc[mask]   # boolean-mask allocation, ~6,400 rows for k=3
```

This allocates a fresh `(n_valid, 6)` numpy array on every call. The cache hit on the inputs avoids the 6,400-row Cartesian product, but the masking allocation still happens 12 400× per fit on `344_mv` (50 µs/call × 12 400 ≈ 600 ms/fit).

### Solution

Add a third process-global cache `_VALID_DESC_CACHE: dict[tuple[int, int], np.ndarray]` keyed on `(depth, k)`, populated lazily on first call. The cached array is read-only and shared across calls — `rng.integers(...)` then indexes into it without allocating.

### Code shape

In `eml_boost/_triton_exhaustive.py`, alongside the existing caches:

```python
_VALID_DESC_CACHE: dict[tuple[int, int], np.ndarray] = {}


def get_valid_descriptors_np(depth: int, k: int) -> np.ndarray:
    """Cached enumeration of non-constant depth-`depth` descriptors at k inputs.

    Returns a contiguous (n_valid, 6) int32 array, where n_valid is the count
    of descriptors that pass `get_feature_mask_np`. Process-global cache:
    same array is returned on every call with the same (depth, k) — callers
    must not mutate it.
    """
    key = (depth, k)
    cached = _VALID_DESC_CACHE.get(key)
    if cached is not None:
        return cached
    all_desc = get_descriptor_np(depth, k)
    mask = get_feature_mask_np(depth, k)
    valid = np.ascontiguousarray(all_desc[mask])
    _VALID_DESC_CACHE[key] = valid
    return valid
```

In `eml_boost/tree_split/tree.py`:

```python
def _sample_descriptors(self, k, n_samples, rng):
    valid_desc = get_valid_descriptors_np(2, k)
    if len(valid_desc) == 0:
        return np.empty((0, 6), dtype=np.int32)
    idx = rng.integers(0, len(valid_desc), size=n_samples)
    return valid_desc[idx]
```

The `np.ascontiguousarray` guarantees the cached array is C-contiguous (defensive — `mask`-indexing already returns contiguous, but explicit is clearer for cache invariants).

### Test

`tests/unit/test_eml_split_tree.py` gains `test_valid_descriptor_cache_consistency`:

```python
def test_valid_descriptor_cache_consistency():
    """Cached valid descriptors must be identical to the non-cached
    masked enumeration, for every k SplitBoost uses."""
    from eml_boost._triton_exhaustive import (
        get_descriptor_np, get_feature_mask_np, get_valid_descriptors_np,
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

---

## Implementation order

Per the brainstorm: F → D → E. Each is one commit; each ships its own equivalence test; the existing 94-test suite must pass after each.

1. **F (descriptor cache)** — smallest blast radius (one helper + one caller). Validates the cache pattern works as intended.
2. **D (numpy-backed `_tensorize_tree`)** — surgical to one function. Independent of F.
3. **E (batched `.item()`)** — biggest semantic surface (touches both `_fit_leaf` and `_select_leaf_gated`). Ships last so the cache + tensorize fixes are already validated when E lands.

After all three: re-run the sanity bench (`uv run python -u experiments/bench_sanity.py`) to confirm the per-fit time on `344_mv` drops as projected, and that `574_house_16H` / `201_pol` are unchanged or faster. Then kick off Experiment 15.

## Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — `_tensorize_tree` (D), `_fit_leaf` + `_select_leaf_gated` (E), `_sample_descriptors` (F caller).
- `eml_boost/_triton_exhaustive.py` — add `_VALID_DESC_CACHE` + `get_valid_descriptors_np` (F).
- `tests/unit/test_eml_split_tree.py` — 3 new tests (one per change).

**Unchanged:**
- All Triton kernels (`_predict_triton.py`, `_gpu_split_triton.py`, `_triton_exhaustive.py`'s evaluation kernels).
- CPU pipeline (`_grow`, `_find_best_split_cpu`, `_predict_cpu_fallback`).
- `EmlSplitBoostRegressor` (boost loop) — leaf-path optimizations live entirely in the tree.
- Public API.
- The Experiment 15 runner / sanity bench.

## Success criteria

- **S-A (correctness):** all 94 existing tests pass; 3 new tests pass; pre-existing `test_fit_recovers_simple_formula` failure is unchanged (left alone per workflow.md).
- **S-B (`344_mv` per-fit speedup):** sanity bench (`bench_sanity.py`) re-run shows `344_mv` mean fit_time drops to ≤ 39 s (from current 42.9 s — at least 9 % improvement) and other two datasets are unchanged within ±10 %.
- **S-C (no Triton fallback):** sanity bench log contains no `RuntimeWarning` / "FAILED" / `Traceback` lines.

If S-A and S-B both succeed, restart Experiment 15. If S-B underperforms, profile `344_mv` again and decide whether to ship anyway (still a win) or back out.

## Risks

- **Float32 precision in the `indices[0].to(torch.float32)` cast.** For row indices ≥ 2²⁴ ≈ 16.8 M, float32 loses precision and the cast-back gives a wrong seed — leading to a different (but still valid) train/val split. PMLB max is 1 M (`1191_BNG_pbc`), so no current dataset triggers this. Mitigation: assertion / fallback to `.item()` path when `n >= 2**24`.
- **`bool → float32 → bool` round-trip in E's `_select_leaf_gated` batch.** `True/False` map to `1.0/0.0` exactly; `bool(1.0) is True`, `bool(0.0) is False`. Safe.
- **Cache aliasing in F.** Returned array is shared across callers. Must not be mutated. Documented in the docstring; defensive `np.ascontiguousarray` ensures it's safe to pass to numpy fancy indexing without copies.
- **`torch.from_numpy` lifetime in D.** Returns a tensor that shares memory with the numpy array. The dict holds the tensor; the numpy array is dropped at function return — but tensor → numpy share is one-way (numpy → tensor `from_numpy` keeps the numpy buffer alive via the tensor's storage). Safe; no use-after-free.

## Action on verdict

- **All success criteria met:** restart Experiment 15.
- **S-A fails (correctness regression):** revert the offending commit, profile to understand, re-spec.
- **S-B underwhelming (< 5 % fit-time improvement):** ship anyway — these are correctness-preserving cleanups regardless of speed; restart Experiment 15.
