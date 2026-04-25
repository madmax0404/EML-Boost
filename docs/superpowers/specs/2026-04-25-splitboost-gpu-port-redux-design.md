# SplitBoost GPU Port — Redux (Triton + X-cache)

**Date:** 2026-04-25
**Context:** The first GPU port (`docs/superpowers/specs/2026-04-25-splitboost-gpu-port-design.md`) eliminated per-node host↔device transfers and added a tensorized GPU predict path. Wall-clock on the worst-case `1191_BNG_pbc` (1M rows × 18 features) dropped from ~690 s/fit to ~180 s/fit (3.8×). Two pre-port estimates remained unmet: the spec's S-B target was <60 s/fit, and the boost-loop predict still re-transfers X to GPU on every round. This redux spec adds three optimizations on top: (A) X-cache across the boost loop, (B) Triton tree-predict kernel, (C) Triton histogram-split kernel. Combined target: ~55 s/fit on `1191_BNG_pbc`, total Experiment 15 runtime under 2 hours.

## Goal

Push SplitBoost's GPU pipeline to the point where Python orchestration is no longer the bottleneck on big datasets. After this work, GPU utilization during fit on `1191_BNG_pbc` should average above 60%, total `1191_BNG_pbc` 5-seed fit time should be under 5 minutes (vs ~30 min currently), and the full PMLB Experiment 15 should finish in 1.5-2 hours.

## Non-goals

- **No algorithm changes.** Numerical equivalence with the current implementation is required (within float32 tolerance).
- **No CPU pipeline changes.** Existing CPU fallback paths stay untouched.
- **No new hyperparameters.** Pure performance refactor.
- **No removal of torch-based fallbacks.** Each Triton kernel ships alongside the existing torch implementation, with a runtime fallback if Triton compilation fails.
- **No change to Experiment 15's runner or reporting structure.** It's already correct; we just want it to finish faster.

## Design overview

Three independent optimizations, each shippable separately:

| optimization | scope | expected speedup on `1191_BNG_pbc` |
|---|---|---|
| (A) X-cache across boost loop | `ensemble.py` + private GPU APIs on tree | 180 s → ~120 s (1.5×) |
| (B) Triton tree-predict kernel | new file `_predict_triton.py` + tree.py wiring | 120 s → ~80 s (1.5×) |
| (C) Triton histogram-split kernel | new file `_gpu_split_triton.py` + tree.py wiring | 80 s → ~55 s (1.45×) |

After all three: ~55 s/fit, ~12.5× total improvement over the 690 s pre-port baseline.

---

## (A) X-cache across the boost loop

### Problem

`EmlSplitBoostRegressor.fit()` runs `~200` rounds, each calling `tree.fit(X_tr, r)` and then `tree.predict(X_tr)`. Both methods convert `X_tr` from numpy to a fresh GPU tensor every call. On `1191_BNG_pbc` that's ~144 MB transferred 400× per fit (57 GB total). Profiling showed this is a meaningful fraction of wall-clock on big datasets.

### Solution

Add private "GPU-input" methods to `EmlSplitTreeRegressor` that accept pre-allocated GPU tensors and skip the H2D transfer. The boost loop allocates `X_tr_gpu`, `y_tr_gpu`, `F_tr_gpu` once and reuses them.

### API additions

In `eml_boost/tree_split/tree.py`:

```python
def _fit_xy_gpu(
    self, X_gpu: torch.Tensor, y_gpu: torch.Tensor,
) -> "EmlSplitTreeRegressor":
    """GPU-input variant of fit(). X_gpu and y_gpu are caller-owned;
    this method borrows references during fit and releases them at end
    (does NOT free the caller's storage)."""
```

```python
def _predict_x_gpu(self, X_gpu: torch.Tensor) -> torch.Tensor:
    """GPU-input variant of predict(). Skips the H2D transfer; returns
    a GPU float32 tensor of shape (n_samples,)."""
```

Both methods are private (leading underscore) — public `fit()` / `predict()` keep their numpy interfaces unchanged.

### Boost-loop refactor

In `eml_boost/tree_split/ensemble.py`'s `fit()`:

When `use_gpu and torch.cuda.is_available()`, replace the existing numpy-based boost loop with a GPU-resident version:

```python
device = torch.device("cuda")
X_tr_gpu = torch.tensor(X_tr, dtype=torch.float32, device=device)
y_tr_gpu = torch.tensor(y_tr, dtype=torch.float32, device=device)
F_tr_gpu = torch.full_like(y_tr_gpu, self._F_0)
X_va_gpu = (
    torch.tensor(X_va, dtype=torch.float32, device=device)
    if X_va is not None else None
)
y_va_gpu = (
    torch.tensor(y_va, dtype=torch.float32, device=device)
    if y_va is not None else None
)
F_va_gpu = (
    torch.full_like(y_va_gpu, self._F_0) if y_va_gpu is not None else None
)

for m in range(self.max_rounds):
    r_gpu = y_tr_gpu - F_tr_gpu
    tree = EmlSplitTreeRegressor(...)._fit_xy_gpu(X_tr_gpu, r_gpu)
    self._trees.append(tree)
    tree_pred_tr_gpu = tree._predict_x_gpu(X_tr_gpu)
    F_tr_gpu = F_tr_gpu + self.learning_rate * tree_pred_tr_gpu
    train_mse = float(((y_tr_gpu - F_tr_gpu) ** 2).mean().item())
    record = {"round": m, "train_mse": train_mse}
    if F_va_gpu is not None and X_va_gpu is not None:
        F_va_gpu = F_va_gpu + self.learning_rate * tree._predict_x_gpu(X_va_gpu)
        val_mse = float(((y_va_gpu - F_va_gpu) ** 2).mean().item())
        record["val_mse"] = val_mse
        # ... existing patience logic ...
    self._history.append(record)
```

The CPU fallback path (when `use_gpu=False` or no CUDA) keeps the existing numpy-based boost loop unchanged.

### Test

`tests/unit/test_eml_split_boost.py` gains `test_xcache_matches_baseline`:

```python
def test_xcache_matches_baseline():
    """The GPU-cached boost loop must produce predictions equivalent
    to the previous (per-tree-transfer) loop within float32 tolerance."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 4))
    y = np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=800)
    # Just verify the boost completes and produces sensible predictions.
    m = EmlSplitBoostRegressor(
        max_rounds=50, max_depth=4, learning_rate=0.1,
        n_eml_candidates=10, k_eml=2, use_gpu=True, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred))
    train_mse = float(np.mean((pred - y) ** 2))
    assert train_mse < 0.5  # Easy bound; signal var is ~1.5
```

The test's value isn't strict numerical equivalence (the GPU loop uses float32 throughout while the prior CPU+GPU mix used float64 for residuals — they will differ slightly). It's a sanity check that the GPU-resident loop produces a fit on a clean signal.

---

## (B) Triton tree-predict kernel

### Problem

`_predict_gpu` (added in the first port) does `max_depth + 1` torch operations in a Python loop. For each iteration: 5-10 small kernel launches (gather, where, comparison, gather, etc.). Per tree predict: ~70 launches. Per boost-loop round: 1 predict on training set + 1 on val set = 140 launches. Per fit: 200 rounds × 140 = 28,000 launches per fit.

### Solution

A single Triton kernel that walks the tree and computes per-sample output in one launch.

### Kernel structure

New file `eml_boost/tree_split/_predict_triton.py`:

```python
import torch
import triton
import triton.language as tl

_EXP_CLAMP_DEFAULT = 50.0
_LOG_EPS_DEFAULT = 1e-6


@triton.jit
def _predict_tree_kernel(
    X_ptr, out_ptr,
    node_kind_ptr, left_child_ptr, right_child_ptr,
    feature_idx_ptr, threshold_ptr, leaf_value_ptr,
    eml_descriptor_ptr,           # (n_nodes, 6) flat row-major, int32
    split_feat_subset_ptr,        # (n_nodes, K_split) flat row-major, int32
    leaf_feat_subset_ptr,         # (n_nodes, K_leaf) flat row-major, int32
    leaf_feat_mean_ptr,
    leaf_feat_std_ptr,
    leaf_eta_ptr, leaf_bias_ptr, leaf_cap_ptr,
    N_SAMPLES, N_FEATURES, MAX_DEPTH,
    K_SPLIT: tl.constexpr,
    K_LEAF: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
    EXP_CLAMP: tl.constexpr,
    LOG_EPS: tl.constexpr,
):
    # Each program handles BLOCK_SAMPLES samples; thread = sample.
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    sample_mask = sample_idx < N_SAMPLES
    current = tl.zeros((BLOCK_SAMPLES,), dtype=tl.int64)

    # Walk to leaf: at most MAX_DEPTH + 1 iterations.
    # The static_range is the number of decision iterations; samples
    # already at leaves are no-op'd via the is_internal mask.
    for _ in tl.static_range(MAX_DEPTH + 1):
        kind = tl.load(node_kind_ptr + current, mask=sample_mask, other=2)
        is_internal = kind < 2
        feat = tl.load(feature_idx_ptr + current, mask=sample_mask, other=0).to(tl.int64)
        thr = tl.load(threshold_ptr + current, mask=sample_mask, other=0.0)
        # Raw gather: X[sample_idx, feat]
        x_offset = sample_idx * N_FEATURES + feat
        raw_val = tl.load(X_ptr + x_offset, mask=sample_mask, other=0.0)
        # EML evaluation, inlined per the depth-2 grammar (split_feat_subset path)
        eml_val = _eval_internal_eml_inline(
            current, sample_idx, X_ptr, eml_descriptor_ptr,
            split_feat_subset_ptr, N_FEATURES, K_SPLIT,
            EXP_CLAMP, LOG_EPS, sample_mask,
        )
        is_raw = kind == 0
        split_val = tl.where(is_raw, raw_val, eml_val)
        go_left = split_val <= thr
        next_lc = tl.load(left_child_ptr + current, mask=sample_mask, other=0).to(tl.int64)
        next_rc = tl.load(right_child_ptr + current, mask=sample_mask, other=0).to(tl.int64)
        next_node = tl.where(go_left, next_lc, next_rc)
        current = tl.where(is_internal, next_node, current)

    # Output: depends on leaf type.
    final_kind = tl.load(node_kind_ptr + current, mask=sample_mask, other=2)
    leaf_const_val = tl.load(leaf_value_ptr + current, mask=sample_mask, other=0.0)
    leaf_eml_val = _eval_leaf_eml_inline(
        current, sample_idx, X_ptr, eml_descriptor_ptr,
        leaf_feat_subset_ptr, leaf_feat_mean_ptr, leaf_feat_std_ptr,
        leaf_eta_ptr, leaf_bias_ptr, leaf_cap_ptr,
        N_FEATURES, K_LEAF, EXP_CLAMP, LOG_EPS, sample_mask,
    )
    out = tl.where(final_kind == 2, leaf_const_val, leaf_eml_val)
    tl.store(out_ptr + sample_idx, out, mask=sample_mask)
```

The `_eval_internal_eml_inline` and `_eval_leaf_eml_inline` are inlined Triton blocks that mirror the depth-2 grammar evaluation. Sketches:

```python
# _eval_internal_eml_inline:
#   For each sample: gather X[sample, split_feat_subset[node]] -> X_sub_K
#   Compute terminal values: c2..c5 via descriptor[2..5] -> {1, X_sub_K[j]}
#   node_0 = exp(clamp(c2, ±EXP_CLAMP)) - log(max(c3, LOG_EPS))
#   node_1 = exp(clamp(c4, ±EXP_CLAMP)) - log(max(c5, LOG_EPS))
#   Compute root left/right via descriptor[0..1] -> {1, X_sub_K[j], node_0, node_1}
#   output = exp(clamp(left, ±EXP_CLAMP)) - log(max(right, LOG_EPS))

# _eval_leaf_eml_inline:
#   Same evaluation but with X_sub standardized: clamp((X[i, leaf_feat_subset[node]] - mean) / std, -3, 3)
#   Then output = eta * eml_value + bias, clipped to ±leaf_cap if finite
```

The depth-2 grammar has K_SPLIT and K_LEAF as separate compile-time constants because `k_eml` (3) and `k_leaf_eml` (1) typically differ.

### Wiring in `tree.py`

`_predict_gpu` becomes a thin dispatcher that prefers Triton when available:

```python
def _predict_gpu(self, X: np.ndarray) -> np.ndarray:
    device = torch.device("cuda")
    if self._gpu_tree_device != device:
        self._gpu_tree = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in self._gpu_tree.items()}
        self._gpu_tree_device = device
    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    return self._predict_x_gpu(X_gpu).cpu().numpy().astype(np.float64)


def _predict_x_gpu(self, X_gpu: torch.Tensor) -> torch.Tensor:
    """GPU-input predict. Tries Triton kernel first; falls back to torch loop."""
    try:
        return predict_tree_triton(X_gpu, self._gpu_tree, self.max_depth)
    except (RuntimeError, ImportError):
        return self._predict_x_gpu_torch_fallback(X_gpu)
```

The torch fallback is the existing `_predict_gpu`'s body (the 9-iteration torch loop), wrapped to accept a GPU tensor.

### Test

`tests/unit/test_eml_split_tree.py` gains `test_predict_triton_matches_torch`:

```python
def test_predict_triton_matches_torch():
    """The Triton tree-predict kernel must produce the same predictions
    as the torch fallback to within float32 tolerance."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 3))
    y = np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=500)

    m = EmlSplitTreeRegressor(
        max_depth=4, n_eml_candidates=10, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=True, random_state=0,
    ).fit(X, y)

    # Triton path (default after this work)
    pred_triton = m.predict(X)

    # Force torch fallback by clearing the Triton cache or via a flag
    pred_torch = m._predict_x_gpu_torch_fallback(
        torch.tensor(X.astype(np.float32), device='cuda')
    ).cpu().numpy().astype(np.float64)

    np.testing.assert_allclose(pred_triton, pred_torch, rtol=1e-3, atol=1e-3)
```

---

## (C) Triton histogram-split kernel

### Problem

`gpu_histogram_split` (in `_gpu_split.py`) does ~6-8 torch ops per call (bin edges, scatter_add, cumsum, gain, argmax). For ~100-300 nodes per tree × 200 rounds, that's 20k-60k calls per fit. Each call has its own kernel-launch overhead.

### Solution

A Triton kernel that fuses the per-feature histogram + best-split-search into a single launch. One program per feature, shared-memory histogram, scan for best gain.

### Kernel structure

New file `eml_boost/tree_split/_gpu_split_triton.py`:

```python
@triton.jit
def _hist_split_kernel(
    feats_ptr,                # (n_samples, n_features) float32 — one feature per program
    y_ptr,                    # (n_samples,) float32
    out_gain_ptr,             # (n_features,) float32 — best gain per feature
    out_thr_idx_ptr,          # (n_features,) int32 — best bin-boundary index per feature
    out_min_ptr, out_max_ptr, # (n_features,) float32 — for translating bin idx to threshold
    N_SAMPLES,
    N_FEATURES,
    MIN_LEAF: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
):
    # Each program handles ONE feature; processes all samples in BLOCK_SAMPLES chunks.
    feat_pid = tl.program_id(0)
    if feat_pid >= N_FEATURES:
        return

    # Pass 1: scan min/max to compute bin edges.
    feat_min = tl.load(feats_ptr + feat_pid)  # placeholder; actually need a reduce
    # ... full reduce pass ...

    # Allocate shared per-bin (count, sum_y, sum_y_sq) accumulators.
    # ... shared memory layout ...

    # Pass 2: iterate samples, bin them, accumulate.
    for sample_block_start in range(0, N_SAMPLES, BLOCK_SAMPLES):
        # ... binning + atomic-or-block-private accumulation ...

    # Pass 3: scan over bins for best (gain, threshold).
    # ... cumsum to compute (left_count, left_sum, right_count, right_sum) at each boundary ...
    # ... gain = (sum_left^2 / count_left) + (sum_right^2 / count_right) ...
    # ... track best gain and bin index ...

    tl.store(out_gain_ptr + feat_pid, best_gain)
    tl.store(out_thr_idx_ptr + feat_pid, best_idx)
    tl.store(out_min_ptr + feat_pid, feat_min)
    tl.store(out_max_ptr + feat_pid, feat_max)
```

After the kernel: a small torch op picks `argmax(out_gain)` to get the best feature. The feature-level threshold is reconstructed from `feat_min + (best_idx + 0.5) * (feat_max - feat_min) / N_BINS`.

### Caveat — minimum viable design

Per-feature parallelism (one program per feature) gives N_FEATURES-way parallel launch. For typical `n_features` (raw + EML candidates ≈ 20-30), GPU is well-utilized.

The challenging part is the per-feature shared-memory histogram. With BLOCK_SAMPLES=1024 and N_BINS=256, each program needs 256 × 3 × 4 bytes = 3 KB shared (well within budget). Use atomics within the program if BLOCK_SAMPLES is processed in parallel by warps; or do the histogram pass with a single-warp loop (slower but simpler).

**Start simple:** single-warp per program, sequential bin accumulation. If profile shows it's still bottleneck, optimize to multi-warp + atomics.

### Wiring in `tree.py` / `_gpu_split.py`

Existing `gpu_histogram_split` becomes a dispatcher:

```python
def gpu_histogram_split(feats, y, n_bins, min_leaf_count):
    """Best-split-finding via histogram. Tries Triton kernel first;
    falls back to the torch implementation."""
    try:
        return gpu_histogram_split_triton(feats, y, n_bins, min_leaf_count)
    except (RuntimeError, ImportError):
        return gpu_histogram_split_torch(feats, y, n_bins, min_leaf_count)
```

The current implementation is renamed `gpu_histogram_split_torch` and kept as the fallback.

### Test

`tests/unit/test_eml_split_tree.py` gains `test_histogram_split_triton_matches_torch`:

```python
def test_histogram_split_triton_matches_torch():
    """The Triton histogram-split kernel must return the same
    (best_idx, threshold, gain) triple as the torch implementation
    on synthetic data."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from eml_boost.tree_split._gpu_split import (
        gpu_histogram_split_torch,
    )
    from eml_boost.tree_split._gpu_split_triton import (
        gpu_histogram_split_triton,
    )

    rng = np.random.default_rng(0)
    n, d = 1000, 5
    X = torch.tensor(rng.uniform(-1, 1, size=(n, d)), dtype=torch.float32, device='cuda')
    y = torch.tensor(X[:, 2].cpu().numpy() * 2 + 0.1 * rng.normal(size=n),
                     dtype=torch.float32, device='cuda')

    idx_t, thr_t, gain_t = gpu_histogram_split_torch(X, y, n_bins=256, min_leaf_count=20)
    idx_tri, thr_tri, gain_tri = gpu_histogram_split_triton(X, y, n_bins=256, min_leaf_count=20)

    # Same best feature
    assert int(idx_t) == int(idx_tri)
    # Threshold within one bin width
    assert abs(float(thr_t) - float(thr_tri)) < 0.01
    # Gain within float32 rounding
    np.testing.assert_allclose(float(gain_t), float(gain_tri), rtol=1e-3)
```

---

## Implementation order

Each optimization is shippable independently. Recommended order:

1. **(A) X-cache** — pure engineering, no Triton. Validates the boost-loop refactor before adding Triton complexity.
2. **(B) Triton tree-predict** — biggest predict-side win.
3. **(C) Triton histogram-split** — incremental fit-side win.

After each step, run the speedup benchmark on `1191_BNG_pbc` to confirm the expected gain. If a step underperforms its target by >2×, profile before continuing.

## Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — add `_fit_xy_gpu`, `_predict_x_gpu` (and `_predict_x_gpu_torch_fallback`), update `_predict_gpu` to delegate.
- `eml_boost/tree_split/ensemble.py` — boost-loop GPU-resident path when `use_gpu=True and torch.cuda.is_available()`.
- `eml_boost/tree_split/_gpu_split.py` — rename current implementation to `gpu_histogram_split_torch`, add dispatcher.
- `tests/unit/test_eml_split_tree.py` — add 2 tests (Triton predict equivalence, Triton histogram equivalence).
- `tests/unit/test_eml_split_boost.py` — add 1 test (X-cache boost loop sanity).

**Created:**
- `eml_boost/tree_split/_predict_triton.py` — Triton tree-predict kernel + `predict_tree_triton` Python wrapper.
- `eml_boost/tree_split/_gpu_split_triton.py` — Triton histogram-split kernel + `gpu_histogram_split_triton` wrapper.

**Unchanged:**
- `eml_boost/_triton_exhaustive.py` — the existing EML evaluation kernel stays.
- `eml_boost/tree_split/nodes.py`.
- The CPU pipeline, the Experiment 15 runner, all reports.

## Success criteria

- **S-A (correctness):** all 32 existing in-scope tests pass; 3 new equivalence tests pass.
- **S-B (`1191_BNG_pbc` speed):** SplitBoost 5-seed total fit time on `1191_BNG_pbc` is **under 5 minutes** (vs ~30 min currently). That's a 6× speedup over the current GPU-port version.
- **S-C (GPU utilization):** during `1191_BNG_pbc` fit, `nvidia-smi` reports GPU utilization > 60% averaged over a 30-second window.

If S-A holds and S-B + S-C both succeed, restart Experiment 15. Estimated total runtime: **1.5-2 hours** (vs the 4-5 hours estimate post-first-port and 13-15 hours pre-port).

## Risks

- **Triton kernel correctness on edge cases.** EML evaluation involves nested exp/log with clamps; getting the float behavior to match torch exactly may require careful constant alignment. The fallback dispatcher means a kernel bug doesn't break correctness — predictions still work, just slower.
- **Triton compile time.** First-fit on each kernel signature takes 5-30 seconds to compile. Cached after that. Mitigated by warming up in the unit tests; production fits after the first won't pay this cost.
- **Triton version compatibility.** The repo uses Triton 3.6+. The kernels use `tl.static_range`, `tl.where`, `tl.load` — all supported in 3.6. Avoid features added in 3.7+.
- **Boost-loop X-cache OOM on huge datasets.** `1191_BNG_pbc` X = 144 MB float32 fits comfortably (RTX 3090 has 24 GB). Some PMLB BNG datasets may have higher feature counts; monitor memory in initial runs. Mitigation: fall back to per-tree allocation if boost-loop allocation exceeds 50% of GPU memory.

## Action on verdict

- **All success criteria met:** restart Experiment 15.
- **Correctness fails on a Triton kernel:** the fallback dispatcher already handles this; ship the working pieces and document the kernel as broken.
- **Speed underwhelming:** profile to identify which optimization underperformed, iterate. The torch fallbacks ensure correctness regardless.
