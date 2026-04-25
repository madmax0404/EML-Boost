# SplitBoost GPU Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the GPU pipeline of `EmlSplitTreeRegressor` end-to-end so fit and predict are no longer Python-bound. Target: 10× speedup on 1M-row datasets, > 50% GPU utilization during fit.

**Architecture:** Three independent code paths get GPU-native treatment: (1) `_grow_gpu` keeps `indices`/`y`/`mask` on GPU throughout recursion; (2) `_top_features_by_corr_gpu` replaces numpy correlation with torch ops; (3) `_predict_vec` replaced by a tensorized GPU traversal using a per-tree tensor representation. Pure torch GPU ops — no new Triton kernels. CPU fallback path stays untouched.

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+ (existing), pytest, PMLB. uv for environment management.

---

## Background an implementer needs

`eml_boost` is a research GBDT regressor whose hot inner loops (EML candidate evaluation, histogram split-finding) are already on GPU via Triton (`eml_boost/_triton_exhaustive.py`) and torch (`eml_boost/tree_split/_gpu_split.py`). But the *outer* orchestration is pure Python:

- `_grow` in `eml_boost/tree_split/tree.py:162` recurses through tree nodes; per-node it does several CPU↔GPU transfers in `_find_best_split_gpu` (`indices.to(device)`, `y.copy(device)`, `mask.cpu().numpy()`).
- `_predict_vec` in `tree.py:764` walks the tree on CPU recursively per sample.
- `_top_features_by_corr` in `tree.py:728` does numpy correlation per node.

Profiling against XGBoost on the 1M-row `1191_BNG_pbc` dataset showed: SplitBoost takes ~690 s/fit, GPU at 11% utilization, Python at 800% CPU (8 cores). XGBoost CUDA does the same fit in 1.1 s. The Python orchestration is the gap.

This plan implements the spec at `docs/superpowers/specs/2026-04-25-splitboost-gpu-port-design.md`. After it lands, Experiment 15 (currently paused with 6 of 122 datasets complete) gets restarted and should complete in 2-3 hours instead of 13-15.

**Before starting Task 1, read:**
- `docs/superpowers/specs/2026-04-25-splitboost-gpu-port-design.md` — the spec.
- `eml_boost/tree_split/tree.py` — the file you're modifying. Especially focus on `__init__` (lines 72-110), `fit` (lines 114-150), `_grow` (162-184), `_find_best_split_gpu` (232-298), `_fit_leaf` (343-465), `_top_features_by_corr` (728-734), and `_predict_vec` (764-805).
- `eml_boost/_triton_exhaustive.py` — the existing `evaluate_trees_torch` and `evaluate_trees_triton` functions. The new GPU predict path needs a per-sample variant (each sample evaluates a different descriptor); we'll add a helper.
- `tests/unit/test_eml_split_tree.py` — the test style; helpers `_count_eml_leaves`, `_count_leaves`, `_mse`.
- `experiments/experiment15/summary.csv` — already has 6 datasets × 5 seeds × 3 models = 90 rows. The Experiment 15 restart at the end of this plan will skip those.

**Things NOT to change:**
- The CPU pipeline (`_grow` for `use_gpu=False`, `_find_best_split_cpu`, the existing `_top_features_by_corr` numpy version).
- The Triton EML kernel itself.
- `EmlSplitBoostRegressor.fit` / `predict` — they call `tree.predict(X_tr)` which gets transparently faster.
- `EmlLeafNode` / `LeafNode` / `InternalNode` dataclasses.
- Any hyperparameter defaults.

---

## Task 1: Foundation — `_y_gpu` allocation and `_top_features_by_corr_gpu`

**Goal:** Add the persistent GPU residual tensor and a GPU-native correlation helper. No behavior change; existing tests stay green. One new test verifies the helper produces the same top-k indices as the numpy version.

**Files:**
- Modify: `eml_boost/tree_split/tree.py`
- Modify: `tests/unit/test_eml_split_tree.py`

- [ ] **Step 1: Write the failing test for the GPU correlation helper.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_top_features_by_corr_gpu_matches_numpy():
    """The GPU correlation helper must return the same top-k feature
    indices as the existing numpy version for the same input."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU correlation requires CUDA")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 8))
    y = X[:, 2] * 2.0 + X[:, 5] * 1.5 + 0.3 * rng.normal(size=500)

    # Existing numpy version
    np_top = EmlSplitTreeRegressor._top_features_by_corr(X, y, k=3)

    # New GPU version
    device = torch.device("cuda")
    X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
    y_gpu = torch.tensor(y, dtype=torch.float32, device=device)
    gpu_top = EmlSplitTreeRegressor._top_features_by_corr_gpu(X_gpu, y_gpu, k=3)

    assert sorted(np_top.tolist()) == sorted(gpu_top.cpu().tolist())
```

- [ ] **Step 2: Run the test; confirm it fails with `AttributeError` on `_top_features_by_corr_gpu`.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_top_features_by_corr_gpu_matches_numpy -v
```
Expected: FAIL with `AttributeError: type object 'EmlSplitTreeRegressor' has no attribute '_top_features_by_corr_gpu'`.

- [ ] **Step 3: Add `_top_features_by_corr_gpu` static method.**

In `eml_boost/tree_split/tree.py`, find the existing `_top_features_by_corr` static method (around line 728). Add the GPU version right below it:

```python
    @staticmethod
    def _top_features_by_corr_gpu(
        X: "torch.Tensor", y: "torch.Tensor", k: int
    ) -> "torch.Tensor":
        """GPU-native version of `_top_features_by_corr`.

        Returns a long tensor on the same device as X with the indices
        of the top-k features by absolute Pearson correlation with y.
        """
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

- [ ] **Step 4: Add `self._y_gpu` allocation in `fit()`.**

In `eml_boost/tree_split/tree.py`, locate `fit()` (around line 114). Find the GPU-storage block that creates `self._X_gpu`, `self._global_mean_gpu`, etc. The current block looks like:

```python
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._X_gpu = torch.tensor(X, dtype=torch.float32, device=self._device)
            self._global_mean_gpu = torch.tensor(
                self._global_mean, dtype=torch.float32, device=self._device,
            )
            self._global_std_gpu = torch.tensor(
                self._global_std, dtype=torch.float32, device=self._device,
            )
        else:
            self._device = None
            self._X_gpu = None
            self._global_mean_gpu = None
            self._global_std_gpu = None
```

Add `self._y_gpu` to both branches:

```python
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._X_gpu = torch.tensor(X, dtype=torch.float32, device=self._device)
            self._y_gpu = torch.tensor(y, dtype=torch.float32, device=self._device)
            self._global_mean_gpu = torch.tensor(
                self._global_mean, dtype=torch.float32, device=self._device,
            )
            self._global_std_gpu = torch.tensor(
                self._global_std, dtype=torch.float32, device=self._device,
            )
        else:
            self._device = None
            self._X_gpu = None
            self._y_gpu = None
            self._global_mean_gpu = None
            self._global_std_gpu = None
```

Then find the post-fit cleanup at the end of `fit()` (the block that sets these attributes back to `None`) and add `self._y_gpu = None` alongside the others:

```python
        # Release GPU handles after fit; tree stores only CPU Node objects.
        self._X_gpu = None
        self._y_gpu = None
        self._global_mean_gpu = None
        self._global_std_gpu = None
        self._device = None
        return self
```

- [ ] **Step 5: Run the new test; confirm it passes.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_top_features_by_corr_gpu_matches_numpy -v
```
Expected: PASS.

- [ ] **Step 6: Run the full unit test suite; confirm no regressions.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 28 in-scope tests pass plus the new one (29 total). One pre-existing unrelated failure in `test_eml_weak_learner.py::test_fit_recovers_simple_formula` remains.

- [ ] **Step 7: Commit.**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: add _y_gpu allocation and _top_features_by_corr_gpu helper

Foundation for the GPU port. Adds self._y_gpu (the residual tensor on
GPU, preallocated once at fit start instead of per-node copies) and a
torch-based correlation helper. No call sites use them yet —
subsequent tasks wire them in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fit-path GPU port — `_grow_gpu` and `_find_best_split_gpu` refactor

**Goal:** Replace the per-node host↔device transfers. Add `_grow_gpu` that keeps `indices` as a torch tensor throughout recursion. Refactor `_find_best_split_gpu` to skip the `.to(device)` calls and return a GPU-native `left_mask`. Modify `_fit_leaf` to accept a torch tensor when in GPU mode (CPU path doesn't call `_fit_leaf` for EML — it uses constant leaves directly via the existing early-out).

**Files:**
- Modify: `eml_boost/tree_split/tree.py`
- Modify: `tests/unit/test_eml_split_tree.py` (one new test)

- [ ] **Step 1: Write the failing equivalence test.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_gpu_grow_matches_cpu_grow():
    """The new GPU-native fit pipeline must produce equivalent
    predictions to the CPU pipeline on the same data + seed."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 4))
    y = np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=500)

    m_gpu = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=5, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=True, random_state=0,
    ).fit(X, y)
    m_cpu = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=5, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=False, random_state=0,
    ).fit(X, y)

    pred_gpu = m_gpu.predict(X)
    pred_cpu = m_cpu.predict(X)
    # Trees may differ slightly due to float32 vs float64 in candidate-eval;
    # require predictions agree to within MSE 0.01 of each other on this fit
    # (the signal MSE itself is ~1.0+ so this is a stable bound).
    diff_mse = float(np.mean((pred_gpu - pred_cpu) ** 2))
    assert diff_mse < 0.01, f"GPU vs CPU prediction MSE diff = {diff_mse:.4f}"
```

- [ ] **Step 2: Run the test; confirm it fails or passes.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_gpu_grow_matches_cpu_grow -v
```
Expected: PASS pre-implementation (the existing pipeline is the only one running). Note this — the test gives a useful regression baseline. After Task 2's refactor, it should still PASS (now exercising the new `_grow_gpu` path on the GPU side).

- [ ] **Step 3: Refactor `_find_best_split_gpu` to accept torch tensor `indices` and return GPU-native `left_mask`.**

In `eml_boost/tree_split/tree.py`, replace the entire `_find_best_split_gpu` method (currently around lines 232-298) with:

```python
    def _find_best_split_gpu(
        self, indices: "torch.Tensor", rng: np.random.Generator
    ) -> tuple[RawSplit | EmlSplit, float, "torch.Tensor"] | None:
        """GPU-batched histogram split-finding. Indices and the returned
        mask are both torch tensors on self._device — no CPU↔GPU
        transfers per call."""
        device = self._device
        assert device is not None and self._X_gpu is not None and self._y_gpu is not None
        X_node = self._X_gpu[indices]                          # (n, n_raw)
        y_node = self._y_gpu[indices]                          # (n,)

        n_raw = X_node.shape[1]
        feat_cols: list[torch.Tensor] = [X_node]
        valid_candidates: np.ndarray | None = None
        top_features_gpu: "torch.Tensor" | None = None
        k_used = 0

        if self.n_eml_candidates > 0 and n_raw > 0:
            k_used = min(self.k_eml, n_raw)
            top_features_gpu = self._top_features_by_corr_gpu(X_node, y_node, k_used)
            candidates = self._sample_descriptors(k_used, self.n_eml_candidates, rng)
            if len(candidates) > 0:
                X_sub = X_node[:, top_features_gpu]
                desc_gpu = torch.tensor(candidates, dtype=torch.int32, device=device)
                eml_values = evaluate_trees_triton(desc_gpu, X_sub, k_used)
                finite = torch.isfinite(eml_values).all(dim=1)
                if finite.any():
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
            split: RawSplit | EmlSplit = RawSplit(
                feature_idx=int(best_idx), threshold=float(best_t),
            )
        else:
            c_idx = int(best_idx) - n_raw
            assert valid_candidates is not None
            assert top_features_gpu is not None
            desc = valid_candidates[c_idx]
            top_features_np = top_features_gpu.cpu().numpy()
            split = EmlSplit(
                snapped=SnappedTree(
                    depth=2, k=k_used,
                    internal_input_count=2, leaf_input_count=4,
                    terminal_choices=tuple(int(v) for v in desc),
                ),
                feature_subset=tuple(int(v) for v in top_features_np),
                threshold=float(best_t),
            )

        # Stay on GPU: return mask as a bool tensor, not numpy.
        left_mask = all_feats[:, best_idx] <= best_t
        return split, float(best_gain), left_mask
```

The key changes vs. the prior version (lines 232-298):
- Signature drops `y_sub: np.ndarray` parameter.
- `idx_gpu = torch.from_numpy(indices).to(device, ...)` — gone (indices is already a GPU tensor).
- `y_node = torch.tensor(y_sub, dtype=torch.float32, device=device)` — replaced by `y_node = self._y_gpu[indices]` (GPU gather).
- `top_features = self._top_features_by_corr(self._X_cpu[indices], y_sub, k_used)` — replaced by GPU call to `_top_features_by_corr_gpu`.
- `left_mask = left_mask_gpu.cpu().numpy()` — gone; return GPU tensor.

- [ ] **Step 4: Refactor `_fit_leaf` to accept torch tensor `indices`.**

In `eml_boost/tree_split/tree.py`, find `_fit_leaf` (around line 343). The current signature is `def _fit_leaf(self, indices: np.ndarray, y_sub: np.ndarray) -> Node`. Change to `def _fit_leaf(self, indices: "torch.Tensor") -> Node`.

The existing body has these lines that need updating:

```python
        n = len(y_sub)
        constant_value = float(y_sub.mean()) if n > 0 else 0.0
```

Replace with:

```python
        n = int(indices.shape[0])
        if n == 0:
            return LeafNode(value=0.0)
        y_sub = self._y_gpu[indices]
        constant_value = float(y_sub.mean().item())
```

Then find this block (around line 350-365):

```python
        # Early-out gates.
        eml_disabled = self.k_leaf_eml <= 0
        too_small = n < self.min_samples_leaf_eml
        no_gpu = self._X_gpu is None or self._device is None
        n_raw = self._X_cpu.shape[1] if self._X_cpu is not None else 0
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=constant_value)

        device = self._device
        assert device is not None and self._X_gpu is not None

        k = min(self.k_leaf_eml, n_raw)
        top_features = self._top_features_by_corr(self._X_cpu[indices], y_sub, k)
        idx_gpu = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        X_sub_raw = self._X_gpu[idx_gpu][:, top_features]
        y_full = torch.tensor(y_sub, dtype=torch.float32, device=device)
```

Replace with:

```python
        # Early-out gates.
        eml_disabled = self.k_leaf_eml <= 0
        too_small = n < self.min_samples_leaf_eml
        no_gpu = self._X_gpu is None or self._device is None
        n_raw = self._X_cpu.shape[1] if self._X_cpu is not None else 0
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=constant_value)

        device = self._device
        assert device is not None and self._X_gpu is not None

        k = min(self.k_leaf_eml, n_raw)
        # X_sub_raw: gather X_gpu by leaf indices, then by top-k features.
        # Use _top_features_by_corr_gpu directly on the leaf's slice.
        X_node = self._X_gpu[indices]
        top_features_gpu = self._top_features_by_corr_gpu(X_node, y_sub, k)
        top_features = top_features_gpu.cpu().numpy()
        X_sub_raw = X_node[:, top_features_gpu]
        y_full = y_sub  # already a GPU float32 tensor
```

The deterministic per-leaf seed line near the val-split point (look for `seed = int(indices[0]) if len(indices) else 0`) needs a small update because `indices` is now a torch tensor:

```python
        seed = int(indices[0].item()) if n > 0 else 0
```

Everything else in `_fit_leaf` stays the same — the rest already operates on GPU.

- [ ] **Step 5: Add `_grow_gpu` method.**

In `eml_boost/tree_split/tree.py`, find the existing `_grow` method (around line 162). Add `_grow_gpu` directly below it:

```python
    def _grow_gpu(
        self, indices: "torch.Tensor", depth: int, rng: np.random.Generator,
    ) -> Node:
        """GPU-native version of `_grow`. `indices` is a long tensor on
        `self._device`; recursion stays on GPU end-to-end."""
        n = int(indices.shape[0])
        if depth >= self.max_depth or n <= 2 * self.min_samples_leaf:
            return self._fit_leaf(indices)

        best = self._find_best_split_gpu(indices, rng)
        if best is None:
            return self._fit_leaf(indices)

        split, _gain, left_mask = best  # left_mask is a GPU bool tensor
        left_count = int(left_mask.sum().item())
        right_count = n - left_count
        if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
            return self._fit_leaf(indices)

        return InternalNode(
            split=split,
            left=self._grow_gpu(indices[left_mask], depth + 1, rng),
            right=self._grow_gpu(indices[~left_mask], depth + 1, rng),
        )
```

- [ ] **Step 6: Dispatch in `fit()` to `_grow_gpu` when on GPU.**

In `eml_boost/tree_split/tree.py`, find the fit() method's tree-growth call. The current line is:

```python
        indices = np.arange(len(X))
        self._root: Node = self._grow(indices, y, depth=0, rng=rng)
```

Replace with:

```python
        if self._device is not None:
            indices_gpu = torch.arange(
                len(X), dtype=torch.long, device=self._device,
            )
            self._root: Node = self._grow_gpu(indices_gpu, depth=0, rng=rng)
        else:
            indices = np.arange(len(X))
            self._root = self._grow(indices, y, depth=0, rng=rng)
```

- [ ] **Step 7: Update the existing CPU `_grow` to not call `_fit_leaf` (which now requires GPU).**

The existing `_grow` (CPU pipeline) currently calls `self._fit_leaf(indices, y_sub)`. Since `_fit_leaf` now requires GPU state, the CPU path needs to inline the constant-leaf return.

In `eml_boost/tree_split/tree.py`, find the existing `_grow` method. The current body:

```python
    def _grow(
        self, indices: np.ndarray, y_sub: np.ndarray, depth: int, rng: np.random.Generator
    ) -> Node:
        if depth >= self.max_depth or len(y_sub) <= 2 * self.min_samples_leaf:
            return self._fit_leaf(indices, y_sub)

        if self._X_gpu is not None:
            best = self._find_best_split_gpu(indices, y_sub, rng)
        else:
            X_node = self._X_cpu[indices]
            best = self._find_best_split_cpu(X_node, y_sub, rng)
        if best is None:
            return self._fit_leaf(indices, y_sub)

        split, _gain, left_mask = best
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return self._fit_leaf(indices, y_sub)

        return InternalNode(
            split=split,
            left=self._grow(indices[left_mask], y_sub[left_mask], depth + 1, rng),
            right=self._grow(indices[~left_mask], y_sub[~left_mask], depth + 1, rng),
        )
```

Since the CPU pipeline doesn't have access to GPU state, EML leaves can't be fit there anyway — the existing `_fit_leaf` short-circuits to `LeafNode` on `no_gpu` regardless. So the CPU path can return a constant leaf directly:

Replace the entire `_grow` method body with:

```python
    def _grow(
        self, indices: np.ndarray, y_sub: np.ndarray, depth: int, rng: np.random.Generator
    ) -> Node:
        # CPU pipeline: EML leaves require GPU, so all leaves are constant.
        def _const_leaf(y):
            return LeafNode(value=float(y.mean()) if len(y) > 0 else 0.0)

        if depth >= self.max_depth or len(y_sub) <= 2 * self.min_samples_leaf:
            return _const_leaf(y_sub)

        X_node = self._X_cpu[indices]
        best = self._find_best_split_cpu(X_node, y_sub, rng)
        if best is None:
            return _const_leaf(y_sub)

        split, _gain, left_mask = best
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return _const_leaf(y_sub)

        return InternalNode(
            split=split,
            left=self._grow(indices[left_mask], y_sub[left_mask], depth + 1, rng),
            right=self._grow(indices[~left_mask], y_sub[~left_mask], depth + 1, rng),
        )
```

Note: the `_X_gpu is not None` branch is gone from `_grow`. GPU dispatch happens at `fit()` level now.

- [ ] **Step 8: Run the equivalence test; confirm it passes.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_gpu_grow_matches_cpu_grow -v
```
Expected: PASS. Both GPU and CPU paths produce predictions within MSE 0.01 of each other.

- [ ] **Step 9: Run the full unit test suite; confirm no regressions.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 29 in-scope tests pass plus the new one (30 total). Pre-existing unrelated failure stays.

- [ ] **Step 10: Quick benchmark: confirm the per-node round-trip elimination matters.**

Run an ad-hoc benchmark on cpu_small (8k rows, ~17s pre-port at d=8):

```bash
PYTHONUNBUFFERED=1 uv run python -u -c "
import time, numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from eml_boost.tree_split import EmlSplitBoostRegressor

X, y = fetch_data('562_cpu_small', return_X_y=True)
X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
m = EmlSplitBoostRegressor(
    max_rounds=200, max_depth=8, learning_rate=0.1, min_samples_leaf=20,
    n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
    use_gpu=True, random_state=0,
)
t0 = time.time(); m.fit(X_tr, y_tr); print(f'fit: {time.time() - t0:.1f}s, rounds={m.n_rounds}')
"
```

Expected: fit time should be similar to or faster than the pre-port baseline (~17 s on cpu_small at d=8 from Experiment 13). On this dataset the per-node transfers are small (8k-row indices = 64 KB), so the speedup is modest (maybe 20-30%). The big speedup comes in Task 3 when predict moves to GPU.

- [ ] **Step 11: Commit.**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: GPU-native fit pipeline (_grow_gpu, _find_best_split_gpu refactor)

Eliminate per-node host↔device transfers by keeping indices, y, and
left_mask on GPU throughout _grow recursion. _find_best_split_gpu now
accepts a torch tensor and returns a GPU bool mask. _fit_leaf accepts
a torch tensor (it already does its work on GPU; just removes the
front-end .to(device) conversion). _top_features_by_corr_gpu replaces
the numpy correlation in the EML-candidate path.

The CPU pipeline now returns constant leaves directly (EML leaves
already required GPU; the dispatch was redundant). _grow_gpu and the
existing _grow are now distinct methods routed at fit() time based
on self._device.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Predict-path GPU port — `_tensorize_tree` and GPU `predict()`

**Goal:** Replace the recursive Python `_predict_vec` with a tensorized GPU traversal. The fitted tree is encoded as a flat dict of CPU tensors at end-of-fit (`_tensorize_tree`), moved to GPU lazily on first predict, then evaluated via a bounded loop of torch ops. This is the single biggest speedup on big datasets — the boost loop's `tree.predict(X_tr)` call scales with `n_samples` per round.

**Files:**
- Modify: `eml_boost/tree_split/tree.py`
- Modify: `tests/unit/test_eml_split_tree.py` (one new test)
- Possibly modify: `eml_boost/_triton_exhaustive.py` (per-sample evaluator wrapper)

- [ ] **Step 1: Write the failing equivalence test for the GPU predict path.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_gpu_predict_matches_cpu_fallback():
    """The GPU predict path and the CPU-fallback predict path must
    produce numerically equivalent outputs on the same fitted tree."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X_tr = rng.uniform(-1, 1, size=(800, 3))
    y_tr = np.exp(X_tr[:, 0]) + 0.5 * X_tr[:, 1] + 0.05 * rng.normal(size=800)
    X_te = rng.uniform(-1, 1, size=(200, 3))

    m = EmlSplitTreeRegressor(
        max_depth=4, min_samples_leaf=20, n_eml_candidates=10, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=True, random_state=0,
    ).fit(X_tr, y_tr)

    # GPU predict (default path after fit on GPU)
    pred_gpu = m.predict(X_te)

    # Force CPU fallback by clearing the tensorized tree
    saved_gpu_tree = m._gpu_tree
    m._gpu_tree = None
    pred_cpu = m.predict(X_te)
    m._gpu_tree = saved_gpu_tree  # restore for any later use

    np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run the test; confirm it fails.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_gpu_predict_matches_cpu_fallback -v
```
Expected: FAIL with `AttributeError: 'EmlSplitTreeRegressor' object has no attribute '_gpu_tree'`.

- [ ] **Step 3: Add `_tensorize_tree` method.**

In `eml_boost/tree_split/tree.py`, add this method right after `_grow_gpu` (the new method from Task 2):

```python
    def _tensorize_tree(self, root: Node) -> dict:
        """Walk the fitted Python tree once and emit a flat dict of CPU
        tensors. The dict is moved to GPU on first predict call.
        Only called in GPU-fit mode."""
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

        K = max(1, self.k_leaf_eml)

        node_kind = torch.zeros(n_nodes, dtype=torch.int8)
        left_child = torch.full((n_nodes,), -1, dtype=torch.int32)
        right_child = torch.full((n_nodes,), -1, dtype=torch.int32)
        feature_idx = torch.zeros(n_nodes, dtype=torch.int32)
        threshold = torch.zeros(n_nodes, dtype=torch.float32)
        leaf_value = torch.zeros(n_nodes, dtype=torch.float32)
        eml_descriptor = torch.zeros((n_nodes, 6), dtype=torch.int32)
        leaf_eta = torch.zeros(n_nodes, dtype=torch.float32)
        leaf_bias = torch.zeros(n_nodes, dtype=torch.float32)
        leaf_cap = torch.full((n_nodes,), float("inf"), dtype=torch.float32)
        leaf_feat_subset = torch.zeros((n_nodes, K), dtype=torch.int32)
        leaf_feat_mean = torch.zeros((n_nodes, K), dtype=torch.float32)
        leaf_feat_std = torch.ones((n_nodes, K), dtype=torch.float32)
        split_feat_subset = torch.zeros((n_nodes, K), dtype=torch.int32)

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
                    eml_descriptor[nid, i] = int(c)
            elif isinstance(node, InternalNode):
                split = node.split
                threshold[nid] = float(split.threshold)
                if isinstance(split, RawSplit):
                    node_kind[nid] = 0
                    feature_idx[nid] = int(split.feature_idx)
                else:
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
            "n_nodes": n_nodes,
        }
```

- [ ] **Step 4: Initialize `self._gpu_tree = None` in `__init__` and populate in `fit()`.**

In `eml_boost/tree_split/tree.py`'s `__init__`, after the existing attribute storage, add:

```python
        self._gpu_tree = None
        self._gpu_tree_device = None
```

In `fit()`, after the line `self._root = ...` (the dispatch you added in Task 2), add this for the GPU branch:

```python
        if self._device is not None:
            indices_gpu = torch.arange(
                len(X), dtype=torch.long, device=self._device,
            )
            self._root: Node = self._grow_gpu(indices_gpu, depth=0, rng=rng)
            self._gpu_tree = self._tensorize_tree(self._root)
        else:
            indices = np.arange(len(X))
            self._root = self._grow(indices, y, depth=0, rng=rng)
            self._gpu_tree = None
```

- [ ] **Step 5: Add `evaluate_trees_torch_per_sample` helper in `_triton_exhaustive.py`.**

The existing `evaluate_trees_torch(descriptors, X, k)` evaluates the same set of descriptors across all samples in X. The predict path needs the inverse: each sample `i` evaluates its own descriptor `descriptors[i]` against its own feature subset `X[i]`. Add this helper to `eml_boost/_triton_exhaustive.py`:

```python
def evaluate_trees_torch_per_sample(
    descriptor: torch.Tensor, X: torch.Tensor, k: int,
) -> torch.Tensor:
    """Evaluate one tree per sample.

    descriptor: (m, 6) int32, one descriptor per sample
    X: (m, k) float, one feature subset per sample
    k: feature count per sample

    Returns: (m,) float — sample i's descriptor evaluated on X[i].

    Implementation: leverages the existing batch evaluator by calling it
    one descriptor at a time and gathering. For predict-time use only,
    where m is on the order of n_samples; vectorized via expand+gather
    rather than a Python loop.
    """
    m = descriptor.shape[0]
    if m == 0:
        return torch.empty(0, dtype=X.dtype, device=X.device)
    n_samples = m
    # Reuse the same depth-2 evaluation scheme but operate per sample.
    # We replicate the core arithmetic from evaluate_trees_torch() inline.
    dtype = X.dtype
    device = X.device

    leaf_terminals = torch.cat([torch.ones(m, 1, device=device, dtype=dtype), X], dim=1)  # (m, k+1)
    # For each sample, gather its leaf-position values.
    # descriptor[:, 2..5] picks leaf input indices in [0, k]
    def gather_leaf(pos):  # pos in {2, 3, 4, 5}
        idx = descriptor[:, pos].long().unsqueeze(1)  # (m, 1)
        return leaf_terminals.gather(1, idx).squeeze(1)  # (m,)

    v_c2 = gather_leaf(2)
    v_c3 = gather_leaf(3)
    v_c4 = gather_leaf(4)
    v_c5 = gather_leaf(5)

    node_0 = (
        torch.exp(v_c2.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c3.clamp(min=_LOG_EPS))
    )
    node_1 = (
        torch.exp(v_c4.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c5.clamp(min=_LOG_EPS))
    )

    # Internal positions 0 and 1 — choice in {0, 1..k, k+1}.
    c0 = descriptor[:, 0].long()
    c1 = descriptor[:, 1].long()

    left = torch.zeros(m, device=device, dtype=dtype)
    right = torch.zeros(m, device=device, dtype=dtype)

    left = left + (c0 == 0).to(dtype)
    right = right + (c1 == 0).to(dtype)
    for j in range(k):
        feat_j = X[:, j]
        left = left + ((c0 == j + 1).to(dtype)) * feat_j
        right = right + ((c1 == j + 1).to(dtype)) * feat_j
    left = left + ((c0 == k + 1).to(dtype)) * node_0
    right = right + ((c1 == k + 1).to(dtype)) * node_1

    output = (
        torch.exp(left.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(right.clamp(min=_LOG_EPS))
    )
    return output
```

Note: this duplicates the core arithmetic of `evaluate_trees_torch` but with `(m,)` shapes throughout instead of `(n_trees, n_samples)`. The `_EXP_CLAMP` and `_LOG_EPS` constants are already module-level.

- [ ] **Step 6: Update import in tree.py.**

In `eml_boost/tree_split/tree.py`, find the existing import line:

```python
from eml_boost._triton_exhaustive import (
    descriptor_feature_mask_numpy,
    enumerate_depth2_descriptor,
    evaluate_trees_torch,
    evaluate_trees_triton,
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
)
```

Add `evaluate_trees_torch_per_sample` to the imported list:

```python
from eml_boost._triton_exhaustive import (
    descriptor_feature_mask_numpy,
    enumerate_depth2_descriptor,
    evaluate_trees_torch,
    evaluate_trees_torch_per_sample,
    evaluate_trees_triton,
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
)
```

- [ ] **Step 7: Replace `predict()` with the GPU path; rename old recursion to `_predict_cpu_fallback`.**

In `eml_boost/tree_split/tree.py`, find the existing `predict()` method (around line 152). The current implementation is a thin wrapper that calls `_predict_vec`. Replace the entire `predict()` body with the GPU path that falls back to the existing recursion when no `_gpu_tree` is available:

```python
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        if self._gpu_tree is None or not torch.cuda.is_available():
            self._predict_cpu_fallback(self._root, X, np.arange(len(X)), out)
            return out
        return self._predict_gpu(X)
```

Then rename `_predict_vec` to `_predict_cpu_fallback` (the rename is just changing the method name on line 764 — search and replace `_predict_vec` everywhere it appears: the definition and the recursive calls inside).

- [ ] **Step 8: Add `_predict_gpu` method.**

In `eml_boost/tree_split/tree.py`, add this method right after `_predict_cpu_fallback`:

```python
    def _predict_gpu(self, X: np.ndarray) -> np.ndarray:
        """Tensorized GPU traversal of the fitted tree."""
        device = torch.device("cuda")
        # Lazy move of the tree dict to GPU on first call.
        if self._gpu_tree_device != device:
            self._gpu_tree = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self._gpu_tree.items()
            }
            self._gpu_tree_device = device

        X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
        n_samples = X_gpu.shape[0]
        if n_samples == 0:
            return np.zeros(0, dtype=np.float64)

        t = self._gpu_tree
        K = t["k_leaf_eml"]
        current = torch.zeros(n_samples, dtype=torch.long, device=device)

        for _ in range(self.max_depth + 1):
            kind_now = t["node_kind"][current]
            is_internal = kind_now < 2
            if not is_internal.any():
                break
            feat = t["feature_idx"][current]            # (n,)
            thr = t["threshold"][current]               # (n,)
            raw_vals = X_gpu.gather(1, feat.long().unsqueeze(1)).squeeze(1)
            is_raw = (kind_now == 0)
            is_eml_split = (kind_now == 1)

            eml_vals = torch.zeros(n_samples, dtype=torch.float32, device=device)
            if is_eml_split.any():
                idx_e = is_eml_split.nonzero(as_tuple=True)[0]
                node_ids_e = current[idx_e]
                descs_e = t["eml_descriptor"][node_ids_e]                # (m, 6)
                feat_subs_e = t["split_feat_subset"][node_ids_e]         # (m, K)
                X_e = X_gpu[idx_e].gather(1, feat_subs_e.long())         # (m, K)
                # Internal EML splits use raw (un-standardized) features
                eml_out_e = evaluate_trees_torch_per_sample(descs_e, X_e, K)
                eml_vals[idx_e] = eml_out_e

            split_vals = torch.where(is_raw, raw_vals, eml_vals)
            go_left = split_vals <= thr
            next_node = torch.where(
                go_left,
                t["left_child"][current].long(),
                t["right_child"][current].long(),
            )
            current = torch.where(is_internal, next_node, current)

        # Compute predictions per leaf type.
        kind_final = t["node_kind"][current]
        out_gpu = torch.zeros(n_samples, dtype=torch.float32, device=device)

        is_leaf_const = (kind_final == 2)
        if is_leaf_const.any():
            out_gpu[is_leaf_const] = t["leaf_value"][current[is_leaf_const]]

        is_leaf_eml = (kind_final == 3)
        if is_leaf_eml.any():
            idx_l = is_leaf_eml.nonzero(as_tuple=True)[0]
            node_ids_l = current[idx_l]
            feat_subs = t["leaf_feat_subset"][node_ids_l]               # (m, K)
            means = t["leaf_feat_mean"][node_ids_l]                     # (m, K)
            stds = t["leaf_feat_std"][node_ids_l]                       # (m, K)
            descs = t["eml_descriptor"][node_ids_l]                     # (m, 6)
            eta = t["leaf_eta"][node_ids_l]                             # (m,)
            bias = t["leaf_bias"][node_ids_l]                           # (m,)
            cap = t["leaf_cap"][node_ids_l]                             # (m,)

            X_leaf_raw = X_gpu[idx_l].gather(1, feat_subs.long())       # (m, K)
            X_leaf_std = torch.clamp((X_leaf_raw - means) / stds, -3.0, 3.0)
            eml_pred = evaluate_trees_torch_per_sample(descs, X_leaf_std, K)
            pred = eta * eml_pred + bias
            cap_finite = cap < float("inf")
            pred = torch.where(cap_finite, torch.clamp(pred, -cap, cap), pred)
            out_gpu[idx_l] = pred

        return out_gpu.cpu().numpy().astype(np.float64)
```

- [ ] **Step 9: Run the equivalence test; confirm it passes.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_gpu_predict_matches_cpu_fallback -v
```
Expected: PASS. The two predict paths agree to `rtol=1e-3, atol=1e-3`.

- [ ] **Step 10: Run the full unit test suite.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 30 in-scope tests + 1 new = 31 pass. Pre-existing unrelated failure stays.

- [ ] **Step 11: Quick benchmark to verify the predict speedup.**

```bash
PYTHONUNBUFFERED=1 uv run python -u -c "
import time, numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from eml_boost.tree_split import EmlSplitBoostRegressor

X, y = fetch_data('562_cpu_small', return_X_y=True)
X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
m = EmlSplitBoostRegressor(
    max_rounds=200, max_depth=8, learning_rate=0.1, min_samples_leaf=20,
    n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
    use_gpu=True, random_state=0,
)
t0 = time.time(); m.fit(X_tr, y_tr); print(f'fit: {time.time() - t0:.1f}s, rounds={m.n_rounds}')
t0 = time.time(); pred = m.predict(X_te); print(f'predict: {time.time() - t0:.3f}s')
print(f'RMSE: {np.sqrt(np.mean((pred - y_te)**2)):.4f}')
"
```

Expected: fit time should be noticeably faster than Task 2's checkpoint, and predict time should be < 0.5s for 1639 samples even on test data. The biggest win shows up on bigger datasets — that's Task 4's job to verify.

- [ ] **Step 12: Commit.**

```bash
git add eml_boost/tree_split/tree.py eml_boost/_triton_exhaustive.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: GPU predict path via tensorized tree

After fit, walk the Python tree once to emit a flat dict of CPU
tensors (_tensorize_tree); predict() lazily moves to GPU and runs a
bounded-depth torch loop instead of Python recursion. Key wins
expected on long boost loops where tree.predict(X_tr) is called per
round on the full training set.

Adds evaluate_trees_torch_per_sample to _triton_exhaustive.py — same
depth-2 grammar evaluation as evaluate_trees_torch but with one
descriptor per sample (predict path needs different leaves to evaluate
different snapped expressions on different feature subsets).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Speedup validation + Experiment 15 restart

**Goal:** Validate the GPU port hits its speedup targets on a representative large dataset, then restart Experiment 15 (resume-from-checkpoint preserves the 6 datasets already complete).

**Files:**
- Modify: `tests/unit/test_eml_split_tree.py` (one new test)
- Execute: `experiments/run_experiment15_full_pmlb.py` (restart)
- Modify: `experiments/experiment15/report.md` (Task 2 of the original Experiment 15 plan, will be done after the restart)

- [ ] **Step 1: Add the speedup sanity test.**

Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_gpu_speedup_on_synthetic_large():
    """Sanity: a 100k-row synthetic at depth=8 must complete in
    under 30 seconds. Loose bound — pre-port this took 60-120 seconds
    extrapolating from cpu_small."""
    import time
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100_000, 10)).astype(np.float64)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] ** 2 + 0.1 * rng.normal(size=100_000)

    m = EmlSplitTreeRegressor(
        max_depth=8, min_samples_leaf=20, n_eml_candidates=10, k_eml=3,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_gpu=True, random_state=0,
    )
    t0 = time.time()
    m.fit(X, y)
    elapsed = time.time() - t0
    pred = m.predict(X[:1000])  # also exercise predict
    assert pred.shape == (1000,)
    assert elapsed < 30.0, f"GPU fit on 100k-row took {elapsed:.1f}s (target < 30s)"
```

- [ ] **Step 2: Run the speedup test; confirm it passes.**

```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_gpu_speedup_on_synthetic_large -v
```
Expected: PASS in well under 30s. If it fails (takes longer), the port didn't deliver enough speedup on this regime — investigate before restarting Exp 15.

- [ ] **Step 3: Sanity-benchmark on a real medium-large PMLB dataset.**

```bash
PYTHONUNBUFFERED=1 uv run python -u -c "
import time, numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from eml_boost.tree_split import EmlSplitBoostRegressor

X, y = fetch_data('564_fried', return_X_y=True)
X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
m = EmlSplitBoostRegressor(
    max_rounds=200, max_depth=8, learning_rate=0.1, min_samples_leaf=20,
    n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
    use_gpu=True, random_state=0,
)
t0 = time.time(); m.fit(X_tr, y_tr); print(f'fit on fried (40k rows): {time.time() - t0:.1f}s, rounds={m.n_rounds}')
t0 = time.time(); pred = m.predict(X_te); print(f'predict: {time.time() - t0:.3f}s')
print(f'RMSE: {np.sqrt(np.mean((pred - y_te)**2)):.4f}')
"
```

Expected: fit on `564_fried` (40k rows × 10 features) at d=8 should take **< 15 seconds** (pre-port: ~30-40 s). If it's still 30+ seconds, the port has gaps; investigate before restarting Exp 15. The headline ratio against XGBoost should be ~1.00 (unchanged from Exp 13).

- [ ] **Step 4: Commit the speedup test.**

```bash
git add tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
test: synthetic 100k-row sanity test for GPU port speedup

Loose 30s wall-clock bound on a depth-8 fit over n=100k, k=10
synthetic data. Pre-port estimate was 60-120s from cpu_small
extrapolation; passes comfortably with GPU port.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Restart Experiment 15 (full run with resume-from-checkpoint).**

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment15_full_pmlb.py 2>&1 | tee -a experiments/experiment15/run.log
```

The runner reads existing `summary.csv` (which has 6 datasets × 5 seeds × 3 models = 90 rows from the pre-port run) and skips those datasets. Remaining: 116 datasets × 5 seeds × 3 models = 1740 fits. Estimated runtime with the GPU port: **2-3 hours** (down from 13-15 h pre-port). The headline statistic should remain comparable to Exp 13's 7-dataset story.

If interrupted, the same command resumes from where it stopped.

Expected console: starts with "resume: 90 (dataset, seed, config) triples already complete; ..." and then iterates the remaining 116 datasets.

- [ ] **Step 6: Read the outputs and write `experiments/experiment15/report.md`.**

This is Task 2 of the original Experiment 15 plan. Follow that plan's Step 3 ("Write `experiments/experiment15/report.md`") to draft the report from `experiments/experiment15/summary.json`'s headline stats. Fill the template's `<…>` placeholders with concrete numbers. Pay special attention to:

- Whether the catastrophic-failure rate (`frac_catastrophic`) is < 5% (S-C in the original plan).
- Whether the within-10% fraction is > 60% (S-A) and outright-wins fraction is > 30% (S-B).
- Whether any of the original 7 datasets behave differently at 5 seeds vs the 3-seed Exp-13 result.

- [ ] **Step 7: Run the full unit test suite one final time.**

```bash
uv run pytest tests/unit/ -v
```

Expected: all 31 in-scope tests pass plus `test_gpu_speedup_on_synthetic_large` from this task = 32 in-scope passes. Pre-existing unrelated failure stays.

- [ ] **Step 8: Commit the Experiment 15 outputs and report.**

```bash
git add experiments/experiment15/summary.csv experiments/experiment15/summary.json \
        experiments/experiment15/pmlb_rmse.png experiments/experiment15/report.md
git add -f experiments/experiment15/run.log experiments/experiment15/failures.json
git commit -m "$(cat <<'EOF'
exp 15 done: full PMLB multi-seed suite with GPU port

122 PMLB regression datasets × 5 seeds × 3 models. Headline:
<fill in once known: e.g., "X/Y within 10% of XGBoost, Z outright
wins, mean ratio R">.

The GPU port (commits before this) cut SplitBoost fit time on big
datasets from minutes to seconds, making the full PMLB suite
tractable. Resume-from-checkpoint preserved the 6 datasets fit
under the pre-port code; the remaining 116 fit under the new
GPU-native pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Edit the heredoc to fill in the headline numbers concretely before running.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- `_y_gpu` allocation → Task 1 step 4.
- `_top_features_by_corr_gpu` → Task 1 step 3.
- `_grow_gpu` → Task 2 step 5.
- `_find_best_split_gpu` refactor → Task 2 step 3.
- `_fit_leaf` GPU-tensor signature → Task 2 step 4.
- `fit()` dispatches based on `self._device` → Task 2 step 6 + Task 3 step 4.
- CPU `_grow` no longer calls `_fit_leaf` → Task 2 step 7.
- `_tensorize_tree` → Task 3 step 3.
- `evaluate_trees_torch_per_sample` helper → Task 3 step 5.
- GPU `predict()` + `_predict_cpu_fallback` rename → Task 3 step 7-8.
- Equivalence tests (gpu_grow_matches_cpu, gpu_predict_matches_cpu) → Task 2 + Task 3.
- Speedup test → Task 4 step 1.
- Experiment 15 restart → Task 4 step 5.

No gaps.

**Placeholder scan:** Task 4 step 6's report-writing step says "follow that plan's Step 3" rather than re-listing the template — acceptable cross-reference since the original Exp-15 plan is already on disk and the implementer has access to it. No TBDs in code blocks.

**Type consistency:** `indices: torch.Tensor` used consistently across `_grow_gpu`, `_find_best_split_gpu`, `_fit_leaf`. `left_mask` is a GPU bool tensor in the new path. `_gpu_tree` is a `dict[str, torch.Tensor | int]` consistently. `evaluate_trees_torch_per_sample(descriptor, X, k)` returns `(m,)` shape used identically in both `_predict_gpu` call sites (split-time and leaf-time).
