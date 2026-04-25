# Leaf L2 Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `leaf_l2: float = 1.0` parameter (XGBoost-style `reg_lambda`) to `EmlSplitTreeRegressor` / `EmlSplitBoostRegressor`, applying L2 shrinkage to constant leaves, EML leaf bias, and split-gain across both GPU and CPU paths. Validate on the 20 Experiment 15 datasets where SplitBoost lost to XGBoost.

**Architecture:** Three sites get the regularizer (constant leaf value `Σy/(n+λ)`, EML bias `(Σy−η·Σp)/(n_fit+λ)`, split-gain SSE denominator `cnt+λ`). Bit-exact at `leaf_l2=0` is preserved by leaving the existing `clamp(cnt, 1.0)` in place and adding `+ leaf_l2` on top of it. Default flips from 0.0 (incremental dev) to 1.0 (final spec value) in Task 7. Validation experiment in Task 8 re-runs SplitBoost on the 20 Exp-15 losers and writes a side-by-side comparison.

**Tech Stack:** Python 3.12 / numpy / torch / Triton / pytest. Run via `uv run`. Tests in `tests/unit/` (97 passing + 1 known-unrelated failure left alone).

**Spec:** `docs/superpowers/specs/2026-04-25-leaf-l2-regularization-design.md`.

---

## File Structure

| file | role | tasks that touch it |
|---|---|---|
| `eml_boost/tree_split/tree.py` | Add `leaf_l2` param + plumb to all leaf-fit and split-finding sites | Tasks 2, 3, 4, 6, 7 |
| `eml_boost/tree_split/ensemble.py` | Add `leaf_l2` param to `EmlSplitBoostRegressor`; forward to per-round tree constructor | Task 2 |
| `eml_boost/tree_split/_gpu_split.py` | Add `leaf_l2` to dispatcher + torch oracle gain formula | Task 4 |
| `eml_boost/tree_split/_gpu_split_triton.py` | Add `leaf_l2` runtime arg to `_hist_scan_kernel` + dispatcher | Task 5 |
| `tests/unit/test_eml_split_tree.py` | 3 new tests (bit-exact, multi-λ Triton equivalence, GPU/CPU equivalence at λ=1.0); triage existing tests | Tasks 1, 5, 6, 7 |
| `experiments/run_experiment16_leaf_l2_validation.py` | NEW — runner | Task 8 |
| `experiments/experiment16/` | NEW — outputs | Task 8 |

---

## Task 1: Capture pre-change golden snapshot + write bit-exact test

**Goal:** Lock in the baseline behavior at `leaf_l2=0` (which is "current behavior") with a deterministic CPU-path test. The test must pass on the current HEAD (`a5c5b6b` — the spec commit, no code change yet) so that future tasks can ensure they preserve bit-exactness at λ=0. CPU path is used because it's deterministic across runs (no atomics); also it covers the constant-leaf shrinkage (B) and the histogram-split-gain change (D, CPU side via `_best_threshold_histogram`).

**Files:**
- Modify: `tests/unit/test_eml_split_tree.py` (add `test_leaf_l2_zero_constant_leaves_bit_exact`)

### Step 1: Write the test scaffold with placeholder snapshot

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_leaf_l2_zero_constant_leaves_bit_exact():
    """At leaf_l2=0 with EML disabled, predictions must be bit-identical
    to the pre-leaf-l2 implementation. Captures a CPU-path snapshot
    (deterministic across runs); the new leaf_l2 plumbing is a
    mathematical no-op at λ=0 and must produce identical floats."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(600, 5)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] - X[:, 2] ** 2
         + 0.05 * rng.normal(size=600))
    m = EmlSplitTreeRegressor(
        max_depth=5,
        min_samples_leaf=20,
        n_eml_candidates=0,        # disable EML internal splits (CPU + GPU paths skip them)
        k_eml=2,
        k_leaf_eml=0,              # disable EML leaves (CPU can't anyway)
        n_bins=256,
        histogram_min_n=500,       # n=600 > 500 → exercises _best_threshold_histogram
        use_gpu=False,             # CPU is deterministic
        leaf_l2=0.0,               # the bit-exactness invariant we're testing
        random_state=0,
    ).fit(X, y)
    pred = m.predict(X[:50])
    expected = np.array([
        # SNAPSHOT — captured during Task 1 Step 3 by running this test
        # body up to `pred = m.predict(X[:50])` on the pre-leaf-l2 HEAD
        # (commit a5c5b6b) and pasting `pred.tolist()` here.
    ], dtype=np.float64)
    assert pred.shape == expected.shape, (
        f"snapshot shape mismatch: {pred.shape} vs {expected.shape}"
    )
    np.testing.assert_array_equal(pred, expected)
```

### Step 2: Note that the test won't pass yet — it has no `leaf_l2` parameter on the regressor

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_l2_zero_constant_leaves_bit_exact -v`
- [ ] Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'leaf_l2'` (the parameter doesn't exist yet — that's added in Task 2).
- [ ] To capture the snapshot for now, temporarily remove the `leaf_l2=0.0` line from the test, then re-run. The test will fail on the snapshot-shape assertion (empty `expected`).

### Step 3: Capture the snapshot via a one-shot script

- [ ] In a Python REPL or a one-shot script, run the same fixture without `leaf_l2`:

```python
import numpy as np
from eml_boost.tree_split import EmlSplitTreeRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(600, 5)).astype(np.float64)
y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] - X[:, 2] ** 2
     + 0.05 * rng.normal(size=600))
m = EmlSplitTreeRegressor(
    max_depth=5, min_samples_leaf=20, n_eml_candidates=0, k_eml=2,
    k_leaf_eml=0, n_bins=256, histogram_min_n=500,
    use_gpu=False, random_state=0,
).fit(X, y)
pred = m.predict(X[:50])
print(repr(pred.tolist()))
```

- [ ] Run: `uv run python -c "<above>"`
- [ ] Copy the printed list of 50 floats into the `expected = np.array([...])` placeholder in the test.
- [ ] Restore the `leaf_l2=0.0` line in the test (it was in the original spec; we'll add the parameter in Task 2 and the test will then run end-to-end).

### Step 4: Verify the test passes once the parameter exists (Task 2 will add it)

- [ ] At this point the test still fails on `TypeError: leaf_l2`. That's expected. **Task 1's job is to land the snapshot, not the parameter** — Task 2 adds the parameter and the test goes green.
- [ ] To leave the test in a runnable state until then, comment out `leaf_l2=0.0` with a TODO marker:

```python
        # leaf_l2=0.0,  # TODO: uncomment after Task 2 adds the leaf_l2 parameter
```

- [ ] Re-run the test. Expected: PASS (the snapshot matches the un-changed CPU code).

### Step 5: Commit

- [ ] Run:

```bash
git add tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
test: capture pre-change snapshot for leaf_l2=0 bit-exact test (Task 1)

Adds test_leaf_l2_zero_constant_leaves_bit_exact with a deterministic
CPU-path fixture and a 50-prediction golden snapshot captured from the
pre-change HEAD. Future tasks must preserve the snapshot's bit-equality
at leaf_l2=0; if the snapshot ever drifts, an op was reordered.

The leaf_l2 parameter doesn't exist yet — Task 2 adds it. The test's
leaf_l2=0.0 line is commented out with a TODO until then.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `leaf_l2` parameter (default 0.0) + constant-leaf shrinkage

**Goal:** Plumb the `leaf_l2: float = 0.0` parameter through both regressors and apply it to the constant-leaf value computation in `_fit_leaf` (early-out path + non-early-out batched-scalar path). Default 0.0 keeps every existing test bit-exact for now (default flip happens in Task 7). After this task, the bit-exact test from Task 1 runs end-to-end and stays green.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (constructor + `_fit_leaf`)
- Modify: `eml_boost/tree_split/ensemble.py` (forward param to per-round tree)
- Modify: `tests/unit/test_eml_split_tree.py` (uncomment the `leaf_l2=0.0` line in the Task 1 test)

### Step 1: Add `leaf_l2` to `EmlSplitTreeRegressor.__init__`

- [ ] In `eml_boost/tree_split/tree.py`, find `EmlSplitTreeRegressor.__init__` (line ~73). Add `leaf_l2: float = 0.0` to the keyword-only params (after `leaf_eml_cap_k`, before `use_stacked_blend`):

```python
        leaf_eml_cap_k: float = 2.0,
        leaf_l2: float = 0.0,                   # NEW; default 0.0 for safety; flipped to 1.0 in Task 7
        use_stacked_blend: bool = False,
        random_state: int | None = None,
    ):
```

- [ ] In the body, validate and store (after the existing `self.leaf_eml_cap_k = leaf_eml_cap_k` assignment):

```python
        if leaf_l2 < 0.0:
            raise ValueError(f"leaf_l2 must be >= 0, got {leaf_l2}")
        self.leaf_l2 = float(leaf_l2)
```

### Step 2: Add `leaf_l2` to `EmlSplitBoostRegressor.__init__` and forward it

- [ ] In `eml_boost/tree_split/ensemble.py`, find `EmlSplitBoostRegressor.__init__`. Add the same `leaf_l2: float = 0.0` parameter (matching the order in `EmlSplitTreeRegressor`):

```python
        leaf_eml_cap_k: float = 2.0,
        leaf_l2: float = 0.0,                   # NEW; mirrors EmlSplitTreeRegressor
        use_stacked_blend: bool = False,
        ...
```

- [ ] Validate and store:

```python
        if leaf_l2 < 0.0:
            raise ValueError(f"leaf_l2 must be >= 0, got {leaf_l2}")
        self.leaf_l2 = float(leaf_l2)
```

- [ ] Find every place where the boost loop constructs an `EmlSplitTreeRegressor` (search for `EmlSplitTreeRegressor(` in `ensemble.py`). At each call site, add `leaf_l2=self.leaf_l2` to the kwargs.

### Step 3: Modify `_fit_leaf` constant-leaf computation

- [ ] In `eml_boost/tree_split/tree.py`, find `_fit_leaf` (line ~431). Two sites change:

**Early-out path** (line ~458). Current:

```python
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=float(y_sub.mean().item()))
```

Replace with:

```python
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=float((y_sub.sum() / (n + self.leaf_l2)).item()))
```

**Non-early-out batched-scalar path** (line ~462-485, the `if cap_k > 0.0:` and `else:` branches). The first stacked element is currently `y_sub.mean()`; replace with `y_sub.sum() / (n + self.leaf_l2)` in BOTH branches:

```python
        cap_k = float(self.leaf_eml_cap_k)
        if cap_k > 0.0:
            scalars_gpu = torch.stack([
                y_sub.sum() / (n + self.leaf_l2),         # CHANGED from y_sub.mean()
                indices[0].to(torch.float32),
                y_sub.abs().max() * cap_k,
            ])
            scalars = scalars_gpu.cpu().numpy()
            constant_value = float(scalars[0])
            seed = int(scalars[1])
            cap_leaf = float(scalars[2])
        else:
            scalars_gpu = torch.stack([
                y_sub.sum() / (n + self.leaf_l2),         # CHANGED from y_sub.mean()
                indices[0].to(torch.float32),
            ])
            scalars = scalars_gpu.cpu().numpy()
            constant_value = float(scalars[0])
            seed = int(scalars[1])
            cap_leaf = float("inf")
```

At `leaf_l2=0`, `y_sub.sum() / (n + 0) == y_sub.mean()` (both are computed as a single torch reduction over the same tensor; PyTorch's `mean()` is internally `sum() / numel`, so the float result is identical).

### Step 4: Modify `_grow` (CPU pipeline) constant-leaf computation

- [ ] In `eml_boost/tree_split/tree.py`, find `_grow` (line ~226). The `_const_leaf` helper:

```python
        def _const_leaf(y):
            return LeafNode(value=float(y.mean()) if len(y) > 0 else 0.0)
```

Replace with:

```python
        def _const_leaf(y):
            return LeafNode(
                value=float(y.sum() / (len(y) + self.leaf_l2)) if len(y) > 0 else 0.0
            )
```

### Step 5: Restore the bit-exact test's `leaf_l2=0.0` line

- [ ] In `tests/unit/test_eml_split_tree.py`, find the `# leaf_l2=0.0,  # TODO:` comment (added in Task 1) and uncomment it:

```python
        leaf_l2=0.0,
```

### Step 6: Run the bit-exact test to verify Task 2's changes are bit-exact at λ=0

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_l2_zero_constant_leaves_bit_exact -v`
- [ ] Expected: PASS. (At `leaf_l2=0`, `y.sum() / (n + 0) == y.mean()` and the snapshot still matches.)

### Step 7: Run the full unit suite to verify no regressions

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `98 passed, 1 failed` (97 prior + 1 new bit-exact test, vs the same pre-existing `test_fit_recovers_simple_formula` failure).

### Step 8: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: add leaf_l2 parameter + constant-leaf shrinkage (Task 2)

Adds leaf_l2: float = 0.0 to EmlSplitTreeRegressor and
EmlSplitBoostRegressor (forwarded to the per-round tree). Modifies
constant-leaf value in _fit_leaf (early-out + batched-scalar paths)
and in _grow (CPU pipeline) from y.mean() to y.sum() / (n + leaf_l2).

At leaf_l2=0 the formulas are mathematically identical to the prior
mean(); the bit-exact test from Task 1 confirms zero behavior change.
Default stays 0.0 in this task; flipped to 1.0 in Task 7 alongside
existing-test threshold triage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: EML leaf bias shrinkage

**Goal:** Apply L2 shrinkage to the EML leaf's bias term in `_fit_leaf`'s closed-form OLS, both `lam == 0` (no ridge on η) and `lam > 0` (existing ridge on η) branches. At `leaf_l2=0`, mathematically identical to current behavior. The η coefficient is left to `leaf_eml_ridge`'s control (orthogonal regularizer).

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (`_fit_leaf` closed-form OLS section, lines ~515-532)

### Step 1: Modify the closed-form OLS bias computation

- [ ] In `eml_boost/tree_split/tree.py`, find the closed-form OLS block in `_fit_leaf` (search for `n_fit = float(X_fit.shape[0])`). The current block is:

```python
        n_fit = float(X_fit.shape[0])
        sum_p = preds_fit.sum(dim=1)
        sum_p2 = (preds_fit * preds_fit).sum(dim=1)
        sum_y_f = y_fit.sum()
        sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
        det = sum_p2 * n_fit - sum_p * sum_p
        lam = float(self.leaf_eml_ridge)
        det_ridged = det + n_fit * lam
        det_safe = torch.where(
            det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
        )
        eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
        if lam == 0.0:
            bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
        else:
            bias = (sum_y_f - eta * sum_p) / n_fit
```

Replace with (only `n_fit_reg` add + the bias post-shrink in the `lam == 0` branch + the `n_fit_reg` denominator in the `lam > 0` branch):

```python
        n_fit = float(X_fit.shape[0])
        n_fit_reg = n_fit + float(self.leaf_l2)        # NEW
        sum_p = preds_fit.sum(dim=1)
        sum_p2 = (preds_fit * preds_fit).sum(dim=1)
        sum_y_f = y_fit.sum()
        sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
        det = sum_p2 * n_fit - sum_p * sum_p
        lam = float(self.leaf_eml_ridge)
        det_ridged = det + n_fit * lam
        det_safe = torch.where(
            det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
        )
        eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
        if lam == 0.0:
            bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
            if self.leaf_l2 > 0.0:                     # NEW: post-shrink the closed-form bias
                bias = bias * n_fit / n_fit_reg
        else:
            bias = (sum_y_f - eta * sum_p) / n_fit_reg # CHANGED denom from n_fit to n_fit_reg
```

At `leaf_l2 = 0`: `n_fit_reg == n_fit`, so the `lam > 0` branch is bit-identical (denominator unchanged). The `lam == 0` branch's `if self.leaf_l2 > 0.0:` guard skips the post-shrink, also bit-identical. ✓

### Step 2: Run the bit-exact test to verify Task 3's changes don't break the CPU path

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_l2_zero_constant_leaves_bit_exact -v`
- [ ] Expected: PASS. (CPU path doesn't exercise EML leaves so the `_fit_leaf` change can't affect it. The test still passes because all the other code paths the test exercises are unchanged.)

### Step 3: Run GPU equivalence tests to verify EML-leaf path stays bit-exact at leaf_l2=0

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_predict_triton_matches_torch tests/unit/test_eml_split_boost.py::test_xcache_matches_baseline -v`
- [ ] Expected: PASS (skipped if no CUDA). These exercise the EML-leaf path on GPU; at default `leaf_l2=0` the new code is a no-op.

### Step 4: Run the full unit suite to verify no regressions

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `98 passed, 1 failed` (the same pre-existing failure left alone).

### Step 5: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py
git commit -m "$(cat <<'EOF'
feat: leaf_l2 shrinkage on EML leaf bias (Task 3)

Modifies the closed-form OLS in _fit_leaf to add L2 regularization on
the bias term, in both the lam == 0 (no leaf_eml_ridge) and lam > 0
branches. Mathematically:

  bias_l2 = (Σy - η·Σp) / (n_fit + leaf_l2)

For lam == 0 we post-shrink the joint OLS bias by n_fit / (n_fit +
leaf_l2) (equivalent to using the conditional-OLS form with the
unregularized eta and adding L2 only on bias). For lam > 0 we replace
the bias denominator directly. At leaf_l2 = 0 both branches are
bit-exact to the prior code.

eta is left to leaf_eml_ridge's orthogonal control, matching the
spec's design choice (XGBoost's reg_lambda regularizes the leaf
weight as a whole; for SplitBoost's two-coefficient EML leaf, bias is
the analog of the leaf weight, eta is a feature-coefficient and gets
its own regularizer).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Split-gain regularization in torch oracle + thread to GPU caller

**Goal:** Update `gpu_histogram_split_torch` (the torch fallback / oracle) and the `gpu_histogram_split` dispatcher to accept and apply `leaf_l2`. Thread the value through from `_find_best_split_gpu` in `tree.py`. At `leaf_l2=0`, bit-exact to current behavior because `clamp(cnt, 1.0) + 0.0 == clamp(cnt, 1.0)`.

**Files:**
- Modify: `eml_boost/tree_split/_gpu_split.py` (dispatcher + torch oracle)
- Modify: `eml_boost/tree_split/tree.py` (`_find_best_split_gpu` call site)

### Step 1: Add `leaf_l2` parameter to `gpu_histogram_split_torch`

- [ ] In `eml_boost/tree_split/_gpu_split.py`, find `gpu_histogram_split_torch` (line ~40). Update the signature:

```python
def gpu_histogram_split_torch(
    values: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
    leaf_l2: float = 0.0,                       # NEW; defaults to 0.0 for backwards-compat
) -> tuple[int, float, float]:
```

- [ ] In the body (around lines ~113-127), update the three SSE denominators. Current:

```python
    total_sse = total_sq - total_sum ** 2 / total_cnt.clamp(min=1)
    # ...
    left_sse = left_sq - left_sum ** 2 / left_cnt.clamp(min=1)
    right_sse = right_sq - right_sum ** 2 / right_cnt.clamp(min=1)
```

Replace with:

```python
    total_sse = total_sq - total_sum ** 2 / (total_cnt.clamp(min=1.0) + leaf_l2)
    # ...
    left_sse = left_sq - left_sum ** 2 / (left_cnt.clamp(min=1.0) + leaf_l2)
    right_sse = right_sq - right_sum ** 2 / (right_cnt.clamp(min=1.0) + leaf_l2)
```

(Three sites; preserve any surrounding lines unchanged.)

### Step 2: Add `leaf_l2` to the `gpu_histogram_split` dispatcher

- [ ] In `eml_boost/tree_split/_gpu_split.py`, find `gpu_histogram_split` (line ~147). Update the signature and body to forward `leaf_l2`:

```python
def gpu_histogram_split(
    feats: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
    leaf_l2: float = 0.0,                       # NEW
) -> tuple[int, float, float]:
    """Best-split-finding via histogram. Tries Triton kernel first;
    falls back to the torch implementation on any error.

    Warns once per process the first time the fallback fires so silent
    Triton failures don't go unnoticed in production runs. Mirrors the
    same dispatcher pattern used for the predict kernel (commit a4df96d).
    """
    try:
        from eml_boost.tree_split._gpu_split_triton import (
            gpu_histogram_split_triton,
        )
        return gpu_histogram_split_triton(feats, y, n_bins, min_leaf_count, leaf_l2)
    except Exception as exc:
        global _TRITON_HIST_FALLBACK_WARNED
        if not _TRITON_HIST_FALLBACK_WARNED:
            import warnings
            warnings.warn(
                f"Triton histogram-split kernel failed; falling back to torch. "
                f"This warning fires once per process. "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _TRITON_HIST_FALLBACK_WARNED = True
        return gpu_histogram_split_torch(feats, y, n_bins, min_leaf_count, leaf_l2)
```

(Note: the call to `gpu_histogram_split_triton` now passes `leaf_l2`; this call will fail until Task 5 updates that function's signature. This is OK temporarily because the existing histogram-split test calls the torch oracle directly, not through the dispatcher, AND the dispatcher's try/except will catch the TypeError and fall back to torch. The warn-once will fire — that's expected during this transition step. Task 5 fixes it.)

### Step 3: Update `_find_best_split_gpu` to pass `leaf_l2`

- [ ] In `eml_boost/tree_split/tree.py`, find `_find_best_split_gpu` (line ~322). Find the call to `gpu_histogram_split` (line ~357):

```python
        best_idx, best_t, best_gain = gpu_histogram_split(
            all_feats, y_node, self.n_bins, min_leaf_count=self.min_samples_leaf,
        )
```

Update to:

```python
        best_idx, best_t, best_gain = gpu_histogram_split(
            all_feats, y_node, self.n_bins,
            min_leaf_count=self.min_samples_leaf,
            leaf_l2=self.leaf_l2,
        )
```

### Step 4: Run the bit-exact test

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_l2_zero_constant_leaves_bit_exact -v`
- [ ] Expected: PASS. (CPU path doesn't go through `gpu_histogram_split`, so this test is unaffected — but running it confirms we haven't broken it.)

### Step 5: Run the full unit suite

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `98 passed, 1 failed`. The dispatcher will warn-once on the first hist-split call (Triton path's signature mismatch, falls back to torch); that's transient and Task 5 fixes it. If the warn-once causes a test failure due to `warnings.simplefilter('error', RuntimeWarning)` in the existing test fixture, comment out that line temporarily — Task 5 will restore it.
- [ ] Specifically run `uv run pytest tests/unit/test_eml_split_tree.py::test_histogram_split_triton_matches_torch -v` and confirm it still passes (it calls `gpu_histogram_split_triton` directly with the OLD signature, which exists until Task 5).

### Step 6: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/_gpu_split.py eml_boost/tree_split/tree.py
git commit -m "$(cat <<'EOF'
feat: leaf_l2 shrinkage in torch oracle gain (Task 4 of 8)

Adds leaf_l2 parameter to gpu_histogram_split_torch and the
gpu_histogram_split dispatcher; threads through from
_find_best_split_gpu (self.leaf_l2 → kwarg). Modifies the three SSE
denominators in the torch oracle from cnt.clamp(min=1) to
cnt.clamp(min=1.0) + leaf_l2 — bit-exact at leaf_l2=0 (the existing
clamp pattern is preserved; +0.0 is a no-op).

The Triton kernel still has the OLD signature (no leaf_l2 param);
Task 5 updates it. During this transition the dispatcher's try/except
falls back to the torch oracle on the first call (and warns once);
the existing histogram-split equivalence test calls the Triton kernel
directly with old-signature args, so it stays green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Split-gain regularization in Triton kernel + multi-λ equivalence test

**Goal:** Update `_hist_scan_kernel` to accept `leaf_l2` as a runtime float and apply it inside the kernel; update `gpu_histogram_split_triton` wrapper accordingly. Restore the dispatcher's Triton-first path (no more fallback). Add a new test that verifies Triton-vs-torch equivalence at multiple λ values.

**Files:**
- Modify: `eml_boost/tree_split/_gpu_split_triton.py` (kernel + wrapper)
- Modify: `tests/unit/test_eml_split_tree.py` (add `test_hist_split_triton_matches_torch_with_l2`)

### Step 1: Update `_hist_scan_kernel` signature and body

- [ ] In `eml_boost/tree_split/_gpu_split_triton.py`, find `_hist_scan_kernel` (line ~120). Update the signature to add `leaf_l2` as a runtime arg (NOT `tl.constexpr` — runtime floats avoid recompilation per λ value):

```python
@triton.jit
def _hist_scan_kernel(
    hist_ptr,       # (N_FEATURES, N_BINS, 3) float32 — populated histogram
    out_gain_ptr,   # (N_FEATURES,) float32 — best gain per feature
    out_bin_ptr,    # (N_FEATURES,) int32   — best bin boundary per feature
    n_features,
    leaf_l2,                              # NEW; runtime float, not constexpr
    MIN_LEAF: tl.constexpr,
    N_BINS: tl.constexpr,
):
```

- [ ] Inside the kernel body, update the three SSE denominators. Current:

```python
    total_cnt_safe = tl.maximum(total_cnt, 1.0)
    total_sse = total_sq - total_sum * total_sum / total_cnt_safe
    # ...
    left_cnt_safe = tl.maximum(left_cnt, 1.0)
    right_cnt_safe = tl.maximum(right_cnt, 1.0)
    left_sse = left_sq - left_sum * left_sum / left_cnt_safe
    right_sse = right_sq - right_sum * right_sum / right_cnt_safe
```

Replace with:

```python
    total_cnt_safe = tl.maximum(total_cnt, 1.0) + leaf_l2
    total_sse = total_sq - total_sum * total_sum / total_cnt_safe
    # ...
    left_cnt_safe = tl.maximum(left_cnt, 1.0) + leaf_l2
    right_cnt_safe = tl.maximum(right_cnt, 1.0) + leaf_l2
    left_sse = left_sq - left_sum * left_sum / left_cnt_safe
    right_sse = right_sq - right_sum * right_sum / right_cnt_safe
```

The `legal` mask (`(left_cnt >= MIN_LEAF) & (right_cnt >= MIN_LEAF)`) uses unmodified counts — unchanged.

### Step 2: Update `gpu_histogram_split_triton` wrapper signature

- [ ] In the same file, find `gpu_histogram_split_triton` (line ~202). Update the signature:

```python
def gpu_histogram_split_triton(
    feats: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
    leaf_l2: float = 0.0,                       # NEW
) -> tuple[int, float, float]:
```

- [ ] In the body, find the call to `_hist_scan_kernel` (line ~288). Add `leaf_l2=leaf_l2` to the call:

```python
    _hist_scan_kernel[grid_scan](
        hist,
        out_gain,
        out_bin,
        n_features=d,
        leaf_l2=leaf_l2,                        # NEW
        MIN_LEAF=min_leaf_count,
        N_BINS=n_bins,
    )
```

### Step 3: Write the new multi-λ equivalence test

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_hist_split_triton_matches_torch_with_l2():
    """The Triton _hist_scan_kernel must agree with the torch oracle
    at multiple leaf_l2 values, including 0.0 (bit-exact-equivalent)
    and 1.0 (the new default). Uses the same tolerance as
    test_histogram_split_triton_matches_torch (rtol=5e-3 on gain,
    one bin width on threshold)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from eml_boost.tree_split._gpu_split import gpu_histogram_split_torch
    from eml_boost.tree_split._gpu_split_triton import (
        gpu_histogram_split_triton,
    )

    rng = np.random.default_rng(0)
    n, d = 1000, 5
    X = torch.tensor(
        rng.uniform(-1, 1, size=(n, d)), dtype=torch.float32, device="cuda"
    )
    y = torch.tensor(
        X[:, 2].cpu().numpy() * 2 + 0.1 * rng.normal(size=n),
        dtype=torch.float32, device="cuda",
    )

    for lam in (0.0, 0.5, 1.0, 2.0):
        idx_t, thr_t, gain_t = gpu_histogram_split_torch(
            X, y, n_bins=256, min_leaf_count=20, leaf_l2=lam,
        )
        idx_tri, thr_tri, gain_tri = gpu_histogram_split_triton(
            X, y, n_bins=256, min_leaf_count=20, leaf_l2=lam,
        )
        assert int(idx_t) == int(idx_tri), f"feature mismatch at lam={lam}"
        assert abs(float(thr_t) - float(thr_tri)) < 0.01, (
            f"threshold mismatch at lam={lam}: torch={thr_t} triton={thr_tri}"
        )
        np.testing.assert_allclose(
            float(gain_t), float(gain_tri), rtol=5e-3,
            err_msg=f"gain mismatch at lam={lam}",
        )
```

### Step 4: Run the new test

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_hist_split_triton_matches_torch_with_l2 -v`
- [ ] Expected: PASS (skipped if no CUDA). All four λ values agree across torch and Triton.

### Step 5: Run the full unit suite

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `99 passed, 1 failed` (98 from after Task 4 + 1 new test, vs the same pre-existing failure). No more transient warnings from the Task 4 dispatcher fallback.

### Step 6: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/_gpu_split_triton.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: leaf_l2 shrinkage in Triton hist-scan kernel (Task 5 of 8)

Updates _hist_scan_kernel to take leaf_l2 as a runtime float (not
constexpr — runtime args avoid one compile per λ value) and applies
it inside the SSE denominators: tl.maximum(cnt, 1.0) + leaf_l2. At
leaf_l2=0, bit-identical to the prior tl.maximum(cnt, 1.0). Wrapper
gpu_histogram_split_triton gains the same parameter and forwards it.

Adds test_hist_split_triton_matches_torch_with_l2: Triton-vs-torch
equivalence at λ ∈ {0.0, 0.5, 1.0, 2.0}, same tolerance as the
existing single-λ equivalence test (rtol=5e-3 on gain, one bin width
on threshold).

After this task the dispatcher's try/except no longer triggers a
fallback on the first call — Triton path now matches the wrapper
signature.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CPU-path parity + GPU/CPU equivalence test at λ=1.0

**Goal:** Apply the same gain-formula change to the CPU split-finding helpers (`_best_threshold` + `_best_threshold_histogram`) and thread `leaf_l2` through `_find_best_split_cpu`. Add a new test that verifies GPU and CPU produce the same fit at `leaf_l2=1.0` within float32 tolerance.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (`_best_threshold`, `_best_threshold_histogram`, `_find_best_split_cpu`)
- Modify: `tests/unit/test_eml_split_tree.py` (add `test_leaf_l2_gpu_cpu_equivalence_at_one`)

### Step 1: Update `_best_threshold_histogram` for `leaf_l2`

- [ ] In `eml_boost/tree_split/tree.py`, find `_best_threshold_histogram` (line ~780). Add `leaf_l2: float = 0.0` parameter:

```python
    def _best_threshold_histogram(
        self, values: np.ndarray, y: np.ndarray, leaf_l2: float = 0.0,
    ) -> tuple[float, float]:
```

- [ ] In the body, update the SSE denominators. Find the lines computing `total_sse`, `left_sse`, `right_sse` (around lines ~818, ~830-835). Currently:

```python
        total_sse = total_sq - total_sum ** 2 / total_count
        # ...
        left_sse = left_sq - left_sum ** 2 / left_count
        right_sse = right_sq - right_sum ** 2 / right_count
```

Replace with:

```python
        total_sse = total_sq - total_sum ** 2 / (np.maximum(total_count, 1.0) + leaf_l2)
        # ...
        left_sse = left_sq - left_sum ** 2 / (np.maximum(left_count, 1.0) + leaf_l2)
        right_sse = right_sq - right_sum ** 2 / (np.maximum(right_count, 1.0) + leaf_l2)
```

At `leaf_l2=0` and counts ≥ 1 (which they will be for legal splits), `np.maximum(cnt, 1.0) + 0.0 == cnt`. Bit-exact.

### Step 2: Update `_best_threshold` for `leaf_l2`

- [ ] In the same file, find `_best_threshold` (line ~389; a `@staticmethod`). Convert from `@staticmethod` to a regular method (so it can access `self.leaf_l2` if we wanted) — actually, keep it `@staticmethod` and just add a `leaf_l2: float = 0.0` parameter:

```python
    @staticmethod
    def _best_threshold(
        values: np.ndarray, y: np.ndarray, leaf_l2: float = 0.0,
    ) -> tuple[float, float]:
```

- [ ] In the body (lines ~412-420), find the SSE computation:

```python
        i = np.arange(1, n)
        left_sum = cumsum[i - 1]
        left_sq = cumsum_sq[i - 1]
        left_sse = left_sq - left_sum ** 2 / i
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        right_sse = right_sq - right_sum ** 2 / (n - i)

        gain = total_sse - left_sse - right_sse
```

Replace with:

```python
        i = np.arange(1, n)
        left_sum = cumsum[i - 1]
        left_sq = cumsum_sq[i - 1]
        left_sse = left_sq - left_sum ** 2 / (np.maximum(i, 1.0) + leaf_l2)
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        right_sse = right_sq - right_sum ** 2 / (np.maximum(n - i, 1.0) + leaf_l2)

        # Recompute total_sse with the same regularizer for symmetric subtraction.
        total_sse_reg = total_sq - total_sum ** 2 / (np.maximum(n, 1.0) + leaf_l2)
        gain = total_sse_reg - left_sse - right_sse
```

The `total_sse` defined earlier in the function (line ~407) was unregularized; here we override it with `total_sse_reg` to match the regularized left/right SSE. At `leaf_l2=0`, `total_sse_reg == total_sse` because `np.maximum(n, 1.0) == n` for n ≥ 1, so this is bit-exact at the default.

### Step 3: Update `_find_best_split_cpu` to pass `leaf_l2`

- [ ] In the same file, find `_find_best_split_cpu` (line ~276). Find the line:

```python
        threshold_fn = self._best_threshold_histogram if use_histogram else self._best_threshold
```

This stores a bound method reference. Then it's called as `threshold_fn(X[:, j], y)` in two places (the raw-feature loop and the EML-candidate loop). We need to pass `leaf_l2` to those calls. Easiest: don't use the dispatch reference, call directly:

```python
        use_histogram = len(y) >= self.histogram_min_n

        def _threshold(values: np.ndarray) -> tuple[float, float]:
            if use_histogram:
                return self._best_threshold_histogram(values, y, leaf_l2=self.leaf_l2)
            return self._best_threshold(values, y, leaf_l2=self.leaf_l2)

        for j in range(n_features):
            t, gain = _threshold(X[:, j])
            # ... unchanged ...

        if self.n_eml_candidates > 0 and n_features > 0:
            # ... unchanged setup ...
            for c_idx in range(candidates.shape[0]):
                if not finite[c_idx]:
                    continue
                t, gain = _threshold(eml_values[c_idx])
                # ... unchanged ...
```

### Step 4: Add the new GPU/CPU equivalence test

- [ ] Append to `tests/unit/test_eml_split_tree.py`:

```python
def test_leaf_l2_gpu_cpu_equivalence_at_one():
    """At leaf_l2=1.0, the GPU and CPU pipelines must produce equivalent
    predictions within float32 tolerance. Validates that the new gain
    formula and constant-leaf shrinkage are consistent across paths."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 4)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=800))

    common_kwargs = dict(
        max_depth=4,
        min_samples_leaf=20,
        n_eml_candidates=10,
        k_eml=2,
        k_leaf_eml=0,                # CPU can't do EML leaves; disable for fair comparison
        n_bins=256,
        histogram_min_n=500,
        leaf_l2=1.0,
        random_state=0,
    )
    m_gpu = EmlSplitTreeRegressor(**common_kwargs, use_gpu=True).fit(X, y)
    m_cpu = EmlSplitTreeRegressor(**common_kwargs, use_gpu=False).fit(X, y)

    pred_gpu = m_gpu.predict(X[:100])
    pred_cpu = m_cpu.predict(X[:100])
    np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-3, atol=1e-3)
```

### Step 5: Run the new test plus the bit-exact test

- [ ] Run: `uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_l2_gpu_cpu_equivalence_at_one tests/unit/test_eml_split_tree.py::test_leaf_l2_zero_constant_leaves_bit_exact -v`
- [ ] Expected: both PASS (the GPU/CPU test skips if no CUDA).

### Step 6: Run the full unit suite

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `100 passed, 1 failed` (99 from after Task 5 + 1 new test, vs the same pre-existing failure).

### Step 7: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: leaf_l2 in CPU-path split-finding for parity (Task 6 of 8)

Adds leaf_l2 parameter to _best_threshold + _best_threshold_histogram
(both with default 0.0 for backwards compat). _find_best_split_cpu
threads self.leaf_l2 through to both via a local closure to avoid
the bound-method-reference dispatch pattern.

Same denominator change as the GPU paths: cnt -> np.maximum(cnt, 1.0)
+ leaf_l2 in the SSE formulas. _best_threshold also recomputes
total_sse with the regularized denominator so the gain subtraction is
symmetric. Bit-exact at leaf_l2=0.

Adds test_leaf_l2_gpu_cpu_equivalence_at_one — at leaf_l2=1.0 the
GPU and CPU pipelines must agree within rtol=1e-3 on a small clean-
signal fit. EML leaves disabled in this test (k_leaf_eml=0) since the
CPU path doesn't support them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Flip default `leaf_l2` from 0.0 to 1.0 + triage existing tests

**Goal:** Change the default for `leaf_l2` from 0.0 to 1.0 in BOTH `EmlSplitTreeRegressor` and `EmlSplitBoostRegressor`. Run the full test suite. Triage any failures: smoke tests (predict-finite, train-MSE-below-some-threshold) should pass; specific-RMSE tests may need their thresholds widened.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (default value)
- Modify: `eml_boost/tree_split/ensemble.py` (default value)
- Modify (possibly): `tests/unit/test_eml_split_tree.py` and `tests/unit/test_eml_split_boost.py` (widen RMSE thresholds where needed)

### Step 1: Flip the default in `EmlSplitTreeRegressor`

- [ ] In `eml_boost/tree_split/tree.py`, find the `leaf_l2` parameter in `__init__` and change `0.0` → `1.0`:

```python
        leaf_l2: float = 1.0,                   # was 0.0; flipped to match XGBoost reg_lambda default
```

### Step 2: Flip the default in `EmlSplitBoostRegressor`

- [ ] In `eml_boost/tree_split/ensemble.py`, same change:

```python
        leaf_l2: float = 1.0,                   # was 0.0; mirrors EmlSplitTreeRegressor
```

### Step 3: Run the full unit suite to surface failures

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected outcomes:
  - `test_leaf_l2_zero_constant_leaves_bit_exact` — PASS (it explicitly passes `leaf_l2=0.0`).
  - `test_leaf_l2_gpu_cpu_equivalence_at_one` — PASS (explicitly `leaf_l2=1.0`).
  - `test_hist_split_triton_matches_torch_with_l2` — PASS (parametrized over λ values).
  - `test_predict_triton_matches_torch` — likely PASS (smoke check on predict path).
  - `test_xcache_matches_baseline` — likely PASS (smoke check on boost loop; train_mse threshold of 0.5 should still hold with leaf_l2=1.0).
  - Other pre-existing tests that fit and check predictions/RMSE — triage individually.
  - `test_fit_recovers_simple_formula` — still failing (pre-existing, leave alone).

### Step 4: Triage any failures

- [ ] For each newly-failing test:
  - **Smoke check (assert finite predictions, assert MSE < some-loose-threshold)**: usually passes; if it doesn't, the threshold may have been very tight. Widen by ~50% (e.g., `< 0.5` → `< 0.75`) and add a comment: `# threshold widened post-leaf_l2-default-flip; see commit <SHA>`.
  - **Hardcoded prediction values or RMSE values**: this should NOT exist in the current suite (no tests pin prediction values that I'm aware of). If found, regenerate the values OR convert the test to a smoke check.
  - **Floating-point allclose with tight rtol**: leaf_l2=1.0 changes the model meaningfully; relax to `rtol=1e-2` and add a comment.
- [ ] DO NOT modify the model behavior to make tests pass. Only relax test assertions.
- [ ] If a test has a clear behavior assertion that breaks at leaf_l2=1.0 (e.g., "this exact split is chosen"), document it in the commit message but do NOT block the task on it — the regularization legitimately changes split decisions.

### Step 5: Re-run the full suite to confirm green

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `100 passed, 1 failed` (the same pre-existing failure).

### Step 6: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/test_eml_split_tree.py tests/unit/test_eml_split_boost.py
git commit -m "$(cat <<'EOF'
feat: flip leaf_l2 default 0.0 -> 1.0 (Task 7 of 8)

Changes the default value of leaf_l2 in both EmlSplitTreeRegressor
and EmlSplitBoostRegressor from 0.0 (incremental-dev safety value)
to 1.0 (the spec's intended default; matches XGBoost reg_lambda).

Anyone calling EmlSplitBoostRegressor() with no args (or the tree
regressor) now gets a different model. To recover the prior bit-
exact behavior, pass leaf_l2=0.0 explicitly. The bit-exact test
(test_leaf_l2_zero_constant_leaves_bit_exact) explicitly does this
and stays green.

Tests with widened thresholds: <list any tests touched and the
specific old → new threshold; if none, write "none — all tests
passed without modification">.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If no test thresholds needed widening, drop the "Tests with widened thresholds" block from the message.)

---

## Task 8: Experiment 16 validation runner + run + comparison.md

**Goal:** Build a runner that re-fits SplitBoost (with the new `leaf_l2=1.0` default) on the 20 datasets where Experiment 15 lost to XGBoost. Read xgb/lgb numbers from `experiment15/summary.csv` (don't re-run them). Write a side-by-side comparison.md showing pre-fix vs post-fix ratios.

**Files:**
- Create: `experiments/run_experiment16_leaf_l2_validation.py`
- Create: `experiments/experiment16/{summary.csv, summary.json, comparison.md}` (outputs)

### Step 1: Write the runner

- [ ] Create `experiments/run_experiment16_leaf_l2_validation.py`:

```python
"""Experiment 16: leaf_l2=1.0 validation on Exp 15 losers.

Re-fits SplitBoost only (with the new leaf_l2=1.0 default, otherwise
Exp-15 defaults) on the 20 PMLB datasets where Exp 15's mean SplitBoost
ratio vs XGBoost was > 1.00. xgb/lgb numbers are read from
experiments/experiment15/summary.csv and not re-fit. Writes a side-by-
side comparison.md.

Estimated runtime: 100 fits at small-medium dataset sizes ≈ 5-15
minutes on RTX 3090.

Usage:
  PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment16_leaf_l2_validation.py 2>&1 | tee experiments/experiment16/run.log
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

EXP15_DIR = Path(__file__).resolve().parent / "experiment15"
EXP16_DIR = Path(__file__).resolve().parent / "experiment16"

# Match Exp 15 config exactly except leaf_l2 (which defaults to 1.0 post-Task-7).
MAX_ROUNDS = 200
MAX_DEPTH = 8
PATIENCE = 15
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF = 20
MIN_SAMPLES_LEAF_EML = 30
LEAF_EML_GAIN_THRESHOLD = 0.05
LEAF_EML_RIDGE = 0.0
LEAF_EML_CAP_K = 2.0
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2, 3, 4]

CSV_HEADER = "dataset,seed,config,rmse,fit_time,n_rounds\n"


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF, n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML, n_bins=N_BINS, histogram_min_n=500, use_gpu=True,
        k_leaf_eml=K_LEAF_EML, min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        leaf_eml_ridge=LEAF_EML_RIDGE, leaf_eml_cap_k=LEAF_EML_CAP_K,
        leaf_l2=1.0,                          # the change being validated
        use_stacked_blend=False,
        patience=PATIENCE, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def _identify_loser_datasets() -> list[tuple[str, dict]]:
    """Read experiment15/summary.json; return [(name, ratios_entry)] for
    datasets where the SplitBoost mean ratio vs XGBoost was > 1.00."""
    summary_path = EXP15_DIR / "summary.json"
    with summary_path.open() as fp:
        exp15 = json.load(fp)
    losers = []
    for name, r in exp15["ratios"].items():
        if r["ratio"] > 1.00:
            losers.append((name, r))
    losers.sort(key=lambda x: -x[1]["ratio"])
    return losers


def _load_completed(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists():
        return set()
    completed = set()
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            completed.add((row["dataset"], int(row["seed"]), row["config"]))
    return completed


def _append_rows(csv_path: Path, rows: list[RunResult]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a") as fp:
        if new_file:
            fp.write(CSV_HEADER)
        for r in rows:
            fp.write(
                f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n"
            )


def main() -> int:
    EXP16_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXP16_DIR / "summary.csv"

    losers = _identify_loser_datasets()
    print(
        f"Identified {len(losers)} Exp-15 loser datasets (SplitBoost mean ratio > 1.00).",
        flush=True,
    )

    completed = _load_completed(csv_path)
    print(f"Resume: {len(completed)} (dataset, seed) triples already complete.", flush=True)

    new_results: dict[str, list[RunResult]] = {}

    for name, exp15_entry in losers:
        print(f"\n=== {name} (Exp-15 ratio: {exp15_entry['ratio']:.3f}) ===", flush=True)
        try:
            X, y = fetch_data(name, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            continue

        rows_for_dataset: list[RunResult] = []
        for seed in SEEDS:
            if (name, seed, "split_boost_l2_1") in completed:
                print(f"  [seed={seed}] SKIPPED (already in CSV)", flush=True)
                continue
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed,
                )
                m, t = _fit_split_boost(X_tr, y_tr, seed)
                rmse = _rmse(m.predict(X_te), y_te)
                n_rounds = getattr(m, "n_rounds", 0)
                rows_for_dataset.append(RunResult(
                    dataset=name, seed=seed, config="split_boost_l2_1",
                    rmse=rmse, fit_time=t, n_rounds=n_rounds,
                ))
                print(
                    f"  [seed={seed}] split_boost_l2_1 ({t:7.1f}s, "
                    f"{n_rounds:>3} rounds)  RMSE={rmse:.4f}",
                    flush=True,
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [seed={seed}] FAILED: {type(e).__name__}: {e}", flush=True)

        if rows_for_dataset:
            _append_rows(csv_path, rows_for_dataset)
            for r in rows_for_dataset:
                completed.add((r.dataset, r.seed, r.config))
            new_results.setdefault(name, []).extend(rows_for_dataset)

    # ---- Build summary.json ----
    print("\nfinalizing summary.json + comparison.md...", flush=True)

    # Reload ALL Exp 16 rows (including resumed ones).
    all_new_rmses: dict[str, list[float]] = {}
    if csv_path.exists():
        with csv_path.open() as fp:
            for row in csv.DictReader(fp):
                all_new_rmses.setdefault(row["dataset"], []).append(float(row["rmse"]))

    # Per-dataset: new SB mean RMSE vs Exp-15's xgb mean.
    with (EXP15_DIR / "summary.json").open() as fp:
        exp15 = json.load(fp)

    comparison_rows = []
    for name, exp15_entry in losers:
        new_rmses = all_new_rmses.get(name, [])
        if not new_rmses:
            comparison_rows.append({
                "dataset": name,
                "exp15_ratio": exp15_entry["ratio"],
                "exp16_ratio": None,
                "delta": None,
                "verdict": "no_data",
            })
            continue
        new_mean = float(mean(new_rmses))
        xgb_mean = exp15_entry["xgboost_mean"]
        new_ratio = new_mean / xgb_mean if xgb_mean > 0 else float("nan")
        delta = new_ratio - exp15_entry["ratio"]
        if new_ratio < 1.00:
            verdict = "now_a_win"
        elif new_ratio < 1.10:
            verdict = "in_band"
        elif new_ratio < 1.50:
            verdict = "improved_but_still_loss"
        elif new_ratio < 2.00:
            verdict = "still_clear_loss"
        else:
            verdict = "still_catastrophic"
        comparison_rows.append({
            "dataset": name,
            "exp15_ratio": exp15_entry["ratio"],
            "exp16_ratio": new_ratio,
            "delta": delta,
            "verdict": verdict,
            "new_rmse_mean": new_mean,
            "exp15_xgb_mean": xgb_mean,
        })

    summary_json = {
        "config": {
            "leaf_l2": 1.0,
            "max_rounds": MAX_ROUNDS,
            "max_depth": MAX_DEPTH,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "n_eml_candidates": N_EML_CANDIDATES,
            "k_eml": K_EML,
            "k_leaf_eml": K_LEAF_EML,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
            "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
            "leaf_eml_ridge": LEAF_EML_RIDGE,
            "leaf_eml_cap_k": LEAF_EML_CAP_K,
            "n_bins": N_BINS,
            "test_size": TEST_SIZE,
            "seeds": SEEDS,
        },
        "comparison": comparison_rows,
        "n_datasets_attempted": len(losers),
        "n_datasets_with_data": sum(1 for c in comparison_rows if c["exp16_ratio"] is not None),
    }

    json_path = EXP16_DIR / "summary.json"
    with json_path.open("w") as fp:
        json.dump(summary_json, fp, indent=2)
    print(f"wrote {json_path}", flush=True)

    # ---- Build comparison.md ----
    md_path = EXP16_DIR / "comparison.md"
    valid = [c for c in comparison_rows if c["exp16_ratio"] is not None]
    n_now_wins = sum(1 for c in valid if c["verdict"] == "now_a_win")
    n_in_band = sum(1 for c in valid if c["verdict"] == "in_band")
    n_still_catastrophic = sum(1 for c in valid if c["verdict"] == "still_catastrophic")
    mean_delta = mean(c["delta"] for c in valid) if valid else 0.0

    with md_path.open("w") as fp:
        fp.write("# Experiment 16: leaf_l2=1.0 validation on Exp 15 losers\n\n")
        fp.write(f"**Date:** 2026-04-25\n")
        fp.write(f"**Config:** Exp-15 defaults + leaf_l2=1.0 (XGBoost reg_lambda match).\n")
        fp.write(f"**Datasets re-fit:** {len(valid)} of {len(losers)} loser datasets from Exp 15.\n\n")
        fp.write("## Headline\n\n")
        fp.write(f"- {n_now_wins}/{len(valid)} are now outright wins (ratio < 1.00).\n")
        fp.write(f"- {n_in_band}/{len(valid)} are within 10% (ratio < 1.10).\n")
        fp.write(f"- {n_still_catastrophic}/{len(valid)} are still catastrophic (ratio > 2.00).\n")
        fp.write(f"- Mean ratio change: **{mean_delta:+.3f}** (negative = improvement).\n\n")
        fp.write("## Per-dataset comparison\n\n")
        fp.write("| dataset | Exp 15 ratio | Exp 16 ratio | Δ | verdict |\n")
        fp.write("|---|---|---|---|---|\n")
        for c in comparison_rows:
            if c["exp16_ratio"] is None:
                fp.write(f"| {c['dataset']} | {c['exp15_ratio']:.3f} | — | — | no_data |\n")
            else:
                fp.write(
                    f"| {c['dataset']} | {c['exp15_ratio']:.3f} | "
                    f"{c['exp16_ratio']:.3f} | {c['delta']:+.3f} | {c['verdict']} |\n"
                )
    print(f"wrote {md_path}", flush=True)

    # ---- Console headline ----
    print("\n=== Headline ===", flush=True)
    print(f"  Now wins (ratio < 1.00):         {n_now_wins}/{len(valid)}", flush=True)
    print(f"  In band (ratio < 1.10):          {n_in_band}/{len(valid)}", flush=True)
    print(f"  Still catastrophic (>2.0):       {n_still_catastrophic}/{len(valid)}", flush=True)
    print(f"  Mean Δ ratio:                    {mean_delta:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Run the validation experiment

- [ ] Run: `PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment16_leaf_l2_validation.py 2>&1 | tee experiments/experiment16/run.log`
- [ ] Expected runtime: 5-15 minutes on RTX 3090. Output streams per-fit lines like `[seed=N] split_boost_l2_1 (Xs, Y rounds)  RMSE=Z` and ends with the headline block.
- [ ] At completion, `experiments/experiment16/{summary.csv, summary.json, comparison.md}` exist.

### Step 3: Verify success criteria from the spec

- [ ] Read the headline from `experiments/experiment16/comparison.md` (or the runner's stdout):
  - **S-B (catastrophic losses fixed):** all 3 of `527_analcatdata_election2000`, `663_rabe_266`, `561_cpu` have ratio ≤ 1.5.
  - **S-C (clear losses materially improved):** mean ratio change across the 20 datasets ≤ −0.20.
  - **S-D (no Triton fallback):** `grep -iE "warning|fallback|FAILED|Traceback|RuntimeWarning" experiments/experiment16/run.log || echo "no issues"` prints `no issues`.

### Step 4: Commit the runner + outputs

- [ ] Run:

```bash
git add experiments/run_experiment16_leaf_l2_validation.py experiments/experiment16/
git commit -m "$(cat <<'EOF'
exp 16 done: leaf_l2=1.0 validation on Exp 15 losers

Re-runs SplitBoost only (now with leaf_l2=1.0) on the <N> Exp-15
datasets where the original mean ratio vs XGBoost was > 1.00.
xgb/lgb numbers reused from experiment15/summary.csv (not re-fit).

Headline: <N>/<M> now outright wins, <N>/<M> in 10% band,
<N>/<M> still catastrophic, mean Δ ratio: <X.XXX>.

(<fill in from the actual run output before committing>)

See experiments/experiment16/comparison.md for the per-dataset
breakdown. The .log is gitignored per *.log; the runner is
self-contained and re-runnable via resume-from-checkpoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] Replace the `<N>/<M>` and `<X.XXX>` placeholders in the commit message with the actual values from the run before committing.

### Step 5: Report and pause for the user

- [ ] Per the user's "checkpoint before long runs" rule and given the visible experiment is now complete, do NOT immediately kick off any further work. Report the headline numbers and the per-dataset table to the user. Wait for their direction on whether to:
  - Accept the results and move on (e.g., to Exp 17 OpenML)
  - Iterate on the regularization (e.g., try `leaf_l2 ∈ {0.5, 2.0}` if S-B/S-C are mixed)
  - Roll back to `leaf_l2=0.0` default (if results are net-negative)

No further commits in this task — just the report.

---

## Implementation order recap

1. **Task 1** — Capture pre-change snapshot, write bit-exact test scaffold.
2. **Task 2** — Add `leaf_l2` parameter (default 0.0) + constant-leaf shrinkage.
3. **Task 3** — EML leaf bias shrinkage in `_fit_leaf` closed-form OLS.
4. **Task 4** — Split-gain regularization in torch oracle + thread to GPU caller.
5. **Task 5** — Split-gain regularization in Triton kernel + multi-λ equivalence test.
6. **Task 6** — CPU-path parity + GPU/CPU equivalence test at λ=1.0.
7. **Task 7** — Flip default 0.0 → 1.0 + triage existing tests.
8. **Task 8** — Build Exp 16 runner, run, write comparison.md, report to user.

Tasks 1-6 maintain bit-exact-at-leaf_l2=0 invariant; Task 7 explicitly breaks that for the user-visible default. Task 8 measures the impact on Exp-15 losers.
