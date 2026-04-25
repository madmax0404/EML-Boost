# Leaf L2 regularization (XGBoost-style `reg_lambda`)

**Date:** 2026-04-25
**Context:** Experiment 15 (full PMLB regression suite) showed 99/119 outright wins vs XGBoost but a clean small-n loss regime: 13 datasets with ratio > 1.05 all have n_train ≤ 200, and 3 catastrophic (ratio > 2.0). A targeted isolation experiment on `663_rabe_266` (n=120, ratio 2.34) and `561_cpu` (n=210, ratio 2.15) showed the catastrophic gap is in the **vanilla GBDT machinery**, not in EML: with `n_eml_candidates=0` and `k_leaf_eml=0` (pure histogram GBDT, no EML), SplitBoost is still 2.3-3.3× worse than XGBoost on these datasets. The mechanism: SplitBoost has no L2 leaf-value regularization. XGBoost's default `reg_lambda=1.0` shrinks leaf values via `Σg / (Σh + λ)` and shrinks split-gain via `(Σg)² / (Σh + λ)`; SplitBoost's constant leaves use plain `mean(y_residual)` and its split-gain is plain SSE-reduction. On tiny noisy data, the unshrunk path overfits aggressively. This spec adds full XGBoost-style L2 regularization (leaf-value shrinkage + regularized split-gain), defaulting on at `leaf_l2 = 1.0`.

## Goal

Close the catastrophic-loss gap on small-n datasets (n_train ≤ ~200) without hurting the architectural lead on the median dataset. Match XGBoost's `reg_lambda=1.0` default behavior on the parts of SplitBoost that mirror vanilla GBDT (constant leaves + split-finding); apply analogous shrinkage to the EML leaf bias term to keep the val-fit gate's comparison fair across leaf types.

After this work, on the 20 datasets where Experiment 15 lost to XGBoost (mean ratio > 1.00), the mean ratio across those 20 should drop materially — target: at least half of them re-enter the "within 10%" band, all 3 catastrophic datasets drop below ratio 1.5.

## Non-goals

- **No re-validation on Experiment 15 winners.** Per the user's direction, the validation experiment (Exp 16) only re-runs the 20 datasets where SplitBoost was worse than XGBoost. Risk acknowledged: a winner could regress at the new default; will catch in Experiment 17 (OpenML) on the new dataset distribution.
- **No tuning of `leaf_l2`.** Single value: 1.0, matching XGBoost's published default. No grid sweep.
- **No new XGBoost-style hyperparameters beyond `leaf_l2`.** `gamma` (split-gain threshold), `alpha` (L1), `min_child_weight` (Hessian-based min leaf), `subsample`, `colsample_*` are all out of scope. Each could close additional gap vs XGBoost but each adds tuning surface.
- **No changes to the EML-leaf η ridge term (`leaf_eml_ridge`).** That parameter stays at its existing default of 0.0; users who want it can opt in.
- **No CPU-path performance work.** The CPU paths get the regularization for parity but the GPU path is the ship target.

## Design overview

| component | change | site |
|---|---|---|
| **A.** parameter | new `leaf_l2: float = 1.0` on `EmlSplitTreeRegressor` and `EmlSplitBoostRegressor` (validated `>= 0`); threaded through to leaf-fit and split-finding | `tree.py` (init + `_fit_leaf` + `_find_best_split_gpu` + `_grow`/`_grow_gpu` plumbing); `ensemble.py` (forwarding) |
| **B.** constant leaf shrinkage | `value = Σy / (n + leaf_l2)` instead of `mean(y)` in both the early-out path and the constant fallback path | `tree.py:_fit_leaf` early-out (line ~458) and the batched-scalar block (line ~462-485) |
| **C.** EML leaf bias shrinkage | bias denominator becomes `(n_fit + leaf_l2)` in the closed-form OLS, in BOTH the `lam == 0` and `lam > 0` ridge branches | `tree.py:_fit_leaf` lines ~515-532 |
| **D.** split-gain regularization | replace `cnt` with `cnt + leaf_l2` in the SSE denominators of the split-gain computation; preserves bit-exactness at `leaf_l2 = 0` | `_gpu_split.py:gpu_histogram_split_torch`; `_gpu_split_triton.py:_hist_scan_kernel` + dispatcher |
| **E.** CPU-path parity | same gain formula change in `_best_threshold` + `_best_threshold_histogram`; constant-leaf shrinkage in `_grow` | `tree.py:_grow`, `_best_threshold*` (lines ~389-429) |

All five live in `eml_boost/tree_split/`. No public API is removed; only `leaf_l2` is added.

---

## A. Parameter

In `eml_boost/tree_split/tree.py`:

```python
class EmlSplitTreeRegressor:
    def __init__(
        self,
        *,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        # ... existing params ...
        leaf_l2: float = 1.0,                # NEW; matches XGBoost reg_lambda default
        random_state: int | None = None,
    ):
        if leaf_l2 < 0.0:
            raise ValueError(f"leaf_l2 must be >= 0, got {leaf_l2}")
        # ... existing assignments ...
        self.leaf_l2 = float(leaf_l2)
```

Same `leaf_l2: float = 1.0` parameter added to `EmlSplitBoostRegressor.__init__` in `ensemble.py`, forwarded to the per-round tree constructor in `_fit_gpu_loop` and `_fit_cpu_loop`.

`leaf_l2 = 0.0` is a valid value and recovers bit-exact pre-change behavior.

---

## B. Constant-leaf shrinkage

`_fit_leaf` produces a `LeafNode(value=...)` in two places: the early-out path and (transitively, via `_select_leaf_gated`) when the EML gate rejects. Both use `constant_value` derived from `y_sub.mean()`. Replace with `Σy / (n + λ)`.

### Early-out path

Currently (post-Task-E batched-`.item()`):

```python
if eml_disabled or too_small or no_gpu or n_raw == 0:
    return LeafNode(value=float(y_sub.mean().item()))
```

Becomes:

```python
if eml_disabled or too_small or no_gpu or n_raw == 0:
    return LeafNode(value=float((y_sub.sum() / (n + self.leaf_l2)).item()))
```

### Batched-scalar block (non-early-out path)

Currently the batched stack reads `y_sub.mean()` for `constant_value`. Replace with `y_sub.sum() / (n + self.leaf_l2)`. The other two stacked scalars (`indices[0].to(torch.float32)` for the seed, and `y_sub.abs().max() * cap_k` for `cap_leaf`) stay unchanged. Only `constant_value`'s formula shifts.

```python
if cap_k > 0.0:
    scalars_gpu = torch.stack([
        y_sub.sum() / (n + self.leaf_l2),       # CHANGED from y_sub.mean()
        indices[0].to(torch.float32),
        y_sub.abs().max() * cap_k,
    ])
    # ... unchanged unpacking ...
else:
    scalars_gpu = torch.stack([
        y_sub.sum() / (n + self.leaf_l2),       # CHANGED
        indices[0].to(torch.float32),
    ])
    # ... unchanged unpacking ...
```

`y_sub.sum() / (n + leaf_l2)` reduces to `y_sub.mean()` exactly when `leaf_l2 = 0` (assuming `n > 0`, which is guaranteed by the `if n == 0: return` at the top). At `leaf_l2 = 1.0`, leaf values are shrunk by `n / (n + 1)` — small datasets shrink more, large datasets approximately unchanged.

---

## C. EML leaf bias shrinkage

The closed-form OLS in `_fit_leaf` (lines ~515-532) computes (η, β) per candidate tree with optional ridge on η controlled by `leaf_eml_ridge`. Add L2 to the bias denominator. New body:

```python
n_fit = float(X_fit.shape[0])
n_fit_reg = n_fit + float(self.leaf_l2)              # NEW
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
    if self.leaf_l2 > 0.0:                            # NEW: post-shrink the closed-form bias
        bias = bias * n_fit / n_fit_reg
else:
    bias = (sum_y_f - eta * sum_p) / n_fit_reg        # CHANGED denom from n_fit to n_fit_reg
```

At `leaf_l2 = 0`, both branches collapse to the existing formulas exactly. At `leaf_l2 > 0`, both branches shrink the bias by `n_fit / (n_fit + leaf_l2)` relative to the unregularized solution. η is left to `leaf_eml_ridge`'s control (orthogonal regularizer).

The `det_safe` guard, the validity mask, and the `_select_leaf_gated`/`_select_leaf_blended` dispatch are unchanged.

---

## D. Split-gain regularization

The current SSE-reduction gain at boundary b — `total_sse - left_sse - right_sse`, where `sse(sum, sq, cnt) = sq - sum²/cnt` — equals `(Σr_L)²/n_L + (Σr_R)²/n_R - (Σr)²/n` after cancellation (the squared-residual terms `Σr²` are partition-invariant and cancel). XGBoost's gain replaces every `n` with `n + λ`, so the kernel-level change is **only the SSE denominators**:

```
sse_reg(sum, sq, cnt) = sq - sum² / (cnt + λ)
gain_reg = total_sse_reg - left_sse_reg - right_sse_reg
        = (Σr_L)²/(n_L + λ) + (Σr_R)²/(n_R + λ) - (Σr)²/(n + λ)
```

That last form is exactly XGBoost's gain (up to the constant 0.5 factor that doesn't affect argmax). To preserve bit-exactness at `λ = 0` while adding regularization at `λ > 0`, the existing `clamp(cnt, 1.0)` stays in place and `λ` is added on top: `denom = clamp(cnt, 1.0) + leaf_l2`. At `λ = 0` this reduces to the current `clamp(cnt, 1.0)` exactly; at `λ > 0` it equals `cnt + λ` for non-empty bins (since `clamp(cnt, 1.0) = cnt` whenever `cnt ≥ 1`) and `1 + λ` for empty bins (where `sum = 0` anyway, so the division contributes 0 to the gain regardless).

### `gpu_histogram_split_torch` in `_gpu_split.py`

Add `leaf_l2: float = 0.0` parameter. (Default 0.0 keeps backwards-compat for any direct callers; the dispatcher will pass the actual value.) Inside, the three SSE denominators stay clamped at 1.0 for empty-bin safety and add `leaf_l2` on top. Lines ~113 / ~126-127:

```python
total_sse = total_sq - total_sum ** 2 / (total_cnt.clamp(min=1.0) + leaf_l2)
# ...
left_sse = left_sq - left_sum ** 2 / (left_cnt.clamp(min=1.0) + leaf_l2)
right_sse = right_sq - right_sum ** 2 / (right_cnt.clamp(min=1.0) + leaf_l2)
```

At `leaf_l2 = 0` this is bit-identical to the current `cnt.clamp(min=1)`. At `leaf_l2 > 0` it equals `cnt + leaf_l2` for non-empty bins. The `total_cnt` is shape `(d, 1)`, the `left_cnt`/`right_cnt` are `(d, n_bins-1)`. Broadcasting works as before. The `legal = (left_cnt >= min_leaf_count) & (right_cnt >= min_leaf_count)` mask uses the unmodified counts (the `min_samples_leaf` constraint is on actual sample counts, not regularized counts).

### `_hist_scan_kernel` in `_gpu_split_triton.py`

Same change inside the kernel — keep the existing `tl.maximum(cnt, 1.0)` clamp for empty-bin safety and add `leaf_l2` on top:

```python
left_cnt_safe = tl.maximum(left_cnt, 1.0) + leaf_l2
right_cnt_safe = tl.maximum(right_cnt, 1.0) + leaf_l2
total_cnt_safe = tl.maximum(total_cnt, 1.0) + leaf_l2
left_sse = left_sq - left_sum * left_sum / left_cnt_safe
right_sse = right_sq - right_sum * right_sum / right_cnt_safe
total_sse = total_sq - total_sum * total_sum / total_cnt_safe
```

At `leaf_l2 = 0` this is bit-identical to the current `tl.maximum(cnt, 1.0)`.

`leaf_l2` is passed as a runtime float kernel argument (NOT `tl.constexpr` — it's a runtime value, and constexpr would force one compiled kernel per λ value, wasting Triton's compile cache). The kernel signature gains:

```python
@triton.jit
def _hist_scan_kernel(
    hist_ptr, out_gain_ptr, out_bin_ptr,
    n_features,
    leaf_l2,                              # NEW; runtime float, not constexpr
    MIN_LEAF: tl.constexpr,
    N_BINS: tl.constexpr,
):
```

The `legal` mask still uses unregularized `left_cnt >= MIN_LEAF` (matches torch oracle).

### Dispatcher in `_gpu_split.py`

`gpu_histogram_split` gains `leaf_l2: float = 0.0` parameter, forwarded to both Triton and torch implementations. The fallback contract stays the same: try Triton, catch `Exception`, warn-once on first fallback.

### Caller in `tree.py`

`_find_best_split_gpu` calls `gpu_histogram_split(all_feats, y_node, self.n_bins, min_leaf_count=self.min_samples_leaf, leaf_l2=self.leaf_l2)`.

---

## E. CPU-path parity

The CPU pipeline (`_grow` + `_find_best_split_cpu` + `_best_threshold` + `_best_threshold_histogram`) is rarely exercised in production but the existing GPU-vs-CPU equivalence test (`test_grow_gpu_matches_cpu` or whatever it's named) must keep passing. Same shape of changes:

- `_best_threshold` (line ~389): replace `cumsum_sq[i-1] - cumsum[i-1]**2 / i` with `cumsum_sq[i-1] - cumsum[i-1]**2 / (np.maximum(i, 1.0) + leaf_l2)` and similar for the right-side and total. Add `leaf_l2` parameter (default 0.0). The `np.maximum(i, 1.0)` is a no-op for the CPU-path numpy `i` array which starts at 1 by construction; the form is kept for symmetry with the GPU path's empty-bin-safe pattern.
- `_best_threshold_histogram`: same. Add `leaf_l2` parameter.
- `_find_best_split_cpu`: pass `leaf_l2=self.leaf_l2` to both threshold helpers.
- `_grow` constant-leaf production: change `LeafNode(value=float(y.mean()) if len(y) > 0 else 0.0)` to `LeafNode(value=float(y.sum() / (len(y) + self.leaf_l2)) if len(y) > 0 else 0.0)`.

CPU path doesn't have the `_select_leaf_gated` / EML-leaf logic — leaves are always constant on CPU per the existing comment "EML leaves require GPU". So the EML bias change (section C) doesn't apply to CPU.

---

## F. Tests

`tests/unit/test_eml_split_tree.py` gains three tests; existing tests need a triage pass.

### New: `test_leaf_l2_zero_is_bit_exact_to_pre_change`

Snapshots a fitted model's predictions at `leaf_l2=0` and compares to a stored baseline (captured from the pre-change code via a one-off script before this PR). Asserts `np.testing.assert_array_equal` (bit-exact, not allclose) — the change is mathematically a no-op at λ = 0 and the floating-point ordering is preserved.

```python
def test_leaf_l2_zero_is_bit_exact_to_pre_change():
    """leaf_l2=0 must produce identical predictions to the pre-change implementation."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 5)).astype(np.float64)
    y = (np.exp(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=500))
    m = EmlSplitTreeRegressor(
        max_depth=4, n_eml_candidates=10, k_eml=2,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_l2=0.0,
        use_gpu=True, random_state=0,
    ).fit(X, y)
    pred = m.predict(X[:50])
    # Golden snapshot — capture by running this fixture on the pre-change
    # HEAD (commit `90957c1`), copying pred.tolist() output, and pasting
    # below. The implementation task includes a step to do this capture
    # before modifying any production code.
    expected = np.array([
        # 50 float values — captured during Step 1 of the implementation plan
    ], dtype=np.float64)
    np.testing.assert_array_equal(pred, expected)
```

### New: `test_leaf_l2_gpu_cpu_equivalence`

At `leaf_l2=1.0`, GPU and CPU paths agree within float32 tolerance. Same harness as the existing `test_gpu_matches_cpu_pipeline` (or whatever the existing equivalence test is named) but with `leaf_l2=1.0`.

### New: `test_hist_split_triton_matches_torch_with_l2`

The Triton `_hist_scan_kernel` and the torch `gpu_histogram_split_torch` agree at `leaf_l2 ∈ {0.0, 0.5, 1.0, 2.0}` to within `rtol=5e-3` on the gain (the same tolerance the existing `test_histogram_split_triton_matches_torch` uses).

### Existing tests

Default flips from `leaf_l2=0` (implicit, current behavior) to `leaf_l2=1.0`. Tests that:
- Assert finite predictions / non-degenerate fits → unchanged, will pass.
- Assert specific RMSE thresholds (e.g., `train_mse < 0.5` on a clean signal) → may need the threshold widened; review during implementation.
- Use `random_state` and assert exact prediction values → will need the snapshots regenerated. None expected in the current suite; flag if any surface.

The `test_gpu_speedup_on_synthetic_large` (100k-row 30s threshold) should still pass; leaf_l2 doesn't affect runtime materially.

---

## G. Validation experiment (Experiment 16)

`experiments/run_experiment16_leaf_l2_validation.py`:

1. Reads `experiments/experiment15/summary.json`. Identifies the 20 datasets where `ratios[name].ratio > 1.00` (SplitBoost lost to XGBoost on the mean).
2. Re-fits SplitBoost only with `leaf_l2=1.0`, otherwise the Exp-15 defaults (`max_depth=8, max_rounds=200, n_eml_candidates=10, ...`), on those 20 datasets × 5 seeds.
3. xgb/lgb numbers are read from the existing `experiment15/summary.csv` — not re-fit.
4. Output: `experiments/experiment16/summary.csv` (per-fit rows for the new SplitBoost numbers), `summary.json` (per-dataset old-vs-new aggregates), `comparison.md` (writeup with side-by-side ratios).

Expected runtime: 100 fits at small-medium dataset sizes ≈ 5-15 minutes on RTX 3090.

### Comparison table format

`comparison.md` includes a per-dataset table:

| dataset | n_train | k | Exp 15 ratio (no L2) | Exp 16 ratio (L2=1.0) | Δ | verdict |
|---|---|---|---|---|---|---|
| 527_analcatdata_election2000 | 53 | 14 | 2.355 | ?.??? | ±?.??? | catastrophic → ??? |
| 663_rabe_266 | 96 | 2 | 2.340 | ?.??? | ±?.??? | catastrophic → ??? |
| 561_cpu | 167 | 7 | 2.148 | ?.??? | ±?.??? | catastrophic → ??? |
| ... 17 more rows ... |

Plus headline: how many of 20 dropped below 1.00, how many below 1.10, mean ratio change across the 20.

---

## H. Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — `leaf_l2` param + plumbing; `_fit_leaf` constant + EML bias paths (sections B + C); `_grow` + `_find_best_split_cpu` + threshold helpers (section E); `_find_best_split_gpu` plumbing.
- `eml_boost/tree_split/ensemble.py` — `leaf_l2` param on `EmlSplitBoostRegressor.__init__`; forward to the per-round tree constructor.
- `eml_boost/tree_split/_gpu_split.py` — `leaf_l2` param on dispatcher and torch oracle; gain formula change.
- `eml_boost/tree_split/_gpu_split_triton.py` — `leaf_l2` runtime arg on `_hist_scan_kernel`; gain formula change.
- `tests/unit/test_eml_split_tree.py` — 3 new tests; review existing tests for threshold widening.

**Created:**
- `experiments/run_experiment16_leaf_l2_validation.py` — runner.
- `experiments/experiment16/{summary.csv, summary.json, comparison.md}` — outputs.

**Unchanged:**
- The leaf-EML closed-form OLS for η (still governed by `leaf_eml_ridge`).
- The val-fit gate / blended leaf selection (`_select_leaf_gated`, `_select_leaf_blended`).
- The Triton kernels for predict and EML evaluation (`_predict_triton.py`, `_triton_exhaustive.py`).
- The boost loop, X-cache, predict path.
- The `EmlBoostRegressor` (older line) — out of scope.

---

## I. Success criteria

- **S-A (correctness):** all 97 existing tests pass after the change; 3 new tests pass; pre-existing `test_fit_recovers_simple_formula` failure unchanged. Existing tests with shifted RMSEs that need threshold widening are documented in the implementation PR.
- **S-B (catastrophic losses fixed):** in Experiment 16, all 3 catastrophic Exp 15 datasets (`527_analcatdata_election2000`, `663_rabe_266`, `561_cpu`) drop to ratio ≤ 1.5 (down from 2.15-2.36).
- **S-C (clear losses materially improved):** in Experiment 16, the mean ratio across the 20 lost datasets drops by at least 0.20 (from a current weighted mean ~1.4 to ≤ 1.2).
- **S-D (no Triton fallback):** Experiment 16's run.log contains no `RuntimeWarning` or fallback messages from the histogram split kernel.

If S-A holds but S-B/S-C don't, escalate before shipping the default flip — could be (a) the L2 mechanism doesn't generalize beyond the two profile targets, or (b) other regularizers (subsampling, gamma, min_child_weight) are needed.

## J. Risks

- **Default flip changes user-visible behavior on every `EmlSplitBoostRegressor()` call.** Anyone with no-arg or partial-arg construction gets a different model. Mitigated by `leaf_l2=0.0` opt-out; documented in the spec and the commit message; covered by the bit-exact equivalence test for `leaf_l2=0`.
- **Wins not validated.** Per user direction. A winner could regress at `leaf_l2=1.0`; Experiment 17 (OpenML) will catch it on the new dataset distribution. Worst case: the default reverts to 0.0 and `leaf_l2=1.0` becomes opt-in.
- **The math is XGBoost-faithful only for L2 leaf-value shrinkage + L2-shrunk gain.** XGBoost's full regularization stack also includes γ (split-gain threshold), α (L1), `min_child_weight` (Hessian-based). This spec ships only the L2 part; the others remain potential follow-ups if S-B/S-C are mixed.
- **Triton kernel regression.** The `_hist_scan_kernel` change is small but the kernel is load-bearing. Mitigated by the existing `test_histogram_split_triton_matches_torch` (which the new λ-sweep test extends) and the warn-once fallback.
- **Float ordering at `leaf_l2 = 0`.** The bit-exact-equivalence test asserts `assert_array_equal` (not `allclose`) at λ=0. The changes are algebraically no-ops at λ=0 (multiplying by `n/n` etc.), and the snapshot-vs-current comparison should hold; if it doesn't, the implementation has accidentally re-ordered an op and needs fixing before merge.

## K. Action on verdict

- **All success criteria met (S-A through S-D):** ship the default flip (`leaf_l2=1.0`), commit the validation report, and queue Experiment 17 (OpenML re-validation) to confirm wins didn't regress.
- **S-A fails (correctness regression on existing tests):** fix the regression before shipping. Most likely culprit is a typo in one of the three sites (constant leaf, EML bias, gain).
- **S-B fails (catastrophic losses still > 1.5):** the L2 mechanism is insufficient; consider adding γ (split-gain threshold) before shipping, OR ship `leaf_l2` as opt-in (default 0.0) and write a follow-up spec for additional regularization.
- **S-D fails (Triton fallback fires):** the `leaf_l2` runtime kernel arg may be incompatible with the Triton version; profile and either fix or pin the kernel to use a compile-time constant per λ value.
