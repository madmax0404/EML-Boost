# Stacked-Blend Leaves for EML-SplitBoost (Experiment 9)

**Date:** 2026-04-24
**Context:** Follows Experiment 8. EML leaves (Phase 4) moved the PMLB 7-dataset benchmark from 4/7 to 5/7 outright wins against XGBoost, but regressed `562_cpu_small` from ratio 0.81 → 0.90 and left two datasets (`210_cloud`, `557_analcatdata_apnea1`) above the 10% band. The current leaf fit is a **binary accept/reject gate**: EML leaf if val-SSE improvement > 5%, else constant. The hypothesis tested here is that a **soft blend** — `prediction = α · constant + (1 − α) · (η · eml(x) + β)` with α fit on a held-out val split — recovers the cpu_small regression without giving up any existing wins, and reduces seed-to-seed variance.

## Goal

Replace the binary leaf gate with a val-fit convex blend between constant and EML contributions. Evaluate across 3 seeds on the Experiment 8 datasets, with blend-on vs. blend-off run at the same commit for a clean ablation.

## Non-goals

- Full PMLB suite (55 datasets) — out of scope; deferred to a separate experiment.
- Cross-validation — single 80/20 shuffle split per seed, matching Experiment 8.
- Capacity sweep — fixed `max_depth=6, max_rounds=200`, same as Experiment 8.
- New leaf node type — the post-fit representation stays as today's `(η, β)` after folding α in.

## Design

### Core algorithm change (per leaf)

1. **Early-out gates unchanged:** `k_leaf_eml > 0` and `n ≥ min_samples_leaf_eml` and GPU available. Otherwise return `LeafNode(value=ȳ)`.
2. **75/25 leaf-local fit/val split** — same deterministic per-leaf seed as today.
3. **For each of the 144 candidate depth-2 EML trees:**
   - OLS fit `(η, β)` on the fit portion.
   - On the val portion, compute `p_val = η · eml(x_val) + β` and `s = ȳ − p_val`.
   - Closed-form optimal blend weight:
     ```
     α* = clip( (s · (y_val − p_val)) / (s · s) , 0, 1 )
     ```
     Derivation: minimize `||y_val − α·ȳ − (1−α)·p_val||²` over α, which gives `α* = s · (y_val − p_val) / (s · s)` where `s = ȳ − p_val`.
   - Blended val residual: `r = y_val − (α*·ȳ + (1−α*)·p_val)`. Record `val_sse = r·r`.
4. **Pick the tree with smallest α-optimized val-SSE.**
5. **Fold** α into `(η, β)`:
   - `η' = (1 − α*) · η`
   - `β' = α* · ȳ + (1 − α*) · β`
6. **Emit leaf:**
   - If `|η'| < 1e-10`, emit `LeafNode(value=β')` (blend collapsed to constant — keeps leaf-type count interpretable).
   - Else emit `EmlLeafNode(snapped=best, feature_subset=top_features, feature_mean, feature_std, eta=η', bias=β')`.

### Why no new node type

The blend `α·ȳ + (1−α)·(η·eml(x) + β)` is algebraically equivalent to `η'·eml(x) + β'` with the folded coefficients. Predict-time code doesn't need to know α existed. This keeps `EmlLeafNode` unchanged and lets blend-off / blend-on share the same `_predict_vec` path.

### Why the closed-form α is safe

The expression `α* = s·(y_val − p_val) / (s·s)` is a 1-D OLS. Numerical failure modes:

- `s · s = 0` iff `p_val ≡ ȳ` (the EML fit is a constant equal to the global mean). In that case the blend is degenerate; we detect `||s||² < 1e-12` and force `α* = 1` (pure constant, since EML provides no signal).
- `s · s > 0` but the closed-form α is unbounded in the absence of clipping. We clip to `[0, 1]` — this is a convex combination by design, and an unclipped α would let the model anti-blend (α < 0 amplifies EML; α > 1 sign-flips the constant). Both are pathological on held-out val.

### Tree selection under the blend

For each candidate tree, compute the α-optimized val-SSE, and pick the tree with the smallest one. This is policy **Y** from the brainstorm (α participates in selection). Extra compute cost: one 1-D closed-form and one val-SSE recompute per candidate — all on GPU, negligible.

This matches the "α replaces the gate" philosophy: a tree whose pure fit overshoots but whose blended form shrinks sensibly can still win.

### Config API — backward compatibility

- `EmlSplitTreeRegressor.__init__` gains one new param:
  ```python
  use_stacked_blend: bool = True
  ```
  When `True` (default), the blend algorithm above runs. When `False`, the existing gate-based algorithm runs. Both code paths live in `_fit_leaf`; neither is deleted. This lets Exp 9 compare blend-on vs blend-off at the same commit.
- `leaf_eml_gain_threshold` is **retained but unused when `use_stacked_blend=True`**. Preserves constructor compat; a later cleanup PR can delete after the Exp 9 results settle.
- `EmlSplitBoostRegressor.__init__` passes `use_stacked_blend` through to each round's tree.

### Instrumentation

- Each `EmlSplitTreeRegressor` grows an internal list `self._leaf_stats: list[dict]` populated when `use_stacked_blend=True`:
  ```python
  {"n_leaf": int, "alpha": float, "leaf_type": Literal["EmlLeafNode", "LeafNode"]}
  ```
- The Exp 9 runner aggregates across all trees of a fitted `EmlSplitBoostRegressor` to report "mean α across all EML leaves" and "fraction of attempted EML leaves that collapsed to constant."

## Experiment 9 plan

### Datasets (same as Experiment 8)

`192_vineyard`, `210_cloud`, `523_analcatdata_neavote`, `557_analcatdata_apnea1`, `529_pollen`, `562_cpu_small`, `564_fried`.

### Models per run

Four per (dataset, seed):

1. **SplitBoost blend-off** (`use_stacked_blend=False, leaf_eml_gain_threshold=0.05`) — emulates the Experiment 8 result.
2. **SplitBoost blend-on** (`use_stacked_blend=True`) — the new algorithm.
3. **XGBoost** (same config as Exp 8).
4. **LightGBM** (same config as Exp 8).

### Config (shared blend-on and blend-off)

```
max_rounds          = 200
max_depth           = 6
learning_rate       = 0.1
patience            = 15
val_fraction        = 0.15
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf_eml = 50
n_bins              = 256
histogram_min_n     = 500
test_size           = 0.20
seeds               = [0, 1, 2]
```

### Runner

`experiments/run_experiment9_stacked_blend.py` — fork of `run_experiment8_pmlb_split.py`, adds a seed loop and a blend-off/blend-on branch per seed.

### Outputs

- `experiments/experiment9/summary.csv` — one row per (dataset, seed, model) combo.
- `experiments/experiment9/summary.json` — same data indexed by dataset, then model, then seed list + mean/std.
- `experiments/experiment9/pmlb_rmse.png` — three-panel figure: bars with error bars for blend-off vs blend-on vs XGBoost across datasets.
- `experiments/experiment9/leaf_stats.json` — per-seed per-dataset aggregated leaf-level α distributions.
- `experiments/experiment9/report.md` — narrative write-up.

## Tests

Added to `tests/unit/test_eml_split_tree.py`:

1. **`test_stacked_blend_alpha_zero_on_clean_signal`** — synthetic `y = exp(x₀) + 0.01·noise`, a single tree fit with `use_stacked_blend=True`. Inspect one leaf's effective `(η, β)` and confirm α-folded prediction is dominated by the EML term (i.e., the leaf is `EmlLeafNode` and its predictions on held-out data beat a constant by ≥ 50%).
2. **`test_stacked_blend_alpha_one_on_pure_noise`** — synthetic `y ~ N(0, 1)`, a single tree with `use_stacked_blend=True, min_samples_leaf_eml=50`. Confirm `≥ 80%` of EML-eligible leaves collapsed to `LeafNode` (α ≈ 1).
3. **`test_stacked_blend_no_numerical_blowup_on_heavy_tails`** — synthetic feature `X[:, 0] = 1e6 · N(0, 1)`, targets `y = small linear combination`. Fit with `use_stacked_blend=True`; predict on held-out data; assert no NaN or ±inf.
4. **`test_blend_off_flag_preserves_current_gate_behavior`** — same synthetic data as the existing `test_eml_leaf_gate_rejects_weak_fits`, fit with `use_stacked_blend=False`, verify the existing observable holds (most leaves stay constant).

Existing tests stay green unchanged.

## Success criteria

Any one of the following in the Experiment 9 results makes the blend a keeper (ratio = SplitBoost RMSE / XGBoost RMSE; lower is better):

- **S-A (primary):** blend-on ratio is within 10% of XGBoost on 5+/7 datasets across all 3 seeds, AND at least one dataset's mean ratio is ≥ 0.03 *lower* with blend-on than blend-off.
- **S-B:** `562_cpu_small` mean ratio < 0.85 with blend-on, without any dataset switching from ratio < 1 (win) with blend-off to ratio ≥ 1 (loss) with blend-on.
- **S-C:** mean σ of ratios across seeds is lower with blend-on than with blend-off.

**Negative outcome:** if blend-on has a ≥ 0.03 *higher* mean ratio than blend-off on 2 or more datasets, declare the blend unhelpful. Revert the default to `use_stacked_blend=False` and keep only the code path and tests. The gate stays as default.

## Risks & mitigations

- **Small val split.** At `min_samples_leaf_eml=50`, val-size is ~12 samples and α's closed form is high-variance. If blend-on hurts small-n datasets (`192_vineyard`, `523_analcatdata_neavote`), that's the cause; mitigation is bumping `min_samples_leaf_eml` to 100 in a follow-up sweep.
- **Backward-compat clutter.** Keeping `leaf_eml_gain_threshold` as a no-op parameter when `use_stacked_blend=True` introduces a small API wart. Acceptable for Experiment 9; cleanup is a follow-up PR.
- **α instability when `ȳ ≈ β`.** If the global leaf mean already equals the OLS intercept (rare, but possible when y is already centered), `s` has very small norm and α blows up pre-clip. The `s · s < 1e-12` check forces α=1 in that case, which collapses to constant — safe default.
- **Seed-to-seed runtime variance.** XGBoost and LightGBM are ~0.2 s each; SplitBoost is ~5-36 s. Total budget for 3 seeds × 4 models × 7 datasets is under 5 minutes on RTX 3090.

## Out of scope — explicit callouts

- Reporting α distributions in the interpretability section is *nice-to-have*, not a success gate. If the stats collection slows down fits materially, skip it.
- Multi-seed CV: not here. Exp 9 uses 3 single-split seeds, not 3 CV folds.
- Capacity sweep (depth/rounds): not here.
- Comparison to Exp 7's old-hybrid architecture: already done in Exp 8, not repeated.

## Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — add `use_stacked_blend` param, rewrite `_fit_leaf` to branch on it, add `_leaf_stats` list.
- `eml_boost/tree_split/ensemble.py` — thread `use_stacked_blend` to per-round trees.
- `tests/unit/test_eml_split_tree.py` — add 4 new tests.

**Created:**
- `experiments/run_experiment9_stacked_blend.py` — runner.
- `experiments/experiment9/` directory (with outputs generated by the runner).

**Unchanged:**
- `eml_boost/tree_split/nodes.py` — schema identical.
- `eml_boost/tree_split/_gpu_split.py`, `eml_boost/_triton_exhaustive.py` — kernel code untouched.
