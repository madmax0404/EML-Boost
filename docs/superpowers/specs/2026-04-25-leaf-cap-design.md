# Adaptive Leaf-Prediction Cap (Experiment 11)

**Date:** 2026-04-25
**Context:** Follows Experiments 9 and 10. The ridge investigation in Experiment 10 dampened η but couldn't prevent the explosions, and a diagnostic on `562_cpu_small` seed 2 confirmed the root cause: **depth-2 nested eml with `_EXP_CLAMP=50` and `_LOG_EPS=1e-6` can produce per-sample predictions up to ~10¹⁴ on test samples that land near the feature-clamp boundary.** One bad tree in the 200-round ensemble contributed `-2.55·10⁸` on a single test sample, matching the observed min-prediction exactly.

This experiment ports XGBoost's `max_delta_step` concept — a hard cap on leaf output magnitude — but adapts it per-leaf using the leaf's own residual scale, so the cap tightens appropriately across boosting rounds.

## Goal

Add a `leaf_eml_cap_k: float = 0.0` hyperparameter. When `k > 0`, each `EmlLeafNode`'s per-sample prediction `η·eml(x) + β` is clipped to `[-k·max|y_leaf|, +k·max|y_leaf|]`, where `y_leaf` is the residuals reaching the leaf at fit time. The cap is stored on the leaf node. Tree selection during fit uses the post-cap val-SSE so it picks the best tree under the clip semantics, not the best uncapped tree.

## Non-goals

- Fixed scalar cap (XGBoost-style `max_delta_step`). The per-leaf multiplier fits the gradient-boosting setting where residual scale shrinks across rounds; a fixed cap would be too tight on early rounds or too loose on late ones.
- Modifying the OLS closed form. The cap is a post-processor on `η·eml + β`; OLS runs uncapped. Iterative cap-aware OLS would need a non-closed-form solver (YAGNI for the first test).
- Fixing the `_EXP_CLAMP` constant in `_triton_exhaustive.py`. The cap addresses the symptom (extreme predictions) regardless of the internal kernel's clamp setting, and leaves the expressivity of the grammar untouched. A tighter `_EXP_CLAMP` remains available as a future follow-up.
- Interacting with ridge. `leaf_eml_ridge` and `leaf_eml_cap_k` are orthogonal; both can be nonzero, but for Experiment 11 we test cap alone to isolate its effect.

## Design

### Per-leaf cap computation

During fit, for each leaf that reaches the EML-decision point (passes the `k_leaf_eml > 0 && n ≥ min_samples_leaf_eml && n_fit > min_samples_leaf_eml/2` gates), compute:

```python
cap_leaf = float(self.leaf_eml_cap_k) * float(torch.abs(y_full).max().item())
```

where `y_full` is the tensor of residuals for every sample that reached the leaf (not just the fit portion — the cap is a leaf-level magnitude statistic, not a fit-stat).

When `self.leaf_eml_cap_k == 0.0`, the cap is conceptually infinite (`cap_leaf = +inf`), and the clipping is a no-op — identical to the pre-Experiment-11 code.

### Tree selection (fit time)

Both `_select_leaf_gated` and `_select_leaf_blended` currently compute `val_pred = η · preds_val + β` and a per-tree `val_sse`. Under this design, they additionally clamp `val_pred` to `[-cap_leaf, +cap_leaf]` before the SSE sum:

```python
val_pred = eta.unsqueeze(1) * preds_val + bias.unsqueeze(1)
if cap_leaf < float("inf"):
    val_pred = torch.clamp(val_pred, -cap_leaf, cap_leaf)
val_res = y_val.unsqueeze(0) - val_pred
val_sse = (val_res * val_res).sum(dim=1)
```

The same adjustment applies in `_select_leaf_blended` right before the blend-SSE computation: the blend value `α·ȳ + (1−α)·val_pred` becomes `α·ȳ + (1−α)·clip(val_pred, ±cap_leaf)` if we apply the cap inside the blend — but note: `ȳ` is the leaf residual mean, already small, so the blend itself is bounded. Simplest is to clip the raw `val_pred` (the `η·eml + β` part) *before* forming the blend. The downstream math is unchanged.

**Interaction with existing validity masks:** the `finite_preds` mask is already computed from `preds_fit` and `preds_val`. Clipping is applied after validity is established, so a tree with overflow preds is rejected before the clip even matters.

### `EmlLeafNode` schema change

Add a new field `cap: float` to `EmlLeafNode`. For backward compat with pre-Exp-11 fits, the field defaults to `float("inf")` (no cap). Existing code paths that don't touch the cap will store `float("inf")` automatically via the dataclass default, and existing serialized trees remain loadable.

```python
@dataclass
class EmlLeafNode:
    snapped: SnappedTree
    feature_subset: tuple[int, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    eta: float
    bias: float
    cap: float = float("inf")
```

### Predict-time clipping

In `_predict_vec`, after computing `out[idx] = node.eta * vals + node.bias`, apply the clip:

```python
pred = node.eta * vals + node.bias
if node.cap < float("inf"):
    pred = np.clip(pred, -node.cap, node.cap)
out[idx] = pred
```

The `inf` sentinel means no clip (backward-compat). `np.clip(..., -inf, inf)` is a no-op anyway, but the explicit conditional skips the vectorized op on the common `cap_k=0` path for minor overhead savings.

### Config API

Added to both class constructors, positioned right after `leaf_eml_ridge`:

```python
class EmlSplitTreeRegressor:
    def __init__(
        self,
        ...,
        leaf_eml_ridge: float = 0.0,
        leaf_eml_cap_k: float = 0.0,  # 0.0 = no cap (backward-compat)
        use_stacked_blend: bool = False,
        ...,
    ): ...
```

Same addition to `EmlSplitBoostRegressor`, threaded through to each round's tree.

## Experiment 11 plan

### Datasets

Same 7 PMLB datasets: `192_vineyard`, `210_cloud`, `523_analcatdata_neavote`, `557_analcatdata_apnea1`, `529_pollen`, `562_cpu_small`, `564_fried`.

### Seeds

`[0, 1, 2]`, same as Experiment 9/10.

### Configurations (5 SplitBoost variants, gated path only)

The gated path is cleaner for this experiment — it isolates the cap effect without tangling with the blend's α-shrinkage dynamic. Blend + cap is a natural follow-up if the cap works.

| id | selection | `leaf_eml_cap_k` | interpretation |
|---|---|---|---|
| C0 | gated | 0.0 | Exp 9 baseline reproduction (no cap) |
| C_tight | gated | 1.0 | cap at max residual (very restrictive) |
| C_loose | gated | 2.0 | cap at 2× max residual |
| C_med | gated | 5.0 | cap at 5× max residual |
| C_wide | gated | 10.0 | cap at 10× max residual (generous) |

### Baselines per (dataset, seed)

XGBoost and LightGBM for reference. No `leaf_eml_ridge` or `use_stacked_blend` (both stay at 0.0/False).

### Total fits

7 datasets × 3 seeds × (5 SplitBoost + XGBoost + LightGBM) = **147 fits**. Estimated runtime ~10-12 min on RTX 3090.

### Outputs

- `experiments/experiment11/summary.csv` — per-(dataset, seed, config) row.
- `experiments/experiment11/summary.json` — config + aggregate means/stds.
- `experiments/experiment11/cap_stats.json` — per-(dataset, config, seed) dict: `{n_eml_leaves, mean_cap, max_cap, median_cap, n_capped_predictions_on_test, pct_capped}`. The "n_capped_predictions_on_test" reflects how often the cap actually fires on the held-out test set.
- `experiments/experiment11/pmlb_rmse.png` — log-scale bar chart with error bars.
- `experiments/experiment11/report.md` — narrative.
- `experiments/experiment11/run.log` — full console output.

## Tests

Added to `tests/unit/test_eml_split_tree.py`:

1. **`test_cap_k_zero_preserves_baseline`** — fit two regressors, one with `leaf_eml_cap_k=0.0`, one with the param omitted; assert `np.allclose` on predictions. Backward-compat pin.

2. **`test_cap_bounds_predictions_on_heavy_tails`** — synthetic `X ~ 1e6 · N(0,1)`, targets in `[0, 99]`, `leaf_eml_cap_k=5.0`, full boost. Assert `max|pred| ≤ 5 · max|y_train|` (any prediction that would have exploded is capped). Directly targets the Exp 9/10 failure mode.

3. **`test_cap_adapts_across_boosting_rounds`** — fit a full boost on `y = exp(x_0)` signal (clean elementary signal, not pathological). Walk the ensemble; verify that the `cap` field values are monotonically non-increasing across rounds (residuals shrink → caps tighten).

4. **`test_eml_leaf_node_default_cap_is_inf`** — construct an `EmlLeafNode` without specifying `cap`; verify `cap == float("inf")`. Protects the backward-compat dataclass default.

## Success criteria

Cap is a keeper if **any** `leaf_eml_cap_k > 0` value satisfies all of:

1. **Stability:** no RMSE > 10× XGBoost RMSE on any dataset × seed. This is the stronger version of Experiment 9/10's S-A that neither ridge nor blend achieved.
2. **cpu_small recovery:** mean ratio for `562_cpu_small` under the best cap config is **below 1.2** (within 20% of XGBoost). Experiment 8's single-seed 0.90 was probably lucky; 1.2 is a realistic target given the multi-seed instability we've seen.
3. **No regression on wins:** datasets where Experiment 8/9 had mean ratio < 1.0 (`192_vineyard`, `523_analcatdata_neavote`, `529_pollen`, `564_fried`) stay within 0.05 of their Experiment 8 ratios.

Cap is a **partial success** if criteria 1 and 2 are met but some winner loses up to 0.10 — acceptable trade-off if it's the cost of stability.

Cap is a **negative outcome** if no `k > 0` value satisfies criterion 1. In that case the root cause is deeper than per-sample magnitude and escalates to a broader investigation (possibly tightening `_EXP_CLAMP` in the kernel after all, or dropping depth-2 EML leaves entirely on heavy-tailed datasets).

## Action on verdict

- **Success:** set `leaf_eml_cap_k` default to the best-grid value and update report language accordingly. Keep the code path so users can still disable it.
- **Partial success:** keep default at 0.0, document the winning value, offer it as a tunable for datasets with heavy-tailed features.
- **Negative outcome:** keep code + parameter; scope `_EXP_CLAMP` tightening as Experiment 12.

## Files changed

**Modified:**
- `eml_boost/tree_split/nodes.py` — add `cap: float = float("inf")` field on `EmlLeafNode`.
- `eml_boost/tree_split/tree.py` — add `leaf_eml_cap_k` param; compute `cap_leaf` in `_fit_leaf`; clip `val_pred` in both `_select_leaf_gated` and `_select_leaf_blended` pre-SSE; store `cap` on the emitted `EmlLeafNode`; apply `np.clip` in `_predict_vec`.
- `eml_boost/tree_split/ensemble.py` — thread `leaf_eml_cap_k` through.
- `tests/unit/test_eml_split_tree.py` — add 4 tests.

**Created:**
- `experiments/run_experiment11_leaf_cap.py` — 5-config cap grid runner.
- `experiments/experiment11/` directory (populated by the runner).

**Unchanged:**
- `eml_boost/_triton_exhaustive.py` — `_EXP_CLAMP` and `_LOG_EPS` stay at their current values. The cap acts on the final prediction, not the kernel's internals.
- `eml_boost/tree_split/_gpu_split.py`, the blend math, the ridge math — all preserved.

## Risks

- **Cap too tight on first round:** round 1 residuals are `y - ȳ ∈ [-max|y|, +max|y|]` roughly, so `max|y_full|` in a leaf is close to `max|y|` overall. With `cap_k=1`, a leaf's cap would be `max|y|` which bounds predictions at the target range — this is intentional but might hurt fit quality. That's what the grid `{1, 2, 5, 10}` is for — we expect `k=5` or `k=10` to be the sweet spot.
- **`cap_k > 0` on small-n datasets:** where EML leaves never activate (vineyard, cloud, neavote), the cap is a no-op. No regression expected on those; verify in the summary table.
- **Fit-time SSE uses capped val-pred, but OLS uses uncapped preds_fit.** This means a tree with extreme OLS η whose capped output happens to fit val well could be picked, leading to a store-time η that looks huge on paper. Predict-time clipping saves the prediction, but the stored (η, β) are misleading. This is a cosmetic concern, not a correctness one — the predictions are what matter.
- **Cap interaction with boosting drift.** If all 200 trees in the ensemble cap-clip frequently, residuals can't shrink as expected. This would show up as slow convergence / early stopping firing late with high train MSE. The `n_capped_predictions_on_test` stat in `cap_stats.json` tracks this.
