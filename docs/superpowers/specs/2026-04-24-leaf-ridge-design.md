# Ridge-Regularized EML Leaves (Experiment 10)

**Date:** 2026-04-24
**Context:** Follows Experiment 9. That experiment revealed catastrophic numerical explosions on `562_cpu_small` (all 3 seeds under blend-on, 1/3 under blend-off) and `564_fried` (2/3 under blend-on). Root cause identified by reading XGBoost v3.2.0 source: our EML-leaf OLS is **unregularized**, whereas XGBoost's leaf weight formula `w* = −G/(H+λ)` has a default `reg_lambda=1.0` that bounds `|w*| ≤ max|residual|`. Our closed-form `η = Sxy / Sxx` has no such guarantee, and on heavy-tailed features `Sxx` can be small, driving η into the thousands.

## Goal

Add a ridge term to the OLS fit inside both `_select_leaf_gated` and `_select_leaf_blended` so the slope estimate `η` is shrunk toward zero by a configurable amount. Run a 6-config grid over ridge strengths on the Experiment 9 datasets and 3 seeds to determine the sweet spot.

## Non-goals

- L1 regularization on η (XGBoost's `reg_alpha`). Ridge first; L1 only if ridge alone is insufficient.
- `max_delta_step`-equivalent hard cap on `|η|`. Same reasoning — if soft shrinkage works, no need for a hard cap.
- Penalizing β. Standard ridge practice: don't regularize the intercept. Regularizing β would bias leaves toward 0 output even when `ȳ` is already perfectly calibrated.
- New leaf-node type. Existing `EmlLeafNode` stores the folded `(η, β)` unchanged.
- Changes to the internal EML-split selection. Ridge applies only to leaf OLS fits, not to the internal-node candidate evaluation.

## Design

### Closed-form modification (per candidate tree, per leaf)

Current (unregularized) OLS on `y ≈ η·p + β` over the fit portion:

```
Sxx = Σp² - n·p̄²       (equivalently det/n where det = n·Σp² − (Σp)²)
Sxy = Σpy - n·p̄·ȳ      (equivalently num_η/n where num_η = n·Σpy − Σp·Σy)
η = Sxy / Sxx
β = ȳ - η·p̄
```

Ridge-regularized (centered ridge on η only):

```
η_ridge = Sxy / (Sxx + λ)
β_ridge = ȳ - η_ridge · p̄
```

Equivalently in the existing code's variables:

```
det_ridge = det + n_fit · λ
η = (n_fit · Σpy − Σp · Σy) / det_ridge
β = (Σy − η · Σp) / n_fit
```

**Why the `det + n_fit · λ` form works:** We're adding `λ·η²` to the objective. The normal-equation diagonal entry for η is `Sxx = (Σp² − n·p̄²) = (Σp² · n − (Σp)²) / n = det / n`. Adding ridge to that entry means `Sxx → Sxx + λ`, or equivalently `det/n → (det + n·λ)/n`. Multiplying through by n gives `det → det + n·λ`.

**β-formula caveat:** The current code computes `β = (Σp²·Σy − Σp·Σpy) / det`, which is the 2-unknown OLS normal-equation solution for β that happens to equal `ȳ − η·p̄` when there's no ridge. Under ridge on η only, this identity still holds — but only if we use the ridge-adjusted η. The cleanest code is:

```python
eta = (n_fit * sum_py_f - sum_p * sum_y_f) / (det_safe + n_fit * lam)
bias = (sum_y_f - eta * sum_p) / n_fit
```

### `leaf_eml_ridge` parameter

Added to both class constructors:

```python
class EmlSplitTreeRegressor:
    def __init__(
        self,
        ...,
        leaf_eml_ridge: float = 0.0,   # 0.0 = current behavior (backward-compat)
        ...,
    ): ...

class EmlSplitBoostRegressor:
    def __init__(
        self,
        ...,
        leaf_eml_ridge: float = 0.0,
        ...,
    ): ...  # threaded through to each round's tree
```

Default `0.0` preserves the Experiment 9 behavior bit-for-bit — the runner can compare against Exp 9 results without rerunning the baseline.

### Where the parameter is consumed

Both `_select_leaf_gated` and `_select_leaf_blended` receive `ctx` from `_fit_leaf` that already includes `eta, bias`. The cleanest change is to move the OLS computation into `_fit_leaf`'s shared setup (it's already there) and have it incorporate `self.leaf_eml_ridge`:

```python
# In _fit_leaf, replace the existing OLS block:
lam = float(self.leaf_eml_ridge)
det_ridged = torch.where(det.abs() > 1e-6, det, torch.ones_like(det)) + float(X_fit.shape[0]) * lam
eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_ridged
bias = (sum_y_f - eta * sum_p) / n_fit
```

(`det_safe` logic folds into `det_ridged`; with `lam > 0` on a nonzero `n_fit`, `det_ridged` is always positive.)

No change to `_select_leaf_gated`'s or `_select_leaf_blended`'s body — they consume the already-computed `eta, bias`.

### Blend path interaction

With ridge applied pre-selection, the blend's closed-form `α*` on the val portion operates on the ridge-shrunk `p_val = η_ridge · eml + β_ridge`. This is correct: the blend shrinks *further* toward the constant on top of ridge's shrinkage of η. No math changes in `_select_leaf_blended`.

### Instrumentation

No change to `_leaf_stats`. The records continue to capture `(n_leaf, alpha, leaf_type)`. For the experiment we also want to inspect the **magnitude distribution of η** across leaves under each ridge setting, so the runner aggregates `|η|` from the fitted leaves by walking each tree's `_root`:

```python
def _collect_eta_magnitudes(boost) -> list[float]:
    out = []
    def walk(node):
        if isinstance(node, EmlLeafNode):
            out.append(abs(node.eta))
        elif isinstance(node, InternalNode):
            walk(node.left); walk(node.right)
    for tree in boost._trees:
        walk(tree._root)
    return out
```

This lives in the Experiment 10 runner, not in the library.

## Experiment 10 plan

### Datasets

Same seven as Experiment 8/9: `192_vineyard`, `210_cloud`, `523_analcatdata_neavote`, `557_analcatdata_apnea1`, `529_pollen`, `562_cpu_small`, `564_fried`.

### Seeds

`[0, 1, 2]`, same as Experiment 9.

### Configurations (6 SplitBoost variants)

The ridge denominator is `det + n_fit · λ`. For a leaf with `Sxx = n_fit · Var(p)` and `Var(p) ≈ 1` on clamped features, the η shrinkage factor is `Sxx / (Sxx + n_fit · λ) = 1/(1+λ)`. So:

- `λ = 0.02` ≈ XGBoost-parity shrinkage (since `1/(1+0.02) ≈ 0.98`)
- `λ = 0.1` ≈ 10% shrinkage
- `λ = 1.0` ≈ 50% shrinkage (halves η)
- `λ = 10.0` ≈ 91% shrinkage (very aggressive)

The four-value ridge grid chosen spans three orders of magnitude so we can see the tuning curve:

| id | selection | `leaf_eml_ridge` | expected η shrinkage |
|---|---|---|---|
| G0 | gated | 0.0 | none (Exp 9 blend-off baseline) |
| G_weak | gated | 0.1 | ~10% |
| G_strong | gated | 1.0 | ~50% |
| G_vstrong | gated | 10.0 | ~91% |
| B0 | blend | 0.0 | none (Exp 9 blend-on reproduction) |
| B_strong | blend | 1.0 | ~50% under blend |

### Baselines per (dataset, seed)

XGBoost (reference only — unchanged) and LightGBM (reference only). Included so the report can compute the familiar "ratio vs XGBoost" numbers without cross-referencing Exp 9's summary.

### Total fits

7 datasets × 3 seeds × (6 SplitBoost configs + XGBoost + LightGBM) = **168 fits**. At the Exp-9 per-fit timing (~5-35 s for the big ones, ~0.1-1 s for the small ones), estimated runtime ~12-15 minutes on RTX 3090.

### Outputs

- `experiments/experiment10/summary.csv` — one row per (dataset, seed, config).
- `experiments/experiment10/summary.json` — same data with per-(dataset, config) aggregates (mean, std, rmses-per-seed).
- `experiments/experiment10/eta_stats.json` — per-(dataset, config, seed) dict: `{count, mean_abs_eta, max_abs_eta, p99_abs_eta}`.
- `experiments/experiment10/pmlb_rmse.png` — bar chart with error bars, ridge-strength on x within each dataset group.
- `experiments/experiment10/report.md` — narrative write-up.
- `experiments/experiment10/run.log` — full console output (force-added).

## Tests

Add to `tests/unit/test_eml_split_tree.py`:

1. **`test_ridge_zero_matches_no_ridge_behavior`** — fit two regressors with identical config except `leaf_eml_ridge=0.0` and the parameter omitted entirely; assert `np.allclose(m1.predict(X), m2.predict(X))`. Protects the backward-compat story.

2. **`test_ridge_shrinks_eta_monotonically`** — on `y = exp(x_0) + small_noise`, fit with `leaf_eml_ridge ∈ {0.0, 0.1, 1.0, 10.0}` and verify the max `|η|` across EML leaves decreases monotonically. Confirms the shrinkage direction.

3. **`test_ridge_prevents_blowup_on_heavy_tails`** — synthetic feature magnitudes ≈ 1e6, targets well-behaved, `leaf_eml_ridge=1.0`, full boost fit. Assert predictions are finite and the max `|η|` stays under 100 (loose — just not 10000+). Directly targets the Exp 9 failure mode.

Existing tests must still pass with the default `leaf_eml_ridge=0.0` (they're unchanged since they use the default).

## Success criteria

Ridge is a keeper if **any** `leaf_eml_ridge > 0` setting among `{0.1, 1.0, 10.0}` satisfies all of:

1. **Stability:** no single-seed explosion (RMSE within 10× of XGBoost RMSE) on any of the 7 datasets × 3 seeds under blend-off + ridge.
2. **No regression:** the 4 datasets Experiment 8/9 had comfortable wins on (vineyard, neavote, pollen, fried on stable seeds) stay within their prior mean ratios.
3. **cpu_small recovery:** mean ratio for `562_cpu_small` under some blend-off + ridge config is under 2.0 (loose — XGBoost is 2.9, so ratio < 2.0 is well-within-XGB-territory). The Exp 9 blend-off mean was 14,543 so even ratio 2.0 is a 7000× improvement.

Ridge is **helpful for blend-on** (secondary): if `B_strong` (blend + ridge=1.0) beats `B0` (blend, no ridge) by 3+ orders of magnitude on cpu_small AND fried, the blend concept is revived as a future research direction (though still not the default).

Ridge is a **negative outcome** if no positive ridge value in the grid brings any explosion-dataset within 10× of XGBoost. In that case, next steps escalate to `max_delta_step`-equivalent (hard cap) or dropping EML leaves on heavy-tailed features entirely.

## Action on verdict

- If ridge works: set the default to the best ridge value and document. Keep both paths (gated default, blend optional).
- If ridge partially works (helps some datasets, not all): set a conservative positive default (e.g., `0.5`), add a report section explaining the residual failure modes, and scope a follow-up Experiment 11 for hard capping.
- If ridge fails entirely: document the finding, keep the parameter in place (for future tuning), and scope hard-capping as the next experiment.

## Files changed

**Modified:**
- `eml_boost/tree_split/tree.py` — add `leaf_eml_ridge` param; modify the OLS computation in `_fit_leaf` to incorporate `det += n_fit · λ`; compute `bias` from `(Σy − η·Σp)/n_fit` post-ridge.
- `eml_boost/tree_split/ensemble.py` — thread `leaf_eml_ridge` through the constructor and per-round tree construction.
- `tests/unit/test_eml_split_tree.py` — add the 3 new tests.

**Created:**
- `experiments/run_experiment10_leaf_ridge.py` — runner.
- `experiments/experiment10/` directory (outputs populated by the runner).

**Unchanged:**
- `eml_boost/tree_split/nodes.py` — `EmlLeafNode` schema unchanged (we still store the folded `(η, β)`).
- `eml_boost/tree_split/_gpu_split.py`, `eml_boost/_triton_exhaustive.py` — kernel code untouched.
- The blend math in `_select_leaf_blended` — ridge pre-shrinks the OLS `η` that the blend then consumes; no change to the α closed-form.

## Risks

- **Ridge too weak:** if the grid (0, 0.1, 1.0, 10.0) isn't aggressive enough to stabilize cpu_small, we'll see `G_vstrong` still exploding. The report will recommend extending the grid upward (50, 100) or switching to a hard cap.
- **Ridge too strong:** ridge biases η toward 0, which on well-calibrated datasets (`529_pollen`, `564_fried` at stable seeds) could hurt test RMSE. `G_weak` at 0.1 covers the mild-shrinkage scenario; `G_vstrong` at 10.0 is intentionally past the reasonable range so we can see the degradation curve.
- **Val-split sample size on blend:** `Bn/2` uses the same 75/25 leaf-local split as Exp 9. If ridge stabilizes the η but the blend's α is still inflated by multiple-testing over 144 candidates, we'll still see some variance — but without the explosions.
