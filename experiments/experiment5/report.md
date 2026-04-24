# Experiment 5: Calibration Rerun After the Standardization Fix

**Date:** 2026-04-24
**Commit:** post-Experiment 4 (exhaustive path no longer standardizes features)
**Runtime:** 3099 s (51 min)
**Scripts:** `experiments/calibration.py`, `experiments/run_calibration_benchmark.py` (unchanged from Experiment 3)

## What the experiment was about

Experiment 4 uncovered a latent bug: the exhaustive snap pipeline was z-scoring features before searching over depth-2 trees, which made the grammar recover `exp(x_0 / σ)` instead of `exp(x_0)`. In-range predictions masked the issue via `learned_η` rescaling; extrapolation exploded. The fix was to skip standardization entirely on the exhaustive path (softmax fallback still standardizes for overflow reasons).

Experiment 5 reruns Experiment 3's calibration sweep — identical config, identical data, identical baselines — to check whether the "graceful degradation: hybrid beats capacity-matched XGBoost by 35% → 67%" claim survived the fix.

Expectation going in: monotonicity preserved, possibly stronger gaps now that the hybrid's recovered formulas are algebraically correct.

## Configuration (unchanged from Experiment 3 v2)

- 5 fractions: `{0.00, 0.25, 0.50, 0.75, 1.00}`
- 2 datasets per fraction, 200 samples each, 70/30 train/test
- `max_rounds = 15`, `depth_eml = 2`, `depth_dt = 2`, `n_restarts = 6`, `k = min(3, n_features)`, patience disabled
- Hybrid = `EmlBoostRegressor` from production code (exhaustive path dispatched at depth 2, `k ≤ 4`)
- DT-only baseline = LightGBM at same capacity
- XGBoost baseline = XGBoost at same capacity (`max_depth=2, n_estimators=15, lr=0.1`)

## Results

**Artifacts:** `calibration_curve.{csv,json,png}`.

| frac | DT-impr (v5) | XGB-impr (v5) | hybrid MSE (v5) | DT-only (v5) | XGBoost (v5) | coverage (v5) | EML-win (v5) |
|---|---|---|---|---|---|---|---|
| 0.00 | **−0.199** | **−0.188** | 1.6113 | 1.3436 | 1.3566 | 0.000 | 0.000 |
| 0.25 | **−0.300** | **−0.258** | 1.0966 | 0.8437 | 0.8715 | 0.000 | 0.000 |
| 0.50 | **−0.429** | **−0.399** | 0.6882 | 0.4815 | 0.4919 | 0.031 | 0.067 |
| 0.75 | +0.260 | +0.270 | 0.1328 | 0.1796 | 0.1821 | 0.305 | 0.100 |
| 1.00 | **+0.859** | **+0.868** | 0.0077 | 0.0545 | 0.0581 | 0.978 | 0.067 |

**Monotonic check: FAIL.** The curve is V-shaped: decreasing from frac=0 to frac=0.5, then increasing through frac=1.

Side-by-side against Experiment 3 v2 (pre-fix) at the extremes:

| frac | DT-impr v3 (pre-fix) | DT-impr v5 (post-fix) | change |
|---|---|---|---|
| 0.00 | +0.338 | **−0.199** | sign flip |
| 1.00 | +0.648 | **+0.859** | +0.21 stronger win |

## What v5 actually shows

Two qualitative changes from the v3 curve:

1. **At frac=1.0, the hybrid's win is substantially stronger** (65% → 86% MSE reduction over capacity-matched XGBoost). The raw-coordinate exhaustive search recovers a clean `exp(x_0)` formula (not `exp(1.73 x_0)`); in-range MSE drops from 0.019 to 0.008.

2. **At low-fraction (categorical-dominated) signal, the hybrid now LOSES to both tree baselines** (by 20% at frac=0, by 30% at frac=0.25, by 43% at frac=0.5). This reverses Experiment 3's "hybrid wins even on pure categorical" finding.

The reversal is illuminating: Experiment 3's +34% at frac=0 was an **artifact of feature standardization**. When the DT regime's integer-valued categorical features `{0, 1, 2, 3, 4}` were z-scored to a compressed smooth range, the exhaustive search could fit smooth elementary formulas through those scaled categorical values and beat the DT baselines. With standardization removed, the raw integers make most candidate EML formulas either produce NaN (log of zero) or extreme values (exp of 4); coverage collapses to 0.000 because almost no EML candidate is even finite. Without an EML contribution, the hybrid falls back to pure DT rounds — but only 15 of them, which under-fits vs the capacity-matched XGBoost and loses.

So the honest post-fix story is:

- **The hybrid's architectural advantage is regime-specific to elementary signals.** When the signal is elementary, the hybrid extracts a correct closed-form expression and extrapolates / fits cleanly. When the signal is categorical lookup, the hybrid can't produce useful elementary approximations and loses.
- **The "graceful degradation curve" claim from Experiment 3 is replaced by a regime-conditional claim.** The curve is not monotonic; it's V-shaped. The hybrid's edge at frac=1 is much larger than before, but the downside at frac=0 is explicitly negative.

## What this means for the paper pitch

The original spec's "monotonic graceful degradation" story is not supported by the honest post-fix numbers. What IS supported:

- **On pure-elementary data, the hybrid beats same-capacity XGBoost by 86% on test MSE.** That's a dramatic headline number.
- **On pure-categorical data, the hybrid loses by ~20% and should not be used.** Honest boundary condition.
- **The crossover happens around `frac ≈ 0.65`** (linear interpolation between 0.5 and 0.75 data points). Below that, prefer XGBoost; above, prefer the hybrid.

This is a cleaner pitch than v2's "hybrid always wins": it's regime-specific, defensible, and consistent with the architectural argument. It also motivates a **fallback heuristic** (defer to DT-only when EML coverage < threshold) that would let the hybrid match tree performance in the unfavorable regime — a concrete and simple algorithmic extension.

## What v5 does NOT show

- Does not test real-world tabular data — still synthetic at `n = 200`.
- Does not compare against capacity-unlocked XGBoost (100 rounds × depth 6). At industrial capacity, XGBoost's in-range interpolation would again dominate at most fractions; the hybrid's unique win stays in the extrapolation regime tested by Experiment 4.
- Does not propose a regime-detection heuristic (e.g., "predict EML coverage before fitting and fall back to DT if low"). That's a followup.
- Does not quantify noise — two datasets per fraction remains the budget-saving setting.

## Reproducing this result

```bash
uv run python experiments/run_calibration_benchmark.py
```

Expected runtime ~55 min on CPU. Requires the exhaustive-path no-standardization fix to be in place (`_fit_eml_tree_exhaustive` with `feature_mean = zeros, feature_std = ones`).

## Consequence for the project

- **Update Experiment 3's report** to note that its monotonic-PASS claim was on the pre-fix codebase and is superseded by this V-shaped curve post-fix. The 65% win at frac=1 is now 86%.
- **Add the regime-conditional claim to the paper draft** alongside Experiment 4's extrapolation result. The hybrid's defensible territory is "elementary-signal regression with correct extrapolation"; the honest caveat is "avoid on pure categorical."
- **Followup: implement the fallback heuristic.** If `model.coverage(X_val) < τ` during validation, skip the EML branch in subsequent rounds or defer predictions to the DT-only path. Would flatten the V-dip and restore monotonicity without hiding it.
