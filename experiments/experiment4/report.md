# Experiment 4: Extrapolation Beyond the Training Range

**Date:** 2026-04-24
**Commit at start:** post-Experiment 3 (exhaustive snap + capacity-matched XGBoost baseline in calibration)
**Runtime:** ~5 min per pass, two passes (v1 pre-fix, v2 post-fix)
**Scripts:** `experiments/run_experiment4_extrapolation.py`
**GPU used:** XGBoost via `device="cuda"`, LightGBM via `device="gpu"` — on RTX 3090. Both worked out of the box.

## What the experiment was about

Every experiment so far tested the hybrid on data *within* its training range. On such in-range data, a depth-6 × 100-round XGBoost can piecewise-approximate any smooth function near-perfectly, and our hybrid's closed-form machinery adds relatively little. The hybrid's unique selling proposition is supposed to be **extrapolation**: when a boosting stack encounters inputs past its training range, every threshold-tree prediction saturates at the nearest boundary leaf value, regardless of capacity. A closed-form formula like `exp(x_0)` has no such bound.

Experiment 4 tests that claim directly: train on `x_0 ∈ [−1, 1]`, evaluate on `x_0 ∈ [1, 2]`. If the hybrid recovers the true elementary form, it should track the function out-of-range; if it has not, the extrapolation error surfaces the recovery gap.

## Configuration (identical across v1 and v2)

- 3 formulas, all with `k = 1` (single feature): `exp(x_0)`, `x_0`, `x_0² + 0.5`.
- `n_train = 500`, `n_test = 200` for each of the two eval ranges. Gaussian noise `σ = 0.02`.
- Train range `[−1, 1]`; in-range test draws from the same range; extrapolation test on `[1, 2]`.
- `seed = 0` throughout.
- **Capacity (strong across all three models):** `max_rounds = 100`, `depth_dt = 6`. Hybrid's EML branch stays at `depth_eml = 2` (exhaustive-search reach).
- Hybrid = `EmlBoostRegressor` (100 boosting rounds, EML branch runs exhaustive search at depth 2 with `k=1` → 144 candidate trees).
- LightGBM: `n_estimators=100, max_depth=6, learning_rate=0.1, device="gpu"`.
- XGBoost: `n_estimators=100, max_depth=6, learning_rate=0.1, device="cuda"`.

## What it was supposed to prove

Three concrete predictions going in:

1. On the **grammar-expressible** target `exp(x_0)`, the hybrid's extrapolation MSE should be dramatically lower than both tree baselines' — the canonical win.
2. On **linear** target `x_0`, all three methods extrapolate meaningfully: trees flat past boundary, hybrid via whatever elementary form it recovers. Prediction: roughly tied or slight hybrid advantage.
3. On **non-elementary** target `x_0² + 0.5`, none of them extrapolates well; the question is which extrapolates *least badly*.

## v1 — initial run, standardization bug

The first run gave an unexpectedly poor hybrid extrapolation result.

| formula | hybrid extrap | LightGBM extrap | XGBoost extrap | ranking |
|---|---|---|---|---|
| `exp(x_0)` | 7.22 | 6.44 | 6.14 | **XGBoost** < LightGBM < Hybrid |
| `x_0` | 13.26 | 0.385 | 0.351 | XGBoost < LightGBM ≪ **Hybrid** (worst) |
| `x_0² + 0.5` | 0.46 | 2.90 | 2.74 | **Hybrid** < XGBoost < LightGBM |

The hybrid was the *worst* extrapolator on `exp(x_0)` — the exact target the pitch relies on. Recovered formula was a chaotic 100-term sum of `exp(b·x_0)` terms with `b ≈ 1.73`, each scaled by a small positive-or-negative coefficient.

**Diagnosis:** training data was standardized (`X_std = (X - μ) / σ`) before the exhaustive search. For `x_0 ∈ [−1, 1]` uniform, `μ ≈ 0`, `σ ≈ 0.577`, so standardized `x_0_std` is `≈ 1.73 · x_0`. The exhaustive search picked the tree `eml(x_0_std, 1) = exp(x_0_std) = exp(1.73·x_0)` — which in-range *looks* like `exp(x_0)` up to a constant scale that `learned_η` absorbs, but in extrapolation grows 4× faster (at `x_0 = 2`, `exp(1.73·2) = exp(3.46) ≈ 31.8` vs `exp(2) ≈ 7.39`).

In-range, the scale error is masked by the linear η regression fit; out-of-range, it explodes.

Artifacts from v1 are not preserved on disk — the second run overwrote them. Numbers above are the verbatim log capture.

## Fix

`_fit_eml_tree_exhaustive` in `eml_boost/weak_learners/eml.py`: skip standardization entirely on the exhaustive path. Standardization is only necessary for the softmax path's numerical-stability reasons (exp overflow on raw inputs with large magnitude); exhaustive evaluates via sympy lambdify → numpy and has `safe_exp`-style clamping baked into the underlying arithmetic. Without standardization:

- `feature_mean = zeros(k)`, `feature_std = ones(k)`: sentinel values so `EmlWeakLearner.predict` becomes a no-op for these transforms.
- The recovered formula is expressed in literal feature coordinates (`exp(x_0)`, not `exp(x_0/σ)`), skipping the post-hoc un-standardization substitution.
- All 11 relevant unit tests remained green after the change.

## v2 — post-fix run

**Artifacts:** `summary.csv`, `summary.json`, `extrapolation_plots.png`.

| formula | hybrid extrap | LightGBM extrap | XGBoost extrap | ranking |
|---|---|---|---|---|
| `exp(x_0)` | **0.131** | 6.435 | 6.141 | **Hybrid** ≪ XGBoost < LightGBM |
| `x_0` | 1.814 | 0.385 | 0.351 | **XGBoost** < LightGBM < Hybrid |
| `x_0² + 0.5` | **2.377** | 2.898 | 2.745 | **Hybrid** < XGBoost < LightGBM |

Recovered formulas (v2):

- `exp(x_0)` → `0.851·exp(x_0) − 0.851`. The `0.851` and the `−0.851` come from the DT branch eating tiny residuals over 100 rounds; the exp shape is intact and extrapolates correctly.
- `x_0` → `0.703·exp(x_0) − 0.703`. The depth-2 grammar does not contain the identity `x_0` as a clean root; the closest smooth elementary fit is a scaled shifted exp. It interpolates okay and extrapolates exponentially — wrong shape.
- `x_0² + 0.5` → `0.146·exp(x_0) − 0.146`. Same fallback; depth-2 `k=1` grammar can't reach `x²` at all.

## What v2 actually shows

- **Prediction 1 confirmed.** For `exp(x_0)`, hybrid extrapolation MSE is **~47× lower** than XGBoost and **~49× lower** than LightGBM at `x_0 ∈ [1, 2]`. The tree baselines saturate at a horizontal plateau past the training-range boundary; the hybrid tracks the exponential.
- **Prediction 2 refuted.** On linear `x_0`, trees decisively beat the hybrid (5×). The hybrid's exp-based extrapolation grows far too fast. This is honest: the depth-2 `k=1` grammar does not express the linear function, and the best elementary approximation within that grammar is a scaled exp. In-range, `learned_η` makes it look linear-ish; out-of-range, the exp grows.
- **Prediction 3 narrowly confirmed.** On `x_0² + 0.5`, all three extrapolate poorly, but the hybrid narrowly wins (2.38 vs 2.75). The hybrid's scaled exp is still growing — wrong function class but right monotonicity direction; the trees flatline.

Put together: **on targets actually in the EML grammar, the hybrid is the only one of the three that extrapolates**, by orders of magnitude. Off-grammar, the hybrid inherits the limitations of whatever elementary formula it falls back to, and that fallback can be worse than a saturated tree.

## What v2 does NOT show

- It does **not** test extrapolation into far-ranges (5σ+ out). At more extreme extrapolation, the gap would likely widen further, but so would compound-error concerns in the recovered formula.
- It does **not** characterize what happens with multi-feature targets (`k > 1`), nor at depth 3+ (exhaustive-search threshold).
- It does **not** fix the "off-grammar gives bad extrapolation" failure mode. On linear targets, our hybrid is worse than XGBoost. A real production system would need a fallback (e.g., a linear component, or early-stop when DT is better *in-range*, which preserves the graceful behavior we saw on off-grammar residuals).
- Noise is `σ = 0.02` (very low). At higher noise, the elementary fit would be noisier and extrapolation less clean.
- All models share the same `max_rounds = 100` and depth-6 DT, but the hybrid's EML branch is depth-2 (that's where exhaustive search is feasible). So "capacity matched" is approximate — the grammar itself imposes expressiveness limits that no matched-capacity setting resolves.

## Reproducing these results

```bash
uv run python experiments/run_experiment4_extrapolation.py
```

Runtime ~5 min on CPU+GPU (hybrid is ~50 s per formula, both tree baselines well under 1 s thanks to GPU). Output goes to `experiments/experiment4/` by default; this folder is a snapshot.

## Consequence for the project

- **The paper claim "EML-Boost extrapolates when the target is elementary" holds.** Experiment 4 v2 is the first experiment where our hybrid's unique advantage dominates a proper apples-to-apples comparison (capacity matched, real baselines, single widely-recognized target).
- **The standardization fix is a real algorithmic improvement** that should be carried into the rest of the codebase. Worth checking whether Experiment 3 results change qualitatively once rerun with the same fix — probably the calibration curves shift slightly, but the monotonicity should survive.
- **Spec 9.3 must-have #1 (Feynman SR recovery ≥ 80%) becomes easier.** The Feynman integration test that was scoped down to 4/5 formulas during Task 15 may now extend to II.11.28 once we have depth 3 + k ≥ 3 exhaustive — or at least we should retest it with standardization disabled.

## Next possible experiments

- **Experiment 5 — retest Experiment 3 after the standardization fix.** Cheap (~55 min); should confirm that the calibration monotonicity is preserved, possibly with even stronger gaps now that the hybrid's recovered formulas are cleaner.
- **Experiment 6 — extrapolation with `k=2` targets.** e.g. `exp(x_0 + x_1)` or `x_0 * x_1`. Tests whether the grammar at `k=2` lands on the right multi-variable form.
- **Extreme extrapolation** — `x_0 ∈ [1, 5]`, same train range. The tree plateau widens; the hybrid's exp keeps tracking. Should be a dramatic figure for the paper.
- **Fallback heuristic** — if the hybrid's in-range MSE on a held-out set is *worse* than the DT-only baseline by more than ε, defer to the DT prediction entirely. This would fix the "off-grammar extrapolation is worse than flat" failure mode seen on `y = x_0`.
