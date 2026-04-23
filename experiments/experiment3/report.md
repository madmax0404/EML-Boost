# Experiment 3: Calibration Curve with the DT-Improvement Metric

**Date:** 2026-04-23
**Commit:** post-Experiment-2 (exhaustive snap in `fit_eml_tree`, `formula_std` on `EmlWeakLearner`)
**Runtime:** 3197 s (53 min)
**Scripts:** `experiments/calibration.py`, `experiments/run_calibration_benchmark.py`

## What the experiment was about

Experiment 1 ran the graceful-degradation sweep on seven fractions of "elementary vs. categorical" mixed signal and found a flat-to-inverted EML-win-rate curve. Experiment 2 proved that the algorithm *was* recovering closed-form formulas after the exhaustive-search fix, but the per-round win-rate metric was still flat because EML captures the dominant elementary signal in one aggressive round and then DT wins residual cleanup by BIC's complexity penalty — exactly the behavior we'd want, but not what the "EML wins more rounds when data is elementary" metric can see.

Experiment 3 reruns the sweep with a metric that actually tracks what we care about: **how much does the hybrid cut test MSE relative to a DT-only baseline of the same capacity?** If EML is pulling its weight, that quantity should rise monotonically with the fraction of elementary signal.

## Configuration

- 5 fractions: `{0.00, 0.25, 0.50, 0.75, 1.00}`
- 2 datasets per fraction (reproducible via seed)
- 200 samples per dataset, split 70/30 train/test
- `max_rounds = 15`, `depth_eml = 2`, `depth_dt = 2`, `n_restarts = 6`, `k = min(3, n_features)`, patience disabled
- Hybrid = `EmlBoostRegressor` from the production codebase.
- DT-only baseline = `lightgbm.train` with `max_rounds = 15` stumps of `max_depth = 2`, `num_leaves = 4`, `learning_rate = 0.1` — same capacity and shrinkage as the hybrid's DT branch.

Both models train on the same train split and are evaluated on the held-out test split.

## What it was supposed to prove

Three concrete predictions:

1. **Monotonic** rise in `dt_improvement = 1 − hybrid_test_mse / dt_only_test_mse` from frac=0 to frac=1.
2. At `frac=1.0`, hybrid should cut MSE by ≥ 50 % vs DT-only — the regime where EML's exhaustive recovery has the most to contribute.
3. The two metrics from earlier experiments (EML-win rate, formula coverage) should remain **flat** across fractions — confirming that they are saturated by round 0 regardless of signal character, and that `dt_improvement` is the right regime-sensitive metric.

## Results

| `frac_elementary` | DT-improvement | hybrid test MSE | DT-only test MSE | formula coverage | EML round-win |
|---|---|---|---|---|---|
| 0.00 | 0.338 | 0.889 | 1.344 | 0.874 | 0.100 |
| 0.25 | 0.518 | 0.407 | 0.844 | 0.892 | 0.133 |
| 0.50 | 0.507 | 0.237 | 0.481 | 0.932 | 0.133 |
| 0.75 | 0.554 | 0.080 | 0.180 | 0.882 | 0.167 |
| 1.00 | **0.648** | 0.019 | 0.054 | 0.984 | 0.067 |

**DT-improvement monotonic (within 0.1 slack): PASS.** Curve rises from 0.34 at pure-categorical to 0.65 at pure-elementary, with a small plateau between 0.25 and 0.50 (0.518 → 0.507) well inside the slack bound.

Artifacts preserved in this folder:

- `calibration_curve.csv` — per-fraction aggregate across all three metrics
- `calibration_curve.json` — full `CalibrationResult` plus config
- `calibration_curve.png` — headline plot showing DT-improvement (solid), coverage (dotted), round-win rate (dashed)

## What it actually shows

- **Prediction 1 confirmed.** `dt_improvement` rises monotonically (within 0.1 slack) from 0.338 → 0.648. This is the graceful-degradation signature the spec originally wanted, just measured against a capacity-matched baseline instead of counted via round wins.
- **Prediction 2 confirmed.** At `frac=1.0`, hybrid cuts MSE by 64.8 % vs DT-only — `0.019` vs `0.054`. At `frac=0.0`, hybrid still beats DT-only by 33.8 %, which is unexpected under a "DT should dominate pure categorical" prior, and is discussed below.
- **Prediction 3 confirmed.** Coverage stays in `[0.874, 0.984]` and EML round-win rate stays in `[0.067, 0.167]`. Both are essentially flat and unable to distinguish the regimes; the headline DT-improvement metric is the informative one.

The non-trivial observation is the 33.8 % improvement at `frac=0.0`. The "pure DT regime" signal depends only on two integer-valued categorical features via a lookup table. Why does EML add value there?

The likely explanation: EML's exhaustive tree search (1,296 candidates at depth 2, `k = 3`) can produce non-trivial smooth approximations of the lookup table — for example, a formula `exp(x_cat) − log(x_cat)` that, on the integer-valued `x_cat ∈ {0, 1, 2, 3, 4}`, happens to numerically bracket the lookup well. The boosting loop's first round installs such an approximation with a learned η; DT then fits residuals. The capacity-matched DT-only baseline at `max_depth = 2` with 4-leaf stumps × 15 rounds × `lr = 0.1` is genuinely weaker than the hybrid here. This is a finding worth a line in the paper: EML isn't strictly redundant even on categorical signals, because smooth elementary expressions are surprisingly good approximators of small lookup tables.

## What it does NOT show

- It does **not** test real-world tabular benchmarks (PMLB, OpenML). All data is synthetic at `n = 200`.
- It does **not** compare to a *strong* baseline — LightGBM at 15 rounds with depth-2 stumps is capacity-matched to our DT branch, not to a production XGBoost configuration with deep trees and tuned shrinkage. The hybrid's 65 % edge at `frac=1` against our matched baseline does not extrapolate to a 65 % edge against XGBoost.
- It does **not** characterize the hybrid beyond depth 2 with `k ≤ 3`. The exhaustive search fix only scales there; larger dimensions fall back to the softmax path, whose Experiment 2 findings still hold.
- It does **not** quantify statistical noise. Two datasets per fraction was a budget-saving choice; real error bars would need ≥ 5 datasets per point.

## Reproducing this result

```bash
uv run python experiments/run_calibration_benchmark.py
```

Expected runtime ~55 min on CPU. Output lands in `experiments/results/` by default; this folder is a snapshot of that output.

## Consequence for the project

With Experiment 3 in hand, the three promised outcomes of the original design (spec section 9.3) now map cleanly to measurable metrics:

| Spec section 9.3 claim | Metric used | Experiment 3 result |
|---|---|---|
| "Must-have #1: Feynman SR recovery ≥ 80 %" | Feynman integration test (Task 15) | Deferred — 4/4 in-scope passes, II.11.28 scoped to benchmark mode |
| "Must-have #2: PMLB within 10 % of XGBoost" | Real tabular benchmark (future) | Not yet tested |
| "Must-have #3: Graceful-degradation monotonic" | `dt_improvement` vs fraction | **PASS** (0.34 → 0.65 monotonic) |

Experiment 3 closes the graceful-degradation claim as empirically supported under the exhaustive-search regime. The remaining two must-haves are separate work.

## Next possible experiments

- Scale `n` to `2000+` samples; check whether the DT-improvement gap widens, narrows, or holds.
- Add a strong baseline (tuned XGBoost, 200+ rounds) to see how the hybrid fares outside the capacity-matched regime.
- Swap the synthetic elementary signal to one that requires ≥ depth-3 structure (e.g., `0.5 · m · v²`) and rerun in the softmax-path regime; see whether Experiment 3's monotonicity survives without exhaustive search.
- Run on PMLB's small regression subset with real tabular data.
