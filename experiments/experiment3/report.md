# Experiment 3: Calibration Curve with the DT-Improvement Metric

**Date:** 2026-04-23 → 2026-04-24
**Commit:** post-Experiment-2 (exhaustive snap in `fit_eml_tree`, `formula_std` on `EmlWeakLearner`)
**Runtime:** 53 min (v1) + 53 min (v2) ≈ 106 min total
**Scripts:** `experiments/calibration.py`, `experiments/run_calibration_benchmark.py`

## What the experiment was about

Experiment 1 ran the graceful-degradation sweep on seven fractions of "elementary vs. categorical" mixed signal and found a flat-to-inverted EML-win-rate curve. Experiment 2 proved that the algorithm *was* recovering closed-form formulas after the exhaustive-search fix, but the per-round win-rate metric was still flat because EML captures the dominant elementary signal in one aggressive round and then DT wins residual cleanup by BIC's complexity penalty — exactly the behavior we'd want, but not what "EML wins more rounds when data is elementary" can see.

Experiment 3 reruns the sweep with a metric that actually tracks what we care about: **how much does the hybrid cut test MSE relative to a same-capacity baseline of the same family structure?** If EML is pulling its weight, that quantity should rise monotonically with the fraction of elementary signal.

The experiment ran in two phases (v1 and v2) separated by an interim result that almost misled us.

## Configuration

- 5 fractions: `{0.00, 0.25, 0.50, 0.75, 1.00}`
- 2 datasets per fraction (reproducible via seed)
- 200 samples per dataset, split 70/30 train/test
- `max_rounds = 15`, `depth_eml = 2`, `depth_dt = 2`, `n_restarts = 6`, `k = min(3, n_features)`, patience disabled
- Hybrid = `EmlBoostRegressor` from the production codebase.
- DT-only baseline = `lightgbm.train` with `max_rounds = 15` stumps of `max_depth = 2`, `num_leaves = 4`, `learning_rate = 0.1` — capacity-matched to the hybrid's DT branch.
- XGBoost baseline — two separate runs:
  - **Strong XGBoost** (interim, dropped): `n_estimators=100`, `max_depth=6`, `learning_rate=0.1` (library defaults).
  - **Capacity-matched XGBoost** (final, v2): `n_estimators=15`, `max_depth=2`, `learning_rate=0.1` — same structural capacity as the hybrid's DT branch.

Both models train on the same train split and are evaluated on the held-out test split.

## What it was supposed to prove

Three concrete predictions:

1. **Monotonic** rise in `dt_improvement = 1 − hybrid_test_mse / dt_only_test_mse` from frac=0 to frac=1.
2. At `frac=1.0`, hybrid should cut MSE by ≥ 50 % vs a capacity-matched baseline — the regime where EML's exhaustive recovery has the most to contribute.
3. The metrics from earlier experiments (EML-win rate, formula coverage) should remain **flat** across fractions — confirming that they are saturated by round 0 regardless of signal character, and that `dt_improvement` is the right regime-sensitive metric.

## Results

### v1 — DT-only baseline only

**Artifacts:** `calibration_curve_v1_dt_only.{csv,json,png}`.

| `frac_elementary` | DT-improvement | hybrid MSE | DT-only MSE | formula coverage | EML round-win |
|---|---|---|---|---|---|
| 0.00 | 0.338 | 0.889 | 1.344 | 0.874 | 0.100 |
| 0.25 | 0.518 | 0.407 | 0.844 | 0.892 | 0.133 |
| 0.50 | 0.507 | 0.237 | 0.481 | 0.932 | 0.133 |
| 0.75 | 0.554 | 0.080 | 0.180 | 0.882 | 0.167 |
| 1.00 | **0.648** | 0.019 | 0.054 | 0.984 | 0.067 |

**DT-improvement monotonic (within 0.1 slack): PASS.** The curve rises from 0.34 at pure-categorical to 0.65 at pure-elementary with a small plateau between 0.25 → 0.50 (0.518 → 0.507) well inside slack. Predictions 1, 2, and 3 all confirmed against the capacity-matched DT baseline.

### Interim — strong XGBoost (depth 6, 100 rounds, lr 0.1)

We added XGBoost at its defaults to sanity-check that v1's "65% MSE reduction" survived exposure to an industrial boosting stack:

| `frac_elementary` | hybrid MSE | DT-only MSE | **strong XGBoost MSE** | XGB-improvement |
|---|---|---|---|---|
| 0.00 | 0.889 | 1.344 | **0.062** | −13.38 |
| 0.25 | 0.407 | 0.844 | **0.032** | −11.72 |
| 0.50 | 0.237 | 0.481 | **0.066** |  −2.58 |
| 0.75 | 0.080 | 0.180 | **0.057** |  −0.41 |
| 1.00 | 0.019 | 0.054 | **0.005** |  −3.11 |

Strong XGBoost beat the hybrid by 3–30× in MSE across every fraction — not even close. The initial read was "the whole pitch is broken." On reflection, this was a **capacity mismatch**: XGBoost had 100 trees × 64 leaves ≈ 6400 leaf-values vs the hybrid's 15 rounds × ≤ 4 leaves per DT stump. Not an architecture comparison; a capacity comparison. Artifact files were not preserved for this interim run — numbers above are the raw record.

### v2 — capacity-matched XGBoost

**Artifacts:** `calibration_curve_v2_capacity_matched.{csv,json,png}`.

XGBoost config dropped to `n_estimators=15, max_depth=2, learning_rate=0.1` — exactly the hybrid's DT branch configuration.

| `frac_elementary` | DT-improvement | XGB-improvement | hybrid MSE | DT-only MSE | XGBoost MSE | coverage | win |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.338 | +0.345 | 0.889 | 1.344 | 1.357 | 0.874 | 0.100 |
| 0.25 | +0.518 | +0.533 | 0.407 | 0.844 | 0.871 | 0.892 | 0.133 |
| 0.50 | +0.507 | +0.518 | 0.237 | 0.481 | 0.492 | 0.932 | 0.133 |
| 0.75 | +0.554 | +0.560 | 0.080 | 0.180 | 0.182 | 0.882 | 0.167 |
| 1.00 | **+0.648** | **+0.670** | 0.019 | 0.054 | 0.058 | 0.984 | 0.067 |

Both improvement curves pass the monotonic-within-0.1-slack check. The two curves lie almost on top of each other — at matched capacity, LightGBM and XGBoost are practically indistinguishable (both are depth-2 threshold-tree boosters with the same learning rate and round count). The hybrid beats both by 34 % at pure-categorical and 67 % at pure-elementary.

The strong-XGBoost result from the interim run is thus reframed: **not "XGBoost is fundamentally better," but "additional tree capacity approximates smooth functions piecewise and closes the gap."** At equal capacity, the architectural advantage of EML's exhaustive closed-form search is real.

## What v2 actually shows

- **Prediction 1 confirmed (v2).** `dt_improvement` and the analogous `xgb_improvement` both rise monotonically (within slack).
- **Prediction 2 confirmed.** At `frac=1.0`, the hybrid cuts MSE by 67 % vs XGBoost and 65 % vs LightGBM of identical structural capacity.
- **Prediction 3 confirmed.** Coverage stays in `[0.87, 0.98]` and EML round-win rate in `[0.07, 0.17]`. Both are flat; the headline DT-/XGB-improvement curves carry the regime signal.

Non-trivial observation: the hybrid still beats both matched baselines by ~34 % at `frac=0.0` (pure categorical). Likely explanation: EML's exhaustive tree search over 1,296 candidates at depth 2, `k=3` happens to produce smooth elementary approximations (e.g., `exp(x_cat) − log(x_cat)` on integer-valued `x_cat ∈ {0..4}`) that numerically bracket a 5-level lookup table better than 4-leaf stumps can. Worth a line in the paper.

## What Experiment 3 does NOT show

- It does **not** test real-world tabular benchmarks (PMLB, OpenML). All data is synthetic at `n = 200`.
- It does **not** prove the hybrid beats *production* XGBoost at industrial capacity. The interim data is unambiguous: depth-6 × 100-round XGBoost wins in-range interpolation 3–30×. The capacity-matched win is an architectural claim, not a drop-in-replacement claim.
- It does **not** characterize the hybrid beyond depth 2 with `k ≤ 3`. The exhaustive-search fix from Experiment 2 only scales there; larger dimensions still fall back to the softmax path whose limitations Experiment 2 documented.
- It does **not** quantify statistical noise. Two datasets per fraction was a budget-saving choice; real error bars need ≥ 5 datasets per point.
- It does **not** test extrapolation — precisely the regime where the closed-form advantage should separate from piecewise-constant approximation regardless of capacity. That's Experiment 5.

## Reproducing these results

```bash
# v2 — capacity-matched XGBoost comparison (current default in calibration.py)
uv run python experiments/run_calibration_benchmark.py
```

Expected runtime ~55 min on CPU. Output goes to `experiments/results/` by default; this folder is a snapshot.

To reproduce the interim strong-XGBoost result, temporarily revert the baseline config in `experiments/calibration.py`:

```python
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=6,          # was depth_dt
    n_estimators=100,     # was max_rounds
    learning_rate=0.1,
    verbosity=0,
    random_state=sub_seed,
)
```

## Consequence for the project

With Experiment 3 v2 in hand, the three promised outcomes of the original design (spec section 9.3) now map cleanly to measurable metrics:

| Spec 9.3 claim | Metric used | Experiment 3 result |
|---|---|---|
| "Must-have #1: Feynman SR recovery ≥ 80 %" | Feynman integration test (Task 15) | Deferred — 4/4 in-scope passes, II.11.28 scoped to benchmark mode |
| "Must-have #2: within 10 % of XGBoost" | XGBoost baseline — capacity-matched vs capacity-unlocked | **Capacity-matched PASS** (hybrid +67 % vs XGBoost at `frac=1`); **capacity-unlocked FAIL** (strong XGBoost wins in-range by 3–30×) |
| "Must-have #3: Graceful-degradation monotonic" | `dt_improvement` and `xgb_improvement` vs fraction | **PASS** (both monotonic, 0.34 → 0.65 and 0.35 → 0.67) |

Must-have #2 is partially answered by v2: at equal capacity, we beat XGBoost. The paper needs to state this boundary condition up front — the hybrid's pitch is not "outperforms XGBoost at any capacity" but "at equal architectural capacity, the closed-form machinery adds legible structure that threshold splits cannot match, AND extrapolates." Experiment 5 will confirm (or refute) the extrapolation part.

## Next experiments

- **Experiment 4 — extrapolation**: train on `x ∈ [−1, 1]`, test on `x ∈ [1, 2]`. The hybrid's closed-form parts (e.g., a recovered `exp(x_0)`) extrapolate correctly; every DT-based model saturates at boundary constants regardless of capacity. This is where the closed-form story uniquely wins.
- **Experiment 5 — strong-XGBoost plus depth-3 hybrid**: bump max_rounds to 100 and depth_dt to 6 on the hybrid side to match XGBoost default, and see whether closed-form weak learners still help when the DT branch has full tree capacity. Different question from Experiment 5; pairs well.
- **PMLB small-regression subset**: real tabular data, both matched- and unlocked-capacity baselines. Needed for the spec's must-have #2 in its strong form.
