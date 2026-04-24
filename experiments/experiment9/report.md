# Experiment 9: Stacked-Blend Leaves

**Date:** 2026-04-24
**Commit:** `609f3c2` (post-Task-5 runner, pre-revert)
**Runtime:** ~6 min on RTX 3090 (84 fits: 7 datasets × 3 seeds × 4 models)
**Scripts:** `experiments/run_experiment9_stacked_blend.py`

## What the experiment was about

Experiment 8 shipped **EML leaves** (Phase 4) gated by a 5% fractional val-SSE improvement threshold. That got to 5/7 outright wins against XGBoost at matched capacity on a single seed, with `562_cpu_small` regressing from ratio 0.81 → 0.90 relative to the constant-leaf baseline. The design hypothesis for Experiment 9 was that replacing the **binary accept/reject gate** with a **val-fit convex blend** `α·ȳ + (1−α)·(η·eml(x)+β)` would:

1. recover the cpu_small regression (success criterion S-B),
2. hold 5/7 within 10% of XGBoost across 3 seeds (S-A),
3. reduce seed-to-seed variance (S-C).

The runner also tests both configurations (`use_stacked_blend=False` and `=True`) at the same commit so we can A/B cleanly.

## Configuration

All matching Experiment 8 except 3 seeds instead of 1.

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
leaf_eml_gain_threshold = 0.05   # blend-off only
n_bins              = 256
histogram_min_n     = 500
test_size           = 0.20
seeds               = [0, 1, 2]
```

## Results (mean ± std over 3 seeds)

| dataset | n | k | blend-off RMSE | blend-on RMSE | XGBoost RMSE | off/xgb | on/xgb | Δ ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 192_vineyard | 52 | 2 | 3.18 ± 1.18 | 3.18 ± 1.18 | 3.31 | **0.96** | **0.96** | 0.000 | identical (both early-out) |
| 210_cloud | 108 | 5 | 0.56 ± 0.17 | 0.56 ± 0.17 | 0.468 | 1.19 | 1.19 | 0.000 | identical (both early-out) |
| 523_analcatdata_neavote | 100 | 2 | 0.87 ± 0.17 | 0.88 ± 0.17 | 1.71 | **0.51** | **0.51** | +0.005 | identical (both early-out) |
| 557_analcatdata_apnea1 | 475 | 3 | 1289 ± 377 | 1276 ± 378 | 1124 | 1.15 | 1.14 | −0.011 | **blend mildly better** |
| 529_pollen | 3848 | 4 | 2.41 ± 1.48 | 1.61 ± 0.08 | 1.58 | 1.53 | **1.02** | **−0.503** | **blend dramatically stabilizes** |
| 562_cpu_small | 8192 | 12 | 42,499 ± 73,606 | 61,937 ± 79,073 | 2.92 | 14,544 | 21,196 | **+6,652** | **both catastrophic; blend worse** |
| 564_fried | 40,768 | 10 | 1.07 ± 0.006 | 4,440 ± 7,673 | 1.07 | **1.00** | 4,146 | **+4,145** | **blend catastrophically unstable** |

Mean-ratio column numbers with explosions are dominated by a single-seed blow-up per dataset; per-seed detail in `summary.csv`.

### Per-seed picture (the detail the means hide)

| | seed 0 RMSE | seed 1 RMSE | seed 2 RMSE |
|---|---|---|---|
| **562_cpu_small blend-off** | 2.69 | **127,492** (explode) | 2.57 |
| **562_cpu_small blend-on** | **151,214** (explode) | **33,875** (explode) | **722** (explode) |
| **564_fried blend-off** | 1.076 | 1.065 | 1.072 |
| **564_fried blend-on** | **19.23** (explode) | **13,300** (explode) | 1.072 |
| **529_pollen blend-off** | 1.57 | **4.12** (2.6× baseline) | 1.53 |
| **529_pollen blend-on** | 1.61 | 1.54 | 1.69 |

- `562_cpu_small`: blend-on explodes on **all 3 seeds**; blend-off explodes on 1/3.
- `564_fried`: blend-on explodes on 2/3; blend-off is perfectly stable.
- `529_pollen`: opposite direction — blend-on is stable (0.075 std); blend-off has a bad seed-1 run (std 1.48).

## Success criteria verdict

- **S-A (primary):** 5/7 within 10% under blend-on **across all 3 seeds**. **NOT MET.** `562_cpu_small` and `564_fried` are orders of magnitude worse under blend-on on multiple seeds.
- **S-B:** `562_cpu_small` mean ratio < 0.85 under blend-on. **NOT MET.** Mean ratio is 21,196.
- **S-C:** mean σ of ratios lower under blend-on than blend-off. **NOT MET.** Averaged across the 7 datasets the blend has higher std (driven by the 7,673-std on `564_fried`, which is stable under blend-off at σ=0.006).

**Negative-outcome check:** "if blend-on has a ≥ 0.03 higher mean ratio than blend-off on 2 or more datasets, declare the blend unhelpful." Two datasets (`562_cpu_small` +6,652, `564_fried` +4,145) blow past the 0.03 threshold by ~5 orders of magnitude. **The negative-outcome criterion is emphatically met.**

**Verdict: BLEND HURTS. Reverting `use_stacked_blend` default to `False`.**

## Leaf-level behavior (blend-on)

Blend-on `_leaf_stats` aggregates (`experiments/experiment9/leaf_stats.json`):

| dataset | total leaf records (3 seeds) | EML-leaf fraction | α mean |
|---|---|---|---|
| 192_vineyard | 0 | — | — |
| 210_cloud | 0 | — | — |
| 523_analcatdata_neavote | 0 | — | — |
| 557_analcatdata_apnea1 | 133 | 73% | 0.44 |
| 529_pollen | 1,629 | 77% | 0.30 |
| 562_cpu_small | 3,848 | 70% | 0.27 |
| 564_fried | 11,363 | 83% | 0.23 |

Observations:
- **Small-n datasets (vineyard, cloud, neavote) never reach the EML-leaf decision.** With `min_samples_leaf_eml=50` and train-set sizes of 41, 86, and 80, no leaf has enough samples. That's why blend-on and blend-off produce identical predictions on those datasets: both hit the `too_small` early-out.
- **On large datasets the blend is nowhere near collapsing.** Mean α of 0.23–0.44 means the EML contribution keeps 56–77% weight. The α *is* shrinking — but not enough to tame the η magnitudes on heavy-tailed features.
- **The EML-leaf *fraction* is high (72–83%)** — the blend accepts most candidate trees because it has no minimum-gain floor to reject marginal ones.

## Why the blend explodes on heavy-tailed features

On `562_cpu_small` (feature magnitudes into the millions pre-standardization) and `564_fried` (10 features, some heavy-tailed), the EML-leaf's post-fit `(η, β)` can land at extreme magnitudes when the per-leaf OLS is fit on ~50 samples with a large-variance response. Under Experiment 8's **5% gate**, most such overfits are rejected: the train/val improvement threshold is harder to cross on pathological leaves. Under the blend, α shrinks η but does not reject it — a 0.3 mean α still leaves 70% of a potentially-extreme η in the prediction. When a boosting round compounds this across 200 trees, test-set predictions spiral into 1e5+ territory.

`529_pollen` is the one case where the blend's continuous shrinkage *helps*: blend-off on that dataset has a bad-seed run (RMSE 4.12) that the blend smooths out (1.54). So the blend isn't universally worse — it's unsafe on heavy-tailed features specifically.

## What's left as a loss

Even under **blend-off**, `562_cpu_small` has 1-of-3 seed failure. Experiment 8's single-seed 0.90 ratio was partly luck — the multi-seed picture shows the EML-leaf mechanism is fragile on heavy-tailed features regardless of gate-vs-blend. That's a finding that outlives this experiment: the Phase-4 EML leaves have a stability problem the original single-seed benchmark didn't expose.

## Action taken

Per the spec's negative-outcome protocol:
1. **Revert the `use_stacked_blend` default to `False`** in both `EmlSplitTreeRegressor.__init__` and `EmlSplitBoostRegressor.__init__`.
2. **Keep the blended code path and its tests intact.** The blend remains available for future experiments via `use_stacked_blend=True`. `_leaf_stats` instrumentation stays in place.
3. **Document the finding in this report** so the next iteration knows the failure mode.

The revert commit follows this one.

## What Experiment 9 does NOT resolve

- **The underlying EML-leaf stability problem on heavy-tailed features.** Both blend-off and blend-on hit this on `562_cpu_small`. A proper fix probably requires one or more of: (a) a magnitude ceiling on `|η|` at fit time, (b) a predict-time Huber-style cap, (c) tighter feature standardization, (d) dropping EML leaves entirely on features with `max(|x|)/std(x)` above a threshold.
- **Whether the blend would help on well-behaved features** if the instability were fixed. `529_pollen` hints at yes — blend-on is dramatically more stable across seeds there. A future experiment that isolates well-behaved datasets could revisit the α shrinkage idea.
- **Full PMLB suite.** 7 datasets is a small sample; the 2-failure / 5-no-difference picture might be different at n=55.

## Reproducing

```bash
uv run python experiments/run_experiment9_stacked_blend.py
```

Expected runtime ~6 min on CUDA (RTX 3090). Requires Triton 3.6+, PyTorch+CUDA, LightGBM 4+ GPU, XGBoost 2.0+.

## Consequence for the project

**Unchanged from Experiment 8's headline.** The pivot architecture (internal EML splits + gated EML leaves) at single seed was a real win, and that's what ships. The stacked-blend idea was worth testing — it surfaced a real stability issue — but it is not the default.

## Next possible experiments

- **Numerical stability of EML leaves at scale.** Instrument η magnitudes across fits on `562_cpu_small`; fit a threshold for rejecting pathological leaves; revisit Experiment 8's 0.90 ratio on multi-seed with the improved guard.
- **Blend on well-behaved features.** Rerun Experiment 9's blend path on a curated subset of datasets with `max(|x|)/std(x) < 10` to test whether the shrinkage helps when the failure mode is absent.
- **Full PMLB multi-seed suite for blend-off.** With Experiment 9 confirming the 3-seed baseline is unstable on large heavy-tailed datasets, the whole 55-dataset benchmark needs multi-seed evaluation before any of the project's prior headline claims can be considered settled.
