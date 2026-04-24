# `min_samples_leaf_eml` Sweep (Experiment 12)

**Date:** 2026-04-25
**Context:** Experiment 11's final "What's left as a loss" noted that `210_cloud` (n_train=86) never activates EML leaves because its depth-6 tree produces leaves in the 20-40 sample range, all below the current `min_samples_leaf_eml=50` gate. The report said: "would need `min_samples_leaf_eml < 50` to enable EML on small datasets, which is its own experiment." This is that experiment.

## Goal

Sweep `min_samples_leaf_eml ∈ {20, 30, 40, 50}` on the 7 PMLB datasets × 3 seeds with all other Experiment-11-best-default hyperparameters held fixed. Determine whether lowering the threshold meaningfully improves `210_cloud` (and potentially `523_analcatdata_neavote`, `192_vineyard`) without regressing the larger-n winners.

## Non-goals

- **No library changes.** The hyperparameter already exists; this is a pure experiment.
- No `min_samples_leaf` sweep. That parameter controls tree structure, not EML eligibility; keep at 20.
- No cap-off ablation. The cap is Experiment 11's critical stability fix for tiny leaves; it stays at `leaf_eml_cap_k=2.0` for all configs (tiny leaves need it more than large ones).
- No full PMLB suite (55 datasets). The 7-dataset story is the lens we've been using.
- No CV, no capacity sweep.

## Design

### Configurations

Four SplitBoost configurations, gated path only:

| id | `min_samples_leaf_eml` | interpretation |
|---|---|---|
| M20 | 20 | matches `min_samples_leaf`; all leaves EML-eligible |
| M30 | 30 | moderate gate |
| M40 | 40 | just below current default |
| M50 | 50 | Experiment 11's default (baseline) |

### Fixed config across all runs

```
max_rounds          = 200
max_depth           = 6
learning_rate       = 0.1
patience            = 15
val_fraction        = 0.15
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0   # Experiment 11's new default — kept on
use_stacked_blend   = False
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2]
```

### Val-split guard considerations

The existing code has this guard in `_fit_leaf` (around tree.py:382):

```python
val_sz = max(n // 4, 5)
if n - val_sz < self.min_samples_leaf_eml // 2:
    return LeafNode(value=constant_value)
```

Check the arithmetic at each sweep value (leaf has exactly `n=min_samples_leaf_eml` samples):

| `min_samples_leaf_eml` | `n` | `val_sz` | `fit_sz` | required fit_sz (= n/2) | passes? |
|---|---|---|---|---|---|
| 20 | 20 | 5 | 15 | 10 | ✓ |
| 30 | 30 | 7 | 23 | 15 | ✓ |
| 40 | 40 | 10 | 30 | 20 | ✓ |
| 50 | 50 | 12 | 38 | 25 | ✓ |

All sweep values satisfy the guard at their own threshold — so the threshold is the effective EML gate, not the val-split check.

### Baselines per (dataset, seed)

XGBoost and LightGBM for reference, same configs as Exp 8-11.

### Total fits

7 datasets × 3 seeds × (4 SplitBoost + XGBoost + LightGBM) = **126 fits**. Estimated runtime ~12 min on RTX 3090.

### Outputs

- `experiments/experiment12/summary.csv` — one row per (dataset, seed, config).
- `experiments/experiment12/summary.json` — config + aggregates.
- `experiments/experiment12/leaf_activation_stats.json` — per-(dataset, config, seed) dict: `{n_eml_leaves, n_total_leaves, eml_fraction, mean_leaf_size, median_leaf_size, min_leaf_size, max_leaf_size}`. Lets us see whether the sweep actually activated more EML leaves on the small-n datasets.
- `experiments/experiment12/pmlb_rmse.png` — bar chart with error bars.
- `experiments/experiment12/report.md` — narrative.
- `experiments/experiment12/run.log`.

## Experiment 12 plan

### Files changed

**Created:**
- `experiments/run_experiment12_min_leaf_sweep.py` — runner.
- `experiments/experiment12/` directory populated by the runner.

**Unchanged:**
- `eml_boost/tree_split/tree.py`, `ensemble.py`, `nodes.py` — no code changes (the `min_samples_leaf_eml` parameter already exists).
- Unit tests — no new tests; the parameter has always worked at any positive int value.

## Success criteria

- **S-A (primary): `210_cloud` mean ratio improves at some `min_samples_leaf_eml < 50` config** — ideally `< 1.10` (within the band). Current Exp 11 ratio: 1.19.
- **S-B: no regression on winners.** `192_vineyard`, `523_analcatdata_neavote`, `529_pollen`, `562_cpu_small`, `564_fried` stay within 0.03 of their Experiment-11 ratios under the best low-threshold config.
- **S-C: no new instability.** No RMSE on any dataset × seed exceeds 10× the XGBoost RMSE on that dataset × seed (the Experiment-11 stability criterion).

A partial success is fine: if `210_cloud` improves but one medium-n dataset regresses slightly, we can set per-dataset guidance or pick a middle-ground default.

## Action on verdict

- If one threshold is cleanly best across datasets (improves cloud without hurting others): flip the default on both constructors.
- If different datasets prefer different thresholds: keep default at 50 and document the tuning guidance.
- If no threshold < 50 improves cloud without regressions: keep default, document the finding, scope as a harder fix (maybe an adaptive per-dataset threshold based on n_train).

## Risks

- **EML leaves on tiny (20-30 sample) leaves may overfit more.** The OLS is fit on ~15-23 samples with 25% held out for val. The cap mitigates this via robust tree selection; without the cap, this would be dangerous. Test-time prediction quality on medium-n datasets (`apnea1` at n=380) is the main thing to watch for regressions.
- **Cap hit-rate may rise on small leaves.** Tiny-leaf OLS η is noisier, so the cap fires more often. Monitor `leaf_activation_stats.json` counts + compare against Exp 11's cap-hit rates.
- **Early-stopping sensitivity.** Shorter trees fit smaller datasets faster, may early-stop earlier. Not a risk per se, just a behavioral shift.
