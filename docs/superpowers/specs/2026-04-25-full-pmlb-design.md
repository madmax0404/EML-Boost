# Full PMLB Multi-Seed Suite (Experiment 15)

**Date:** 2026-04-25
**Context:** Experiment 13 established that SplitBoost at `max_depth=8, max_rounds=200` is 6/7 within 10% of XGBoost across 3 seeds on a curated 7-dataset subset. Experiment 14 confirmed the architectural lead extends to higher capacity but with cloud regressing past the band. The headline still rests on a 7-dataset story. To turn it into a real, defensible benchmark statistic, we need the full PMLB regression suite at the established d=8, r=200 default — across enough seeds to produce error bars.

## Goal

Run a single SplitBoost config (`max_depth=8, max_rounds=200, leaf_eml_cap_k=2.0, min_samples_leaf_eml=30`) against matched-capacity XGBoost and LightGBM on **all 122 PMLB regression datasets × 5 seeds**. Aggregate to: fraction-within-10%, fraction-outright-wins, mean RMSE ratio with std, and dataset-level percentile distributions. Produce a final report that turns "6/7 on 7 datasets" into a defensible aggregate statistic.

## Non-goals

- **No library changes.** All hyperparameters at established defaults; runner-only.
- No hyperparameter sweep. Single config.
- No CV. Same 80/20 shuffle split per seed as prior experiments.
- No capacity-unlocked rerun. Experiment 14 already showed d=10 is partial-success on a subset; we don't generalize that finding here.
- No per-dataset tuning or fallback. If a config produces a bad fit on some dataset, that's a real result.

## Design

### The single SplitBoost config

```
max_rounds          = 200       # Exp 12 default
max_depth           = 8         # Exp 13 default
learning_rate       = 0.1
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 20
min_samples_leaf_eml = 30       # Exp 12 default
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0       # Exp 11 default
use_stacked_blend   = False
n_bins              = 256
patience            = 15
val_fraction        = 0.15
test_size           = 0.20
seeds               = [0, 1, 2, 3, 4]
```

### Matched-capacity baselines

- **XGBoost** at `max_depth=8, n_estimators=200`. Other params unchanged.
- **LightGBM** at `max_depth=8, num_boost_round=200, num_leaves=2**8=256, min_data_in_leaf=20`.

### Datasets

All 122 PMLB regression datasets from `pmlb.regression_dataset_names`. Including BNG_* synthetics (12 datasets) which can have 1M+ rows. The `try/except` around each fit handles failures.

### Reliability machinery

- **Per-fit try/except.** A single model failure on a single (dataset, seed) doesn't crash the run. Failed fits go to a `failures.json` log with the dataset, seed, model, and exception type.
- **Incremental CSV append.** After each dataset's seed sweep completes (i.e., 5 seeds × 3 models = 15 fits per dataset), append those rows to `summary.csv` immediately. This way a mid-run crash leaves the data for completed datasets safely on disk.
- **Resume-from-checkpoint.** On startup, the runner reads any existing `summary.csv` and skips datasets that already have full coverage (15 rows for that dataset). Partial-coverage datasets are restarted from scratch (we don't have to handle partial seeds — the per-dataset block runs all seeds before appending).
- **Per-fit timeout.** Hard cap of 600 seconds per individual fit to prevent a single broken dataset from hanging the run. Timeout logs as a failure.

### Total fits

122 datasets × 5 seeds × 3 models = **1830 fits**. Estimated runtime ~6-8 hours on RTX 3090. Dominated by BNG_* and 1595_poker (1M-row datasets) where each SplitBoost fit may take 10+ minutes at d=8/r=200.

### Outputs

- `experiments/experiment15/summary.csv` — appended row-by-row as datasets complete. Columns: `dataset, seed, config, rmse, fit_time, n_rounds`.
- `experiments/experiment15/summary.json` — written at end. Has `config`, `aggregate` (per-dataset means/stds), `ratios` (per-dataset SplitBoost/XGBoost ratio), and `headline_stats` (fraction-within-10%, fraction-outright-wins, mean ratio, percentiles).
- `experiments/experiment15/failures.json` — list of `{dataset, seed, config, error_type, error_message, error_phase}` for each failed fit.
- `experiments/experiment15/pmlb_rmse.png` — TWO panels stacked vertically:
  - Top: Sorted bar chart of all 122 mean ratios (SplitBoost / XGBoost), one bar per dataset, color-coded by within-10% (blue) or outside (red).
  - Bottom: Histogram of mean ratios with bin width 0.05; vertical lines at 1.0 (parity) and 1.1 (10% band).
- `experiments/experiment15/report.md` — narrative.
- `experiments/experiment15/run.log` — full console output.

### Headline statistics computed at end

```python
total_datasets = number of datasets with at least one successful seed for all three models
fraction_within_10pct = fraction with mean(SplitBoost) / mean(XGBoost) ≤ 1.10
fraction_outright_wins = fraction with mean(SplitBoost) / mean(XGBoost) < 1.00
fraction_within_5pct = fraction with mean ratio ≤ 1.05
mean_ratio = mean over datasets of mean ratio
median_ratio = median over datasets of mean ratio
p25_ratio, p75_ratio = quartiles over datasets
```

These go into the report's headline section.

## Success criteria

This experiment is descriptive, not pass/fail — but for the report's headline:

- **S-A: > 60% of datasets within 10% of XGBoost.** A 60% lower bound is the published "GBDT competitive on most datasets" baseline; SplitBoost should clear that.
- **S-B: > 30% of datasets outright wins (ratio < 1.00).** Strong indicator of architectural value beyond just "competitive."
- **S-C: < 5% of datasets with catastrophic failure (ratio > 2.0 or fit failure).** The leaf cap and cap_k=2.0 default should keep instability rare.

If all three are met, the project's headline becomes "competitive across PMLB regression with X% outright wins, stable across 5 seeds." If any fail, the report identifies which dataset categories caused the issues (small-n, heavy-tailed features, etc.) and scopes follow-ups.

## Risks

- **Long-tail runtime.** A few BNG_* datasets at 1M rows could take 30+ min per fit. With 5 seeds × 3 models, that's 7-10 hours just on that one dataset. The 600s per-fit timeout limits damage but those datasets contribute zero data.
- **OOM on very wide datasets.** Some PMLB datasets may have 200+ features at 100k+ rows. GPU memory could fill. The try/except handles it; we'll get failures and move on.
- **PMLB cache.** First-time download of all 122 datasets is several GB — make sure the network is up and the cache directory has space. PMLB caches to `~/.pmlb_cache` typically.
- **Non-numeric features.** Some PMLB datasets may have categorical or string features that XGBoost handles via internal encoding but our `np.asarray(X, dtype=np.float64)` cast will fail. The try/except logs these as failures; we pre-screen `np.isfinite(X).all(axis=1)` already, but type errors hit before that.

## Files changed

**Created:**
- `experiments/run_experiment15_full_pmlb.py` — runner.
- `experiments/experiment15/` directory populated by the runner.

**Unchanged:**
- `eml_boost/tree_split/tree.py`, `ensemble.py`, `nodes.py` — no library changes.
- Unit tests — no changes.

## Action on verdict

Whatever the result, the report becomes the project's reference benchmark. The recommended action is:
- **If S-A/S-B/S-C all met:** ship the report's headline as the project's primary claim. Update `experiments/experiment8/report.md` and similar prior reports with a footer pointing to Experiment 15's aggregate.
- **If S-A holds but S-B is weak:** "competitive on PMLB" is the more conservative claim.
- **If S-A fails:** the strong "GBDT-competitive" framing is wrong. The report should identify which dataset categories drag the average and propose targeted experiments. The 7-dataset story remains what it was — anecdotal but consistent.
