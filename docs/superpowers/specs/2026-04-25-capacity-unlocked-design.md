# Capacity-Unlocked Mode (Experiment 14)

**Date:** 2026-04-25
**Context:** Experiment 13 flipped the `max_depth` default from 6 to 8 and made every dataset's ratio strictly improve or tie. The natural follow-up: does the architectural win hold up further when both SplitBoost and XGBoost are run at substantially higher capacity? Specifically, does SplitBoost continue to lead at `max_depth=10, max_rounds=500`, or does XGBoost (with its mature regularization) catch up?

## Goal

Run **one** capacity-unlocked config — `max_depth=10, max_rounds=500` — for SplitBoost and matched baselines (XGBoost, LightGBM at the same capacity) on the same 7 PMLB datasets × 3 seeds. Compare the resulting ratios against Experiment 13's D8_R200 reference (already on disk). Determine whether the 7/7-within-10% picture holds at unlocked capacity, and whether any dataset's ratio worsens by more than 0.05.

## Non-goals

- **No library changes.** Both `max_depth` and `max_rounds` already work at any positive integer. Pure runner-only experiment.
- No 2×2 sweep (depth × rounds). The Experiment-13 result already showed depth > candidates; the question here is whether the trend continues, not which lever drove it.
- No `n_eml_candidates` sweep. Experiment 13 confirmed C30/C100 don't help; stick with C10.
- No baseline rerun at D8_R200. Already on disk in `experiments/experiment13/summary.json`. Reference comparisons happen during report-writing.
- No full PMLB suite (Experiment 15's job).

## Design

### The single SplitBoost config

| param | value | source |
|---|---|---|
| `max_depth` | **10** | unlocked from Exp-13's 8 |
| `max_rounds` | **500** | unlocked from Exp-13's 200 |
| `n_eml_candidates` | 10 | Exp-12/13 default |
| `learning_rate` | 0.1 | unchanged |
| `min_samples_leaf` | 20 | unchanged |
| `min_samples_leaf_eml` | 30 | Exp-12 default |
| `leaf_eml_cap_k` | 2.0 | Exp-11 default |
| `leaf_eml_ridge` | 0.0 | Exp-10 verdict |
| `use_stacked_blend` | False | Exp-9 verdict |
| `k_eml` | 3 | unchanged |
| `k_leaf_eml` | 1 | unchanged |
| `patience` | 30 | bumped from 15 — at 500 rounds, 15 may early-stop too aggressively |
| `val_fraction` | 0.15 | unchanged |

Single SplitBoost config: `D10_R500_C10`.

**Rationale for `patience=30`:** with 500 max rounds, the default `patience=15` lets boosting stop after 3% of allowed rounds without improvement — too aggressive at high capacity. 30 corresponds to 6% of allowed rounds, matching the spirit of the prior 7.5% (15/200) ratio.

### Matched baselines

- **XGBoost** at `max_depth=10, n_estimators=500`. Other XGBoost params unchanged from prior experiments.
- **LightGBM** at `max_depth=10, num_boost_round=500, num_leaves=2**10=1024`. Other LightGBM params unchanged.

`min_data_in_leaf=20` matches `min_samples_leaf=20` on the SplitBoost side.

### Total fits

7 datasets × 3 seeds × 3 models = **63 fits**. Estimated runtime ~25 min on RTX 3090.

Per-fit time estimate at `D10_R500`:
- Small datasets (vineyard, cloud, neavote): ~0.3-1 s (early-stops fast)
- Medium (apnea1): ~10-15 s
- Large (pollen, cpu_small): ~30-60 s
- Largest (fried): ~120-150 s
- Total per seed: ~5-7 min × 3 seeds × 3 models = ~25 min after batching savings on small datasets

### Outputs

- `experiments/experiment14/summary.csv` — one row per (dataset, seed, model).
- `experiments/experiment14/summary.json` — config + aggregates + ratios vs the matched-capacity XGBoost.
- `experiments/experiment14/pmlb_rmse.png` — bar chart with error bars + ratio panel; D10_R500 vs XGBoost only (D8_R200 reference goes in the report's table form).
- `experiments/experiment14/report.md` — narrative comparing D10_R500 to Exp 13's D8_R200.
- `experiments/experiment14/run.log`.

## Success criteria

- **S-A (primary): the 7/7-within-10% picture holds at unlocked capacity.** Specifically, every dataset's mean ratio under D10_R500 stays at ≤ 1.10 (with `557_analcatdata_apnea1`'s "marginal miss" at the same threshold tolerance as Experiment 13's 1.104).
- **S-B: no dataset regresses by > 0.05 in mean ratio vs the D8_R200 baseline.** A small regression on some datasets is plausible (capacity-unlocked can overfit on small-n datasets), but a 0.05 swing on a winning dataset would mean SplitBoost's regularization isn't keeping up with XGBoost's at this capacity.
- **S-C: no RMSE on any dataset × seed exceeds 10× the matched-capacity XGBoost RMSE.** Stability check; the leaf cap should still hold.

## Action on verdict

- **Clean win (S-A + S-B + S-C):** keep current Exp-13 defaults (`max_depth=8`) — capacity-unlocked is *available* but not the default, since the 2-3× runtime cost isn't justified by a small ratio improvement. Document in the report that SplitBoost holds up at unlocked capacity.
- **Partial success:** if S-A holds but S-B fails on one or two datasets, document as "SplitBoost favors moderate capacity over unlocked." Still keep d=8 defaults.
- **Negative outcome (S-A fails on > 1 new dataset):** that's a finding too — SplitBoost's regularization may not scale to depth=10. Document the cap-unlocked failure mode and scope a follow-up to investigate (cap tuning at high depth, or `min_samples_leaf_eml` adjustment).

## Risks

- **Runtime on `564_fried`** at depth=10. The fit will be 5-10× the depth=6 time. Could push the seed-2 fit to 3-5 minutes. Acceptable in absolute terms; flag if any fit exceeds 10 minutes (would suggest a deeper bug).
- **XGBoost overfit at depth=10.** Already observed in Experiment 13 at depth=8; expect more pronounced at d=10. SplitBoost's leaf cap should still hold but if it doesn't, S-B will fail.
- **Early stopping behavior.** Bumping `patience=30` may let SplitBoost run all 500 rounds on small-n datasets where it's already converged at round 50. Inspecting `n_rounds_per_seed` in the summary will tell us if this is happening.
- **Memory during long fits.** GPU memory is fine for batched candidate evaluation (we've run depth-2 EML on 6400-tree spaces at much higher batch sizes); long-running 500-round fits don't accumulate state. No concern.

## Files changed

**Created:**
- `experiments/run_experiment14_capacity_unlocked.py` — the runner.
- `experiments/experiment14/` directory populated by the runner.

**Unchanged:**
- `eml_boost/tree_split/tree.py`, `ensemble.py`, `nodes.py` — no library changes.
- Unit tests — no new tests.
