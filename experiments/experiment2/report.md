# Experiment 2: Diagnosing and Fixing the "snap_ok = 0" Pipeline

**Date:** 2026-04-23
**Commit at start:** post-`8d399a1` (EML-Boost v1 on master)
**Runtime:** ~15 min across three diagnostic traces + two fix attempts
**Scripts:** `experiments/run_experiment2_trace.py` (parametrized by `EXPERIMENT2_DEPTH` and `EXPERIMENT2_FORMULA` env vars)

## What the experiment was about

Experiment 1's graceful-degradation curve was flat-to-inverted: EML-win rate hovered at 10–20% across all fractions. The public-facing outcome was unambiguous, but it didn't localize the cause. Experiment 2's job was to dig into a single pure-elementary run and identify, per boosting round, which weak learner BIC picked, whether EML's snap succeeded, and what the snapped closed-form expression actually said.

Two open questions going in:

1. **Q1.** Are depth-2 EML fits producing `snap_ok=True` often enough to matter, or is the verification gate rejecting legitimate fits?
2. **Q2.** Is the problem (a) training dynamics, (b) verification tolerance, (c) capacity at fixed depth, or (d) something else?

## What it was supposed to prove (or disprove)

If `snap_ok` rate at depth 2 on a simple elementary signal was high (≥ 50%) but the selector still favored DT, the story was "BIC calibration off, needs a simple tolerance or penalty tweak." If `snap_ok` rate was low (near 0), we'd need to fix the snap pipeline before any calibration work could be meaningful.

## Traces and findings

### Part A — depth 2, `exp(x_0) + 0.5 * x_1²`

**Artifacts:** `trace.csv`, `summary.json`, `bic_per_round.png`.

Per-round trace across 20 boosting rounds, 6 restarts per round, on `n = 200` samples. Per-round output showed `snap_ok=N` for every single round. `eml_win_rate = 0.20` (4/20 wins), but with `formula=None` in every row — EML was selected based on its soft torch output, never as a closed-form result.

**Finding:** 100 % of rounds produced a numerically usable EML weak learner *without* a closed-form artifact. Snap verification failed universally.

### Part B — depth 3, same formula (flip test)

**Artifacts:** `trace_depth3.csv`, `summary_depth3.json`, `bic_per_round_depth3.png`.

Motivation: spec 5.2's claim that the full calibration regime uses depth 3 with ≥ 20 restarts. If the problem was just "depth 2 is too small," depth 3 should flip the picture.

Result: worse, not better. `snap_ok = 0/20` as at depth 2. EML-win rate dropped from 20 % to 5 %. Mean params (EML) rose from 4.7 → 10.8 because the larger tree had more non-"1" positions in the `count_active_positions` fallback. BIC penalized EML harder and DT won harder.

**Finding:** capacity wasn't the bottleneck. The verification failure reproduced cleanly at a different depth.

### Diagnostic C — inspect trained softmax distributions

Before throwing more hyperparameters at the problem, we ran a standalone inspection (`/tmp/inspect_softmax.py`) on a single depth-2 fit. For each of the six logit positions, printed the softmax distribution at end of training.

```
logit #0: argmax=0 max_prob=0.986 probs = [0.986 0.009 0.001 0.004]
logit #1: argmax=3 max_prob=0.992 probs = [0.    0.001 0.007 0.992]
logit #2: argmax=1 max_prob=0.998 probs = [0.001 0.998 0.001]
logit #3: argmax=2 max_prob=0.983 probs = [0.015 0.001 0.983]
logit #4: argmax=0 max_prob=0.998 probs = [0.998 0.001 0.001]
logit #5: argmax=0 max_prob=0.622 probs = [0.622 0.353 0.025]

||unsnapped − snapped|| / ||unsnapped|| = 0.2500
snapped range: [1.718, 1.718]   ← constant e − 1
unsnapped range: [0.540, 1.851] ← varies with x_0
```

Five of six positions were ≥ 98 % one-hot. One position was soft (62/35). After argmax, the snapped tree evaluated to a **constant** `e − 1 ≈ 1.718` — it did not depend on `x_0` at all.

**The bug:** the outer EML's right input snapped to `f_prev` (child-1's output), and child-1 itself snapped to all-1s. Child-0 computed `eml(x_0, x_1)` but its output was never referenced by the root. The continuous optimum had been exploiting **soft mixing** — logit #5's 35 % leak let `x_0` flow into the root — and argmax destroyed it, producing a constant-valued "dead-branch" tree.

This was a **topology problem**, not a softmax-sharpness problem.

### Fix #2 (dead-branch prune) — partial success

Added a check in `fit_eml_tree`: after `snap_constants`, if the simplified formula has empty `free_symbols`, the tree is dead-branch and the restart is discarded via `continue`.

Re-ran depth 2 with `exp(x_0)` (simplest possible elementary signal):

- 4/20 restarts produced pure-constant formulas → correctly pruned
- 16/20 restarts produced non-constant formulas like `E − log(E − log(x_0))` or `exp(x_1) − log(x_1)` (wrong feature!) — these passed the prune but failed `reproduces_numerically`

Confirmed that hyperparameter bumps (`_ENTROPY_MAX = 0.5 → 2.0 → 5.0`, `_SNAP_TOL = 1e-6 → 1e-3 → 1e-2`) did not close the gap. The non-constant 16/20 formulas were **sharp-but-wrong**: the optimizer was settling at local minima with non-ideal discrete snap structure.

Fix #2 rejected the most degenerate cases but couldn't fix the core problem.

### Fix #1 (exhaustive search) — full success

At depth 2 with `k = 2` features, the discrete tree space is:

```
internal_positions = 2 × (k+2)^2 = 2 × 16 = ?
leaf_positions     = 4 × (k+1)^? …
                   = (k+2)^2 × (k+1)^4 = 16 × 81 = 1,296 trees
```

Small enough to enumerate every configuration. Implemented `_fit_eml_tree_exhaustive` in `eml_boost/weak_learners/eml.py`: builds the sympy expression for each snapped tree, evaluates on inner-val, picks the one with minimum MSE. `fit_eml_tree` dispatches to exhaustive when `_tree_space_size(depth, k) ≤ _EXHAUSTIVE_THRESHOLD = 50 000` (covers depth 1 and depth 2 with `k ≤ 4`).

Re-ran depth 2 on `exp(x_0)`:

```
snap_ok rate:    20/20
best formula:    exp(x_0)   (recovered exactly on training data)
```

**Artifacts:** `trace_exp_x0_.csv`, `summary_exp_x0_.json`, `bic_per_round_exp_x0_.png`.

Two follow-up cleanups once exhaustive worked:

1. Added `formula_std` field to `EmlWeakLearner`; `params_count` now uses the standardized formula's RPN length for BIC. Without this, un-standardization's float coefficients (`0.86 * exp(1.74 * x_0) − 1`) inflated params count from ~5 to ~9, costing EML 4–5 BIC points per round.
2. 52/52 unit tests stayed green across all fix stages.

## What it actually shows

- The softmax-training + argmax-snap pipeline (spec section 5.3) fails in a specific, repeatable way: the optimizer finds **continuous optima that use soft mixing across simplex components**, and argmax destroys that mixing, producing dead-branch or sharp-but-wrong topologies. This happens at any depth.
- The problem is **not** verification tolerance, entropy strength, or tree capacity — we ruled all three out by direct measurement.
- At small depth (≤ 2 for `k ≤ 4`), exhaustive search is both feasible (≤ 1 300 trees) and dramatically more reliable — `snap_ok` went from 0 % → 100 % and the true formula is recovered exactly on noiseless data.
- Beyond small depth, the pipeline falls back to the existing softmax path. That path's limitations remain; Experiment 2 does not fix them.

## What it does NOT show

- This experiment does not validate the spec's original end-to-end claim ("EML-win rate ≥ 95 % at frac=1 at depth 3 with 20 restarts"). It refutes that claim for the softmax path at any depth we tested, and replaces it with an exhaustive-search alternative that only scales to `k ≤ 4`.
- It does not test real-data benchmarks — all traces are synthetic, small, and noise-light.
- It does not prove the hybrid algorithm's overall value vs. DT-only boosting. That's Experiment 3.

## Consequence for the codebase

After Experiment 2, `eml_boost/weak_learners/eml.py` carries:

- `_tree_space_size(depth, k)` helper.
- `_enumerate_snapped_trees(depth, k)` iterator.
- `_fit_eml_tree_exhaustive(X, y, depth, k, random_state)` — the exhaustive variant.
- A size-based dispatch in `fit_eml_tree`: exhaustive if the space is ≤ 50 000 configs, softmax + snap otherwise.
- `formula_std` field on `EmlWeakLearner` so BIC's `params_count` reflects structural complexity rather than the un-standardization's float-coefficient blow-up.

All 52 unit tests pass on the merged state.

## Reproducing each part

```bash
# Part A — depth 2 with the quadratic (original Experiment 1 signal)
EXPERIMENT2_DEPTH=2 uv run python experiments/run_experiment2_trace.py

# Part B — depth 3, same signal (flip test)
EXPERIMENT2_DEPTH=3 uv run python experiments/run_experiment2_trace.py

# Part C — depth 2 with exp(x_0) (control; after fix #1)
EXPERIMENT2_FORMULA="exp(x0)" EXPERIMENT2_DEPTH=2 uv run python experiments/run_experiment2_trace.py
```

Each run takes 2–5 minutes. Output files land in `experiments/experiment2/` with suffixes derived from the env vars (default: no suffix).
