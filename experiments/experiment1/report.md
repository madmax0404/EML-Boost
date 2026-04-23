# Experiment 1: Graceful-Degradation Calibration (depth-2 baseline)

**Date:** 2026-04-23
**Commit:** post-`8d399a1` (EML-Boost v1 on master)
**Runtime:** 3360 s (56 min)
**Scripts:** `experiments/calibration.py`, `experiments/run_calibration_benchmark.py`

## What the experiment was about

This is the first end-to-end validation of EML-Boost's central design claim from the spec (sections 4.2 and 9.3): *a BIC-based per-round selector between an EML weak learner and a DT weak learner should pick EML when the underlying signal is an elementary function, and DT when the signal is threshold-like (categorical / piecewise).*

We generated synthetic datasets whose signal is a convex blend of two regimes:

- **Elementary regime**: `y = exp(x_0) + 0.5 * x_1²` on `x ∈ U(-1, 1)²`
- **DT regime**: signal depends only on two integer-valued "categorical" features via an arbitrary lookup table

The blend parameter is `frac_elementary ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}`. At `frac=0.0`, the signal is entirely DT-style; at `frac=1.0`, entirely elementary.

For each fraction we ran EML-Boost on 3 independent datasets (`n=200` samples each, `max_rounds=20`, `depth_eml=2`, `depth_dt=2`, `n_restarts=6`, patience disabled) and recorded the **fraction of boosting rounds where the EML weak learner won the BIC selection** against the DT weak learner.

## What it was supposed to prove

If EML-Boost's selection rule is well-calibrated, the EML-win rate should rise **monotonically** with `frac_elementary` — low at the DT end (near 0%), high at the elementary end (near 100%), with a smooth transition through the middle. This is the spec's "graceful degradation" plot (section 9.3, must-have #3).

A monotonic curve would be direct empirical evidence that the data decides which family wins, which is the interpretability-and-generalization pitch of the whole project.

## Results

| frac_elementary | EML-win rate | total rounds |
|---|---|---|
| 0.00 | 18.3% | 60 |
| 0.10 | 15.0% | 60 |
| 0.25 | 20.0% | 60 |
| 0.50 | 10.0% | 60 |
| 0.75 | 11.7% | 60 |
| 0.90 | 6.7% | 60 |
| 1.00 | 6.7% | 60 |

**The curve is flat-to-slightly-inverted at ~10–20%. DT wins almost every round, even on pure-elementary data.**

Artifacts preserved in this folder:

- `calibration_curve.csv` — per-fraction aggregate
- `calibration_curve.json` — full result dataclass + config
- `calibration_curve.png` — plot of the curve

## What it actually shows

The depth-2 EML weak learner is not expressive enough to beat DT on this dataset's elementary regime. Root cause: the test formula includes a quadratic term `0.5 * x_1²`. Per source paper Odrzywołek 2026 Table 4, `x²` requires an RPN program of length ≥ 17, which corresponds to a tree of depth 3+. A depth-2 EML tree cannot express `x²` exactly, so it has to approximate it with multiple shallow weak learners summed by the boosting loop — and the BIC tradeoff then prefers a single depth-2 DT stump, which can approximate a quadratic via piecewise splits at lower structural complexity.

This matches the earlier Feynman integration-test finding: the four simple formulas that passed at depth 2 (`μ·N`, `½mv²` restricted to tabulated values, `ω/c`, Gaussian PDF) are all shallow; the one that failed (`p·cos(θ)/r²`) requires depth 3+.

The `run_calibration_benchmark.py` script's "Monotonic (within 0.1 slack): PASS" line is misleading here — the check is `all(Δ ≥ -0.1)`, which accepts a flat-or-slightly-decreasing curve because the deltas are small in magnitude. A stricter check — e.g., `eml_win_rates[-1] − eml_win_rates[0] ≥ 0.4` — would correctly fail.

## What this run does NOT prove

- It does **not** refute the algorithm's core claim. The spec's calibration target (`eml_win_rate ≥ 95% at frac=1.0`) explicitly assumes **depth 3** with the full 20-restart budget; we deliberately ran under-spec to fit the time budget.
- It does **not** mean the BIC selector is mis-calibrated. The BIC is correctly doing its job — picking whichever weak learner gives lower `n·log(MSE) + k·log(n)`. DT wins here because its approximation of `x²` is more parameter-efficient than a sum of depth-2 EML trees.
- It does **not** invalidate the pipeline end-to-end. The pipeline fit, selected, boosted, and reported for 420 EML+DT pairs across 21 datasets without a single crash or non-terminating run.

## Next experiment

See Experiment 2 (planned): single-dataset trace on pure elementary (`frac=1.0`) with `depth_eml=2` and then `depth_eml=3`. The goal is to answer:

1. How often does a depth-2 EML fit produce `snap_ok=True`? (If rarely, then the verification gate is rejecting legitimate fits; if often, then the BIC penalty is the limiting factor.)
2. Does bumping to depth 3 flip the BIC selector's preference toward EML on a simple elementary signal?

That will tell us whether the graceful-degradation story needs a depth-3 calibration rerun, a different synthetic test formula, or a BIC recalibration.

## Reproducing this result

```bash
uv run python experiments/run_calibration_benchmark.py
```

Expected runtime ~55 min on CPU. Output goes to `experiments/results/` by default; this folder is a snapshot.
