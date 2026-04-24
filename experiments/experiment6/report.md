# Experiment 6: Two-Feature Extrapolation

**Date:** 2026-04-24
**Commit:** post-Experiment 5 (no-standardization exhaustive path, V-shaped calibration curve documented)
**Runtime:** ~24 min (3 formulas × ~8 min per hybrid fit + negligible tree fits)
**Scripts:** `experiments/run_experiment6_k2_extrapolation.py`
**GPU used:** XGBoost (`device="cuda"`), LightGBM (`device="gpu"`) on RTX 3090.

## What the experiment was about

Experiment 4 showed that on a `k=1` grammar-expressible target (`exp(x_0)`), the hybrid's extrapolation MSE was **47× lower** than XGBoost on a disjoint out-of-range test set. Off-grammar targets (`x_0`, `x_0² + 0.5`) fell back to whatever elementary approximation the exhaustive search found.

Experiment 6 repeats that test with two features, expanding BOTH dimensions into the extrapolation range. Goals:

1. Does the closed-form extrapolation advantage survive when the hybrid has to pick which of two features to build structure over?
2. Does the depth-2 grammar at `k=2` recover genuinely two-feature formulas, or does it fall back to single-feature elementary expressions?
3. How do the failure modes on off-grammar `k=2` targets (linear sum, product) compare to the `k=1` case?

## Configuration

- Three formulas, all trained on `(x_0, x_1) ∈ [−1, 1]²`:
  - `F1: y = exp(x_0) − log(x_1 + 2)` — grammar-expressible structure (offset so `log` argument is positive on both ranges).
  - `F2: y = x_0 + x_1` — linear sum; not cleanly in the depth-2 grammar.
  - `F3: y = x_0 · x_1` — product; requires depth 3+ to express via `exp(log(a) + log(b))`.
- `n_train = 500`, `n_in_range_test = n_extrap_test = 200`, Gaussian noise `σ = 0.02`.
- Train range `[−1, 1]²`; extrapolation range `[1, 2]²` (both dimensions expanded).
- `seed = 0`.
- Capacity: `max_rounds = 100`, `depth_dt = 6`, `depth_eml = 2`, `k = 2`. Same across all three models; equal capacity comparison.
- Hybrid's EML branch uses exhaustive search over 1,296 candidate trees per round at depth 2, `k = 2` (space size = `(k+2)² × (k+1)⁴ = 16 × 81`).
- LightGBM / XGBoost: identical configuration to Experiment 4 (`n_estimators=100, max_depth=6, lr=0.1`) on GPU.

## What it was supposed to prove

1. **Prediction A.** On the grammar-expressible `F1 = exp(x_0) − log(x_1 + 2)`, the hybrid should extrapolate correctly and beat both tree baselines by a large margin.
2. **Prediction B.** On the linear sum `F2 = x_0 + x_1`, trees extrapolate as horizontal constants past the training boundary; the hybrid's exp-based fallback grows too fast and overshoots. Trees should win.
3. **Prediction C.** On the product `F3 = x_0 · x_1`, neither method expresses the target; both fail but possibly for different reasons — the interesting question is whether the hybrid's elementary fallback does better or worse than the trees' boundary plateau.

## Results

**Artifacts:** `summary.csv`, `summary.json`, `extrapolation_plots.png`.

| formula | hybrid extrap | LightGBM extrap | XGBoost extrap | ranking |
|---|---|---|---|---|
| `exp(x_0) − log(x_1 + 2)` | **0.135** | 4.951 | 4.900 | **Hybrid** ≪ XGBoost < LightGBM |
| `x_0 + x_1` | 4.587 | 1.596 | **1.278** | XGBoost < LightGBM < Hybrid |
| `x_0 · x_1` | 3.062 | **2.445** | 2.497 | LightGBM < XGBoost < Hybrid |

Recovered formulas (top-level sympy after `learned_η` scaling + DT contributions):

- F1 → `0.8454 · exp(x_0) − 0.8454`
- F2 → `0.5949 · exp(x_0) + 0.7342 · exp(x_1) − 1.3291`
- F3 → `0.0648 · exp(x_0) + 0.0965 · exp(x_1) − 0.1613`

## What it actually shows

**Prediction A confirmed.** On F1, the hybrid extrapolates 36× better than XGBoost and 37× better than LightGBM. Trees saturate at the boundary of their training range; the hybrid's recovered exp keeps tracking. This is the k=2 analog of Experiment 4's k=1 headline result.

**Prediction A (with a caveat).** The recovered formula is `0.8454 · exp(x_0) − 0.8454` — it only captured the `exp(x_0)` part of F1, **not** the `log(x_1 + 2)` part. The `− 0.8454` constant tail suggests a single-feature elementary approximation plus a constant correction from DT rounds, rather than a true two-feature EML expression. In-range MSE ≈ 0.15 (worse than the Experiment 4 single-variable case's ~0.07) reflects this: the hybrid is missing up to 1.1 of signal variance from the log term. It still extrapolates correctly because the captured `exp(x_0)` dominates the output variance.

This is an honest limitation worth stating: **the depth-2 grammar at k=2 can express `eml(x_0, x_1)` but not `eml(x_0, x_1 + 2)`** — the grammar lacks a way to add a literal constant to a feature without continuous coefficients (spec's Option A). Because `log(x_1)` diverges on `x_1 ∈ [−1, 1]` (values where `x_1 ≤ 0` produce NaN), `eml(x_0, x_1)` is ruled out by the finite-output filter, and the exhaustive search settles on `eml(x_0, 1) = exp(x_0)` as the best remaining candidate. The offset `+2` is structurally unreachable.

Despite this gap, F1's hybrid extrapolation MSE is still 36× better than trees — so the grammar-match advantage dominates the single-feature-only fallback.

**Prediction B confirmed.** On F2, the hybrid loses to trees by ~3-4× on extrapolation. Recovered formula is `a · exp(x_0) + b · exp(x_1) + c` — sum of two exps, which in-range can fit a bilinear surface acceptably but extrapolates exponentially rather than linearly. In-range MSE (~0.18) is worse than trees (~0.004), and out-of-range the gap widens. This is the expected k=2 version of Experiment 4's linear-target failure.

**Prediction C narrowly refuted.** On F3 (product), trees narrowly beat the hybrid (2.45 vs 3.06). Hybrid's recovered sum-of-exps is a poor match for the multiplicative structure — at `(x_0, x_1) = (2, 2)` the hybrid predicts `0.065 · e² + 0.097 · e² − 0.161 ≈ 0.88` while truth is `4.0`. Trees plateau around `1.0` from the boundary leaf. Both are catastrophically wrong on extrapolation; trees' plateau is closer.

## Cross-experiment summary

Experiments 4 and 6 together now cover both k=1 and k=2 extrapolation:

| target class | k=1 result (Exp 4) | k=2 result (Exp 6) | conclusion |
|---|---|---|---|
| Grammar-expressible exp/eml | **Hybrid wins ~47× (MSE 0.13 vs 6.14)** | **Hybrid wins ~36× (MSE 0.13 vs 4.90)** | Robust headline claim |
| Linear target | Trees win ~5× | Trees win ~3-4× | Consistent hybrid weakness |
| Non-expressible smooth | Hybrid wins narrowly (x²) | Trees win narrowly (x·x) | Hybrid's fallback quality depends on specific target |

The paper's strongest defensible claim: **when the target is a closed-form elementary expression that the EML grammar at the chosen depth can reach, the hybrid is the only one of the three methods that extrapolates correctly**, and the gap is 36-47× on MSE at `x ∈ [1, 2]` after training on `[−1, 1]`. For off-grammar targets, the hybrid inherits the failure modes of whatever elementary approximation it defaults to.

## What v6 does NOT show

- Does not test higher depths. At depth 3+, the softmax fallback path is in charge, and the exhaustive-search guarantee doesn't hold.
- Does not cover continuous-coefficient targets with offsets or scalings (e.g., `log(x_1 + c)` for `c ≠ 0`). The spec's Option A explicitly lacks this expressive power; addressing it requires Option B (continuous feature coefficients in the grammar) or a richer exhaustive search.
- Does not explore the "stop EML when fallback is worse than DT" heuristic flagged by Experiment 5. Would likely help on F2 and F3.
- Does not test higher noise, much larger extrapolation distance, or multi-seed variance.

## Reproducing these results

```bash
uv run python experiments/run_experiment6_k2_extrapolation.py
```

Runtime ~24 min on CPU + GPU. Output goes to `experiments/experiment6/` by default.

## Consequence for the project

- **Extrapolation pitch is now triangulated by two experiments** (k=1 and k=2). Both confirm: grammar-match → dramatic win, otherwise → honest loss.
- **Paper Section "Extrapolation":** can cite both experiments with one bar chart comparing across formula / dimensionality.
- **The grammar's inability to express `log(x_1 + 2)` is a concrete Option-A limitation.** If a future version adopts Option B (continuous β coefficients), F1 should jump from "0.135 extrap, approximate" to near-exact recovery. Worth trying as a proof-of-concept extension.
- **Known failure mode on off-grammar targets stands** — both linear and product fall back to scaled sum-of-exps, which interpolates okay and extrapolates exponentially in the wrong direction. The "fallback to DT when EML coverage is low" heuristic from Experiment 5 applies here too.

## Next possible experiments

- **Extreme-range extrapolation plot** (`x ∈ [1, 5]` after training on `[−1, 1]`). The hybrid's win on F1 should widen dramatically; trees stay flat. Would make a visually striking figure for the paper.
- **Option B mini-experiment** — implement continuous-coefficient EML for a single formula like F1 and see whether it recovers `exp(x_0) − log(x_1 + 2)` exactly.
- **Fallback heuristic experiment** — add a "predict DT-only when EML coverage < τ" switch to the production code, rerun Experiments 5 and 6, confirm that the off-grammar regressions are eliminated without hurting the on-grammar headline.
- **Real tabular extrapolation** — find a PMLB/OpenML dataset where the test set covers ranges outside the training set, and reproduce the extrapolation advantage there.
