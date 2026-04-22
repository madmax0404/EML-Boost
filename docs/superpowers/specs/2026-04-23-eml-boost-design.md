# EML-Boost: Design Specification

**Date:** 2026-04-23
**Status:** Draft — awaiting review
**Source paper:** Odrzywołek, A. "All elementary functions from a single binary operator." arXiv:2603.21852v2 (April 2026).

## 1. Summary

EML-Boost is a gradient boosting regressor whose per-round weak learner is adaptively chosen between two heterogeneous families:

1. A shallow trainable EML tree — a parameterized realization of the `eml(x, y) = exp(x) − ln(y)` Sheffer operator introduced in Odrzywołek (2026), universal over elementary functions. When the fit snaps successfully to the vertex of its softmax parameter simplex, the weak learner resolves to a closed-form elementary expression.
2. A shallow LightGBM decision tree — standard GBDT weak learner with native categorical and missing-value handling.

A BIC-style scoring rule selects one of the two per boosting round. The resulting ensemble decomposes into a closed-form "formula part" (sum of snapped EML expressions) and a "residual part" (sum of shallow DTs). On data generated from an elementary law, EML-Boost recovers the formula exactly; on messy tabular data, it degrades gracefully to a competitive boosting ensemble.

## 2. Motivation

The source paper provides two elements that together create this opportunity:

1. **A universal elementary operator.** Every elementary function (including `sin`, `cos`, `√`, `log`, constants like `π`, `e`, `i`) can be expressed as a binary tree of `eml` nodes with constant-1 leaves. The space of elementary functions has a uniform combinatorial structure — `S → 1 | eml(S, S)` — directly analogous to NAND circuits in Boolean logic.

2. **A known training barrier.** The paper reports that gradient-based recovery of an elementary function from data via parameterized EML trees succeeds in 100% of runs at tree depth 2, ~25% at depth 3–4, and below 1% at depth 5. Blind recovery from random initialization fails past depth 6.

Boosting is designed exactly for this shape of problem: it builds expressive models by summing shallow weak learners. If the easy subproblem (shallow EML fit) is reliably solvable, additively combining many shallow fits can express functions a single deep fit could not, without ever solving the hard deep-tree problem directly. This converts the paper's recovery barrier from a weakness into the structural justification for a boosting approach.

The tension with "pure" gradient boosting is that EML trees are smooth, differentiable, algebraic objects — not piecewise-constant threshold trees. A true drop-in replacement is not meaningful. The hybrid approach here preserves EML's advantage on smooth elementary structure while delegating threshold / categorical / missing-value structure to a standard DT weak learner, with the boosting framework deciding per round which family to use.

## 3. Goals and non-goals

### Goals (v1)

- Regression (squared-error loss).
- Numerical and categorical features; single-machine, in-memory, tabular data up to ~10⁵ rows and ~100 features.
- Scikit-learn-style API: `EmlBoostRegressor().fit(X, y).predict(X)`.
- Exact-formula recovery on Feynman SR benchmark at ≥ 80% rate.
- Test-set RMSE within 10% of XGBoost averaged over PMLB regression benchmarks.
- Interpretable model surface: `model.describe()` returns a recovered formula (if any) plus a characterization of the residual ensemble.

### Non-goals (deferred)

- Classification, quantile regression, multi-target regression.
- Text, image, sequential data.
- Distributed training, GPU acceleration, out-of-core training.
- Online / streaming fit.
- Custom loss functions beyond L2.
- Second-order (Newton-style) boosting.
- Mixture-per-round weak learners (strictly one-or-the-other in v1).
- Adaptive EML depth within a single training run.
- Higher-rank Sheffer operators (EDL, ternary variants from the paper) — EML only for v1.

## 4. Algorithm

### 4.1 Outer boosting loop

```
Inputs:
  X, y                dataset
  M                   max boosting rounds (default 500)
  η_DT                DT shrinkage (default 0.1)  — EML step is learned per round
  depth_eml           EML tree depth (default 3)
  depth_dt            DT depth (default 3)
  n_restarts          EML random restarts per round (default 20)
  k                   top-k feature count for EML (default 3)
  patience            early-stopping patience (default 15)
  outer_val_frac      outer val split size (default 0.15)
  inner_val_frac      per-round inner val split size (default 0.20)

Procedure:
  split X, y  → (X_trainval, y_trainval), (X_outerval, y_outerval)

  F_0(x) = mean(y_trainval)
  history = []

  for m = 1..M:
    r_i = y_trainval_i − F_{m-1}(x_i)                       # L2 pseudo-residual
    split (X_trainval, r) → (X_tr, r_tr), (X_iv, r_iv)      # inner val

    h_eml = fit_eml_tree(X_tr, r_tr, depth_eml, n_restarts, k)
    h_dt  = fit_dt_stump(X_tr, r_tr, depth_dt)

    # Per-round learned scale for EML (closed-form 1-D least squares on inner val);
    # DT uses a fixed shrinkage η_DT (standard GBDT regularization).
    η_eml_m = <r_iv, h_eml(X_iv)> / (||h_eml(X_iv)||² + ε)

    score_eml = BIC(η_eml_m · h_eml, X_iv, r_iv)
    score_dt  = BIC(η_DT     · h_dt,  X_iv, r_iv)

    if score_eml < score_dt:
        (h_m, type_m, η_m) = (h_eml, EML, η_eml_m)
    else:
        (h_m, type_m, η_m) = (h_dt, DT, η_DT)

    F_m(x) = F_{m-1}(x) + η_m · h_m(x)
    history.append(record)

    if no outer-val improvement for `patience` rounds: break

  return EmlBoostModel(F_m, history)
```

### 4.2 BIC selection rule

```
BIC(h, X_val, r_val) = n_val · log(MSE_val(h)) + params(h) · log(n_val)

params(h_eml) = K(h_eml) + 1
                where K = RPN length of the snapped, simplified expression
                         (the paper's complexity measure — cf. Section 2 of Odrzywołek 2026)
                the +1 accounts for the per-round learned η_eml_m scalar.
                If snap fails, fall back to:
                  (# input positions not snapped to "1") + 1

params(h_dt)  = number of leaves
```

Rationale: BIC gives consistent model selection when the true hypothesis is representable in the class. Under exact-recovery regime (true law is elementary), BIC favors the EML fit once it snaps cleanly. Under non-elementary regime, BIC favors DT's compact leaf structure.

For EML, since the snapped expression contains no continuous free parameters (all simplex positions collapse to vertices), complexity is measured structurally — via the RPN encoding length of the simplified formula. This matches how the source paper itself measures formula complexity (Table 4 of Odrzywołek 2026).

Selection rule will be validated by a calibration experiment (see Section 9). The experiment produces a plot of EML-win-rate as a function of signal compositionality — a direct empirical demonstration of the graceful-degradation claim. If BIC (with penalty weight `log(n)`) fails a ≥ 95% correct-family threshold on either of the two pure-regime synthetic benchmarks, the penalty weight will be reduced to `2` (AIC-style); the decision is recorded as a hyperparameter pinned by the calibration experiment, not as a runtime tunable.

### 4.3 Step sizes — learned EML, fixed DT

**EML rounds: learned per-round η.** For each EML round, the scalar step size `η_eml_m` is chosen by a closed-form 1-D least-squares fit against the inner-val residual:

```
η_eml_m = <r_iv, h_eml(X_iv)> / (||h_eml(X_iv)||² + ε)
```

Rationale: under Option A (paper-faithful grammar), the snapped EML expression has no continuous free parameters — its output is scale-fixed. Arbitrary coefficients in the true law (e.g., the `0.5` in kinetic energy, the `2π` in pendulum period) cannot be expressed structurally at shallow depth. Per-round learned η closes this gap: if the snapped formula captures the correct functional form, the coefficient is recovered exactly by 1-D least squares. This preserves interpretability (each round contributes `η · formula`, a scalar × closed-form term) while enabling arbitrary-constant recovery.

ε is a small regularizer (default `1e-8`) preventing divide-by-zero when a weak learner produces identically-zero predictions.

**DT rounds: fixed shrinkage.** `η_DT = 0.1` by default, user-overridable. Rationale: standard GBDT shrinkage regularization; DT weak learners approximate residual structure and benefit from the classical bias-variance tradeoff. Learning η_DT per round is deferred to v2 (would remove shrinkage, risk overfitting).

Ablation studies report: learned η_EML (default) vs fixed η_EML ∈ {1.0, 0.5, 0.1}; and η_DT ∈ {0.1, 0.05, 0.3}.

## 5. EML weak learner

### 5.1 Parameterization

Based on the paper's master formula construction, extended to the multivariate case following the paper's grammar `S → 1 | x_1 | x_2 | ... | x_k | eml(S, S)`.

Each input position to an internal EML node is a convex combination (softmax) over `(k + 2)` possibilities: the constant 1, each of the k selected input features, and the previous EML output `f_prev`. Each input is therefore parameterized by a logit vector of length `k + 2`:

```
input_i = softmax(logits_i) · [1, x_1, ..., x_k, f_prev]
        = α_i · 1 + β_{i,1} · x_1 + ... + β_{i,k} · x_k + γ_i · f_prev
     with  α_i + β_{i,1} + ... + β_{i,k} + γ_i = 1  (softmax output)
```

Leaf input positions (at tree depth `depth_eml`) use a `(k + 1)`-way simplex over `[1, x_1, ..., x_k]`; the `γ` component is omitted because there is no EML node below.

After `argmax` snap, exactly one simplex component equals 1 and the rest are 0; the input resolves to either the constant 1, one of the features, or the previous output. The snapped expression contains **no continuous free parameters** — all complexity is structural, carried by the tree topology and the choice of terminal at each input position.

Parameter count (pre-snap logits) for depth `n`, `k` selected features:

```
internal_inputs = 2^n − 2          # 2 × number of non-root, non-leaf internal nodes
leaf_inputs     = 2^n              # 2 × number of leaf-parents
logits_total    = (k + 2) · internal_inputs + (k + 1) · leaf_inputs
                = (2k + 3) · 2^n − 2·(k + 2)
```

For `n=3, k=3`: 62 logits. Manageable for Adam optimization within the boosting loop.

**Note:** This is faithful to the paper's grammar (terminals are 1 and input variables — not arbitrary linear combinations of features). A richer alternative would treat β as a continuous k-vector of feature coefficients (Option B in design discussion). v1 commits to the paper's Option A for fidelity and because Option A has a zero-parameter snapped form, which gives a cleaner BIC story. Option B is flagged for v2 ablation if Option A proves insufficiently expressive per round.

### 5.2 Feature selection (k-variate per round)

Before each EML fit:

1. Compute Pearson correlation between each feature and the current residual.
2. Select top `k` features by |correlation|. Default `k = 3`.
3. EML sees only those k features; remaining features are invisible to EML for this round.

This prevents per-round parameter explosion on wide datasets and improves optimization stability. The DT weak learner always sees all features.

Categorical features are excluded from EML's correlation computation and from EML's input set unconditionally. They participate only in the DT weak learner.

### 5.3 Training procedure

Three-phase optimization per restart:

1. **Exploration.** Adam optimizer, learning rate `1e-2`, 500 steps. Random logit initialization from N(0, 1). Softmax with temperature 1.
2. **Hardening.** Continued Adam, lr decaying `1e-2 → 1e-4`, 500 additional steps. Add annealed entropy penalty: `+ λ_H · Σ_i H(softmax(logits_i))`, with `λ_H` ramping from 0 to 0.5. Pushes each `(k+2)`-way (or `(k+1)`-way at leaves) softmax toward a simplex vertex.
3. **Snap and verify.** For each input-position logit vector, take argmax — setting the winning component to 1 and others to 0. The input resolves to the selected terminal (constant 1, one of the k features, or the previous EML output). Evaluate the resulting symbolic expression at held-out points. If reproduction MSE < `ε` (default `1e-6` relative to unsnapped EML's output range), record the snapped expression as the weak learner output. Otherwise, retain the unsnapped EML (still a valid, differentiable predictor) and flag this restart as "soft" — that is, usable numerically but not convertible to closed form for the final report.

Restarts: `n_restarts = 20` per round with independent random inits, run in parallel via vectorized PyTorch (stacked in a leading batch dimension; single optimizer call). Keep the restart with lowest inner-val MSE for selection.

### 5.4 Numerical handling

The paper's numerical lessons are adopted without modification:

- **Data type:** `torch.complex128` internally. The EML operator requires complex arithmetic to generate constants like `π`, `i` via principal-branch logarithm of negative reals.
- **Exp argument clamping:** `|arg| ≤ 50` before applying `exp()`. Without this, overflow → NaN → irreversible training failure.
- **Divergence detection:** any NaN in forward or backward pass aborts the current restart. No penalty; next restart proceeds.
- **Real-output enforcement:** after snap, evaluate on real data; require `|imaginary part| < tol` (default `1e-8`) across the inner-val set. If violated, the restart is flagged non-physical and discarded.
- **Feature standardization:** before fitting, features are z-scored. Un-standardization is pushed into the sympy pipeline during simplify — the reported formula is in the original feature coordinates.

## 6. DT weak learner

LightGBM single-tree mode (`num_boost_round = 1`) with the following configuration:

| Parameter | Default | Notes |
| --- | --- | --- |
| `objective` | `regression_l2` | matches squared-error pseudo-residuals |
| `max_depth` | 3 | shallow; comparable capacity to EML depth 3 |
| `num_leaves` | 8 | consistent with max_depth |
| `min_data_in_leaf` | 20 | overfitting guard |
| `learning_rate` | 1.0 | no internal shrinkage — outer η does shrinkage |
| `categorical_feature` | auto-detected | LightGBM native handling |
| `verbose` | -1 | suppress log spam |

Missing values are handled natively by LightGBM's default-direction split mechanism. Categorical columns are identified by pandas `category` dtype, `object` dtype with low cardinality, or explicit user override via `EmlBoostRegressor(categorical_features=...)`.

## 7. Post-processing and model surface

### 7.1 Decomposition

After training:

```
formula_part(x)    = Σ_{m: type=EML, snap_ok} η_eml_m · snapped_expr_m(x)
soft_eml_part(x)   = Σ_{m: type=EML, not snap_ok} η_eml_m · h_m(x)
dt_part(x)         = Σ_{m: type=DT} η_DT · h_m(x)
F_M(x)             = F_0 + formula_part(x) + soft_eml_part(x) + dt_part(x)
```

where `η_eml_m` is the per-round learned EML step size from Section 4.3 and `η_DT` is the fixed DT shrinkage.

### 7.2 Formula simplification pipeline

1. **Symbolic sum** of all snapped expressions via sympy.
2. **Algebraic simplification** via `sympy.simplify`. Combines like terms, applies trig/exp identities, reduces rationals.
3. **Constant snapping.** Numerical coefficients within `1e-6` of common values (`π`, `e`, `π/2`, `1/2`, `1`, `2`, small integers) are rounded to the exact symbolic form. Limited whitelist — no general inverse-symbolic-calculator in v1 (too many false positives).
4. **Verification.** The simplified expression is numerically evaluated on test data; compared against the raw `formula_part` output. If MSE > `1e-6`, the rounding step is reverted and the un-simplified form is reported.

### 7.3 Coverage metric and exact-recovery flag

```
coverage = 1 − Var(soft_eml_part + dt_part) / Var(F_M − F_0)
```

If `coverage > 0.99`, the model is marked as an **exact recovery**: a single closed-form formula explains ≥ 99% of the learned signal variance, and the residual ensemble is negligible. The reported summary elevates the formula as the primary result.

### 7.4 Public API surface

```
model = EmlBoostRegressor(...).fit(X, y)

model.predict(X)                       # full ensemble
model.formula_predict(X)               # formula_part only (closed-form output)
model.describe()                       # human-readable summary
model.formula                          # sympy expression (if recovered)
model.residual_model                   # LightGBM-compatible object for DT part
model.history                          # list[RoundRecord]
model.coverage                         # float in [0, 1]
model.is_exact_recovery                # bool
```

`describe()` output shape:

```
EmlBoostRegressor summary
  Total rounds:              52 (48 EML, 4 DT)
  Early-stopped:             yes, at round 52 (patience 15)
  Recovered formula:         3.1416 * x_1**2 + exp(-x_2)
  Formula coverage:          99.7%
  Residual ensemble:         4 DT learners (depth ≤ 3), MSE 0.0031
  Outer val RMSE (final):    0.0456
```

## 8. Scope and package layout

```
eml_boost/
  __init__.py                    # exports EmlBoostRegressor
  ensemble.py                    # main class, fit/predict/describe
  training.py                    # outer boosting loop
  weak_learners/
    base.py                      # WeakLearner protocol + record dataclasses
    eml.py                       # fit_eml_tree entrypoint, EmlTree class
    dt.py                        # fit_dt_stump (LightGBM wrapper)
  symbolic/
    master_formula.py            # parameterized EML trees (PyTorch nn.Module)
    snap.py                      # simplex rounding + active-logit counting
    simplify.py                  # sympy simplification pipeline
    verify.py                    # numerical cross-check of snapped form
  selection.py                   # BIC scoring, tie-breaking
  datasets.py                    # Feynman loader, synthetic generators
  metrics.py                     # coverage, exact-recovery, graceful-degradation plot
  _numerics.py                   # complex128 guards, exp clamp
  tests/
    unit/                        # fast, per-component
    integration/                 # end-to-end on Feynman subset
    benchmarks/                  # slow, nightly
```

### 8.1 Dependencies

- **Core:** `torch>=2.0`, `numpy`, `lightgbm>=4.0`, `sympy>=1.12`, `scikit-learn>=1.3`.
- **Benchmark / data:** `pysr` (baseline), `pmlb`, `openml`.
- **Dev:** `pytest`, `pytest-benchmark`, `ruff`.

## 9. Benchmarks and success criteria

### 9.1 Benchmark suites

| Suite | Regime | Metric |
| --- | --- | --- |
| Feynman SR (100 formulas) | exact recovery | recovery rate, formula MSE, wall-clock |
| Nguyen / Koza SR | exact recovery (classical SR sanity) | recovery rate |
| PMLB regression subset (~55 problems) | messy tabular | test RMSE |
| OpenML-CC18 regression | messy tabular | test RMSE |
| Synthetic calibration (pure EML / pure DT / mixed) | graceful-degradation proof | EML-win fraction vs signal compositionality |

### 9.2 Baselines

- **Exact recovery:** PySR, AI Feynman 2.0, GP-Learner.
- **Messy tabular:** XGBoost, LightGBM, CatBoost.
- **Sanity:** linear regression, random forest.

### 9.3 Must-have success criteria

1. Feynman SR recovery rate ≥ 80% (PySR is ~55–65% at equivalent wall-clock).
2. Average RMSE on PMLB regression within 10% of XGBoost's.
3. Graceful-degradation plot is monotonic and continuous: EML-win-fraction per round is a smooth function of the signal's elementary-function content.

### 9.4 Nice-to-have criteria

4. Wall-clock ≤ 2× PySR on Feynman at equivalent recovery rate.
5. ≥ 3 real-world PMLB/OpenML datasets where `model.describe()` produces a human-meaningful formula.

### 9.5 Kill criteria

- Feynman recovery < 40% after full implementation: depth-3 × 20-restart recipe insufficient; revisit depth policy or optimization approach.
- PMLB average > 30% worse than XGBoost: EML-Boost is net harmful even in mixed regimes; pivot to Approach B (two-phase) or C (EML leaves per DT).

## 10. Testing strategy

- **Unit tests.** Master formula forward and backward passes; snap correctness on synthetic logit configurations; BIC selection rule on toy fits; sympy simplify pipeline on canonical elementary expressions; single weak-learner fit on fixed small data. Run per-commit (target: < 30 seconds).
- **Integration tests.** End-to-end exact recovery on a 5-formula Feynman subset. Run per-commit (target: < 10 minutes).
- **Benchmark tests.** Full Feynman + PMLB + OpenML with all baselines. Nightly. Produces the headline numbers and graceful-degradation plot.

## 11. Open questions

These are known uncertainties whose resolution depends on implementation-stage evidence, not design decisions:

1. **Inner-val fraction.** Default 0.20 is conventional; the tradeoff is selection-rule noise vs training-data loss. May need tuning via the calibration experiment.
2. **Feature-selection correlation vs mutual information.** Correlation is linear and may miss strong nonlinear dependencies on the residual. MI is more expensive per round but more sensitive. Default to correlation; revisit if calibration shows systematic EML miss.
3. **Hardening-phase entropy penalty schedule.** Starting `λ_H` and ramp shape are specified as defaults but not validated. Ablation needed.
4. **Parameterization fidelity (Option A vs Option B).** v1 uses the paper's Option A: each input position is a `(k+2)`-way simplex over `[1, x_1, ..., x_k, f_prev]`, snapped to a single terminal, with per-round learned `η_eml_m` recovering arbitrary scalar coefficients. Option B would replace the feature-selection simplex with a continuous k-vector of feature coefficients — richer per-round capacity but non-zero snapped-expression complexity. If Option A rounds saturate too early on Feynman formulas with non-multiplicative coefficient structure, evaluate Option B in v2.
5. **Learned η_DT.** v1 keeps DT shrinkage fixed at 0.1 for regularization reasons. If ablation shows EML rounds consistently dominate DT rounds (due to their learned-η advantage) in cases where DT should win, revisit: either learn η_DT per round or remove the fixed shrinkage asymmetry.

## 12. References

- Odrzywołek, A. (2026). *All elementary functions from a single binary operator.* arXiv:2603.21852v2.
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Cranmer, M. (2023). *Interpretable machine learning for science with PySR and SymbolicRegression.jl.*
- Udrescu, S.-M., Tegmark, M. (2020). *AI Feynman: a physics-inspired method for symbolic regression.* Science Advances 6(16):eaay2631.
- Schwarz, G. (1978). *Estimating the dimension of a model.* Annals of Statistics 6(2):461–464.
