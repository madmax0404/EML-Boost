# Experiment 19: Level-wise growth engine — RMSE parity + fit-time on OpenML-CTR23

**Date:** 2026-07-12
**Commits:** `fa3c06a` (Task 2A: deterministic shared histogram core) / `46ebc29`
(Task 4: Stage 1 batched leaves default-on) / `d2e1fcb` (Task 8: level-wise engine)
/ `3ddeb65` (Task 9: `tree_growth` wired through the ensemble) / `435a32d` (Task 10:
same-seed determinism gate) / `5b76d05`..`8f5b4bc` (Task 14: launch-consolidation pass)
/ `71a1616` (Task 12: parity runner) / `12aa25a` (Task 13: parity results)
**Runtime:** idle RTX 3090, post-Task-14 HEAD (`12aa25a`); SB suite fit-time 167 s
(Exp-18: 691 s), inside the spec's ~15-25 min post-speedup wall-clock estimate
**Scripts:** `experiments/run_experiment19_levelwise_parity.py`
**Spec:** `docs/superpowers/specs/2026-07-11-levelwise-growth-design.md`
**Plan:** `docs/superpowers/plans/2026-07-11-levelwise-growth.md` (Amendments 1 and 2)

## What the experiment was about

Experiment 18 established the reportable headline — OpenML-CTR23, matched
hyperparameters, 76.5% wins / 0% catastrophic / 0.987 median vs XGBoost — but left one
axis explicitly out of scope: **speed**. SplitBoost fit 20-105× slower than matched
XGBoost across the suite (per-dataset median ~59×; suite totals 691 s vs 12 s), because
its GPU growth engine visited ~500 tree nodes per round sequentially from Python, each
visit launching ~25 tiny CUDA ops and blocking on 2-4 device syncs to make one scalar
decision. The April optimization campaign had made each per-node operation faster but
kept the per-node execution structure; the level-wise plan removes the structure itself.

Experiment 19 is the **validation run** for that rebuild, and it carries a dual focus,
one per success criterion in the spec:

- **(a) RMSE parity.** The new engine draws its EML descriptors in breadth-first (BFS)
  order instead of the old depth-first visit order — an accepted, distribution-neutral
  semantic change that means Exp-18's per-dataset numbers do not reproduce bit-for-bit.
  The parity gate proves the *headline* is unmoved: win rate within ±5 pp of 76.5%,
  median ratio within ±0.01 of 0.987, zero catastrophic, no per-dataset delta beyond
  seed noise.
- **(b) Fit-time transformation.** The spec's goal is CTR23 suite-total SB fit time
  **≤ 10× matched XGBoost** (from ~60×), measured by re-running the Exp-18 protocol with
  all three models fresh in one session.

The run also delivers a **first for this project: same-seed determinism.** The old
node-wise engine was run-to-run nondeterministic — `gpu_histogram_split` accumulated
float32 histogram sums in atomic (nondeterministic) order, and 1-ULP gain wobble flipped
near-tied split argmaxes and cascaded; two same-seed fits of the *unmodified* code
differed on 419/500 predictions (max 5%). The new fixed-point integer-histogram core
(Amendment 1, `fa3c06a`), shared by both engines, makes two same-seed fits byte-identical.

The engine landed in two independently-tested stages, per the spec:

- **Stage 1 — batched leaf fitting** (`46ebc29`). Growth emits `_PendingLeaf`
  placeholders; a single post-growth `_finalize_leaves()` pass fits all leaves at once.
  Tree *structure* is bit-identical to the node-wise path; leaf params match within
  float32 reduction-order tolerance. The per-leaf `_fit_leaf` is retained as the test
  oracle.
- **Stage 2 — level-wise split growth** (`d2e1fcb`, `3ddeb65`), behind a
  `tree_growth="levelwise"` / `"nodewise"` flag. The tree grows breadth-first, all
  frontier nodes batched per level. Per-node uniform binning, the gain formula (incl.
  `leaf_l2`), `min_samples_leaf` legality, and leaf OLS/gate/cap semantics are preserved
  exactly. The only accepted behavioral change is the BFS RNG draw order. With EML
  disabled (`n_eml_candidates=0, k_leaf_eml=0`) the RNG is never consumed and levelwise
  reproduces nodewise structure exactly — the no-EML structural oracle isolates
  batching-math correctness from the RNG change.

This report's run used `tree_growth="levelwise"` **explicitly**; the library-default flip
lands in the commit *after* this report (see Action taken).

## Configuration

Identical protocol to Experiment 18 — same 35-dataset OpenML-CTR23 suite, seeds 0-4,
80/20 shuffle-split, matched XGBoost / LightGBM, early stopping with patience=15 on a 15%
inner-val hold-out. The only intended change is SplitBoost's growth engine.

### Matched hyperparameters across all three algorithms

| axis | SplitBoost | XGBoost | LightGBM |
|---|---|---|---|
| leaf-floor | `min_samples_leaf=1` (library default) | `min_child_weight=1` | `min_data_in_leaf=1` |
| L2 leaf-weight | `leaf_l2=1.0` (library default) | `reg_lambda=1.0` | `reg_lambda=1.0` |
| early stopping | `patience=15, val_fraction=0.15` | `early_stopping_rounds=15` + 15% inner-val | `early_stopping(15)` + 15% inner-val |

```
# SplitBoost — Exp-19 config (Exp-18 config + explicit tree_growth)
tree_growth         = "levelwise"   # <-- the only change from Exp-18
max_rounds          = 200
max_depth           = 8
patience            = 15
inner_val_fraction  = 0.15
learning_rate       = 0.1
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 1             # library default
leaf_l2             = 1.0           # library default
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2, 3, 4]
```

Mean SplitBoost round count was 129.6 (Exp-18: 128.4) — the BFS engine and early
stopping converge to essentially the same tree budget, so the fit-time deltas below are a
per-round cost result, not a fewer-rounds artifact.

## Coverage

- **Datasets attempted:** 35 (full OpenML-CTR23 suite)
- **Datasets with full 5-seed × 3-model coverage:** 34
- **Fetch failures (excluded from ratios):** 1 — `red_wine`, the same
  `IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed`
  at the OpenML data-fetch stage seen in Exp 18. This is now the **third consecutive
  occurrence** (Exp 17 flagged the OpenML payload-shape edge case, Exp 18 hit it, Exp 19
  hits it again). It is an OpenML data-integrity issue, not a SB bug. Recorded in
  `failures.json`.

## Headline results (dual gate: RMSE parity + timing)

### (a) RMSE parity — SplitBoost / matched-XGBoost — ALL GATES PASS

| metric | Exp 19 (levelwise) | Exp 18 (nodewise) | gate | verdict |
|---|---|---|---|---|
| Outright wins (ratio < 1.00) | **25/34 (73.5%)** | 26/34 (76.5%) | within ±5 pp of 76.5% | **PASS** (−3.0 pp) |
| Median ratio | **0.987** | 0.987 | within ±0.01 of 0.987 | **PASS** (Δ 0.000) |
| Catastrophic (ratio > 2.0) | **0/34 (0%)** | 0/34 (0%) | 0 | **PASS** |
| Per-dataset \|Δratio\| beyond 2× seed noise | **0/34** | — | investigate any | **PASS** (none) |
| Within 10% of XGBoost | **33/34 (97.1%)** | 31/34 (91.2%) | — | (+5.9 pp) |
| Within 5% of XGBoost | **31/34 (91.2%)** | 31/34 (91.2%) | — | (unchanged) |
| Mean ratio | **0.977** | 0.979 | — | (−0.002) |
| P25 / P75 | **0.933 / 1.001** | 0.931 / 0.999 | — | |
| Min / Max ratio | **0.723 / 1.720** | 0.720 / 1.706 | — | |

Every RMSE parity gate passes. The win rate moved −3.0 pp (well inside the ±5 pp band),
the median is identical to three decimals, the catastrophic regime stays empty, and no
single dataset moved beyond twice its 5-seed noise envelope. The within-10% rate actually
*improved* by 5.9 pp (91.2% → 97.1%): under the BFS engine `solar_flare` (1.098) and
`forest_fires` (1.089) landed just below the 10% line instead of just above it, which is
exactly the kind of ±seed-noise reshuffle the parity framing anticipates. The aggregate
shape of the Exp-18 headline is preserved.

### (b) Timing — SplitBoost vs matched-XGBoost — GATE UNMET

| metric | Exp 19 (levelwise) | Exp 18 (nodewise) | goal | verdict |
|---|---|---|---|---|
| SB suite fit-time total | **167 s** | 691 s | — | |
| XGB suite fit-time total | 12 s | 12 s | — | |
| **SB / XGB suite ratio** | **14.17×** | 58.7× | **≤ 10×** | **UNMET** |
| SB suite speedup (Exp-18 → Exp-19) | **4.14×** | — | — | |
| Per-dataset SB/XGB timing (median) | **12.4×** | 58.5× | — | |
| Per-dataset SB speedup (median) | **4.1×** | — | — | |
| Per-dataset SB speedup (range) | **1.55× – 8.37×** | — | — | |
| LGB suite fit-time total | 26 s | 26 s | — | |

The engine delivered a genuine **4.1× suite-wide fit-time reduction** and pulled the
per-dataset SB/XGB gap down from a ~59× median to ~12×. But the spec's headline goal —
≤ 10× XGBoost suite-total — is **not met**: SB lands at 14.17×. The gap is a factor of
~1.4× short, and it is *structural*, not a tuning miss (see Methodological caveats). The
timing gate is documented here as unmet; the accuracy gates are the load-bearing result
of the rebuild, and they pass cleanly.

The **headline plot** (`openml_rmse.png`) renders the RMSE ratios as sorted bars with a
histogram overlay of the per-dataset distribution.

## Per-dataset fit time (seconds, mean across 5 seeds)

Top 8 datasets by SplitBoost-Exp-19 fit time, plus the 4 smallest. `sp` = SB
Exp-18→Exp-19 speedup; `SB/XGB` = the Exp-19 per-dataset timing ratio.

| dataset | SB e18 | SB e19 | XGB e19 | LGB e19 | sp | SB/XGB |
|---|---:|---:|---:|---:|---:|---:|
| wave_energy | 82.0 | 13.83 | 1.15 | 2.23 | 5.9× | 12.0× |
| video_transcoding | 56.1 | 13.49 | 0.47 | 1.39 | 4.2× | 28.4× |
| sarcos | 46.9 | 11.35 | 0.77 | 1.53 | 4.1× | 14.8× |
| physiochemical_protein | 39.7 | 11.12 | 0.53 | 1.24 | 3.6× | 20.9× |
| diamonds | 29.2 | 10.21 | 0.44 | 1.20 | 2.9× | 23.0× |
| superconductivity | 25.4 | 9.39 | 0.81 | 1.49 | 2.7× | 11.5× |
| kings_county | 28.3 | 8.00 | 0.50 | 1.06 | 3.5× | 15.9× |
| naval_propulsion_plant | 28.0 | 6.77 | 0.34 | 1.15 | 4.1× | 20.0× |
| … | | | | | | |
| geographical_origin_of_music | 5.3 | 1.01 | 0.25 | 0.36 | 5.2× | 4.0× |
| student_performance_por | 3.6 | 0.94 | 0.08 | 0.28 | 3.9× | 12.2× |
| solar_flare | 2.1 | 0.55 | 0.03 | 0.21 | 3.9× | 17.5× |
| forest_fires | 0.8 | 0.46 | 0.04 | 0.22 | 1.6× | 12.4× |

The speedup is largest on the big-n datasets where the old engine's per-node overhead
compounded most (`wave_energy` 5.9×, `superconductivity`/`diamonds` grow the raw
seconds but the ratio to XGB stays 12-28×). The smallest datasets show the floor of the
new engine: `forest_fires` (n≈400) only sped up 1.6×, because at ~16 rounds × 9 levels
the fixed per-level Python/CUDA dispatch cost dominates and there is little per-node work
to batch away. Even at 12× on `forest_fires`, XGB's 0.04 s is so small that the ratio is
dominated by SB's launch overhead, not by compute — the same dispatch-bound floor that
keeps the suite total above 10×.

## Distribution of ratios (SplitBoost / XGBoost)

| ratio band | meaning | Exp 19 (levelwise) | Exp 18 (nodewise) |
|---|---|---|---|
| `< 0.85` | deep win (>15% better) | 3 | 2 |
| `0.85 - 0.95` | clear win (5-15% better) | 9 | 10 |
| `0.95 - 1.00` | narrow win (0-5% better) | 13 | 14 |
| `1.00 - 1.05` | narrow loss (0-5% worse) | 6 | 5 |
| `1.05 - 1.10` | loss (5-10% worse) | 2 | 0 |
| `1.10 - 2.00` | clear loss (10-100% worse) | 1 | 3 |
| `>= 2.00` | catastrophic | **0** | 0 |

The distribution is the Exp-18 distribution jittered by seed noise. The modal band is
still the narrow-win band (0.95-1.00, 13 datasets); combined with the narrow-loss band it
still puts ~56% of datasets within ±5% of XGB. The one visible shift is the tail:
Exp-18's empty 1.05-1.10 band now holds 2 datasets (`solar_flare`, `forest_fires`) and
the 1.10-2.00 band drops from 3 to 1 (`brazilian_houses` only) — the same two borderline
losses crossed from just-above to just-below the 10% line, which is why within-10% rose.
No dataset approaches the catastrophic regime.

## Top 10 wins (lowest ratios, Exp 19)

| dataset | ratio | SB RMSE (std) | XGB RMSE (std) |
|---|---|---|---|
| fps_benchmark | 0.723 | 10.81 (2.68) | 14.95 (2.84) |
| naval_propulsion_plant | 0.731 | 0.000880 (5.9e-5) | 0.001204 (9.3e-5) |
| socmob | 0.823 | 15.60 (4.90) | 18.97 (2.68) |
| cars | 0.856 | 0.466 (0.061) | 0.544 (0.020) |
| cpu_activity | 0.879 | 2.201 (0.076) | 2.503 (0.390) |
| energy_efficiency | 0.882 | 0.451 (0.031) | 0.511 (0.144) |
| auction_verification | 0.895 | 664.9 (194) | 743.0 (228) |
| kin8nm | 0.932 | 0.1148 (0.0018) | 0.1231 (0.0027) |
| QSAR_fish_toxicity | 0.933 | 0.914 (0.071) | 0.980 (0.073) |
| student_performance_por | 0.933 | 2.887 (0.326) | 3.094 (0.313) |

The win roster is the same architectural story as Exp 18: medium-to-large real-world
tasks where the EML mechanism extracts smooth sub-signals XGB's piecewise-constant leaves
miss — `fps_benchmark`, `naval_propulsion_plant`, `cpu_activity`, `cars`. The individual
ratios wobble at the second decimal from the BFS RNG reorder (`fps_benchmark` 0.720 →
0.723; `naval_propulsion_plant` 0.781 → 0.731 — its widest single swing, still a deep
win), and `student_performance_por` (0.933) edges into the top-10 where `Moneyball`
(0.918 in Exp 18, now ~0.94) edged out. None of these movements is beyond the per-dataset
seed-noise envelope; the gate check flagged 0/34.

## Top losses (highest ratios, Exp 19)

| dataset | ratio | SB RMSE (std) | XGB RMSE (std) | notes |
|---|---|---|---|---|
| brazilian_houses | 1.720 | 7398 (4560) | 4301 (2670) | High seed variance: SB std/mean = 0.62; per-seed RMSEs span 2504–15201 |
| solar_flare | 1.098 | 0.766 (0.052) | 0.698 (0.061) | Borderline; now *within* 10% (was 1.105) |
| forest_fires | 1.089 | 81.2 (22.6) | 74.58 (26.0) | Borderline; now *within* 10% (was 1.108) |
| wave_energy | 1.020 | 21456 (167) | 21041 (186) | Tied within 2% |
| concrete_compressive_strength | 1.005 | 5.029 (0.436) | 5.003 (0.415) | Tied |
| geographical_origin_of_music | 1.005 | 17.67 (0.83) | 17.58 (0.91) | Tied |
| pumadyn32nh | 1.003 | 0.02187 (3.3e-4) | 0.02179 (3.7e-4) | Tied |
| health_insurance | 1.002 | 14.64 (0.06) | 14.62 (0.02) | Loss boundary; effectively tied |
| miami_housing | 1.001 | 88305 (5550) | 88180 (6950) | Tied |

The only **structural loss greater than 10%** is `brazilian_houses`, unchanged from
Exp 18 as both the sole >10% loss and the driver of the max ratio. Its per-seed SB RMSEs
span 2504–15201 with std=4560 ≈ 62% of the mean — this is a high-variance dataset where
seed luck dominates, not a clean architectural loss (XGB shows the same ~62% relative
variance). The Exp-18 verdict stands: a targeted 20-seed re-run would likely shrink it
toward parity. The two borderline losses `solar_flare` and `forest_fires` are now *inside*
the 10% band under the BFS engine. Everything from `wave_energy` down is a statistical tie
counted as a loss by the mean-ratio convention; readers should treat ratios in 1.00-1.02
as parity outcomes.

## vs LightGBM (matched)

| metric | Exp 19 (levelwise) | Exp 18 (nodewise) |
|---|---|---|
| Outright wins (SB < LGB) | **24/34 (70.6%)** | 26/34 (76.5%) |
| Within 10% of LGB | 32/34 (94.1%) | 32/34 (94.1%) |
| Within 5% of LGB | 31/34 (91.2%) | 31/34 (91.2%) |
| Mean ratio (SB / LGB) | 0.976 | 0.980 |
| Median ratio (SB / LGB) | **0.988** | 0.987 |
| Min / Max ratio | 0.725 / 1.119 | 0.722 / 1.126 |

The SB-vs-LGB story tracks the SB-vs-XGB story, as in Exp 18. Wins moved −5.9 pp (24 vs
26) but the median holds at ~0.988 and within-10%/within-5% are identical to Exp 18. The
largest SB-vs-LGB win is still `fps_benchmark` (0.725); the largest loss is still
`auction_verification` (1.119) — a dataset SB beats XGB on (0.895) but loses to LGB on,
the standout where LGB's leaf-wise growth policy remains an unmatched algorithmic axis.

## Comparison to Exp 18 (nodewise)

The comparison here is *within one dataset universe* — the same CTR23 datasets, the same
seeds, the same matched baselines. The only moving part is SplitBoost's growth engine, so
the deltas isolate the engine change.

| dimension | Exp 18 (nodewise) | Exp 19 (levelwise) | shift |
|---|---|---|---|
| Outright wins vs XGB | 76.5% | 73.5% | −3.0 pp (within gate) |
| Median ratio vs XGB | 0.987 | 0.987 | 0.000 |
| Within 10% vs XGB | 91.2% | 97.1% | +5.9 pp |
| Catastrophic | 0% | 0% | unchanged |
| Max loss ratio | 1.706 | 1.720 | +0.014 (same dataset) |
| SB suite fit-time | 691 s | 167 s | **4.1× faster** |
| SB/XGB suite ratio | 58.7× | 14.17× | **4.1× closer to XGB** |
| Same-seed determinism | no (float atomics) | **yes (fixed-point)** | first for project |

The engine is **statistically equivalent on accuracy and strictly better on every other
axis**: 4.1× faster, deterministic, and no worse in the tail. That dominance is the basis
for the default-flip decision (Action taken).

## What Exp 19 actually shows

**The level-wise engine preserves the Exp-18 accuracy headline.** All four RMSE parity
gates pass: 73.5% wins (−3.0 pp, inside ±5 pp), 0.987 median (identical), 0 catastrophic,
0/34 datasets beyond seed noise. The BFS RNG reorder moves individual per-dataset numbers
at the second decimal but does not move the aggregate. The rebuild is accuracy-safe.

**The engine is 4.1× faster suite-wide and deterministic.** SB suite fit-time dropped
691 s → 167 s; the per-dataset SB/XGB gap fell from ~59× to ~12×; and — a first for the
project — two same-seed fits now produce byte-identical predictions, because the shared
fixed-point integer histogram (Amendment 1) removed the float-atomic accumulation that
made the old engine nondeterministic.

**Level-wise strictly dominates node-wise.** Same accuracy distribution, 4× the speed,
deterministic, with no new tail risk. There is no axis on which the old node-wise engine
is preferable — which is why it is being demoted to test-oracle status rather than kept
as a co-equal option.

**But the ≤10× timing goal is not met.** SB lands at 14.17× XGB suite-total, ~1.4× short
of the spec goal. The remaining cost is CPU-dispatch overhead intrinsic to a per-level
engine at CTR23's small n (details below). This is logged as an open goal, not a passed
gate.

## What Exp 19 does NOT show

- **No baseline tuning beyond the 3 matched axes** (unchanged from Exp 18). SB, XGB, and
  LGB each have further hyperparameters (subsample, colsample, learning-rate schedule,
  per-dataset depth) that were not swept. An Optuna-tuned per-dataset comparison would
  represent each algorithm's true ceiling; expect SB's lead to shrink on medium/large
  real-world datasets where XGB and LGB benefit most from tuning. The matched-default
  headline is the defensible baseline, not the strong-baseline test.
- **The timing gate is unmet.** 14.17× vs the spec's ≤10×. The engine did not reach the
  headline speed goal; it reached a 4.1× improvement that leaves a ~1.4× structural gap.
  This report does not claim the speed goal was achieved.
- **Cross-version bit-stability caveat.** A first Exp-19 run (pre-Task-14 code, also idle
  box) measured the same win rate (73.5%) and 14.6× timing but median 0.980 and
  within-10% 94.1%; the committed run is statistically equivalent but NOT bit-identical to
  it — Task 14's Item-2 buffer-layout change legitimately reordered float accumulation in
  the correlation pass, flipping EML top-k near-ties. Same-seed determinism holds WITHIN a
  code version, not across versions. The first run's raw CSV was deleted before the re-run
  (a process mistake); only its aggregate numbers survive in the execution ledger. The
  determinism guarantee is therefore: *identical code + identical seed → identical bits*;
  it does not span optimization commits that reshuffle float reductions, and the parity
  gate — not bit-equality — is what certifies the headline across code versions.
- **Single 80/20 shuffle-split per seed, 5 seeds, no K-fold, no significance test**
  (unchanged from Exp 18). Aggregate headline statistics survive 5-seed sampling;
  per-dataset loss claims (`brazilian_houses` especially) do not, and would benefit from
  a 20-seed paired re-validation.
- **SB's EML mechanism and LGB's leaf-wise growth remain unmatched** — the intentional
  architectural differentiators, not confounds to remove.

## Methodological caveats specific to Exp 19

**Why the timing gate was missed (measured, not assumed).** Task 14's profiling
(`task-14-report.md`) traced the residual cost precisely: `grow_levelwise` is
**CPU-dispatch-bound**, not GPU-bound. Its cumulative CPU-dispatch time (~24 ms/round)
≈ its wall time (~24.7 ms/round) — across ~9 tree levels the CPU issues ~3000 tiny torch
ops/round and stays saturated, so each GPU kernel finishes before the CPU can queue the
next and kernel time hides under dispatch. The four sanctioned launch-consolidation items
each removed only tens of ops out of ~3000 (<1%), moving the per-round cost 26.6 → 25.4
ms — wall-neutral by construction. Reaching ≤10× requires **dispatch elimination** (CUDA
graphs, `torch.compile`, or fused mega-kernels that collapse a level's ~15-20 histogram
ops into one launch), not further micro-fusion. That is a new-kernel effort with its own
bit-exactness budget; it was logged as future work and deliberately **not attempted** in
this experiment (Task 14 Item 5, STOP-first).

**The 4.1× is real and load-bearing, the ≤10× is an estimate that missed.** The spec
flagged "≤10× is an estimate; if torch-first lands at ~15×, promote per-level hot spots to
Triton before reconsidering the goal." Torch-first landed at 14.17×. The recommendation
carries forward as the named next phase.

**OpenML data hygiene** (unchanged from Exp 18). `red_wine` failed the OpenML fetch with
the same `IndexError` for the third experiment running — an OpenML payload-shape edge
case, not a SB fault. Categorical features are one-hot encoded (`pd.get_dummies`) with
NaN-row drops; XGB/LGB native categorical handling is not used, so all three algorithms
receive the same float matrix.

**Inner-val split same-seed-different-RNG** (unchanged from Exp 18). XGB/LGB use
`sklearn.train_test_split` for the early-stopping hold-out; SB uses its own RNG with the
same seed value. Negligible on large datasets; a per-seed variance contributor on the
small-n cluster (`brazilian_houses`, `solar_flare`, `forest_fires`).

## Action taken

- **Default flip approved (lands next commit).** With all RMSE parity gates passing and
  the level-wise engine strictly dominating node-wise (4.1× faster, deterministic,
  statistically equivalent accuracy), the user approved flipping the library default
  `tree_growth` to `"levelwise"` **with the timing gate documented as unmet**. The
  rationale is explicit: the flip carries no accuracy risk (excluded by the parity gates),
  and the only unmet criterion — suite timing — is a strict improvement (14× beats the old
  59×) whose remaining gap is dispatch overhead, not a correctness or quality concern.
  This report's run used `tree_growth="levelwise"` explicitly; the one-line default flip
  is the commit *after* this report.
- **Node-wise retained as the test oracle.** Not deleted — it backs the no-EML structural
  oracle and the Stage-1 A/B tests. Deletion is deferred to a later cleanup.
- **Determinism is now a project invariant.** The same-seed byte-identical guarantee is
  covered by a CI gate (`435a32d`). Note the scope: it holds within a code version.
  Consequently the per-seed values of Exp 15-18 (old nondeterministic engine) were never
  exactly re-runnable — their aggregate headlines stand, but their per-seed CSVs are one
  draw among many. Exp 19 is the first re-runnable experiment.

## Next experiments

- **Dispatch-elimination phase (named follow-up to close the timing gate).** The ≤10×
  goal is reachable only by removing per-level CPU dispatch, not by more micro-fusion.
  Candidate levers, in the spec's own priority order: a Triton mega-kernel folding the
  histogram pipeline (`bin_idx` + 3 scatters + cumsum) into one launch (the single largest
  dispatch consumer, ~29% of round CPU time); CUDA graph capture of the per-level op
  sequence; or `torch.compile` on the growth loop. Each must keep the fixed-point
  determinism and no-EML structural oracles `torch.equal`-green — a real bit-exactness
  budget. This is the direct continuation of Experiment 19's unmet gate.

- **Optuna-tuned baselines** (inherited from Exp 18). Re-run CTR23 with per-dataset
  Optuna-tuned XGB and LGB for an honest strong-baseline comparison; expect SB's median
  lead to compress on medium/large datasets.

- **Grinsztajn-2022 cross-suite validation** (inherited from Exp 18). The ~25-dataset
  "trees vs DL" tabular benchmark as a third-suite cross-check on whether the ≥73% wins /
  0% catastrophic / ~0.987 median story is distribution-robust.

- **20-seed re-validation of `brazilian_houses`** (inherited from Exp 18). Now cheaper
  than ever on the 4× faster deterministic engine — a targeted 20-seed run would settle
  whether the sole >10% loss is a real algorithmic loss or a high-variance noise artifact.
