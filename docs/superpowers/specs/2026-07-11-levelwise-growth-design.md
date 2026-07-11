# SplitBoost Level-wise Growth Engine (batched leaves + level-wise splits)

**Date:** 2026-07-11
**Context:** After the April optimization campaign (GPU port, Triton predict/histogram kernels, X-cache, descriptor cache — 9.3× combined on the 1M-row worst case), SplitBoost still fits 20-105× slower than matched XGBoost across the OpenML-CTR23 suite (median 60×; suite totals 691 s vs 12 s, Exp 18 `summary.csv`). Those optimizations made each per-node operation faster but kept the per-node execution structure. This spec removes the structure itself.

## Diagnosis (evidence)

Three independent measurements agree:

1. **Exp-18 timings (34 datasets).** SB per-round cost floors at ~40-100 ms even on n≈400 datasets (`forest_fires`: 45 ms/round vs XGB's 13 ms); per-round cost, not round count, drives the gap.
2. **cProfile on `344_mv`** (32.6k × 10, 50 rounds, 26.5 s): each round grows ~500 nodes (saturated depth-8 tree). Per round: ~1,200 sequential Triton launches, ~2,800 forced syncs (`.item()`/`.cpu()`). `_fit_leaf` = 50% of wall time (13.3 s / 12,450 calls ≈ 1.07 ms/leaf), `_find_best_split_gpu` = 41% (0.87 ms/node) — per-node cost is ~1 ms regardless of node size, while the per-node GPU math is microseconds.
3. **Controlled scaling probe** (synthetic, d=10, depth 8): growing n 64× (2k→128k) grows s/round only 3× (0.195→0.595). Fit time tracks node count, not data size.

**Root cause:** `_grow_gpu` visits ~500 nodes per round sequentially from Python; each visit launches ~25 tiny CUDA ops and blocks on 2-4 device syncs to make a scalar decision. ~1 ms × 500 nodes × 200 rounds ≈ the whole gap. XGBoost/LightGBM process one tree *level* at a time with all nodes batched; that is the fix.

## Goal

- **CTR23 suite-total SB fit time ≤ 10× matched XGBoost** (from ~60×; ~691 s → ≤ ~120 s), measured by re-running the Exp-18 protocol with all three models fresh.
- **Statistical equivalence, not bit-exactness:** RMSE headline (win rate, median ratio, zero-catastrophic) within Exp-18's seed-noise envelope. Bit-level guarantees are scoped per stage below.
- **Same-seed determinism:** two fits with the same seed produce identical trees.

## Non-goals

- **No CPU-pipeline changes.** The no-CUDA fallback paths stay untouched.
- **No hyperparameter or split-math changes.** Gain formulas (incl. `leaf_l2`), per-node uniform binning, min-leaf legality, leaf OLS/gate/cap semantics are preserved exactly.
- **No predict-path changes.** Predict is already batched (Triton).
- **No removal of the node-wise path in this effort.** It remains behind a flag as the test oracle; deletion is a later cleanup.
- **No latency work on datasets beyond CTR23/PMLB scale** (n < 2^24 assert stays).

## Design overview

Two stages, independently landable, testable, benchmarkable:

| stage | what | equivalence guarantee | expected effect (344_mv, 0.53 s/round) |
|---|---|---|---|
| 1 | batched leaf fitting (defer + finalize) | tree structure bit-identical; leaf params within float32 reduction-order tolerance; leaf-type decisions identical | ~2× (≤ ~0.28 s/round) |
| 2 | level-wise split growth behind `tree_growth` flag | statistically equivalent (BFS RNG order); no-EML mode structure-identical to node-wise | ~10× cumulative (≤ ~0.06 s/round) |

---

## Stage 1 — batched leaf fitting

### Problem

`_fit_leaf` runs once per leaf (~250/round), each call doing ~30 kernel launches (two 144-descriptor Triton evaluations, segmented OLS, gate) plus 2-3 batched-but-still-per-leaf `.cpu()` syncs. Even constant-leaf early-outs pay one `.item()` sync each.

### Solution

Growth never fits leaves inline. When a leaf condition triggers, `_grow_gpu` returns a `PendingLeaf` placeholder capturing the leaf's GPU row-index tensor. After the tree skeleton is complete, a single `_finalize_leaves()` pass fits every leaf batched, then patches real `LeafNode`/`EmlLeafNode` objects over the placeholders.

### Batched finalize pipeline

1. Partition pending leaves into *constant-only* (n < `min_samples_leaf_eml`, or `k_leaf_eml<=0`, or the `n − val_sz < min_samples_leaf_eml//2` bail) and *EML-eligible*. Constant values for **all** leaves come from one segmented sum (`index_add_` over `leaf_id`) — zero per-leaf syncs.
2. EML-eligible leaves: concatenate rows in leaf order; build `leaf_id`, offsets.
3. Per-leaf top-k features: segmented two-pass centered Pearson correlation (pass 1 means, pass 2 centered products via `index_add_`), mirroring `_top_features_by_corr_gpu`'s math; batched topk.
4. Per-leaf train/val split: CPU loop drawing each leaf's permutation with the **existing seed rule** (`seed = first row index of the leaf`) — leaf RNG has never consumed the shared tree RNG, which is what makes Stage 1 structure-safe. ~250 tiny numpy permutations ≈ sub-ms, no device syncs.
5. One evaluation of the shared 144-descriptor set over **all** leaf rows: per-row gather of the row's leaf's chosen feature(s), standardize with that feature's **global** mean/std, clamp to [-3, 3], single Triton call on the (144, N_total) grid. Chunk over descriptor blocks when N_total > ~500k rows (memory ceiling ~70 MB per 128k rows at 144 trees).
6. Segmented OLS sufficient statistics per (leaf, tree) via `index_add_` over fit rows; closed-form η/β identical to current code (incl. `leaf_eml_ridge` branch and `leaf_l2` bias shrink); validity mask (feature mask, finiteness, det guard) vectorized over the (L, 144) grid.
7. Vectorized gate: per-leaf constant val-SSE (segmented), per-(leaf,tree) EML val-SSE with cap applied, argmin per leaf, `leaf_eml_gain_threshold` accept/reject.
8. **One** `.cpu()` readback of all per-leaf outputs → Python loop builds node objects and patches placeholders.

`use_stacked_blend=True` keeps the per-leaf reference path (it's non-default and Exp-9-rejected); the batched path covers the gated policy only, and the A/B test enforces that flag routing.

### Guarantees & tests

- Split path untouched ⇒ identical partitions ⇒ identical structure and identical leaf row sets, guaranteed.
- Leaf params: float32 reduction-order tolerance. Leaf-type (gate) decisions must match the per-leaf reference exactly on the A/B test suite; a flip fails the test.
- The per-leaf `_fit_leaf` is retained as the reference implementation; a private toggle selects it in tests only.

---

## Stage 2 — level-wise split growth

### Problem

Even with leaves batched, ~250 split-node visits per round each pay ~0.9 ms of dispatch+sync (top-k corr, candidate eval, histogram, argmax readback, partition), sequentially.

### Solution

New growth engine behind `tree_growth: str` on both `EmlSplitTreeRegressor` and `EmlSplitBoostRegressor`: `"levelwise"` (new) / `"nodewise"` (current, oracle). Default stays `"nodewise"` until the CTR23 parity run passes, then one commit flips it (same rollout pattern as the `leaf_l2` and `min_samples_leaf` default flips).

State per fit: `sample_node` int32 tensor (row → frontier node), plus a small Python-side frontier list (tree position, depth, parent linkage). Per level:

1. **Counts & eligibility** — segmented bincount; one small readback (~L ints) decides split-attempt vs `PendingLeaf` (depth/min-samples gates identical to today's).
2. **Top-k per node** — segmented centered correlation → (L, d) → per-row topk → (L, k).
3. **Descriptor sampling** — CPU RNG, **BFS order** (level ascending, node index ascending within level): L×`n_eml_candidates` draws from the cached valid-descriptor table; one H2D upload per level.
4. **Candidate evaluation** — new Triton kernel variant generalizing `evaluate_trees_torch_per_sample` from 1 to C descriptors per row: `out[c, i] = eval(desc[node_of[i], c], X_topk[i])` → (C, n_active). Per-(node, candidate) finiteness via segmented all-finite reduction; non-finite candidates take gain = −inf (same outcome as today's pre-histogram drop).
5. **Histograms** — per (node, column), columns = d raw + C EML values: segmented per-(node,col) min/max (preserving per-node uniform binning), scatter into (L, d+C, 256) sum/sq/count.
6. **Split decision** — bin cumsums + current gain formula (incl. `leaf_l2`, `min_samples_leaf` legality), argmax per node over (cols × bins).
7. **Level readback** — one `.cpu()` for all L decisions (best col, threshold, gain, EML descriptor + feature subset for EML winners) → Python builds `InternalNode`s, extends frontier.
8. **Partition update** — recompute winning column per split node's rows, `go_left = val <= thr`, one kernel updates `sample_node`.

~20 launches + 2 small syncs per level × 8 levels, replacing ~25 launches + 3 syncs × ~250 nodes.

### Kernel strategy — torch first

Steps 2/5/8 are implemented first as `index_add_`/`scatter_add_` torch code with the same dispatcher-plus-fallback pattern used by the existing Triton kernels. Profile after Stage 2 lands; promote to Triton only what the profile justifies (multi-node histogram is the expected candidate). Step 4 requires the new Triton variant from the start (no efficient torch expression exists for per-row descriptor indirection at C>1).

### Determinism

Atomic float scatter-adds are accumulation-order nondeterministic and would break same-seed reproducibility. Requirements and mechanism:

- **Histogram sums** (drive thresholds — the load-bearing decision): accumulate in fixed-point int64 atomics. Quantize residual contributions with a per-round scale derived from max|y| (residuals are bounded per round); integer adds are associative ⇒ order-independent ⇒ deterministic. Count histograms are integer natively; y² uses the same treatment.
- **Correlation top-k**: float accumulation retained. A ulp wobble can only flip *exact* ties between candidate features (measure-zero on continuous data); any resolution is a legitimate draw. If the determinism test ever flakes here, corr moves to the same fixed-point treatment.
- **Acceptance test:** two same-seed fits → byte-identical predictions, run on multiple datasets in CI.

### RNG & equivalence policy

The tree RNG's descriptor draws move from DFS-visit order to BFS order. Distribution unchanged; realized trees differ. Consequences accepted per the equivalence decision:

- Published Exp-18 per-dataset numbers will not reproduce exactly under `levelwise`.
- The parity run (below) is the gate proving the headline is unmoved.
- With `n_eml_candidates=0, k_leaf_eml=0` the RNG is never consumed ⇒ levelwise must match nodewise structure exactly (thresholds/values within float32 tolerance, since fixed-point histograms shift sums at ulp scale) — this isolates batching-math correctness from the RNG change.

### Memory ceiling (1M-row worst case, d=18, C=10)

Value matrix (n_active × 28 cols) ≈ 112 MB; histograms 128 × 28 × 256 × 3 stats ≈ 11 MB; well within the 3090's 24 GB. CTR23's largest datasets are ~50k rows — trivial.

---

## Testing

1. **No-EML bit-structure oracle** — levelwise vs nodewise, `n_eml_candidates=0, k_leaf_eml=0`, several datasets × seeds: identical structure, thresholds/leaf values within float32 tolerance.
2. **Stage-1 A/B** — batched vs per-leaf reference on real fits (EML enabled): identical structure, identical leaf-type decisions, leaf params within tolerance.
3. **Component oracles** — segmented corr vs `_top_features_by_corr_gpu`; multi-node histogram vs `gpu_histogram_split` per node; batched OLS/gate vs `_fit_leaf` internals; C-descriptor evaluator vs `evaluate_trees_torch_per_sample` looped. Fixed synthetic data, tight tolerances.
4. **Determinism** — same-seed double-fit equality (predictions byte-identical).
5. **Speed regression gate** — extend the existing GPU-speedup test pattern: levelwise per-round time on a fixed synthetic config must beat nodewise by a conservative factor (exact threshold set from measured results, with slack for CI noise).
6. Existing suite (95 tests) keeps passing; nodewise path untouched semantically.

## Benchmarks

- Scaling probe (n = 2k/8k/32k/128k, d=10, depth 8): today 0.19/0.22/0.40/0.60 s/round. Stage-2 target: ≤ ~0.06 s/round at 32k and near-flat curve retained.
- `344_mv` 50-round fit: 26.5 s today → Stage 1 ≤ ~15 s → Stage 2 ≤ ~3 s.
- `1191_BNG_pbc` sanity check post-Stage-2 (big-n regime where real compute dominates; expect smaller relative win, no regression).

## Validation run & rollout

1. Stage 1 lands default-on after its A/B test (structure-exact ⇒ low risk).
2. Stage 2 lands behind `tree_growth="nodewise"` default.
3. **Parity run:** full CTR23, 34 datasets × 5 seeds × {SB-levelwise, XGB, LGB}, all fresh for a same-session timing claim (Exp-18 runner, seeds, and protocol unchanged). Checkpoint with the user before launching (~15-25 min expected post-speedup). Gates:
   - suite-total SB ≤ 10× XGB;
   - win rate / median ratio / zero-catastrophic within Exp-18 seed-noise; per-dataset |Δratio| beyond noise investigated before proceeding.
4. Default flip commit + report (experiment numbering per project convention at run time).
5. Nodewise-path removal deferred to a later cleanup.

## Risks

- **Multi-node histogram + fixed-point determinism** is the riskiest new code. Mitigations: torch-first reference implementations, component oracles, no-EML structural oracle.
- **≤10× is an estimate.** If torch-first lands at ~15×, promote per-level hot spots to Triton before reconsidering the goal.
- **Fixed-point quantization error** shifts bin sums at ~2^-20 relative; thresholds move within histogram approximation error (already O(1/256)). The no-EML oracle's float32 tolerance covers this; if it doesn't, widen scale bits.
- **BFS RNG shift** moves per-dataset numbers within seed noise (accepted; gated by parity run).

## Success criteria

1. CTR23 suite-total SB fit time ≤ 10× XGB under the matched Exp-18 protocol.
2. RMSE headline within Exp-18 noise envelope (win rate ±5 pp, median ratio ±0.01, zero catastrophic).
3. Same-seed determinism test passes.
4. All unit tests pass (existing + new oracles).
5. `tree_growth="levelwise"` is the library default; nodewise retained as oracle.
