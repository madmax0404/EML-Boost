# Current state — 2026-04-25 (post-redux, pre-Experiment-15-restart)

## What we're working on

**Experiment 15** — full PMLB regression suite (122 datasets × 5 seeds ×
3 models = 1,830 fits) comparing SplitBoost (Exp-13 defaults) against
matched-capacity XGBoost and LightGBM. Runner:
`experiments/run_experiment15_full_pmlb.py`.

## Progress so far

- **6 datasets / 90 rows already complete** in
  `experiments/experiment15/summary.csv` (preserved across the whole
  redux work). Header + 90 data rows = 91 lines.
- **Run was paused** because the original GPU port left fit time at
  ~155s on the worst-case `1191_BNG_pbc` (1M rows × 18 features),
  projecting 5-15 hours total — too long to run unattended.
- **The redux GPU port** (X-cache + Triton predict + Triton histogram)
  delivered correct kernels but only modest end-to-end gains because
  predict and histogram-split weren't the dominant bottleneck.
- **Profile-driven follow-up** identified `enumerate_depth2_descriptor`
  taking 46% of fit time due to a missed cache. One-line fix +
  `.item()` batching in `_select_leaf_gated` brought 1191_BNG_pbc fit
  time to **~74s**. With the 6 datasets-already-done as a sample, full
  Experiment 15 should finish in 2-3 hours.

## Performance progression on `1191_BNG_pbc` (200 rounds, depth=8)

| Stage                                          | Time | Notes |
|------------------------------------------------|------|-------|
| Pre-port (CPU-bound)                           | 690s | The original baseline |
| Original GPU port (commits c238977..eb24a0c)   | 180s | Per-node H2D eliminated, tensorized predict |
| Post T1 — X-cache across boost loop            | 156s | `_fit_xy_gpu`, `_predict_x_gpu` |
| Post T2 — Triton predict kernel                | 153s | 239× per-call but predict was small share |
| Post T3 — Triton histogram-split kernel        | 143s | 2.5× per-call, ~7% e2e |
| **Post descriptor-cache + .item() batching**   | **74s** | The big win |
| **Total speedup vs pre-port**                  | **9.3×** ||

## Tests

- 94 unit tests pass.
- 1 pre-existing unrelated failure:
  `tests/unit/test_eml_weak_learner.py::test_fit_recovers_simple_formula`
  (in `fit_eml_tree`, not in any of the changed modules). Leave alone.
- Equivalence tests for both new Triton kernels
  (`test_predict_triton_matches_torch`,
  `test_histogram_split_triton_matches_torch`) verify Triton path
  actually executes (no silent fallback) within float32 tolerance.

## Recent commits (most recent first)

```
2e77881 chore: remove orphaned imports after descriptor cache fix
c9e3264 fix: cache enumerate_depth2_descriptor + feature mask in
        _sample_descriptors; batch D2H syncs in _select_leaf_gated
421fec1 test: tighten GPU speedup threshold after Triton port
e3ee5a6 fix: validate n_bins power-of-2 in Triton histogram-split +
        rephrase comment
27f8ea4 feat: Triton kernel for histogram-based best-split-finding
a4df96d fix: broaden Triton predict fallback exception + warn-once on
        first fallback
c0b183a feat: Triton kernel for whole-tree GPU prediction
529bb55 fix: use population std in _fit_xy_gpu for parity with numpy path
c610c3a feat: cache X on GPU across the boost loop (X-cache optimization)
f90646a plan: SplitBoost GPU port redux
d3ea969 spec: SplitBoost GPU port redux
```

## Spec & plan

- Spec: `docs/superpowers/specs/2026-04-25-splitboost-gpu-port-redux-design.md`
- Plan: `docs/superpowers/plans/2026-04-25-splitboost-gpu-port-redux.md`
- The descriptor-cache fix was profile-driven and not from the original
  plan; covered post-hoc by `c9e3264`'s commit message.

## Pending decision (this is where the user paused)

User asked for a checkpoint before restarting Experiment 15. Three
options were on the table:

1. **Kick off Experiment 15 in the background now.** Resume-from-
   checkpoint preserves the 6 datasets / 90 rows already done. Notify
   when complete. Estimated ~2-3 hours.
2. **Run a 5-seed sanity bench on 2-3 representative non-1191_BNG_pbc
   datasets first** to confirm the speedup holds outside the worst-case
   profile target. ~5 min, then start Experiment 15.
3. **Hold off** and do something else first.

Pick one when resuming; ping the assistant with the option number.

## Active task IDs

- #61: Redux T4 — validate combined speedup + restart Experiment 15
  (in_progress; Steps 1-3 done, Step 4 paused on user decision)
- #62: Redux Tx — descriptor cache fix + .item() batching (completed)
- #58, #59, #60: Redux T1, T2, T3 (completed)

## How to resume after reboot

1. `cd /home/max1024/Workspaces/company/new-gbdt-angle`
2. Verify state: `git log --oneline -3` should show `2e77881` at HEAD.
3. Quick sanity: `uv run pytest tests/unit/ -q` should report
   `94 passed, 1 failed`.
4. Read this file and `experiments/workflow.md`.
5. Tell the assistant which option (1, 2, or 3) you want for restarting
   Experiment 15.

## Files touched in the redux

```
eml_boost/tree_split/_predict_triton.py     (NEW — Triton predict kernel)
eml_boost/tree_split/_gpu_split_triton.py   (NEW — Triton histogram kernel)
eml_boost/tree_split/tree.py                (X-cache, predict dispatcher,
                                              descriptor-cache fix, batching)
eml_boost/tree_split/ensemble.py            (boost-loop GPU residency)
eml_boost/tree_split/_gpu_split.py          (rename + dispatcher)
tests/unit/test_eml_split_tree.py           (3 new tests + threshold tighten)
tests/unit/test_eml_split_boost.py          (1 new test)
docs/superpowers/specs/2026-04-25-splitboost-gpu-port-redux-design.md
docs/superpowers/plans/2026-04-25-splitboost-gpu-port-redux.md
```

The pre-existing baseline files are unchanged in behavior on the CPU
path; all GPU changes preserve correctness within float32 tolerance.

## Profile artifacts (not tracked in git)

`profile_redux/` contains `cum.txt`, `tot.txt`, `profile.pstats`,
`run_profile.py`, and `run.log` from the cProfile-instrumented 50-round
fit on `1191_BNG_pbc` that drove the descriptor-cache fix. Useful as
reference if a new bottleneck appears later.
