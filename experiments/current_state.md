# Current state — 2026-07-12 (post-Experiment-19, levelwise engine is the default)

## What just happened

The level-wise growth engine effort (spec
`docs/superpowers/specs/2026-07-11-levelwise-growth-design.md`, plan + 2 amendments in
`docs/superpowers/plans/2026-07-11-levelwise-growth.md`) is complete: 31 commits,
`f98c541..171281d`, all task-reviewed, final whole-branch review passed with its fix
wave landed.

- **`tree_growth="levelwise"` is the library default** (both regressors). The old
  recursive engine remains as `tree_growth="nodewise"` — the reference/oracle, GPU-only
  distinction unchanged (CPU fallback ignores the flag).
- **Exp 19** (`experiments/experiment19/report.md`) is the validation record:
  RMSE parity vs Exp-18 passed every gate (win rate 73.5%, median ratio 0.987,
  0 catastrophic, 0/34 datasets beyond seed noise); CTR23 suite fit time 691s → 167s
  (4.1×); SB/XGB suite ratio 59.6× → **14.17×** — the spec's ≤10× goal is **UNMET**
  (documented, user-accepted): the level loop is CPU-dispatch-bound (~3000 tiny torch
  ops/round; launch consolidation was wall-neutral).
- **Same-seed determinism now holds** (a first): fixed-point integer histograms with
  per-segment scale + int64 cumsum, shared by both engines; scoped deterministic
  algorithms in the batched leaf finalize. The pre-existing engine was nondeterministic
  (float-atomic hist), meaning Exp-15..18 per-seed values were never exactly
  re-runnable; their aggregates stand.

## Tests

- 138 passed / 1 pre-existing failure (`test_fit_recovers_simple_formula` — old
  weak-learner module, red since April; standing instruction: leave alone).
- Key gates in `tests/unit/test_levelwise.py`: bit-exact no-EML structural oracle
  (levelwise ≡ nodewise), EML invariants, same-seed determinism (incl. adversarial
  duplicated/correlated-feature fixture), 3× speed gate (measures ~6.3×).

## Known follow-up queue (none blocking)

1. **Dispatch-elimination phase** — the named path to ≤10×: CUDA-graph capture /
   torch.compile of the level loop, or mega-kernel consolidation. Needs its own spec.
2. **Fixed-point integer corr** (mirror `_multinode_hist`) if the determinism gate ever
   flakes — the float `index_add_` corr top-k is the one theoretically-uncovered path
   (empirically stable across 13+ runs; scoped-deterministic alternative measured at 2×
   fit cost and rejected — see `171281d` commit body).
3. Exp-18 report's inherited proposals: Optuna-tuned baselines (Exp 20?), Grinsztajn
   suite cross-check, 20-seed `brazilian_houses` re-validation.
4. `red_wine` OpenML fetch is deterministically broken (IndexError; Exp 18 + both Exp-19
   runs) — consider reporting upstream or pinning a cached copy.
5. Cross-version RMSE bit-stability does not hold (float near-tie reshuffles under code
   changes) — same-seed determinism is per-version. Caveated in the Exp-19 report.

## How to resume after reboot

1. `git log --oneline -3` should show `171281d` at HEAD.
2. `uv run pytest tests/unit/ -q` → 138 passed, 1 pre-existing failure.
3. Execution ledger (task-by-task history + carried findings): `.superpowers/sdd/progress.md`.
