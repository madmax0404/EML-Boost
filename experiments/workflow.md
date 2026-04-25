# Experiment workflow

This file describes how experiments are structured in this repo so any
agent picking up the project can follow the same conventions.

## The loop: brainstorm → spec → plan → implement

For every experiment, optimization, or non-trivial change we use the
superpowers skills:

1. **`superpowers:brainstorming`** — clarifying questions, 2-3 approach
   options, present design section by section, get approval after each.
2. **Spec** — write the validated design to
   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` and commit.
3. **`superpowers:writing-plans`** — turn the spec into a step-by-step
   implementation plan at `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
   and commit.
4. **`superpowers:subagent-driven-development`** — execute the plan with
   one fresh subagent per task, two-stage review per task (spec
   compliance, then code quality).

Master is the working branch. The user has explicitly opted out of
worktrees for this project — implementers commit directly on master.

## Naming conventions

- **Specs**: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
  `<topic>` is short kebab-case (e.g. `leaf-cap`,
  `splitboost-gpu-port-redux`).
- **Plans**: `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`. Same
  `<topic>` as the spec — same date is fine; if the plan slips a day,
  rev the date.
- **Experiment runners**: `experiments/run_experimentN_<short_name>.py`
  where N is the sequential experiment number.
- **Experiment outputs**: `experiments/experimentN/` containing:
  - `run.log` — captured stdout/stderr from the run
  - `summary.csv` — per-fit results (header in the runner)
  - `summary.json` — aggregated headline stats (mean ratio vs xgboost,
    win count, etc.)
  - `pmlb_rmse.png` (or similar) — the headline plot
  - `report.md` — the writeup; placeholders filled with concrete numbers

## Per-fit reliability machinery

Long-running experiments must:

- Wrap each fit in `try/except`. On exception, log the dataset/seed/model
  and continue.
- Write the per-fit row to `summary.csv` immediately, not at the end.
- On restart, read existing `summary.csv` and skip already-done rows.
  This is the resume-from-checkpoint pattern — see
  `experiments/run_experiment15_full_pmlb.py` for the canonical
  implementation.

This means you can `Ctrl+C`, reboot, kill the process — and just rerun
the same command to pick up where you left off.

## Defaults (Exp-13 calibrated)

The current "production" SplitBoost defaults, validated through
Experiments 11-14, are:

- `max_rounds=200`
- `max_depth=8`
- `learning_rate=0.1`
- `min_samples_leaf=20`
- `n_eml_candidates=10`
- `k_eml=3`
- `k_leaf_eml=1`
- `min_samples_leaf_eml=30`
- `leaf_eml_cap_k=2.0` (per-leaf magnitude cap from Exp-11)
- `leaf_eml_gain_threshold=0.05`
- `n_bins=256`
- `patience=15`
- `val_fraction=0.15`
- `use_gpu=True` (with torch fallback when CUDA absent)

Any new experiment that varies one of these from default should say so
explicitly in its spec and report.

## Test discipline

- `tests/unit/` is the gate. Run `uv run pytest tests/unit/ -v`.
- 94 tests pass + 1 pre-existing unrelated failure
  (`test_eml_weak_learner.py::test_fit_recovers_simple_formula`). Don't
  try to "fix" that one — it's a known pre-existing issue in an unrelated
  module (`fit_eml_tree`).
- Equivalence tests live alongside the implementations they validate
  (e.g. `test_predict_triton_matches_torch`,
  `test_histogram_split_triton_matches_torch`).
- Speedup regression test: `test_gpu_speedup_on_synthetic_large` —
  100k-row synthetic fit must complete in <30s.

## Triton + GPU patterns

- Each Triton kernel ships with a torch fallback. The dispatcher tries
  Triton, catches `Exception`, and falls back. On first fallback it
  emits a one-time `RuntimeWarning` so silent failures get noticed.
- Equivalence tests must verify the Triton path actually executes (e.g.
  by setting `warnings.simplefilter('error', RuntimeWarning)` and
  running the dispatcher) — silent fallback to torch would mask kernel
  bugs and make tests trivially pass.
- The Triton kernels live in private modules (`_predict_triton.py`,
  `_gpu_split_triton.py`) and are dispatched from the public-facing
  modules.

## Commits

- One feature/fix per commit.
- Commit messages explain the WHY, not the WHAT (e.g. "broaden Triton
  predict fallback exception" with body explaining the silent-failure
  risk).
- Co-author trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Spec and plan are committed before implementation begins.
- Implementation tasks commit at the end of each task as called for in
  the plan.

## Common gotchas

- **`enumerate_depth2_descriptor` and `descriptor_feature_mask_numpy`
  must be accessed via the cached `get_descriptor_np` /
  `get_feature_mask_np` accessors** in `_triton_exhaustive.py`. The raw
  functions rebuild a 6,400-row Cartesian product every call — once
  cost a 46% performance regression by being called 11k+ times per fit.
- **`_X_cpu` is sometimes a `(0, d)` sentinel** in the GPU path
  (`_fit_xy_gpu`). Code reading `_X_cpu` must only access `.shape[1]`
  in that path, never index rows.
- **`torch.std` defaults to `unbiased=True`** (Bessel-corrected, ddof=1).
  Use `unbiased=False` to match `np.std`'s default ddof=0.
- **Tree-fit's per-tree seeds** are derived in `EmlSplitBoostRegressor.fit()`
  before dispatching to `_fit_gpu_loop` / `_fit_cpu_loop`. Both loops
  consume the same `tree_seeds` list — don't recompute in either branch.
- **Profile output goes in `profile_redux/` (or similar one-off dirs)**.
  These directories are NOT tracked in git. Don't `git add` profile
  artifacts.

## Skill priority reminder

Process skills (`brainstorming`, `debugging`, `writing-plans`,
`subagent-driven-development`, `finishing-a-development-branch`)
override the default Claude Code system prompt where they conflict.
User instructions in CLAUDE.md / direct messages override the skills.
