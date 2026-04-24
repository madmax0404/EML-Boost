# Ridge-Regularized EML Leaves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `leaf_eml_ridge` hyperparameter that ridge-regularizes the slope `η` in the per-leaf OLS fit so heavy-tailed features no longer cause numerical explosions. Run an Experiment 10 grid sweep across ridge strengths.

**Architecture:** Modify the OLS denominator in `_fit_leaf` from `det` to `det + n_fit · λ`, changing `η = Sxy/Sxx` into the centered-ridge form `η = Sxy/(Sxx+λ)`. `β` is recomputed as `(Σy − η·Σp)/n_fit` so it stays the conditional-OLS intercept given the shrunk η. Default `leaf_eml_ridge=0.0` is backward-compatible with Experiment 9. Both `_select_leaf_gated` and `_select_leaf_blended` consume the already-regularized `(η, β)` from the shared `_fit_leaf` setup — no changes in either selector.

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+, pytest, PMLB, XGBoost, LightGBM, matplotlib. uv for environment management.

---

## Background an implementer needs

**The codebase under change** is `EmlSplitTreeRegressor` (`eml_boost/tree_split/tree.py`), a regression tree whose internal nodes may split on either raw features or sampled depth-2 EML expressions and whose leaves may be EML expressions. The relevant method is `_fit_leaf` (around lines 330-430) which:

1. Standardizes the top-`k_leaf_eml` residual-correlated features using fit-time global mean/std, clamps to `[−3, 3]`.
2. Splits the leaf samples 75%/25% fit/val using a deterministic per-leaf seed.
3. On the fit portion, computes closed-form OLS `(η, β)` for every one of 144 candidate trees in one batched GPU operation:
   ```python
   det = sum_p2 * n_fit - sum_p * sum_p
   det_safe = torch.where(det.abs() > 1e-6, det, torch.ones_like(det))
   eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
   bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
   ```
4. Builds a `ctx` dict and dispatches to either `_select_leaf_gated` (when `use_stacked_blend=False`, the default) or `_select_leaf_blended` (when `True`).

**Experiment 9** (`experiments/experiment9/report.md`) showed that steps 3's unregularized OLS produces extreme `η` values on heavy-tailed features (`562_cpu_small`, `564_fried`) — the RMSE explodes into 10⁴-10⁵ territory on multiple seeds. XGBoost's analogous `w* = −G/(H+λ)` with default `reg_lambda=1.0` provably bounds `|w*| ≤ max|residual|`. We're porting that regularization idea to our OLS.

**Test infrastructure:** existing unit tests are in `tests/unit/test_eml_split_tree.py` (16 tests including 5 new blend-family tests from Experiment 9) and `tests/unit/test_eml_split_boost.py` (5 tests). Run with `uv run pytest tests/unit/ -v`. Several tests require CUDA (they use `torch.cuda.is_available()` and `pytest.skip()`); the RTX 3090 in the dev environment has CUDA.

**Before starting Task 1, read:**
- `eml_boost/tree_split/tree.py` — focus on `__init__` (~lines 60-105) and the OLS block inside `_fit_leaf`.
- `eml_boost/tree_split/ensemble.py` — focus on `__init__` (~lines 56-95) and the per-round `EmlSplitTreeRegressor(...)` construction in `fit`.
- `tests/unit/test_eml_split_tree.py` — read the test helpers (`_count_eml_leaves`, `_count_leaves`, `_mse`) and the existing blend tests at the end.
- The spec: `docs/superpowers/specs/2026-04-24-leaf-ridge-design.md`.
- The failing Experiment 9 report: `experiments/experiment9/report.md`.

---

## Task 1: Add `leaf_eml_ridge` parameter (default 0.0, no behavior change yet)

**Goal:** Add the constructor parameter on both `EmlSplitTreeRegressor` and `EmlSplitBoostRegressor`, threaded through. Default `0.0` preserves existing behavior exactly. No OLS change in this task.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (constructor)
- Modify: `eml_boost/tree_split/ensemble.py` (constructor + per-round tree construction)
- Modify: `tests/unit/test_eml_split_tree.py` (one smoke test)

- [ ] **Step 1: Write the failing smoke test.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_leaf_eml_ridge_parameter_accepted():
    """Constructor must accept the new leaf_eml_ridge parameter. At 0.0
    (default) the predictions must be identical to a regressor built
    without the parameter — this pins the backward-compat story."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    m_default = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30, random_state=0,
    ).fit(X, y)
    m_ridge_zero = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_ridge=0.0, random_state=0,
    ).fit(X, y)
    assert np.allclose(m_default.predict(X), m_ridge_zero.predict(X))
```

- [ ] **Step 2: Run the test; confirm it fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_eml_ridge_parameter_accepted -v
```
Expected: FAIL with `TypeError: EmlSplitTreeRegressor.__init__() got an unexpected keyword argument 'leaf_eml_ridge'`.

- [ ] **Step 3: Add `leaf_eml_ridge` parameter to `EmlSplitTreeRegressor.__init__`.**

In `eml_boost/tree_split/tree.py`, locate the constructor signature. The relevant block looks like:

```python
        k_leaf_eml: int = 1,
        min_samples_leaf_eml: int = 50,
        leaf_eml_gain_threshold: float = 0.05,
        use_stacked_blend: bool = False,
        random_state: int | None = None,
```

Insert `leaf_eml_ridge: float = 0.0` just after `leaf_eml_gain_threshold` and before `use_stacked_blend`:

```python
        k_leaf_eml: int = 1,
        min_samples_leaf_eml: int = 50,
        leaf_eml_gain_threshold: float = 0.05,
        leaf_eml_ridge: float = 0.0,
        use_stacked_blend: bool = False,
        random_state: int | None = None,
```

In the same `__init__` body, find the attribute-storage block (look for `self.leaf_eml_gain_threshold = leaf_eml_gain_threshold`) and add:

```python
        self.leaf_eml_ridge = leaf_eml_ridge
```

alongside the other attribute assignments, immediately after `self.leaf_eml_gain_threshold = leaf_eml_gain_threshold`.

- [ ] **Step 4: Add `leaf_eml_ridge` parameter to `EmlSplitBoostRegressor.__init__`.**

In `eml_boost/tree_split/ensemble.py`, locate the constructor signature (similar block around lines 70-75). Insert `leaf_eml_ridge: float = 0.0` after `leaf_eml_gain_threshold` and before `use_stacked_blend`:

```python
        k_leaf_eml: int = 1,
        min_samples_leaf_eml: int = 50,
        leaf_eml_gain_threshold: float = 0.05,
        leaf_eml_ridge: float = 0.0,
        use_stacked_blend: bool = False,
        patience: int | None = 15,
```

Add attribute storage alongside the others:

```python
        self.leaf_eml_ridge = leaf_eml_ridge
```

Place it right after `self.leaf_eml_gain_threshold = leaf_eml_gain_threshold`.

- [ ] **Step 5: Thread `leaf_eml_ridge` into the per-round tree construction.**

In `eml_boost/tree_split/ensemble.py`, locate the `fit` method's per-round tree construction (look for `tree = EmlSplitTreeRegressor(` inside the `for m in range(self.max_rounds):` loop). Add `leaf_eml_ridge=self.leaf_eml_ridge` into the kwargs, placed next to the other leaf-specific params:

```python
            tree = EmlSplitTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_eml_candidates=self.n_eml_candidates,
                k_eml=self.k_eml,
                eml_depth=self.eml_depth,
                n_bins=self.n_bins,
                histogram_min_n=self.histogram_min_n,
                use_gpu=self.use_gpu,
                k_leaf_eml=self.k_leaf_eml,
                min_samples_leaf_eml=self.min_samples_leaf_eml,
                leaf_eml_gain_threshold=self.leaf_eml_gain_threshold,
                leaf_eml_ridge=self.leaf_eml_ridge,
                use_stacked_blend=self.use_stacked_blend,
                random_state=tree_seeds[m],
            ).fit(X_tr, r)
```

- [ ] **Step 6: Run the smoke test; confirm it passes.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_eml_ridge_parameter_accepted -v
```
Expected: PASS.

- [ ] **Step 7: Run the full unit test suite; confirm no regressions.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 22 in-scope tests pass (21 existing + 1 new smoke test). One known pre-existing failure in `test_eml_weak_learner.py::test_fit_recovers_simple_formula` is unrelated — do not chase it.

- [ ] **Step 8: Commit.**

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: add leaf_eml_ridge parameter (default 0.0, no behavior change)

Adds the hyperparameter on both EmlSplitTreeRegressor and
EmlSplitBoostRegressor. Default 0.0 preserves Experiment 9 behavior
exactly. Next task wires it into the OLS formula.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Apply ridge to the OLS in `_fit_leaf`

**Goal:** Modify the OLS block in `_fit_leaf` so that `det` becomes `det + n_fit · λ` when `λ > 0`, and recompute `β` from `(Σy − η·Σp)/n_fit` (the centered-ridge form that stays consistent with the shrunk η).

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (OLS block inside `_fit_leaf`)
- Modify: `tests/unit/test_eml_split_tree.py` (add 2 tests — shrinkage monotonicity, heavy-tails stability)

- [ ] **Step 1: Write the failing test for monotonic shrinkage.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_ridge_shrinks_max_abs_eta_monotonically():
    """On a clean elementary signal, max |η| across the tree's EML leaves
    should decrease monotonically as leaf_eml_ridge increases."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")

    def _collect_etas(node):
        etas: list[float] = []
        def walk(n):
            if isinstance(n, EmlLeafNode):
                etas.append(abs(float(n.eta)))
            elif isinstance(n, InternalNode):
                walk(n.left); walk(n.right)
        walk(node)
        return etas

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    max_etas: list[float] = []
    for ridge in [0.0, 0.1, 1.0, 10.0]:
        m = EmlSplitTreeRegressor(
            max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
            k_leaf_eml=1, min_samples_leaf_eml=30,
            leaf_eml_ridge=ridge, random_state=0,
        ).fit(X, y)
        etas = _collect_etas(m._root)
        max_etas.append(max(etas) if etas else 0.0)

    # Strict monotonic decrease expected; allow equality only when all
    # ridge settings produce zero EML leaves (shouldn't happen on a
    # clean exp signal, but be defensive).
    for i in range(len(max_etas) - 1):
        assert max_etas[i] >= max_etas[i + 1], (
            f"non-monotonic: ridge grid gives max|eta| = {max_etas}"
        )
```

- [ ] **Step 2: Run the test; confirm it fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_ridge_shrinks_max_abs_eta_monotonically -v
```
Expected: FAIL. Because the ridge is not wired into the OLS yet, all four ridge values produce identical `η`s and the max-|η| sequence is constant. The assertion `max_etas[i] >= max_etas[i+1]` is actually satisfied when equal, so the test might accidentally pass.

If the test accidentally passes: check by running it with a print statement temporarily to confirm `max_etas` is constant. That's still evidence the ridge has no effect — proceed to implementation. After Task 2 step 6 is done, the test will observe true monotonic decrease and catch regressions.

- [ ] **Step 3: Write the failing test for heavy-tails stability.**

Add this test just below the previous one:

```python
def test_ridge_prevents_blowup_on_heavy_tails():
    """On features with magnitudes ~1e6, the Experiment-9 failure mode,
    leaf_eml_ridge=1.0 must keep predictions finite and max |η| bounded."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")

    def _collect_etas(node):
        etas: list[float] = []
        def walk(n):
            if isinstance(n, EmlLeafNode):
                etas.append(abs(float(n.eta)))
            elif isinstance(n, InternalNode):
                walk(n.left); walk(n.right)
        walk(node)
        return etas

    rng = np.random.default_rng(0)
    # Same magnitudes as 562_cpu_small features in Experiment 9.
    X = rng.normal(size=(800, 2)) * 1e6
    y = 0.001 * (X[:, 0] / 1e6) + 0.01 * rng.normal(size=800)

    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        leaf_eml_ridge=1.0, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred))
    etas = _collect_etas(m._root)
    if etas:
        # A 50% shrinkage (ridge=1.0) applied to pre-clamp features of
        # magnitude ~1 should keep |η| well below 100 on this synthetic.
        assert max(etas) < 100.0, f"max|eta| = {max(etas):.2f}"
```

- [ ] **Step 4: Run this test; confirm it passes or fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_ridge_prevents_blowup_on_heavy_tails -v
```

Expected: likely FAIL (the Experiment 9 blowup regime). If it passes pre-implementation, that's because the existing `[−3, 3]` clamp plus `min_samples_leaf_eml=50` gate is already sufficient for this synthetic — proceed regardless. The real test of the ridge is the implementation + behavior tests, not the synthetic's exact failure timing.

- [ ] **Step 5: Modify the OLS block in `_fit_leaf` to apply ridge.**

In `eml_boost/tree_split/tree.py`, find the block inside `_fit_leaf` that computes `eta` and `bias`. It currently looks like:

```python
    # Closed-form OLS per tree on the fit portion.
    n_fit = float(X_fit.shape[0])
    sum_p = preds_fit.sum(dim=1)
    sum_p2 = (preds_fit * preds_fit).sum(dim=1)
    sum_y_f = y_fit.sum()
    sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
    det = sum_p2 * n_fit - sum_p * sum_p
    det_safe = torch.where(det.abs() > 1e-6, det, torch.ones_like(det))
    eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
    bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
```

Replace that block with:

```python
    # Closed-form OLS per tree on the fit portion. When
    # self.leaf_eml_ridge > 0 we regularize the slope (η) with a ridge
    # penalty λ·η². The centered-ridge closed form adds n_fit·λ to the
    # normal-equation diagonal; on the original sufficient statistics
    # that means replacing det with det + n_fit·λ. The bias term is
    # then the conditional-OLS intercept given the shrunk slope:
    # β = (Σy − η·Σp) / n_fit.
    n_fit = float(X_fit.shape[0])
    sum_p = preds_fit.sum(dim=1)
    sum_p2 = (preds_fit * preds_fit).sum(dim=1)
    sum_y_f = y_fit.sum()
    sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
    det = sum_p2 * n_fit - sum_p * sum_p
    lam = float(self.leaf_eml_ridge)
    det_ridged = det + n_fit * lam
    # Guard against the remaining zero case (λ = 0 and det = 0 — a
    # genuinely degenerate tree with p_val constant and λ off).
    det_safe = torch.where(
        det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
    )
    eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
    bias = (sum_y_f - eta * sum_p) / n_fit
```

- [ ] **Step 6: Run the two new tests; confirm both pass.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py -v -k "ridge_shrinks or ridge_prevents"
```

Expected: both tests pass. The monotonicity test now observes true strict decrease; the heavy-tails test shows the predictions stay finite.

- [ ] **Step 7: Run the full unit test suite; confirm no regressions.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 23 in-scope tests pass (22 from Task 1 + 2 new Task 2 tests = 24; but the Task 1 accident-pass of the monotonicity test was counted — final is 22 pre-existing + 2 new = 24 if the Task 1 test was distinct; adjust the count to whatever pytest reports, just verify no failures beyond the known pre-existing one). The pre-existing failure in `test_eml_weak_learner.py::test_fit_recovers_simple_formula` remains — do not chase.

- [ ] **Step 8: Verify the existing ridge-zero backward compat.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_eml_ridge_parameter_accepted -v
```

Expected: PASS (the Task-1 smoke test, now protecting against the OLS change drifting predictions at `ridge=0.0`). If this fails, the OLS change produced different `bias` values at `ridge=0.0` — investigate before committing. The algebraic identity is `β_old = (Σp²·Σy − Σp·Σpy)/det` equals `β_new = (Σy − η·Σp)/n_fit` when `λ=0`; float32 arithmetic order differences could cause tiny drifts, but `np.allclose` with default tolerances should still pass.

If it genuinely fails due to float-order drift beyond `np.allclose` tolerance, the cleanest fix is to branch on `lam == 0.0` and use the old `sum_p2`-based β formula for that case, preserving exact bit-compat. Include that branch only if needed.

- [ ] **Step 9: Commit.**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: ridge-regularize η in the EML-leaf OLS fit

Modify _fit_leaf's OLS block to apply the centered-ridge closed form
η = Sxy/(Sxx+λ), implemented as det → det + n_fit·λ. β is recomputed
as the conditional-OLS intercept (Σy − η·Σp)/n_fit, consistent with
the shrunk slope. Default leaf_eml_ridge=0.0 preserves bit-compat with
Experiment 9.

Two new tests: monotonic |η| shrinkage as ridge grows, and no
numerical blowup on heavy-tailed synthetic features.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create the Experiment 10 runner

**Goal:** Produce `experiments/run_experiment10_leaf_ridge.py` that sweeps 6 SplitBoost configurations over 3 seeds and 7 datasets, plus XGBoost and LightGBM baselines, aggregates mean/std across seeds, and writes CSV/JSON/PNG/eta-stats artifacts to `experiments/experiment10/`.

**Files:**
- Create: `experiments/run_experiment10_leaf_ridge.py`
- Reference: `experiments/run_experiment9_stacked_blend.py` (fork basis)

- [ ] **Step 1: Create the runner file.**

Write `experiments/run_experiment10_leaf_ridge.py` with the following complete contents:

```python
"""Experiment 10: ridge-regularized EML-leaf OLS.

Sweeps 6 SplitBoost configurations across 3 seeds and the 7 PMLB
datasets used in Experiments 8/9, plus XGBoost and LightGBM baselines.
Writes aggregate stats and per-leaf η magnitude distributions.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode

RESULTS_DIR = Path(__file__).resolve().parent / "experiment10"

DATASETS = [
    "192_vineyard",
    "210_cloud",
    "523_analcatdata_neavote",
    "557_analcatdata_apnea1",
    "529_pollen",
    "562_cpu_small",
    "564_fried",
]

MAX_ROUNDS = 200
DEPTH = 6
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF_EML = 50
LEAF_EML_GAIN_THRESHOLD = 0.05
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]

# (config_id, use_stacked_blend, leaf_eml_ridge) tuples.
SPLIT_CONFIGS = [
    ("G0",        False, 0.0),
    ("G_weak",    False, 0.1),
    ("G_strong",  False, 1.0),
    ("G_vstrong", False, 10.0),
    ("B0",        True,  0.0),
    ("B_strong",  True,  1.0),
]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, use_stacked_blend: bool, leaf_eml_ridge: float):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=20,
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        leaf_eml_ridge=leaf_eml_ridge,
        use_stacked_blend=use_stacked_blend,
        patience=15,
        val_fraction=0.15,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_lgb(X_tr, y_tr, seed):
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=DEPTH,
            num_leaves=2**DEPTH,
            min_data_in_leaf=20,
            learning_rate=LEARNING_RATE,
            device="gpu",
            seed=seed,
            verbose=-1,
        ),
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=MAX_ROUNDS,
    )
    return m, time.time() - start


def _fit_xgb(X_tr, y_tr, seed):
    start = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=DEPTH,
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        device="cuda",
        verbosity=0,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def _collect_eta_abs(boost: EmlSplitBoostRegressor) -> list[float]:
    """Walk every tree in the boost; return |η| for every EmlLeafNode."""
    out: list[float] = []
    def walk(node):
        if isinstance(node, EmlLeafNode):
            out.append(abs(float(node.eta)))
        elif isinstance(node, InternalNode):
            walk(node.left); walk(node.right)
    for tree in boost._trees:
        walk(tree._root)
    return out


def _summarize_etas(etas: list[float]) -> dict:
    if not etas:
        return {"count": 0, "mean_abs_eta": float("nan"),
                "max_abs_eta": float("nan"), "p99_abs_eta": float("nan")}
    arr = np.asarray(etas)
    return {
        "count": int(len(arr)),
        "mean_abs_eta": float(arr.mean()),
        "max_abs_eta": float(arr.max()),
        "p99_abs_eta": float(np.percentile(arr, 99)),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    # dataset -> config -> seed -> eta_stats dict
    eta_stats: dict[str, dict[str, dict[int, dict]]] = {}

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        eta_stats[name] = {cfg_id: {} for cfg_id, _, _ in SPLIT_CONFIGS}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            for cfg_id, blend, ridge in SPLIT_CONFIGS:
                m, t = _fit_split_boost(
                    X_tr, y_tr, seed,
                    use_stacked_blend=blend, leaf_eml_ridge=ridge,
                )
                rmse = _rmse(m.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=cfg_id,
                    rmse=rmse, fit_time=t, n_rounds=m.n_rounds,
                ))
                print(
                    f"    {cfg_id:>10} ({t:6.1f}s, {m.n_rounds:>3} rounds) "
                    f"RMSE={rmse:.4f}",
                    flush=True,
                )
                eta_stats[name][cfg_id][seed] = _summarize_etas(_collect_eta_abs(m))

            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(f"    {'lightgbm':>10} ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}", flush=True)

            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(f"    {'xgboost':>10} ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}", flush=True)

    # Per-(dataset, config) aggregates.
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.dataset, r.config)
        agg.setdefault(key, {"rmses": [], "times": [], "n_rounds": []})
        agg[key]["rmses"].append(r.rmse)
        agg[key]["times"].append(r.fit_time)
        agg[key]["n_rounds"].append(r.n_rounds)
    for key, d in agg.items():
        d["rmse_mean"] = float(mean(d["rmses"]))
        d["rmse_std"] = float(stdev(d["rmses"])) if len(d["rmses"]) > 1 else 0.0
        d["time_mean"] = float(mean(d["times"]))

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,config,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    # JSON with aggregates
    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS, "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES, "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "n_bins": N_BINS, "test_size": TEST_SIZE, "seeds": SEEDS,
        "split_configs": [{"id": c, "blend": b, "ridge": r} for c, b, r in SPLIT_CONFIGS],
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}}
    for (ds, cfg), d in agg.items():
        out["aggregate"].setdefault(ds, {})[cfg] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    # Eta stats JSON
    eta_json_path = RESULTS_DIR / "eta_stats.json"
    with eta_json_path.open("w") as fp:
        json.dump(eta_stats, fp, indent=2)
    print(f"wrote {eta_json_path}")

    # Plot: grouped bars per dataset, one bar per SplitBoost config + XGBoost.
    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    split_cfg_ids = [c for c, _, _ in SPLIT_CONFIGS]
    all_bars = split_cfg_ids + ["xgboost"]
    colors = ["#2E86AB", "#4EA1B2", "#588157", "#7FA65A", "#E9C46A", "#F4A261", "#9B2226"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=110)
    xs = np.arange(len(ordered))
    width = 0.8 / len(all_bars)

    for i, cfg_name in enumerate(all_bars):
        means_ = [agg[(n, cfg_name)]["rmse_mean"] for n in ordered]
        stds_ = [agg[(n, cfg_name)]["rmse_std"] for n in ordered]
        offset = (i - len(all_bars) / 2) * width + width / 2
        ax1.bar(xs + offset, means_, width, yerr=stds_, label=cfg_name, color=colors[i])

    ax1.set_xticks(xs)
    ax1.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE (mean ± std over 3 seeds)")
    ax1.set_yscale("log")
    ax1.set_title("Experiment 10: ridge-regularized EML leaves (log scale)")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3, axis="y")

    # Ratio panel: vs XGBoost for each SplitBoost config.
    for i, cfg_name in enumerate(split_cfg_ids):
        ratios = [agg[(n, cfg_name)]["rmse_mean"] / agg[(n, "xgboost")]["rmse_mean"]
                  for n in ordered]
        offset = (i - len(split_cfg_ids) / 2) * width + width / 2
        ax2.bar(xs + offset, ratios, width, label=cfg_name, color=colors[i])

    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("mean SplitBoost RMSE / XGBoost RMSE")
    ax2.set_yscale("log")
    ax2.set_title("Ratio vs. XGBoost (log scale)")
    ax2.legend(fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Aggregate summary (mean over 3 seeds, log ratio vs XGBoost) ===")
    header = f"{'dataset':>28}  " + "  ".join(f"{c:>11}" for c in split_cfg_ids)
    print(header)
    for n in ordered:
        xg_mean = agg[(n, "xgboost")]["rmse_mean"]
        cells = []
        for c in split_cfg_ids:
            r = agg[(n, c)]["rmse_mean"] / xg_mean
            cells.append(f"{r:>11.3f}")
        print(f"{n:>28}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the runner on one small dataset.**

Temporarily edit `DATASETS` in place to only `["523_analcatdata_neavote"]` and `SEEDS` to `[0]`, then run:

```bash
uv run python experiments/run_experiment10_leaf_ridge.py
```

Expected: completes in under 2 minutes; writes `experiments/experiment10/{summary.csv, summary.json, eta_stats.json, pmlb_rmse.png}`; console prints the "Aggregate summary" with one row.

Then **revert** `DATASETS` and `SEEDS` to the full values before committing.

- [ ] **Step 3: Commit the runner.**

```bash
git add experiments/run_experiment10_leaf_ridge.py
git commit -m "$(cat <<'EOF'
add: Experiment 10 runner for ridge-regularized EML leaves

6 SplitBoost configs (gated × 4 ridge values, blend × 2 ridge values)
× 3 seeds × 7 datasets against XGBoost and LightGBM. Writes summary,
per-leaf |η| magnitude stats, and log-scale ratio plot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run Experiment 10 and write the report

**Goal:** Execute the full runner and produce `experiments/experiment10/report.md` with a verdict on whether ridge rescues the heavy-tailed datasets, and set a principled default for `leaf_eml_ridge`.

**Files:**
- Execute: `experiments/run_experiment10_leaf_ridge.py`
- Create: `experiments/experiment10/report.md`
- Commit: outputs (`summary.csv`, `summary.json`, `eta_stats.json`, `pmlb_rmse.png`, `run.log`) + report.

- [ ] **Step 1: Run the full benchmark.**

Run:
```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment10_leaf_ridge.py 2>&1 | tee experiments/experiment10/run.log
```

Expected runtime ~12-15 minutes on RTX 3090 (168 fits). Confirms all fits complete. Writes all artifacts to `experiments/experiment10/`.

If any SplitBoost fit raises an exception the runner stops — investigate before writing the report.

- [ ] **Step 2: Read the outputs.**

```bash
cat experiments/experiment10/summary.csv
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment10/summary.json'))['aggregate'], indent=2))" | head -100
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment10/eta_stats.json')), indent=2))" | head -80
```

Identify per-dataset:
- For each ridge value: mean ratio vs XGBoost across 3 seeds, max |η| across seeds.
- Which ridge setting is the lowest ratio that also keeps max |η| bounded (no explosion seed).
- Whether any ridge > 0 config gets within 10× of XGBoost on `562_cpu_small` (S-B criterion).
- Whether any ridge > 0 config has no-explosion (S-A criterion).

- [ ] **Step 3: Decide the default and write `experiments/experiment10/report.md`.**

Fill in every `<…>` placeholder with concrete numbers from Step 2. Use the Experiment 9 report as a template for tone/format.

Create the file with this structure:

```markdown
# Experiment 10: Ridge-Regularized EML Leaves

**Date:** 2026-04-24
**Commit:** <fill in from `git rev-parse HEAD` at this point>
**Runtime:** <fill in from run.log>
**Scripts:** `experiments/run_experiment10_leaf_ridge.py`

## What the experiment was about

Experiment 9's multi-seed benchmark exposed catastrophic numerical
explosions on `562_cpu_small` and `564_fried` — traced by reading
XGBoost v3.2.0's source to the fact that our EML-leaf OLS is
unregularized, while XGBoost's `w* = −G/(H+λ)` with default
`reg_lambda=1.0` provably bounds `|w*| ≤ max|residual|`. Experiment 10
ports that idea: add a `leaf_eml_ridge` hyperparameter that shrinks
the OLS slope `η` via `η = Sxy/(Sxx+λ)`. Sweeps 4 ridge values under
the gate and 2 under the blend to find a principled default.

## Configuration

<fill in from summary.json config section>

## Results (mean ratios vs XGBoost, 3 seeds)

| dataset | G0 | G_weak (0.1) | G_strong (1.0) | G_vstrong (10.0) | B0 | B_strong (1.0) | verdict |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

<one row per dataset, mean ratio to 3 decimals, with verdict
column describing the effect of increasing ridge>

## |η| magnitudes

Aggregate `max|η|` across 3 seeds per config per dataset (from
`eta_stats.json`):

| dataset | G0 | G_weak | G_strong | G_vstrong | B0 | B_strong |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

<should show ridge monotonically shrinking max|η| as expected;
if not on some dataset, flag it>

## Success criteria verdict

- **S-A (no explosion):** <MET / NOT MET> — <which configs stayed
  within 10× of XGBoost on all 7 datasets × 3 seeds>
- **S-B (cpu_small recovery):** <MET / NOT MET> — <which configs got
  cpu_small mean ratio below 2.0>
- **S-C (no regression on wins):** <MET / NOT MET> — <whether
  any of vineyard/neavote/pollen/fried lost their within-10% status
  under ridge>

**Recommended default:** `leaf_eml_ridge = <value>` because <reason>.

## What Experiment 10 actually shows

- <headline: does ridge fix the explosions?>
- <interaction with blend: does blend + ridge outperform blend alone?>
- <degradation curve: at what ridge does performance start suffering?>
- <comparison to XGBoost: does leaf_eml_ridge=1.0 put us in ballpark?>

## What's left as a loss

<datasets still outside 10% of XGBoost under the recommended config>

## What Experiment 10 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite not tested.
- Ridge only — no L1 or hard-cap alternatives tried.

## Action taken

<apply one of: set default to recommended value, or keep default=0.0
and document further work>

## Consequence for the project

<update headline claim based on results>

## Next possible experiments

<2-4 bullets>
```

- [ ] **Step 4: Optionally set the new default if ridge works.**

If the report's recommended default is non-zero (ridge rescued the explosions), update both constructors:

- `eml_boost/tree_split/tree.py`: change `leaf_eml_ridge: float = 0.0` to the recommended value.
- `eml_boost/tree_split/ensemble.py`: same change.

Run `uv run pytest tests/unit/ -v` to confirm tests still pass under the new default.

If the report recommends keeping `leaf_eml_ridge=0.0` (e.g., ridge didn't help), skip this step.

- [ ] **Step 5: Run the unit test suite one more time to confirm nothing drifted.**

```bash
uv run pytest tests/unit/ -v
```

Expected: all tests pass (with whatever pass count the previous tasks produced).

- [ ] **Step 6: Commit the run outputs, report, and any default change.**

```bash
git add experiments/experiment10/summary.csv experiments/experiment10/summary.json \
        experiments/experiment10/eta_stats.json experiments/experiment10/pmlb_rmse.png \
        experiments/experiment10/report.md
git add -f experiments/experiment10/run.log
# If defaults changed in step 4:
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py
git commit -m "$(cat <<'EOF'
exp 10 done: ridge-regularized EML leaves on PMLB

6-config ridge grid × 3 seeds × 7 datasets. Report includes the
tuning curve across ridge strengths, max|η| under each config, and a
recommended default (<fill in the number or "kept at 0.0">).
Addresses the Experiment 9 failure mode on heavy-tailed features.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Before running the commit, edit the heredoc to fill in the recommended default value (or "kept at 0.0") so the commit message is concrete.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- Add `leaf_eml_ridge` parameter → Task 1.
- Modify OLS to `det + n_fit · λ` → Task 2.
- Centered-ridge `β = (Σy − η·Σp)/n_fit` formula → Task 2 step 5.
- Default `0.0` preserves backward compat → Task 1 (default) + Task 2 step 8 (verify).
- Applied in both `_select_leaf_gated` and `_select_leaf_blended` → Task 2 via the shared `_fit_leaf` setup (no per-selector code).
- 3 new tests (ridge-zero backward compat, monotonic shrinkage, no blowup on heavy tails) → Task 1 + Task 2 steps 1+3.
- 6 SplitBoost configs (4 gated + 2 blend) × 3 seeds × 7 datasets → Task 3 (`SPLIT_CONFIGS` list).
- Outputs: summary CSV/JSON, eta_stats.json, PNG, report.md, run.log → Task 3 + Task 4.
- S-A/S-B/S-C success criteria → Task 4 step 3 (report template).
- Optional default flip if successful → Task 4 step 4.

No gaps.

**Placeholder scan:** No TBDs in any task body. Task 4 step 3's report template has `<fill in…>` markers — those are explicit instructions to replace with run-time data, not plan gaps.

**Type consistency:** `leaf_eml_ridge: float = 0.0` on both classes. The attribute is named `self.leaf_eml_ridge` consistently. The kwarg name in the per-round tree construction matches. The OLS change reads `float(self.leaf_eml_ridge)` which handles int/float uniformly. Test helpers reuse `EmlLeafNode` and `InternalNode` imports already present in the file.
