"""EML weak learner training.

Three-phase optimization per restart (spec section 5.3):
  1. Exploration — Adam, lr=1e-2, 500 steps, random init.
  2. Hardening — continued Adam with entropy penalty driving each
     softmax toward a simplex vertex.
  3. Snap + verify — argmax each softmax, convert to sympy, verify.

Parallel restarts: n_restarts independent MasterFormulas trained
concurrently via vmap-like stacking in a leading batch dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
import torch

from eml_boost.symbolic.master_formula import MasterFormula
from eml_boost.symbolic.simplify import (
    rpn_length,
    snap_constants,
    snapped_to_sympy,
)
from eml_boost.symbolic.snap import (
    SnappedTree,
    count_active_positions,
    snap_master_formula,
)
from eml_boost.symbolic.verify import reproduces_numerically
from eml_boost._numerics import is_real_valued

_EXPLORATION_STEPS = 500
_HARDENING_STEPS = 500
_LR_INIT = 1e-2
_LR_MIN = 1e-4
_ENTROPY_MAX = 0.5
_SNAP_TOL = 1e-6


@dataclass
class EmlWeakLearner:
    """Result of fitting an EML tree to a residual target.

    If snap_ok, `formula` is the closed-form sympy expression.
    Otherwise, predictions come from the trained MasterFormula as-is.
    """

    trained_module: MasterFormula
    snapped: SnappedTree
    snap_ok: bool
    formula: sp.Expr | None
    feature_names: tuple[str, ...]
    feature_idx: np.ndarray  # indices into the full feature matrix (length k)
    feature_mean: np.ndarray  # mean of the k selected features (length k)
    feature_std: np.ndarray  # std of the k selected features (length k)
    inner_val_mse: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new inputs; returns real-valued (n,) array."""
        X_sel = X[:, self.feature_idx]
        X_std = (X_sel - self.feature_mean) / self.feature_std
        with torch.no_grad():
            X_t = torch.tensor(X_std, dtype=torch.complex128)
            out = self.trained_module(X_t)
            return out.real.cpu().numpy().astype(np.float64)

    def params_count(self) -> int:
        """BIC complexity proxy. +1 for the per-round learned eta."""
        if self.snap_ok and self.formula is not None:
            return rpn_length(self.formula) + 1
        return count_active_positions(self.snapped) + 1


def fit_eml_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    n_restarts: int,
    k: int,
    random_state: int | None = None,
) -> EmlWeakLearner:
    """Fit an EML weak learner to targets y given features X.

    Selects the top-k features by |correlation| with y; standardizes them;
    runs n_restarts independent trainings; returns the best by inner-val MSE.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    depth : int
        Depth of the MasterFormula tree.
    n_restarts : int
        Number of independent random restarts to try.
    k : int
        Number of top features to select. Must be <= X.shape[1].
    random_state : int or None
        If None, each restart uses a freshly drawn torch seed (non-deterministic).
        If int, seeds are derived from random_state so results are reproducible.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # I3: Validate k before any computation.
    if k > X.shape[1]:
        raise ValueError(f"k={k} exceeds number of features ({X.shape[1]})")

    # I2: Use np.random.default_rng so random_state=None is truly non-deterministic
    #     and random_state=0 is distinct from None.
    rng = np.random.default_rng(random_state)

    # 1. Feature selection by |correlation|
    col_std = X.std(axis=0) + 1e-12
    y_centered = y - y.mean()
    corrs = (X - X.mean(axis=0)).T @ y_centered / (col_std * y.std() * len(y) + 1e-12)
    feature_idx = np.argsort(-np.abs(corrs))[:k]
    feature_names = tuple(f"x_{int(i)}" for i in feature_idx)

    X_sel = X[:, feature_idx]
    mean = X_sel.mean(axis=0)
    std = X_sel.std(axis=0) + 1e-12
    X_std = (X_sel - mean) / std

    # 2. Inner train/val split.
    n = len(X_std)
    perm = rng.permutation(n)
    val_sz = max(int(0.2 * n), 1)
    val_idx, tr_idx = perm[:val_sz], perm[val_sz:]
    X_tr = torch.tensor(X_std[tr_idx], dtype=torch.complex128)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.complex128)
    X_iv = torch.tensor(X_std[val_idx], dtype=torch.complex128)
    y_iv_np = y[val_idx]

    # I2: Precompute per-restart torch seeds from the rng so that
    #     random_state=None gives fresh (non-deterministic) seeds each call,
    #     while an integer random_state gives fully reproducible seeds.
    restart_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_restarts)]

    # 3. Run n_restarts. For simplicity in v1 we do them sequentially; parallel
    #    stacking is a performance optimization flagged for v2.
    best: EmlWeakLearner | None = None
    for seed in restart_seeds:
        torch.manual_seed(seed)
        mf = MasterFormula(depth=depth, k=k)
        _train_single(mf, X_tr, y_tr)

        # Quick divergence check on the raw network output before snapping.
        with torch.no_grad():
            raw_iv = mf(X_iv)
        if torch.isnan(raw_iv).any():
            continue  # training diverged — skip this restart

        unsnapped_pred = raw_iv.real.cpu().numpy().astype(np.float64)
        if np.isnan(unsnapped_pred).any():
            continue  # defensive guard

        snapped = snap_master_formula(mf)
        formula = snapped_to_sympy(snapped, feature_names)
        formula = snap_constants(formula)

        # C1 (spec 5.4): After snap, evaluate the formula on real inner-val data
        # and require |imag part| < 1e-8. This catches snapped formulas that
        # still produce complex-valued outputs on real inputs (e.g., log of a
        # negative constant that wasn't caught by snap_constants).
        snap_iv_vals = _eval_formula_numpy(formula, feature_names, X_std[val_idx])
        if snap_iv_vals is None or not is_real_valued(
            torch.tensor(snap_iv_vals, dtype=torch.complex128), tol=1e-8
        ):
            continue  # snapped formula is non-physical — discard

        # Verify snap reproduces the unsnapped output.
        snap_ok = reproduces_numerically(
            formula,
            feature_names,
            X_std[val_idx],
            unsnapped_pred,
            tol=_SNAP_TOL,
        )

        mse = float(np.mean((unsnapped_pred - y_iv_np) ** 2))

        learner = EmlWeakLearner(
            trained_module=mf,
            snapped=snapped,
            snap_ok=snap_ok,
            formula=formula if snap_ok else None,
            feature_names=feature_names,
            feature_idx=feature_idx,
            feature_mean=mean,
            feature_std=std,
            inner_val_mse=mse,
        )
        if best is None or learner.inner_val_mse < best.inner_val_mse:
            best = learner

    # I1: If every restart was skipped, raise a clear error instead of
    #     returning a NaN learner or hitting an assert.
    if best is None:
        raise RuntimeError(
            f"All {n_restarts} restarts diverged or produced non-real outputs. "
            "Consider increasing n_restarts or reducing depth."
        )
    return best


def _eval_formula_numpy(
    formula: sp.Expr,
    feature_names: tuple[str, ...],
    X_val: np.ndarray,
) -> np.ndarray | None:
    """Evaluate a sympy formula numerically on X_val (shape n x k).

    Returns a complex numpy array, or None if evaluation raises an exception.
    The caller can inspect the imaginary part to detect non-real outputs.
    """
    try:
        syms = [sp.Symbol(name) for name in feature_names]
        f_lambda = sp.lambdify(syms, formula, modules="numpy")
        cols = [X_val[:, j] for j in range(len(feature_names))]
        result = f_lambda(*cols)
        return np.asarray(result, dtype=np.complex128)
    except Exception:
        return None


def _train_single(mf: MasterFormula, X_tr: torch.Tensor, y_tr: torch.Tensor) -> None:
    """Run the 3-phase training on a single MasterFormula in place."""
    optimizer = torch.optim.Adam(mf.parameters(), lr=_LR_INIT)

    # Phase 1: exploration.
    for _ in range(_EXPLORATION_STEPS):
        optimizer.zero_grad()
        pred = mf(X_tr)
        if torch.isnan(pred).any():
            return  # divergence — abort this restart cleanly
        loss = ((pred.real - y_tr.real) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Phase 2: hardening with entropy penalty ramp.
    for step in range(_HARDENING_STEPS):
        frac = step / max(_HARDENING_STEPS - 1, 1)
        lr = _LR_INIT * (1 - frac) + _LR_MIN * frac
        entropy_weight = _ENTROPY_MAX * frac
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad()
        pred = mf(X_tr)
        if torch.isnan(pred).any():
            return
        loss = ((pred.real - y_tr.real) ** 2).mean()
        for p in mf.logits_list:
            probs = torch.softmax(p, dim=0)
            entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
            loss = loss + entropy_weight * entropy
        loss.backward()
        optimizer.step()
