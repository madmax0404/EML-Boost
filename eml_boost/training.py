"""Outer boosting loop — spec section 4.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from eml_boost.selection import select_winner
from eml_boost.symbolic.simplify import snap_constants
from eml_boost.weak_learners.base import RoundRecord, WeakLearner, WeakLearnerKind
from eml_boost.weak_learners.dt import fit_dt_stump
from eml_boost.weak_learners.eml import fit_eml_tree

_OUTER_VAL_FRAC = 0.15
_INNER_VAL_FRAC = 0.20


@dataclass
class EmlBoostModel:
    F_0: float
    weak_learners: list[tuple[WeakLearner, float, WeakLearnerKind]] = field(default_factory=list)
    history: list[RoundRecord] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), self.F_0, dtype=np.float64)
        for learner, eta, _kind in self.weak_learners:
            out = out + eta * learner.predict(X)
        return out

    def formula_predict(self, X: np.ndarray) -> np.ndarray:
        """Sum of snapped EML contributions only."""
        out = np.zeros(len(X), dtype=np.float64)
        for learner, eta, kind in self.weak_learners:
            if kind == WeakLearnerKind.EML:
                from eml_boost.weak_learners.eml import EmlWeakLearner
                if isinstance(learner, EmlWeakLearner) and learner.snap_ok:
                    out = out + eta * learner.predict(X)
        return out

    def coverage(self, X: np.ndarray) -> float:
        total = self.predict(X) - self.F_0
        formula = self.formula_predict(X)
        total_var = float(np.var(total)) + 1e-12
        residual_var = float(np.var(total - formula))
        return max(0.0, 1.0 - residual_var / total_var)

    @property
    def formula(self) -> sp.Expr | None:
        """Top-level recovered closed-form expression, if any (spec 7.2).

        Sums all snapped EML weak learners' contributions scaled by their
        learned eta, then simplifies via sympy. Returns None if no EML
        rounds produced snap_ok formulas.
        """
        from eml_boost.weak_learners.eml import EmlWeakLearner

        contributions: list[sp.Expr] = []
        for learner, eta, kind in self.weak_learners:
            if (
                kind == WeakLearnerKind.EML
                and isinstance(learner, EmlWeakLearner)
                and learner.snap_ok
                and learner.formula is not None
            ):
                contributions.append(sp.Float(eta) * learner.formula)
        if not contributions:
            return None
        total = sp.simplify(sp.Add(*contributions))
        return snap_constants(total)

    def is_exact_recovery(self, X: np.ndarray, threshold: float = 0.99) -> bool:
        """True when formula-part coverage exceeds threshold (spec 7.3)."""
        return self.coverage(X) > threshold

    def describe(self, X: np.ndarray | None = None) -> str:
        n_eml = sum(1 for r in self.history if r.kind == WeakLearnerKind.EML)
        n_dt = sum(1 for r in self.history if r.kind == WeakLearnerKind.DT)
        lines = [
            "EmlBoostRegressor summary",
            f"  Total rounds:         {len(self.history)} ({n_eml} EML, {n_dt} DT)",
        ]
        f = self.formula
        if f is not None:
            lines.append(f"  Recovered formula:    {f}")
        else:
            lines.append(
                "  Recovered formula:    (none — no EML rounds produced closed-form output)"
            )
        if X is not None:
            cov = self.coverage(X)
            lines.append(f"  Formula coverage:     {cov * 100:.1f}%")
        return "\n".join(lines)


def boost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    M: int,
    depth_eml: int,
    depth_dt: int,
    n_restarts: int,
    k: int,
    patience: int = 15,
    eta_dt: float = 0.1,
    random_state: int | None = None,
) -> EmlBoostModel:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    n = len(X)
    perm = rng.permutation(n)
    outer_val_n = max(int(_OUTER_VAL_FRAC * n), 1)
    outer_val_idx = perm[:outer_val_n]
    trainval_idx = perm[outer_val_n:]
    X_trval, y_trval = X[trainval_idx], y[trainval_idx]
    X_oval, y_oval = X[outer_val_idx], y[outer_val_idx]

    F_0 = float(y_trval.mean())
    model = EmlBoostModel(F_0=F_0)

    best_oval_mse = float("inf")
    since_improve = 0

    for m in range(M):
        r_trval = y_trval - model.predict(X_trval)

        inner_perm = rng.permutation(len(X_trval))
        inner_val_n = max(int(_INNER_VAL_FRAC * len(X_trval)), 1)
        iv_idx = inner_perm[:inner_val_n]
        tr_idx = inner_perm[inner_val_n:]

        X_tr = X_trval[tr_idx]
        r_tr = r_trval[tr_idx]
        X_iv = X_trval[iv_idx]
        r_iv = r_trval[iv_idx]

        seed = None if random_state is None else random_state * 31 + m
        h_eml = fit_eml_tree(X_tr, r_tr, depth=depth_eml, n_restarts=n_restarts, k=k, random_state=seed)
        h_dt = fit_dt_stump(X_tr, r_tr, depth=depth_dt)

        winner, kind, eta, score = select_winner(h_eml, h_dt, X_iv, r_iv, eta_dt=eta_dt)
        model.weak_learners.append((winner, eta, kind))

        # Inner-val MSE of the selected scaled weak learner.
        scaled = eta * winner.predict(X_iv)
        mse_iv = float(np.mean((r_iv - scaled) ** 2))
        model.history.append(
            RoundRecord(round_index=m, kind=kind, eta=eta, score=score, mse_inner_val=mse_iv)
        )

        oval_mse = float(np.mean((y_oval - model.predict(X_oval)) ** 2))
        if oval_mse < best_oval_mse - 1e-10:
            best_oval_mse = oval_mse
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    return model
