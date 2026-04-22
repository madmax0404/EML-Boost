"""Outer boosting loop — spec section 4.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eml_boost.selection import select_winner
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
