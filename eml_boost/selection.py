"""BIC-based weak-learner selection and per-round learned EML step size."""

from __future__ import annotations

import math

import numpy as np

from eml_boost.weak_learners.base import WeakLearner, WeakLearnerKind
from eml_boost.weak_learners.dt import DtWeakLearner
from eml_boost.weak_learners.eml import EmlWeakLearner

_ETA_DT_DEFAULT = 0.1
_LS_EPS = 1e-8


def bic_score(targets: np.ndarray, predictions: np.ndarray, params: int) -> float:
    """BIC = n * log(MSE) + params * log(n). Lower is better."""
    n = len(targets)
    residual = targets - predictions
    mse = max(float(np.mean(residual**2)), 1e-30)
    return n * math.log(mse) + params * math.log(n)


def learned_eta(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Closed-form 1-D least-squares scale: <r, h(X)> / ||h(X)||^2."""
    num = float(np.dot(predictions, targets))
    den = float(np.dot(predictions, predictions) + _LS_EPS)
    return num / den


def select_winner(
    eml: EmlWeakLearner,
    dt: DtWeakLearner,
    X_val: np.ndarray,
    r_val: np.ndarray,
    eta_dt: float = _ETA_DT_DEFAULT,
) -> tuple[WeakLearner, WeakLearnerKind, float, float]:
    """Pick the lower-BIC of the two. Returns (learner, kind, eta_used, score)."""
    eml_pred = eml.predict(X_val)
    dt_pred = dt.predict(X_val)

    eta_eml = learned_eta(eml_pred, r_val)
    scaled_eml = eta_eml * eml_pred
    scaled_dt = eta_dt * dt_pred

    score_eml = bic_score(r_val, scaled_eml, eml.params_count())
    score_dt = bic_score(r_val, scaled_dt, dt.params_count())

    if score_eml <= score_dt:
        return eml, WeakLearnerKind.EML, eta_eml, score_eml
    return dt, WeakLearnerKind.DT, eta_dt, score_dt
