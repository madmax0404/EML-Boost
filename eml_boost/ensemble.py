"""sklearn-style public API for EML-Boost."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from eml_boost.training import EmlBoostModel, boost


class EmlBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        max_rounds: int = 500,
        depth_eml: int = 3,
        depth_dt: int = 3,
        n_restarts: int = 20,
        k: int = 3,
        patience: int = 15,
        eta_dt: float = 0.1,
        random_state: int | None = None,
    ):
        self.max_rounds = max_rounds
        self.depth_eml = depth_eml
        self.depth_dt = depth_dt
        self.n_restarts = n_restarts
        self.k = k
        self.patience = patience
        self.eta_dt = eta_dt
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmlBoostRegressor":
        self._model: EmlBoostModel = boost(
            X,
            y,
            M=self.max_rounds,
            depth_eml=self.depth_eml,
            depth_dt=self.depth_dt,
            n_restarts=self.n_restarts,
            k=min(self.k, X.shape[1]),
            patience=self.patience,
            eta_dt=self.eta_dt,
            random_state=self.random_state,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def describe(self, X: np.ndarray | None = None) -> str:
        return self._model.describe(X)

    def coverage(self, X: np.ndarray) -> float:
        return self._model.coverage(X)

    def formula_predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.formula_predict(X)

    @property
    def formula(self):
        """Recovered closed-form sympy expression, or None (spec 7.2)."""
        return self._model.formula

    def is_exact_recovery(self, X: np.ndarray, threshold: float = 0.99) -> bool:
        """True when formula coverage exceeds threshold (spec 7.3)."""
        return self._model.is_exact_recovery(X, threshold)

    @property
    def history(self):
        return self._model.history
