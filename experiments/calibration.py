"""Calibration experiment from spec section 4.2 / 9.3.

Sweeps datasets with varying fractions of elementary signal vs categorical
signal. Trains EML-Boost on each. Records EML-win rate per boosting round.
Aggregated, this produces the graceful-degradation curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eml_boost import EmlBoostRegressor
from eml_boost.datasets import generate_mixed_regime, generate_pure_elementary, generate_pure_dt_regime
from eml_boost.weak_learners.base import WeakLearnerKind


@dataclass
class CalibrationResult:
    fractions: list[float]
    eml_win_rates: list[float] = field(default_factory=list)
    per_fraction_round_counts: list[int] = field(default_factory=list)


def _compose_dataset(frac_elementary: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if frac_elementary <= 0.0:
        return generate_pure_dt_regime(n=n, n_numeric=1, n_cat=2, seed=seed)
    if frac_elementary >= 1.0:
        return generate_pure_elementary(
            formula="exp(x0) + 0.5 * x1**2", n=n, n_features=2, seed=seed
        )[:2]
    # Blend: generate both, mix targets
    X_e, y_e, _ = generate_pure_elementary(
        formula="exp(x0) + 0.5 * x1**2", n=n, n_features=2, seed=seed
    )
    X_d, y_d = generate_pure_dt_regime(n=n, n_numeric=1, n_cat=2, seed=seed + 1)
    # Concatenate feature columns; blend targets
    X = np.concatenate([X_e, X_d], axis=1)
    y = frac_elementary * y_e + (1 - frac_elementary) * y_d
    return X, y


def run_calibration(
    *,
    elementary_fractions: list[float],
    n_datasets_per_fraction: int,
    n: int,
    seed: int,
    max_rounds: int,
    n_restarts: int,
    depth_eml: int,
    depth_dt: int,
) -> CalibrationResult:
    result = CalibrationResult(fractions=list(elementary_fractions))
    rng = np.random.default_rng(seed)
    for frac in elementary_fractions:
        eml_wins = 0
        total_rounds = 0
        for rep in range(n_datasets_per_fraction):
            sub_seed = int(rng.integers(0, 10**9))
            X, y = _compose_dataset(frac, n=n, seed=sub_seed)
            model = EmlBoostRegressor(
                max_rounds=max_rounds,
                depth_eml=depth_eml,
                depth_dt=depth_dt,
                n_restarts=n_restarts,
                k=min(3, X.shape[1]),
                patience=max_rounds,  # disable early stopping for clean win-rate signal
                random_state=sub_seed,
            )
            model.fit(X, y)
            for rec in model.history:
                total_rounds += 1
                if rec.kind == WeakLearnerKind.EML:
                    eml_wins += 1
        rate = eml_wins / max(total_rounds, 1)
        result.eml_win_rates.append(rate)
        result.per_fraction_round_counts.append(total_rounds)
    return result
