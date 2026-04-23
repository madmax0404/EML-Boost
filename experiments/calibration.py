"""Calibration experiment from spec section 4.2 / 9.3.

Sweeps datasets with varying fractions of elementary signal vs categorical
signal. Trains EML-Boost on each. Records EML-win rate per boosting round.
Aggregated, this produces the graceful-degradation curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from eml_boost import EmlBoostRegressor
from eml_boost.datasets import generate_mixed_regime, generate_pure_elementary, generate_pure_dt_regime
from eml_boost.weak_learners.base import WeakLearnerKind


@dataclass
class CalibrationResult:
    fractions: list[float]
    eml_win_rates: list[float] = field(default_factory=list)
    per_fraction_round_counts: list[int] = field(default_factory=list)
    # Per-fraction mean coverage (formula_part's share of explained variance,
    # spec 7.3). Robust to "EML succeeds in one round then DT fits noise":
    # a single strong EML round produces high coverage even if it only wins
    # 1 of N rounds.
    eml_coverages: list[float] = field(default_factory=list)
    # Per-fraction mean held-out MSE for the hybrid regressor and the
    # DT-only baseline (same capacity). `dt_improvement` is the fractional
    # reduction of the hybrid's test MSE against the baseline's — the true
    # test for "EML is pulling its weight on elementary signals."
    hybrid_test_mse: list[float] = field(default_factory=list)
    dt_only_test_mse: list[float] = field(default_factory=list)
    dt_improvement: list[float] = field(default_factory=list)


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
        coverages: list[float] = []
        hybrid_mses: list[float] = []
        dt_only_mses: list[float] = []
        for rep in range(n_datasets_per_fraction):
            sub_seed = int(rng.integers(0, 10**9))
            X, y = _compose_dataset(frac, n=n, seed=sub_seed)

            # Train/test split: 70/30 on a fresh permutation per dataset.
            perm = np.random.default_rng(sub_seed + 7).permutation(len(X))
            n_tr = int(0.7 * len(X))
            tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            # Hybrid
            model = EmlBoostRegressor(
                max_rounds=max_rounds,
                depth_eml=depth_eml,
                depth_dt=depth_dt,
                n_restarts=n_restarts,
                k=min(3, X_tr.shape[1]),
                patience=max_rounds,
                random_state=sub_seed,
            )
            model.fit(X_tr, y_tr)
            for rec in model.history:
                total_rounds += 1
                if rec.kind == WeakLearnerKind.EML:
                    eml_wins += 1
            coverages.append(float(model.coverage(X_tr)))
            hybrid_test_pred = model.predict(X_te)
            hybrid_mses.append(float(np.mean((y_te - hybrid_test_pred) ** 2)))

            # DT-only baseline: same capacity (max_rounds stumps) via LightGBM.
            # shrinkage=0.1 matches the hybrid's DT branch step size.
            dt_only = lgb.train(
                dict(
                    objective="regression_l2",
                    max_depth=depth_dt,
                    num_leaves=2**depth_dt,
                    min_data_in_leaf=20,
                    learning_rate=0.1,
                    verbose=-1,
                ),
                lgb.Dataset(X_tr, label=y_tr),
                num_boost_round=max_rounds,
            )
            dt_only_test_pred = dt_only.predict(X_te)
            dt_only_mses.append(float(np.mean((y_te - dt_only_test_pred) ** 2)))

        rate = eml_wins / max(total_rounds, 1)
        result.eml_win_rates.append(rate)
        result.per_fraction_round_counts.append(total_rounds)
        result.eml_coverages.append(float(np.mean(coverages)))
        mean_hybrid = float(np.mean(hybrid_mses))
        mean_dt = float(np.mean(dt_only_mses))
        result.hybrid_test_mse.append(mean_hybrid)
        result.dt_only_test_mse.append(mean_dt)
        result.dt_improvement.append(1.0 - mean_hybrid / max(mean_dt, 1e-12))
    return result
