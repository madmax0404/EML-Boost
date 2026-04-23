"""Experiment 4: Extrapolation beyond training range.

Piecewise-constant boosting (LightGBM, XGBoost) saturates at the nearest
training-range boundary when asked to predict outside that range. Closed-form
formulas don't. If EML-Boost's hybrid recovers a true elementary form in one
round, it should extrapolate correctly; trees should extrapolate as constants
regardless of capacity.

This experiment fits the hybrid and two tree baselines at a capacity well
above the calibration sweep (100 rounds, depth-6 DT branch), on three
formulas that cover different extrapolation personalities:

  F1: y = exp(x_0)           — expressible at depth 2; should be the clean
                                 win for the hybrid on extrapolation.
  F2: y = x_0                 — linear; every method extrapolates correctly.
  F3: y = x_0**2 + 0.5        — not expressible at depth 2 under Option A;
                                 honest test that the hybrid isn't magic and
                                 extrapolation depends on recovery.

For each formula we fit on x_0 in [-1, 1] and evaluate on two disjoint
test sets: an in-range holdout x_0 in [-1, 1] and an extrapolation range
x_0 in [1, 2]. Metric is MSE on each set.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from eml_boost import EmlBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment4"
N_TRAIN = 500
N_TEST = 200
NOISE_STD = 0.02
SEED = 0

# Strong but equal capacity for all models.
MAX_ROUNDS = 100
DEPTH_DT = 6      # hybrid's DT branch and both tree baselines
DEPTH_EML = 2    # depth 2 exhaustive — the hybrid's closed-form branch

# Formulas — keep to k=1 (single feature) for clean extrapolation visualization.
FORMULAS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("exp(x_0)",       lambda x: np.exp(x)),
    ("x_0",             lambda x: x.copy()),
    ("x_0**2 + 0.5",   lambda x: x**2 + 0.5),
]

TRAIN_LO, TRAIN_HI = -1.0, 1.0
EXTRAP_LO, EXTRAP_HI = 1.0, 2.0


@dataclass
class ModelResult:
    model_name: str
    train_mse: float
    in_range_mse: float
    extrap_mse: float
    elapsed_seconds: float
    pred_curve: np.ndarray  # predictions over the full plot range


def _make_dataset(
    f: Callable[[np.ndarray], np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(TRAIN_LO, TRAIN_HI, size=N_TRAIN).reshape(-1, 1)
    y_tr = f(x_tr[:, 0]) + rng.normal(0, NOISE_STD, size=N_TRAIN)

    x_in = rng.uniform(TRAIN_LO, TRAIN_HI, size=N_TEST).reshape(-1, 1)
    y_in = f(x_in[:, 0]) + rng.normal(0, NOISE_STD, size=N_TEST)

    x_ex = rng.uniform(EXTRAP_LO, EXTRAP_HI, size=N_TEST).reshape(-1, 1)
    y_ex = f(x_ex[:, 0]) + rng.normal(0, NOISE_STD, size=N_TEST)
    return x_tr, y_tr, x_in, y_in, x_ex, y_ex


def _fit_hybrid(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int
) -> tuple[EmlBoostRegressor, float]:
    start = time.time()
    model = EmlBoostRegressor(
        max_rounds=MAX_ROUNDS,
        depth_eml=DEPTH_EML,
        depth_dt=DEPTH_DT,
        n_restarts=6,
        k=1,
        patience=MAX_ROUNDS,  # disable early stopping
        eta_dt=0.1,
        random_state=seed,
    )
    model.fit(x_tr, y_tr)
    return model, time.time() - start


def _fit_lgb_gpu(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int
) -> tuple[lgb.Booster, float]:
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=DEPTH_DT,
            num_leaves=2**DEPTH_DT,
            min_data_in_leaf=20,
            learning_rate=0.1,
            device="gpu",
            seed=seed,
            verbose=-1,
        ),
        lgb.Dataset(x_tr, label=y_tr),
        num_boost_round=MAX_ROUNDS,
    )
    return m, time.time() - start


def _fit_xgb_gpu(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int
) -> tuple[xgb.XGBRegressor, float]:
    start = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=DEPTH_DT,
        n_estimators=MAX_ROUNDS,
        learning_rate=0.1,
        device="cuda",
        verbosity=0,
        random_state=seed,
    )
    m.fit(x_tr, y_tr)
    return m, time.time() - start


def _evaluate(
    name: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    elapsed: float,
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_in: np.ndarray, y_in: np.ndarray,
    x_ex: np.ndarray, y_ex: np.ndarray,
    curve_x: np.ndarray,
) -> ModelResult:
    tr_mse = float(np.mean((predict_fn(x_tr) - y_tr) ** 2))
    in_mse = float(np.mean((predict_fn(x_in) - y_in) ** 2))
    ex_mse = float(np.mean((predict_fn(x_ex) - y_ex) ** 2))
    curve = np.asarray(predict_fn(curve_x.reshape(-1, 1)), dtype=np.float64)
    return ModelResult(
        model_name=name,
        train_mse=tr_mse,
        in_range_mse=in_mse,
        extrap_mse=ex_mse,
        elapsed_seconds=elapsed,
        pred_curve=curve,
    )


def _plot_all(
    formula_results: dict[str, list[ModelResult]],
    f_targets: dict[str, Callable[[np.ndarray], np.ndarray]],
    curve_x: np.ndarray,
    path: Path,
) -> None:
    n_formulas = len(formula_results)
    fig, axes = plt.subplots(2, n_formulas, figsize=(5 * n_formulas, 8), dpi=110)
    if n_formulas == 1:
        axes = axes.reshape(2, 1)
    colors = {"Hybrid": "#2E86AB", "LightGBM": "#588157", "XGBoost": "#9B2226"}
    linestyles = {"Hybrid": "-", "LightGBM": "--", "XGBoost": "-."}

    for col, (formula_str, results) in enumerate(formula_results.items()):
        # Top row: prediction curves over the full range
        ax_top = axes[0, col]
        truth = f_targets[formula_str](curve_x)
        ax_top.plot(curve_x, truth, color="black", linewidth=2.5, label="truth")
        for r in results:
            ax_top.plot(
                curve_x, r.pred_curve,
                color=colors[r.model_name],
                linestyle=linestyles[r.model_name],
                linewidth=1.5,
                label=r.model_name,
            )
        ax_top.axvspan(TRAIN_LO, TRAIN_HI, color="gray", alpha=0.08, label="train range")
        ax_top.axvspan(EXTRAP_LO, EXTRAP_HI, color="gold", alpha=0.08, label="extrap range")
        ax_top.set_title(f"y = {formula_str}")
        ax_top.set_xlabel("x_0")
        ax_top.set_ylabel("y")
        ax_top.legend(loc="best", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom row: grouped bar chart of MSEs
        ax_bot = axes[1, col]
        xs = np.arange(3)
        width = 0.25
        for i, r in enumerate(results):
            offsets = xs + (i - 1) * width
            values = [r.train_mse, r.in_range_mse, r.extrap_mse]
            ax_bot.bar(
                offsets, values, width,
                color=colors[r.model_name],
                label=r.model_name,
            )
        ax_bot.set_xticks(xs)
        ax_bot.set_xticklabels(["train", "in-range test", "extrap test"])
        ax_bot.set_ylabel("MSE")
        ax_bot.set_yscale("symlog", linthresh=1e-3)
        ax_bot.set_title("MSE per split (symlog)")
        ax_bot.grid(True, alpha=0.3, axis="y")
        ax_bot.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(path)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    formula_results: dict[str, list[ModelResult]] = {}
    f_targets: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    curve_x = np.linspace(TRAIN_LO - 0.3, EXTRAP_HI + 0.3, 400)

    for formula_str, f in FORMULAS:
        print(f"\n=== formula: y = {formula_str} ===")
        f_targets[formula_str] = f

        x_tr, y_tr, x_in, y_in, x_ex, y_ex = _make_dataset(f, SEED)

        results: list[ModelResult] = []

        # Hybrid
        m_hy, t_hy = _fit_hybrid(x_tr, y_tr, SEED)
        r_hy = _evaluate(
            "Hybrid", lambda X: m_hy.predict(X), t_hy,
            x_tr, y_tr, x_in, y_in, x_ex, y_ex, curve_x,
        )
        results.append(r_hy)
        print(
            f"  Hybrid    ({t_hy:5.1f}s): "
            f"train={r_hy.train_mse:.4f}  in={r_hy.in_range_mse:.4f}  extrap={r_hy.extrap_mse:.4f}"
        )
        if m_hy.formula is not None:
            print(f"    recovered formula: {m_hy.formula}")

        # LightGBM (GPU)
        m_lgb, t_lgb = _fit_lgb_gpu(x_tr, y_tr, SEED)
        r_lgb = _evaluate(
            "LightGBM", lambda X: m_lgb.predict(X), t_lgb,
            x_tr, y_tr, x_in, y_in, x_ex, y_ex, curve_x,
        )
        results.append(r_lgb)
        print(
            f"  LightGBM  ({t_lgb:5.1f}s): "
            f"train={r_lgb.train_mse:.4f}  in={r_lgb.in_range_mse:.4f}  extrap={r_lgb.extrap_mse:.4f}"
        )

        # XGBoost (GPU)
        m_xgb, t_xgb = _fit_xgb_gpu(x_tr, y_tr, SEED)
        r_xgb = _evaluate(
            "XGBoost", lambda X: m_xgb.predict(X), t_xgb,
            x_tr, y_tr, x_in, y_in, x_ex, y_ex, curve_x,
        )
        results.append(r_xgb)
        print(
            f"  XGBoost   ({t_xgb:5.1f}s): "
            f"train={r_xgb.train_mse:.4f}  in={r_xgb.in_range_mse:.4f}  extrap={r_xgb.extrap_mse:.4f}"
        )

        formula_results[formula_str] = results

    # Write CSV + JSON
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("formula,model,train_mse,in_range_mse,extrap_mse,elapsed_seconds\n")
        for fstr, rs in formula_results.items():
            for r in rs:
                fp.write(
                    f'"{fstr}",{r.model_name},{r.train_mse},{r.in_range_mse},'
                    f"{r.extrap_mse},{r.elapsed_seconds}\n"
                )
    print(f"\nwrote {csv_path}")

    json_path = RESULTS_DIR / "summary.json"
    with json_path.open("w") as fp:
        out = {}
        for fstr, rs in formula_results.items():
            out[fstr] = {
                r.model_name: {
                    "train_mse": r.train_mse,
                    "in_range_mse": r.in_range_mse,
                    "extrap_mse": r.extrap_mse,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in rs
            }
        json.dump(
            {
                "config": {
                    "n_train": N_TRAIN,
                    "n_test": N_TEST,
                    "noise_std": NOISE_STD,
                    "max_rounds": MAX_ROUNDS,
                    "depth_dt": DEPTH_DT,
                    "depth_eml": DEPTH_EML,
                    "train_range": [TRAIN_LO, TRAIN_HI],
                    "extrap_range": [EXTRAP_LO, EXTRAP_HI],
                    "seed": SEED,
                },
                "results": out,
            },
            fp, indent=2,
        )
    print(f"wrote {json_path}")

    # Plot
    plot_path = RESULTS_DIR / "extrapolation_plots.png"
    _plot_all(formula_results, f_targets, curve_x, plot_path)
    print(f"wrote {plot_path}")

    # Winners at extrap
    print("\n=== Extrapolation MSE ranking per formula ===")
    for fstr, rs in formula_results.items():
        ordered = sorted(rs, key=lambda r: r.extrap_mse)
        ranking = " < ".join(f"{r.model_name}({r.extrap_mse:.3f})" for r in ordered)
        print(f"  {fstr:>18}: {ranking}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
