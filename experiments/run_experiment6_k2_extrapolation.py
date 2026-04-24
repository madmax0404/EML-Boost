"""Experiment 6: two-feature extrapolation.

Experiment 4 showed that on a grammar-expressible k=1 target (exp(x_0)),
the hybrid extrapolates ~50× better than XGBoost; off-grammar targets
(linear, quadratic) fell back to whatever elementary approximation the
exhaustive search chose. Experiment 6 repeats that test with two
features, expanding BOTH dimensions into the extrapolation range.

Formulas (all train in [-1, 1]^2, extrap in [1, 2]^2):

  F1: y = exp(x_0) - log(x_1 + 2)
      — the EML operator applied to (x_0, x_1+2). Grammar-expressible at
        depth 2 up to the "+2" offset, which exhaustive should approach
        with scaled-and-shifted exp/log. Expected: clean hybrid win.
  F2: y = x_0 + x_1
      — linear sum, not in grammar at depth 2. Expected: trees win via
        boundary-flat extrapolation; hybrid's elementary fallback grows
        too fast.
  F3: y = x_0 * x_1
      — product, requires depth 3 to express via exp(log(a)+log(b)).
        Expected: both bad, see how they fail differently.
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

RESULTS_DIR = Path(__file__).resolve().parent / "experiment6"
N_TRAIN = 500
N_TEST = 200
NOISE_STD = 0.02
SEED = 0

MAX_ROUNDS = 100
DEPTH_DT = 6
DEPTH_EML = 2
K = 2

FORMULAS: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
    ("exp(x_0) - log(x_1 + 2)", lambda x0, x1: np.exp(x0) - np.log(x1 + 2)),
    ("x_0 + x_1",                lambda x0, x1: x0 + x1),
    ("x_0 * x_1",                lambda x0, x1: x0 * x1),
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
    extrap_y_true: np.ndarray
    extrap_y_pred: np.ndarray


def _make_dataset(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    def sample(n: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        X = rng.uniform(lo, hi, size=(n, 2))
        y = f(X[:, 0], X[:, 1]) + rng.normal(0, NOISE_STD, size=n)
        return X, y
    X_tr, y_tr = sample(N_TRAIN, TRAIN_LO, TRAIN_HI)
    X_in, y_in = sample(N_TEST, TRAIN_LO, TRAIN_HI)
    X_ex, y_ex = sample(N_TEST, EXTRAP_LO, EXTRAP_HI)
    return X_tr, y_tr, X_in, y_in, X_ex, y_ex


def _fit_hybrid(X_tr, y_tr, seed):
    start = time.time()
    m = EmlBoostRegressor(
        max_rounds=MAX_ROUNDS,
        depth_eml=DEPTH_EML,
        depth_dt=DEPTH_DT,
        n_restarts=6,
        k=K,
        patience=MAX_ROUNDS,
        eta_dt=0.1,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_lgb_gpu(X_tr, y_tr, seed):
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
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=MAX_ROUNDS,
    )
    return m, time.time() - start


def _fit_xgb_gpu(X_tr, y_tr, seed):
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
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _evaluate(
    name: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    elapsed: float,
    X_tr, y_tr, X_in, y_in, X_ex, y_ex,
):
    def mse(X, y):
        p = np.asarray(predict_fn(X), dtype=np.float64)
        return float(np.mean((p - y) ** 2))
    ex_pred = np.asarray(predict_fn(X_ex), dtype=np.float64)
    return ModelResult(
        model_name=name,
        train_mse=mse(X_tr, y_tr),
        in_range_mse=mse(X_in, y_in),
        extrap_mse=float(np.mean((ex_pred - y_ex) ** 2)),
        elapsed_seconds=elapsed,
        extrap_y_true=y_ex.copy(),
        extrap_y_pred=ex_pred,
    )


def _plot_all(formula_results: dict[str, list[ModelResult]], path: Path) -> None:
    n_formulas = len(formula_results)
    fig, axes = plt.subplots(2, n_formulas, figsize=(5 * n_formulas, 9), dpi=110)
    if n_formulas == 1:
        axes = axes.reshape(2, 1)
    colors = {"Hybrid": "#2E86AB", "LightGBM": "#588157", "XGBoost": "#9B2226"}
    markers = {"Hybrid": "o", "LightGBM": "s", "XGBoost": "D"}

    for col, (formula_str, results) in enumerate(formula_results.items()):
        # Top row: (y_true, y_pred) scatter on extrap test set.
        ax_top = axes[0, col]
        # Diagonal reference
        y_all = np.concatenate([r.extrap_y_true for r in results])
        lo, hi = float(y_all.min()), float(y_all.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax_top.plot(
            [lo - pad, hi + pad], [lo - pad, hi + pad],
            "--", color="black", linewidth=1, label="perfect",
        )
        for r in results:
            ax_top.scatter(
                r.extrap_y_true, r.extrap_y_pred,
                s=14, alpha=0.6,
                c=colors[r.model_name],
                marker=markers[r.model_name],
                label=r.model_name,
                edgecolors="none",
            )
        ax_top.set_xlabel("y_true (extrap test)")
        ax_top.set_ylabel("y_pred")
        ax_top.set_title(f"y = {formula_str}")
        ax_top.legend(loc="best", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom row: MSE bars per split
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

    for formula_str, f in FORMULAS:
        print(f"\n=== formula: y = {formula_str} ===")
        X_tr, y_tr, X_in, y_in, X_ex, y_ex = _make_dataset(f, SEED)
        results: list[ModelResult] = []

        m_hy, t_hy = _fit_hybrid(X_tr, y_tr, SEED)
        r_hy = _evaluate("Hybrid", lambda X: m_hy.predict(X), t_hy,
                         X_tr, y_tr, X_in, y_in, X_ex, y_ex)
        results.append(r_hy)
        print(
            f"  Hybrid    ({t_hy:5.1f}s): "
            f"train={r_hy.train_mse:.4f}  in={r_hy.in_range_mse:.4f}  extrap={r_hy.extrap_mse:.4f}"
        )
        if m_hy.formula is not None:
            f_str = str(m_hy.formula)
            if len(f_str) > 200:
                f_str = f_str[:200] + " ... [truncated]"
            print(f"    recovered: {f_str}")

        m_lgb, t_lgb = _fit_lgb_gpu(X_tr, y_tr, SEED)
        r_lgb = _evaluate("LightGBM", lambda X: m_lgb.predict(X), t_lgb,
                          X_tr, y_tr, X_in, y_in, X_ex, y_ex)
        results.append(r_lgb)
        print(
            f"  LightGBM  ({t_lgb:5.1f}s): "
            f"train={r_lgb.train_mse:.4f}  in={r_lgb.in_range_mse:.4f}  extrap={r_lgb.extrap_mse:.4f}"
        )

        m_xgb, t_xgb = _fit_xgb_gpu(X_tr, y_tr, SEED)
        r_xgb = _evaluate("XGBoost", lambda X: m_xgb.predict(X), t_xgb,
                          X_tr, y_tr, X_in, y_in, X_ex, y_ex)
        results.append(r_xgb)
        print(
            f"  XGBoost   ({t_xgb:5.1f}s): "
            f"train={r_xgb.train_mse:.4f}  in={r_xgb.in_range_mse:.4f}  extrap={r_xgb.extrap_mse:.4f}"
        )

        formula_results[formula_str] = results

    # CSV
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

    # JSON
    json_path = RESULTS_DIR / "summary.json"
    with json_path.open("w") as fp:
        out = {
            fstr: {
                r.model_name: {
                    "train_mse": r.train_mse,
                    "in_range_mse": r.in_range_mse,
                    "extrap_mse": r.extrap_mse,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in rs
            }
            for fstr, rs in formula_results.items()
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
                    "k": K,
                    "train_range": [TRAIN_LO, TRAIN_HI],
                    "extrap_range": [EXTRAP_LO, EXTRAP_HI],
                    "seed": SEED,
                },
                "results": out,
            },
            fp, indent=2,
        )
    print(f"wrote {json_path}")

    plot_path = RESULTS_DIR / "extrapolation_plots.png"
    _plot_all(formula_results, plot_path)
    print(f"wrote {plot_path}")

    print("\n=== Extrapolation MSE ranking per formula ===")
    for fstr, rs in formula_results.items():
        ordered = sorted(rs, key=lambda r: r.extrap_mse)
        ranking = " < ".join(f"{r.model_name}({r.extrap_mse:.3f})" for r in ordered)
        print(f"  {fstr:>26}: {ranking}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
