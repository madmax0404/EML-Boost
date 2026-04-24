"""Experiment 11: adaptive leaf-prediction cap.

Sweeps 5 SplitBoost configurations (gated path, varying leaf_eml_cap_k)
across 3 seeds and the 7 PMLB datasets used in Experiments 8-10, plus
XGBoost and LightGBM baselines. Writes aggregate stats plus per-leaf
cap distributions to experiments/experiment11/.
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

RESULTS_DIR = Path(__file__).resolve().parent / "experiment11"

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

# (config_id, leaf_eml_cap_k) tuples. Gated path only.
SPLIT_CONFIGS = [
    ("C0",      0.0),
    ("C_tight", 1.0),
    ("C_loose", 2.0),
    ("C_med",   5.0),
    ("C_wide",  10.0),
]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, leaf_eml_cap_k: float):
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
        leaf_eml_cap_k=leaf_eml_cap_k,
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


def _collect_caps(boost: EmlSplitBoostRegressor) -> list[float]:
    """Return the stored cap for every EmlLeafNode in the ensemble."""
    out: list[float] = []
    def walk(node):
        if isinstance(node, EmlLeafNode):
            out.append(float(node.cap))
        elif isinstance(node, InternalNode):
            walk(node.left); walk(node.right)
    for tree in boost._trees:
        walk(tree._root)
    return out


def _count_capped_preds(boost: EmlSplitBoostRegressor, X_te: np.ndarray) -> tuple[int, int]:
    """Approximate: for each tree, compare η·eml+β pre-clip vs post-clip on X_te.

    Returns (n_capped, n_total) aggregated across all trees. A prediction
    is "capped" if the pre-clip magnitude exceeded the leaf cap. This
    requires re-evaluating the leaf forward pass without the clip; done
    here with a local copy that walks the tree and records the hit count.
    """
    caps_seen = _collect_caps(boost)
    if not caps_seen or all(np.isinf(c) for c in caps_seen):
        return 0, 0
    n_capped = 0
    n_total = 0
    import torch
    from eml_boost._triton_exhaustive import evaluate_trees_torch
    from eml_boost.tree_split.nodes import LeafNode, EmlLeafNode as _Eml, InternalNode as _Int
    def _walk_predict(node, X_local):
        nonlocal n_capped, n_total
        if isinstance(node, LeafNode):
            return
        if isinstance(node, _Eml):
            if len(X_local) == 0:
                return
            X_leaf = X_local[:, list(node.feature_subset)]
            mean = np.asarray(node.feature_mean, dtype=np.float64)
            std = np.asarray(node.feature_std, dtype=np.float64)
            X_leaf_std = np.clip((X_leaf - mean) / std, -3.0, 3.0)
            X_t = torch.tensor(X_leaf_std, dtype=torch.float64)
            desc_t = torch.tensor([node.snapped.terminal_choices], dtype=torch.int32)
            preds = evaluate_trees_torch(desc_t, X_t, node.snapped.k)
            vals = preds.squeeze(0).cpu().numpy().astype(np.float64)
            pre = node.eta * vals + node.bias
            if node.cap < float("inf"):
                n_capped += int(np.sum(np.abs(pre) > node.cap))
                n_total += len(pre)
            return
        from eml_boost.tree_split.tree import EmlSplitTreeRegressor as _Reg
        mask = _Reg._evaluate_split(node.split, X_local)
        _walk_predict(node.left, X_local[mask])
        _walk_predict(node.right, X_local[~mask])
    for tree in boost._trees:
        _walk_predict(tree._root, X_te)
    return n_capped, n_total


def _summarize_caps(caps: list[float], n_capped_preds: int, n_total_preds: int) -> dict:
    """Aggregate per-leaf cap distribution + hit-rate stats."""
    if not caps:
        return {
            "n_eml_leaves": 0,
            "mean_cap": float("nan"),
            "median_cap": float("nan"),
            "max_cap": float("nan"),
            "n_capped_preds_on_test": n_capped_preds,
            "n_total_eml_preds_on_test": n_total_preds,
            "pct_capped": float("nan"),
        }
    finite_caps = [c for c in caps if not np.isinf(c)]
    if not finite_caps:
        return {
            "n_eml_leaves": len(caps),
            "mean_cap": float("inf"),
            "median_cap": float("inf"),
            "max_cap": float("inf"),
            "n_capped_preds_on_test": 0,
            "n_total_eml_preds_on_test": 0,
            "pct_capped": 0.0,
        }
    arr = np.asarray(finite_caps)
    return {
        "n_eml_leaves": len(caps),
        "mean_cap": float(arr.mean()),
        "median_cap": float(np.median(arr)),
        "max_cap": float(arr.max()),
        "n_capped_preds_on_test": n_capped_preds,
        "n_total_eml_preds_on_test": n_total_preds,
        "pct_capped": float(n_capped_preds / n_total_preds) if n_total_preds else 0.0,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    cap_stats: dict[str, dict[str, dict[int, dict]]] = {}

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        cap_stats[name] = {cfg_id: {} for cfg_id, _ in SPLIT_CONFIGS}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            for cfg_id, cap_k in SPLIT_CONFIGS:
                m, t = _fit_split_boost(X_tr, y_tr, seed, leaf_eml_cap_k=cap_k)
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
                caps = _collect_caps(m)
                n_capped, n_total_preds = _count_capped_preds(m, X_te)
                cap_stats[name][cfg_id][seed] = _summarize_caps(caps, n_capped, n_total_preds)

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

    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,config,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS, "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES, "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "n_bins": N_BINS, "test_size": TEST_SIZE, "seeds": SEEDS,
        "split_configs": [{"id": c, "leaf_eml_cap_k": k} for c, k in SPLIT_CONFIGS],
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

    cap_json_path = RESULTS_DIR / "cap_stats.json"
    with cap_json_path.open("w") as fp:
        json.dump(cap_stats, fp, indent=2)
    print(f"wrote {cap_json_path}")

    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    split_cfg_ids = [c for c, _ in SPLIT_CONFIGS]
    all_bars = split_cfg_ids + ["xgboost"]
    colors = ["#2E86AB", "#4EA1B2", "#588157", "#7FA65A", "#E9C46A", "#9B2226"]

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
    ax1.set_title("Experiment 11: adaptive leaf-prediction cap (log scale)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3, axis="y")

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
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    print("\n=== Aggregate summary (mean over 3 seeds, ratio vs XGBoost) ===")
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
