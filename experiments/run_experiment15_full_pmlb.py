"""Experiment 15: full PMLB regression suite, 5 seeds.

SplitBoost (D8_R200_C10 with leaf_eml_cap_k=2.0, min_samples_leaf_eml=30)
vs matched-capacity XGBoost and LightGBM on all 122 PMLB regression
datasets × 5 seeds. Reliability machinery: per-fit try/except,
incremental CSV append, resume-from-checkpoint.

Estimated runtime: 6-8 hours on RTX 3090. Run with:
  PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment15_full_pmlb.py 2>&1 | tee experiments/experiment15/run.log
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev, median

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data, regression_dataset_names
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment15"

DATASETS = list(regression_dataset_names)  # all 122 regression datasets

# Established Exp-13 default config.
MAX_ROUNDS = 200
MAX_DEPTH = 8
PATIENCE = 15
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF = 20
MIN_SAMPLES_LEAF_EML = 30
LEAF_EML_GAIN_THRESHOLD = 0.05
LEAF_EML_RIDGE = 0.0
LEAF_EML_CAP_K = 2.0
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2, 3, 4]

CSV_HEADER = "dataset,seed,config,rmse,fit_time,n_rounds\n"


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


@dataclass
class Failure:
    dataset: str
    seed: int | str  # may be "ALL_SEEDS" for data-fetch failures
    config: str       # "data_fetch" / "split_boost" / "xgboost" / "lightgbm"
    error_type: str
    error_message: str
    error_phase: str  # "fetch" / "split" / "fit" / "predict"


def _fit_split_boost(X_tr, y_tr, seed):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        leaf_eml_ridge=LEAF_EML_RIDGE,
        leaf_eml_cap_k=LEAF_EML_CAP_K,
        use_stacked_blend=False,
        patience=PATIENCE,
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
            max_depth=MAX_DEPTH,
            num_leaves=2**MAX_DEPTH,
            min_data_in_leaf=MIN_SAMPLES_LEAF,
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
        max_depth=MAX_DEPTH,
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        device="cuda",
        verbosity=0,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


FIT_FNS = [
    ("split_boost", _fit_split_boost),
    ("xgboost", _fit_xgb),
    ("lightgbm", _fit_lgb),
]


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def _load_completed(csv_path: Path) -> set[tuple[str, int, str]]:
    """Return set of (dataset, seed, config) triples already in summary.csv."""
    if not csv_path.exists():
        return set()
    completed = set()
    with csv_path.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            completed.add((row["dataset"], int(row["seed"]), row["config"]))
    return completed


def _append_rows(csv_path: Path, rows: list[RunResult]) -> None:
    """Append rows to summary.csv, creating with header if needed."""
    new_file = not csv_path.exists()
    with csv_path.open("a") as fp:
        if new_file:
            fp.write(CSV_HEADER)
        for r in rows:
            fp.write(
                f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n"
            )


def _save_failures(json_path: Path, failures: list[Failure]) -> None:
    with json_path.open("w") as fp:
        json.dump([f.__dict__ for f in failures], fp, indent=2)


def _load_failures(json_path: Path) -> list[Failure]:
    if not json_path.exists():
        return []
    with json_path.open() as fp:
        return [Failure(**d) for d in json.load(fp)]


def _summarize_failure(name: str, seed: int | str, config: str,
                        phase: str, exc: Exception) -> Failure:
    return Failure(
        dataset=name,
        seed=seed,
        config=config,
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],  # truncate
        error_phase=phase,
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "summary.csv"
    failures_path = RESULTS_DIR / "failures.json"

    completed = _load_completed(csv_path)
    failures = _load_failures(failures_path)
    print(
        f"resume: {len(completed)} (dataset, seed, config) triples already complete; "
        f"{len(failures)} prior failures recorded.",
        flush=True,
    )

    for name in DATASETS:
        # Skip if every seed × every model for this dataset is already done.
        expected_count = len(SEEDS) * len(FIT_FNS)
        present = sum(
            1 for (d, s, c) in completed if d == name
        )
        if present >= expected_count:
            print(f"\n=== dataset: {name} (skipping; {present} rows already)", flush=True)
            continue

        print(f"\n=== dataset: {name} ===", flush=True)

        # Fetch + sanitize.
        try:
            X, y = fetch_data(name, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            if len(X) < 20 or X.shape[1] < 1:
                raise ValueError(
                    f"dataset too small after sanitization: n={len(X)}, k={X.shape[1] if len(X) else 0}"
                )
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(_summarize_failure(name, "ALL_SEEDS", "data_fetch", "fetch", e))
            _save_failures(failures_path, failures)
            continue

        rows_for_dataset: list[RunResult] = []

        for seed in SEEDS:
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed,
                )
            except Exception as e:
                print(f"  [seed={seed}] SPLIT FAILED: {type(e).__name__}: {e}", flush=True)
                failures.append(_summarize_failure(name, seed, "data_split", "split", e))
                continue

            for cfg_name, fit_fn in FIT_FNS:
                if (name, seed, cfg_name) in completed:
                    print(f"  [seed={seed}] {cfg_name:>11} SKIPPED (already in CSV)",
                          flush=True)
                    continue
                try:
                    m, t = fit_fn(X_tr, y_tr, seed)
                    rmse = _rmse(m.predict(X_te), y_te)
                    n_rounds = getattr(m, "n_rounds", 0)
                    rows_for_dataset.append(RunResult(
                        dataset=name, seed=seed, config=cfg_name,
                        rmse=rmse, fit_time=t, n_rounds=n_rounds,
                    ))
                    print(
                        f"  [seed={seed}] {cfg_name:>11} ({t:7.1f}s, "
                        f"{n_rounds:>3} rounds)  RMSE={rmse:.4f}",
                        flush=True,
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  [seed={seed}] {cfg_name:>11} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    failures.append(_summarize_failure(name, seed, cfg_name, "fit", e))

        if rows_for_dataset:
            _append_rows(csv_path, rows_for_dataset)
            for r in rows_for_dataset:
                completed.add((r.dataset, r.seed, r.config))
            print(f"  appended {len(rows_for_dataset)} rows to {csv_path.name}",
                  flush=True)
        if failures:
            _save_failures(failures_path, failures)

    # ---- Final aggregation ----
    print("\nfinalizing summary.json + plot + headline stats...", flush=True)

    rows: list[RunResult] = []
    with csv_path.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(RunResult(
                dataset=row["dataset"],
                seed=int(row["seed"]),
                config=row["config"],
                rmse=float(row["rmse"]),
                fit_time=float(row["fit_time"]),
                n_rounds=int(row["n_rounds"]),
            ))

    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.dataset, r.config)
        agg.setdefault(key, {"rmses": [], "times": []})
        agg[key]["rmses"].append(r.rmse)
        agg[key]["times"].append(r.fit_time)
    for key, d in agg.items():
        d["rmse_mean"] = float(mean(d["rmses"]))
        d["rmse_std"] = float(stdev(d["rmses"])) if len(d["rmses"]) > 1 else 0.0
        d["time_mean"] = float(mean(d["times"]))
        d["n_seeds"] = len(d["rmses"])

    # Per-dataset ratios — only datasets where ALL three configs have all 5 seeds.
    full_coverage_datasets = []
    ratios = {}
    for name in DATASETS:
        sb = agg.get((name, "split_boost"))
        xg = agg.get((name, "xgboost"))
        lg = agg.get((name, "lightgbm"))
        if sb and xg and lg and sb["n_seeds"] == len(SEEDS) and xg["n_seeds"] == len(SEEDS) and lg["n_seeds"] == len(SEEDS):
            full_coverage_datasets.append(name)
            ratios[name] = {
                "split_boost_mean": sb["rmse_mean"],
                "split_boost_std": sb["rmse_std"],
                "xgboost_mean": xg["rmse_mean"],
                "lightgbm_mean": lg["rmse_mean"],
                "ratio": sb["rmse_mean"] / xg["rmse_mean"] if xg["rmse_mean"] > 0 else float("nan"),
            }

    ratio_values = [ratios[n]["ratio"] for n in full_coverage_datasets if not np.isnan(ratios[n]["ratio"])]
    n_total = len(ratio_values)
    n_within_10pct = sum(1 for r in ratio_values if r <= 1.10)
    n_within_5pct = sum(1 for r in ratio_values if r <= 1.05)
    n_outright_wins = sum(1 for r in ratio_values if r < 1.00)
    n_catastrophic = sum(1 for r in ratio_values if r > 2.0)
    sorted_ratios = sorted(ratio_values)

    headline_stats = {
        "n_total_datasets": n_total,
        "n_within_10pct": n_within_10pct,
        "n_within_5pct": n_within_5pct,
        "n_outright_wins": n_outright_wins,
        "n_catastrophic": n_catastrophic,
        "frac_within_10pct": n_within_10pct / n_total if n_total else 0.0,
        "frac_within_5pct": n_within_5pct / n_total if n_total else 0.0,
        "frac_outright_wins": n_outright_wins / n_total if n_total else 0.0,
        "frac_catastrophic": n_catastrophic / n_total if n_total else 0.0,
        "mean_ratio": float(mean(ratio_values)) if ratio_values else float("nan"),
        "median_ratio": float(median(ratio_values)) if ratio_values else float("nan"),
        "p25_ratio": float(np.percentile(ratio_values, 25)) if ratio_values else float("nan"),
        "p75_ratio": float(np.percentile(ratio_values, 75)) if ratio_values else float("nan"),
        "min_ratio": float(min(ratio_values)) if ratio_values else float("nan"),
        "max_ratio": float(max(ratio_values)) if ratio_values else float("nan"),
    }

    # ---- summary.json ----
    json_path = RESULTS_DIR / "summary.json"
    out = {
        "config": {
            "max_rounds": MAX_ROUNDS,
            "max_depth": MAX_DEPTH,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "n_eml_candidates": N_EML_CANDIDATES,
            "k_eml": K_EML,
            "k_leaf_eml": K_LEAF_EML,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
            "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
            "leaf_eml_ridge": LEAF_EML_RIDGE,
            "leaf_eml_cap_k": LEAF_EML_CAP_K,
            "n_bins": N_BINS,
            "test_size": TEST_SIZE,
            "seeds": SEEDS,
            "n_datasets_attempted": len(DATASETS),
        },
        "headline_stats": headline_stats,
        "aggregate": {
            ds: {cfg: agg[(ds, cfg)] for cfg in ("split_boost", "xgboost", "lightgbm")
                 if (ds, cfg) in agg}
            for ds in DATASETS
        },
        "ratios": ratios,
        "n_failures": len(failures),
    }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"wrote {json_path}", flush=True)

    # ---- Plot: sorted bars + histogram ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=110,
                                    gridspec_kw={"height_ratios": [2, 1]})
    sorted_pairs = sorted(
        [(n, ratios[n]["ratio"]) for n in full_coverage_datasets],
        key=lambda p: p[1],
    )
    bar_xs = np.arange(len(sorted_pairs))
    bar_ys = [r for _, r in sorted_pairs]
    bar_colors = [
        "#2E86AB" if r < 1.00 else "#588157" if r <= 1.10 else "#E63946" if r <= 2.0 else "#9B2226"
        for r in bar_ys
    ]
    ax1.bar(bar_xs, bar_ys, color=bar_colors, width=0.85)
    ax1.axhline(1.0, color="black", linewidth=1, label="parity")
    ax1.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10%")
    ax1.set_xlim(-0.5, len(sorted_pairs) - 0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("SplitBoost RMSE / XGBoost RMSE (log)")
    ax1.set_title(
        f"Experiment 15: full PMLB regression suite — "
        f"{n_within_10pct}/{n_total} within 10% of XGBoost, "
        f"{n_outright_wins}/{n_total} outright wins"
    )
    ax1.set_xticks([])
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    # Histogram (clip to [0, 2.5] for legibility; catastrophic outliers reported numerically).
    hist_clip = np.clip(ratio_values, 0.0, 2.5)
    ax2.hist(hist_clip, bins=np.arange(0.0, 2.55, 0.05), color="#2E86AB",
             edgecolor="white", alpha=0.85)
    ax2.axvline(1.0, color="black", linewidth=1, label="parity")
    ax2.axvline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10%")
    ax2.set_xlabel("ratio (clipped to [0, 2.5] for histogram; catastrophic >2.5 counted in stats)")
    ax2.set_ylabel("# datasets")
    ax2.set_title("Histogram of mean ratios")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}", flush=True)

    # ---- Console headline ----
    print("\n=== Headline statistics ===", flush=True)
    print(f"Total datasets with full 5-seed coverage on all 3 models: "
          f"{n_total}/{len(DATASETS)}", flush=True)
    print(f"  Within 10%: {n_within_10pct}/{n_total} "
          f"({100 * n_within_10pct / n_total if n_total else 0:.1f}%)", flush=True)
    print(f"  Within 5%:  {n_within_5pct}/{n_total} "
          f"({100 * n_within_5pct / n_total if n_total else 0:.1f}%)", flush=True)
    print(f"  Outright wins: {n_outright_wins}/{n_total} "
          f"({100 * n_outright_wins / n_total if n_total else 0:.1f}%)", flush=True)
    print(f"  Catastrophic (ratio > 2.0): {n_catastrophic}/{n_total}", flush=True)
    print(f"  Mean ratio:   {headline_stats['mean_ratio']:.3f}", flush=True)
    print(f"  Median ratio: {headline_stats['median_ratio']:.3f}", flush=True)
    print(f"  P25 / P75:    {headline_stats['p25_ratio']:.3f} / "
          f"{headline_stats['p75_ratio']:.3f}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
