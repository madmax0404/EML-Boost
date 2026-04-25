"""Experiment 16: leaf_l2=1.0 validation on Exp 15 losers.

Re-fits SplitBoost only (with the new leaf_l2=1.0 default, otherwise
Exp-15 defaults) on the 20 PMLB datasets where Exp 15's mean SplitBoost
ratio vs XGBoost was > 1.00. xgb/lgb numbers are read from
experiments/experiment15/summary.csv and not re-fit. Writes a side-by-
side comparison.md.

Estimated runtime: 100 fits at small-medium dataset sizes ≈ 5-15
minutes on RTX 3090.

Usage:
  PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment16_leaf_l2_validation.py 2>&1 | tee experiments/experiment16/run.log
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

EXP15_DIR = Path(__file__).resolve().parent / "experiment15"
EXP16_DIR = Path(__file__).resolve().parent / "experiment16"

# Match Exp 15 config exactly except leaf_l2 (which defaults to 1.0 post-Task-7).
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


def _fit_split_boost(X_tr, y_tr, seed):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF, n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML, n_bins=N_BINS, histogram_min_n=500, use_gpu=True,
        k_leaf_eml=K_LEAF_EML, min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        leaf_eml_ridge=LEAF_EML_RIDGE, leaf_eml_cap_k=LEAF_EML_CAP_K,
        leaf_l2=1.0,                          # the change being validated
        use_stacked_blend=False,
        patience=PATIENCE, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def _identify_loser_datasets() -> list[tuple[str, dict]]:
    """Read experiment15/summary.json; return [(name, ratios_entry)] for
    datasets where the SplitBoost mean ratio vs XGBoost was > 1.00."""
    summary_path = EXP15_DIR / "summary.json"
    with summary_path.open() as fp:
        exp15 = json.load(fp)
    losers = []
    for name, r in exp15["ratios"].items():
        if r["ratio"] > 1.00:
            losers.append((name, r))
    losers.sort(key=lambda x: -x[1]["ratio"])
    return losers


def _load_completed(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists():
        return set()
    completed = set()
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            completed.add((row["dataset"], int(row["seed"]), row["config"]))
    return completed


def _append_rows(csv_path: Path, rows: list[RunResult]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a") as fp:
        if new_file:
            fp.write(CSV_HEADER)
        for r in rows:
            fp.write(
                f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n"
            )


def main() -> int:
    EXP16_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXP16_DIR / "summary.csv"

    losers = _identify_loser_datasets()
    print(
        f"Identified {len(losers)} Exp-15 loser datasets (SplitBoost mean ratio > 1.00).",
        flush=True,
    )

    completed = _load_completed(csv_path)
    print(f"Resume: {len(completed)} (dataset, seed) triples already complete.", flush=True)

    new_results: dict[str, list[RunResult]] = {}

    for name, exp15_entry in losers:
        print(f"\n=== {name} (Exp-15 ratio: {exp15_entry['ratio']:.3f}) ===", flush=True)
        try:
            X, y = fetch_data(name, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            continue

        rows_for_dataset: list[RunResult] = []
        for seed in SEEDS:
            if (name, seed, "split_boost_l2_1") in completed:
                print(f"  [seed={seed}] SKIPPED (already in CSV)", flush=True)
                continue
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed,
                )
                m, t = _fit_split_boost(X_tr, y_tr, seed)
                rmse = _rmse(m.predict(X_te), y_te)
                n_rounds = getattr(m, "n_rounds", 0)
                rows_for_dataset.append(RunResult(
                    dataset=name, seed=seed, config="split_boost_l2_1",
                    rmse=rmse, fit_time=t, n_rounds=n_rounds,
                ))
                print(
                    f"  [seed={seed}] split_boost_l2_1 ({t:7.1f}s, "
                    f"{n_rounds:>3} rounds)  RMSE={rmse:.4f}",
                    flush=True,
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [seed={seed}] FAILED: {type(e).__name__}: {e}", flush=True)

        if rows_for_dataset:
            _append_rows(csv_path, rows_for_dataset)
            for r in rows_for_dataset:
                completed.add((r.dataset, r.seed, r.config))
            new_results.setdefault(name, []).extend(rows_for_dataset)

    # ---- Build summary.json ----
    print("\nfinalizing summary.json + comparison.md...", flush=True)

    # Reload ALL Exp 16 rows (including resumed ones).
    all_new_rmses: dict[str, list[float]] = {}
    if csv_path.exists():
        with csv_path.open() as fp:
            for row in csv.DictReader(fp):
                all_new_rmses.setdefault(row["dataset"], []).append(float(row["rmse"]))

    # Per-dataset: new SB mean RMSE vs Exp-15's xgb mean.
    with (EXP15_DIR / "summary.json").open() as fp:
        exp15 = json.load(fp)

    comparison_rows = []
    for name, exp15_entry in losers:
        new_rmses = all_new_rmses.get(name, [])
        if not new_rmses:
            comparison_rows.append({
                "dataset": name,
                "exp15_ratio": exp15_entry["ratio"],
                "exp16_ratio": None,
                "delta": None,
                "verdict": "no_data",
            })
            continue
        new_mean = float(mean(new_rmses))
        xgb_mean = exp15_entry["xgboost_mean"]
        new_ratio = new_mean / xgb_mean if xgb_mean > 0 else float("nan")
        delta = new_ratio - exp15_entry["ratio"]
        if new_ratio < 1.00:
            verdict = "now_a_win"
        elif new_ratio < 1.10:
            verdict = "in_band"
        elif new_ratio < 1.50:
            verdict = "improved_but_still_loss"
        elif new_ratio < 2.00:
            verdict = "still_clear_loss"
        else:
            verdict = "still_catastrophic"
        comparison_rows.append({
            "dataset": name,
            "exp15_ratio": exp15_entry["ratio"],
            "exp16_ratio": new_ratio,
            "delta": delta,
            "verdict": verdict,
            "new_rmse_mean": new_mean,
            "exp15_xgb_mean": xgb_mean,
        })

    summary_json = {
        "config": {
            "leaf_l2": 1.0,
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
        },
        "comparison": comparison_rows,
        "n_datasets_attempted": len(losers),
        "n_datasets_with_data": sum(1 for c in comparison_rows if c["exp16_ratio"] is not None),
    }

    json_path = EXP16_DIR / "summary.json"
    with json_path.open("w") as fp:
        json.dump(summary_json, fp, indent=2)
    print(f"wrote {json_path}", flush=True)

    # ---- Build comparison.md ----
    md_path = EXP16_DIR / "comparison.md"
    valid = [c for c in comparison_rows if c["exp16_ratio"] is not None]
    n_now_wins = sum(1 for c in valid if c["verdict"] == "now_a_win")
    n_in_band = sum(1 for c in valid if c["verdict"] == "in_band")
    n_still_catastrophic = sum(1 for c in valid if c["verdict"] == "still_catastrophic")
    mean_delta = mean(c["delta"] for c in valid) if valid else 0.0

    with md_path.open("w") as fp:
        fp.write("# Experiment 16: leaf_l2=1.0 validation on Exp 15 losers\n\n")
        fp.write(f"**Date:** 2026-04-25\n")
        fp.write(f"**Config:** Exp-15 defaults + leaf_l2=1.0 (XGBoost reg_lambda match).\n")
        fp.write(f"**Datasets re-fit:** {len(valid)} of {len(losers)} loser datasets from Exp 15.\n\n")
        fp.write("## Headline\n\n")
        fp.write(f"- {n_now_wins}/{len(valid)} are now outright wins (ratio < 1.00).\n")
        fp.write(f"- {n_in_band}/{len(valid)} are within 10% (ratio < 1.10).\n")
        fp.write(f"- {n_still_catastrophic}/{len(valid)} are still catastrophic (ratio > 2.00).\n")
        fp.write(f"- Mean ratio change: **{mean_delta:+.3f}** (negative = improvement).\n\n")
        fp.write("## Per-dataset comparison\n\n")
        fp.write("| dataset | Exp 15 ratio | Exp 16 ratio | Δ | verdict |\n")
        fp.write("|---|---|---|---|---|\n")
        for c in comparison_rows:
            if c["exp16_ratio"] is None:
                fp.write(f"| {c['dataset']} | {c['exp15_ratio']:.3f} | — | — | no_data |\n")
            else:
                fp.write(
                    f"| {c['dataset']} | {c['exp15_ratio']:.3f} | "
                    f"{c['exp16_ratio']:.3f} | {c['delta']:+.3f} | {c['verdict']} |\n"
                )
    print(f"wrote {md_path}", flush=True)

    # ---- Console headline ----
    print("\n=== Headline ===", flush=True)
    print(f"  Now wins (ratio < 1.00):         {n_now_wins}/{len(valid)}", flush=True)
    print(f"  In band (ratio < 1.10):          {n_in_band}/{len(valid)}", flush=True)
    print(f"  Still catastrophic (>2.0):       {n_still_catastrophic}/{len(valid)}", flush=True)
    print(f"  Mean Δ ratio:                    {mean_delta:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
