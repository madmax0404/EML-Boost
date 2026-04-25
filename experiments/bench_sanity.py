"""Pre-Exp-15 sanity bench: 3 medium-regime PMLB datasets.

Runs the same SplitBoost / XGBoost / LightGBM configs as
`run_experiment15_full_pmlb.py` on three datasets that sit between the
6 already-done tiny ones and the 1M-row `1191_BNG_pbc` profile target.
Confirms the descriptor-cache + Triton kernels generalize before
committing to the full ~2-3h run.

Rows are appended to `experiment15/summary.csv` via the same helpers
the full runner uses, so resume-from-checkpoint will skip them when
the full Exp-15 run starts.

Usage:
  PYTHONUNBUFFERED=1 uv run python -u experiments/bench_sanity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

# Allow `python experiments/bench_sanity.py` to import its sibling.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_experiment15_full_pmlb import (  # noqa: E402
    FIT_FNS,
    RESULTS_DIR,
    SEEDS,
    TEST_SIZE,
    RunResult,
    _append_rows,
    _load_completed,
    _load_failures,
    _rmse,
    _save_failures,
    _summarize_failure,
)

SANITY_DATASETS = [
    "574_house_16H",   # 22,784 x 16  — real-world, mid-row mid-dim
    "201_pol",         # 15,000 x 48  — high-dim, stresses per-feature mask cache
    "344_mv",          # 40,768 x 10  — largest of three; closes scaling gap
]


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

    for name in SANITY_DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        try:
            X, y = fetch_data(name, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(_summarize_failure(name, "ALL_SEEDS", "data_fetch", "fetch", e))
            _save_failures(failures_path, failures)
            continue

        rows: list[RunResult] = []
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
                    rows.append(RunResult(
                        dataset=name, seed=seed, config=cfg_name,
                        rmse=rmse, fit_time=t, n_rounds=n_rounds,
                    ))
                    print(
                        f"  [seed={seed}] {cfg_name:>11} ({t:7.1f}s, "
                        f"{n_rounds:>3} rounds)  RMSE={rmse:.4f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [seed={seed}] {cfg_name:>11} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    failures.append(_summarize_failure(name, seed, cfg_name, "fit", e))

        if rows:
            _append_rows(csv_path, rows)
            for r in rows:
                completed.add((r.dataset, r.seed, r.config))
            print(f"  appended {len(rows)} rows to {csv_path.name}", flush=True)
        if failures:
            _save_failures(failures_path, failures)

    # Per-dataset summary, eyeballable against the success criteria.
    print("\n=== sanity bench summary ===", flush=True)
    by_dataset: dict[str, dict[str, list[RunResult]]] = {
        d: {"split_boost": [], "xgboost": [], "lightgbm": []} for d in SANITY_DATASETS
    }
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            d = row["dataset"]
            cfg = row["config"]
            if d in by_dataset and cfg in by_dataset[d]:
                by_dataset[d][cfg].append(RunResult(
                    dataset=d,
                    seed=int(row["seed"]),
                    config=cfg,
                    rmse=float(row["rmse"]),
                    fit_time=float(row["fit_time"]),
                    n_rounds=int(row["n_rounds"]),
                ))

    for d in SANITY_DATASETS:
        print(f"  {d}:")
        for cfg in ("split_boost", "xgboost", "lightgbm"):
            results = by_dataset[d][cfg]
            if not results:
                print(f"    {cfg:<11}: (no rows)")
                continue
            times = [r.fit_time for r in results]
            rmses = [r.rmse for r in results]
            print(
                f"    {cfg:<11}: fit_time mean={np.mean(times):6.2f}s "
                f"max={np.max(times):6.2f}s   "
                f"rmse mean={np.mean(rmses):.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
