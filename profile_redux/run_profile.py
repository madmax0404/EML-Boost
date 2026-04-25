"""Profile 1191_BNG_pbc fit to find the next bottleneck after redux A+B+C.

Strategy: run a 50-round fit (1/4 of the full 200-round Experiment 15
config) under cProfile, then dump cumulative + tottime sorted reports.
50 rounds is enough to amortize startup costs and surface the per-round
hot path.

Output: profile_redux/cum.txt (cumulative), profile_redux/tot.txt (tottime).
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor


def main() -> None:
    print("Loading 1191_BNG_pbc...", flush=True)
    X, y = fetch_data("1191_BNG_pbc", return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    X_tr, _X_te, y_tr, _y_te = train_test_split(
        X, y, test_size=0.2, random_state=0,
    )
    print(f"X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}", flush=True)

    model = EmlSplitBoostRegressor(
        max_rounds=50,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        n_eml_candidates=10,
        k_eml=3,
        n_bins=256,
        histogram_min_n=500,
        use_gpu=True,
        random_state=0,
    )

    print("Profiling 50-round fit...", flush=True)
    t0 = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    model.fit(X_tr, y_tr)
    profiler.disable()
    dt = time.time() - t0
    print(f"50-round fit took {dt:.1f}s ({dt / 50:.2f}s/round)", flush=True)

    out_dir = "profile_redux"
    stats = pstats.Stats(profiler)
    with open(f"{out_dir}/cum.txt", "w") as f:
        stats.stream = f
        stats.sort_stats("cumulative").print_stats(60)
    with open(f"{out_dir}/tot.txt", "w") as f:
        stats.stream = f
        stats.sort_stats("tottime").print_stats(60)

    profiler.dump_stats(f"{out_dir}/profile.pstats")
    print(f"Wrote {out_dir}/cum.txt, tot.txt, profile.pstats", flush=True)


if __name__ == "__main__":
    main()
