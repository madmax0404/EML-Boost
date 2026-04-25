"""Profile 344_mv fit to understand why it's 4x slower than 574_house_16H
on a similar n*k budget. 50-round cProfile, mirrors profile_redux/run_profile.py.

Output: profile_344mv/cum.txt, tot.txt, profile.pstats.
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
    print("Loading 344_mv...", flush=True)
    X, y = fetch_data("344_mv", return_X_y=True)
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
        k_leaf_eml=1,
        min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05,
        leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0,
        use_stacked_blend=False,
        patience=15,
        val_fraction=0.15,
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

    out_dir = "profile_344mv"
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
