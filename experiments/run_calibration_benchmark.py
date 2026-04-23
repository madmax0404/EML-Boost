"""Graceful-degradation calibration benchmark.

Sweeps synthetic datasets with elementary-signal fractions in
[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], fits EML-Boost, and records the
fraction of boosting rounds where the EML weak learner wins the BIC
selection against the DT weak learner.

Writes:
  experiments/results/calibration_curve.csv   — raw per-fraction data
  experiments/results/calibration_curve.json  — full result dataclass
  experiments/results/calibration_curve.png   — headline plot
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.calibration import run_calibration

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_DATASETS_PER_FRACTION = 2
N = 200
MAX_ROUNDS = 15
N_RESTARTS = 6
DEPTH_EML = 2
DEPTH_DT = 2


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Running calibration sweep: {len(FRACTIONS)} fractions × "
        f"{N_DATASETS_PER_FRACTION} datasets × EML-Boost("
        f"max_rounds={MAX_ROUNDS}, n_restarts={N_RESTARTS}, depth_eml={DEPTH_EML})"
    )
    print(f"  fractions: {FRACTIONS}")
    print("  started:", time.strftime("%Y-%m-%d %H:%M:%S"))

    start = time.time()
    result = run_calibration(
        elementary_fractions=FRACTIONS,
        n_datasets_per_fraction=N_DATASETS_PER_FRACTION,
        n=N,
        seed=0,
        max_rounds=MAX_ROUNDS,
        n_restarts=N_RESTARTS,
        depth_eml=DEPTH_EML,
        depth_dt=DEPTH_DT,
    )
    elapsed = time.time() - start
    print(f"  finished in {elapsed:.1f}s")

    # Raw CSV
    csv_path = RESULTS_DIR / "calibration_curve.csv"
    with csv_path.open("w") as f:
        f.write(
            "fraction_elementary,eml_win_rate,eml_coverage,"
            "hybrid_test_mse,dt_only_test_mse,dt_improvement,"
            "xgboost_test_mse,xgb_improvement,total_rounds\n"
        )
        for frac, rate, cov, hmse, dmse, imp, xmse, ximp, total in zip(
            result.fractions,
            result.eml_win_rates,
            result.eml_coverages,
            result.hybrid_test_mse,
            result.dt_only_test_mse,
            result.dt_improvement,
            result.xgboost_test_mse,
            result.xgb_improvement,
            result.per_fraction_round_counts,
        ):
            f.write(
                f"{frac},{rate},{cov},{hmse},{dmse},{imp},"
                f"{xmse},{ximp},{total}\n"
            )
    print(f"  wrote {csv_path}")

    # JSON
    json_path = RESULTS_DIR / "calibration_curve.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "fractions": FRACTIONS,
                    "n_datasets_per_fraction": N_DATASETS_PER_FRACTION,
                    "n_samples": N,
                    "max_rounds": MAX_ROUNDS,
                    "n_restarts": N_RESTARTS,
                    "depth_eml": DEPTH_EML,
                    "depth_dt": DEPTH_DT,
                    "elapsed_seconds": elapsed,
                },
                "result": asdict(result),
            },
            f,
            indent=2,
        )
    print(f"  wrote {json_path}")

    # Plot — two headline metrics side-by-side: MSE improvement over DT-only
    # (capacity-matched weak baseline) and MSE improvement over XGBoost (strong
    # industry baseline). Plus reference curves for coverage and win rate.
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    ax.plot(
        result.fractions,
        result.dt_improvement,
        marker="o",
        linewidth=2.5,
        markersize=9,
        color="#2E86AB",
        label="MSE improvement vs DT-only (capacity-matched)",
    )
    ax.plot(
        result.fractions,
        result.xgb_improvement,
        marker="D",
        linewidth=2.5,
        markersize=9,
        color="#9B2226",
        label="MSE improvement vs XGBoost (capacity-matched)",
    )
    ax.plot(
        result.fractions,
        result.eml_coverages,
        marker="^",
        linewidth=1.2,
        markersize=6,
        color="#588157",
        linestyle=":",
        alpha=0.7,
        label="Formula coverage (spec 7.3)",
    )
    ax.plot(
        result.fractions,
        result.eml_win_rates,
        marker="s",
        linewidth=1.2,
        markersize=6,
        color="#E63946",
        linestyle="--",
        alpha=0.5,
        label="EML round-win rate",
    )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Fraction of signal that is elementary")
    ax.set_ylabel("Metric")
    ax.set_title(
        "EML-Boost graceful degradation\n"
        f"(N={N}, {N_DATASETS_PER_FRACTION} datasets/fraction, "
        f"{MAX_ROUNDS} rounds, depth-{DEPTH_EML} EML)"
    )
    ax.set_xlim(-0.05, 1.05)
    lo = min(-0.05, min(result.dt_improvement) - 0.05, min(result.xgb_improvement) - 0.05)
    hi = max(1.05, max(result.xgb_improvement) + 0.05)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "calibration_curve.png"
    plt.savefig(plot_path)
    print(f"  wrote {plot_path}")

    # Console summary
    print()
    print("=== Calibration curve ===")
    print(
        f"{'frac_elem':>9}  {'DT-impr':>8}  {'XGB-impr':>9}  "
        f"{'hybrid':>9}  {'DT_only':>9}  {'XGB':>9}  "
        f"{'cov':>6}  {'win':>6}"
    )
    for frac, rate, cov, hmse, dmse, imp, xmse, ximp in zip(
        result.fractions,
        result.eml_win_rates,
        result.eml_coverages,
        result.hybrid_test_mse,
        result.dt_only_test_mse,
        result.dt_improvement,
        result.xgboost_test_mse,
        result.xgb_improvement,
    ):
        print(
            f"{frac:>9.2f}  {imp:>+8.3f}  {ximp:>+9.3f}  "
            f"{hmse:>9.4f}  {dmse:>9.4f}  {xmse:>9.4f}  "
            f"{cov:>6.3f}  {rate:>6.3f}"
        )

    # Monotonicity check on both headline metrics
    for label, series in (
        ("DT-improvement", result.dt_improvement),
        ("XGB-improvement", result.xgb_improvement),
    ):
        deltas = [b - a for a, b in zip(series, series[1:])]
        if all(d >= -0.1 for d in deltas):
            print(
                f"\n{label} monotonic (within 0.1 slack): PASS "
                "— rises cleanly with frac_elementary"
            )
        else:
            print(f"\n{label} monotonic: FAIL — dips in the curve")
            for i, d in enumerate(deltas):
                if d < -0.1:
                    print(
                        f"  drop from {result.fractions[i]:.2f} "
                        f"({series[i]:+.2f}) to {result.fractions[i + 1]:.2f} "
                        f"({series[i + 1]:+.2f}): Δ={d:+.2f}"
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
