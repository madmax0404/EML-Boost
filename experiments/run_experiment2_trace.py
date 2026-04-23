"""Experiment 2: per-round trace of a single pure-elementary run.

Answers two questions left open by Experiment 1:

  Q1. Are depth-2 EML fits producing `snap_ok=True` often enough to
      matter, or is the verification gate rejecting legitimate fits?
  Q2. Does bumping depth to 3 flip the BIC selector's preference toward
      EML on a simple elementary signal?

Three traces per run:
  A. `exp(x_0) - log(|x_1| + 1)` at depth_eml=2 — a depth-2-expressible
     control; EML should win most rounds here if the pipeline is healthy.
  B. `exp(x_0) + 0.5 * x_1**2` at depth_eml=2 — the spec formula that
     drove Experiment 1's flat curve; expected to be dominated by DT.
  C. `exp(x_0) + 0.5 * x_1**2` at depth_eml=3 — the flip test; if EML
     wins here, depth 2 is the limiter and Experiment 1 will work at
     higher depth.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eml_boost.datasets import generate_pure_elementary
from eml_boost.selection import bic_score, learned_eta
from eml_boost.weak_learners.base import WeakLearnerKind
from eml_boost.weak_learners.dt import fit_dt_stump
from eml_boost.weak_learners.eml import EmlWeakLearner, fit_eml_tree

RESULTS_DIR = Path(__file__).resolve().parent / "experiment2"
ETA_DT = 0.1
INNER_VAL_FRAC = 0.20
OUTER_VAL_FRAC = 0.15


@dataclass
class RoundTrace:
    round_index: int
    kind: str                  # "EML" or "DT"
    eta_eml: float
    eta_dt: float
    bic_eml: float
    bic_dt: float
    params_eml: int
    params_dt: int
    eml_snap_ok: bool
    eml_formula: str | None
    outer_val_mse: float


@dataclass
class TraceRun:
    name: str
    formula: str
    depth_eml: int
    depth_dt: int
    n_restarts: int
    k: int
    max_rounds: int
    rounds: list[RoundTrace] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _traced_boost(
    *,
    X: np.ndarray,
    y: np.ndarray,
    max_rounds: int,
    depth_eml: int,
    depth_dt: int,
    n_restarts: int,
    k: int,
    random_state: int,
) -> TraceRun:
    """Mirror of `eml_boost.training.boost` with verbose per-round logging.

    Keeps tracing out of production code; if this drifts from `boost()`
    the diagnostic value weakens, so when updating the real loop also
    touch this function.
    """
    run = TraceRun(
        name="(filled by caller)",
        formula="(filled by caller)",
        depth_eml=depth_eml,
        depth_dt=depth_dt,
        n_restarts=n_restarts,
        k=k,
        max_rounds=max_rounds,
    )

    rng = np.random.default_rng(random_state)
    n = len(X)
    perm = rng.permutation(n)
    outer_val_n = max(int(OUTER_VAL_FRAC * n), 1)
    outer_val_idx = perm[:outer_val_n]
    trainval_idx = perm[outer_val_n:]
    X_trval, y_trval = X[trainval_idx], y[trainval_idx]
    X_oval, y_oval = X[outer_val_idx], y[outer_val_idx]

    F_0 = float(y_trval.mean())
    weak_learners: list[tuple[object, float, WeakLearnerKind]] = []

    def _predict_trval() -> np.ndarray:
        out = np.full(len(X_trval), F_0)
        for learner, eta, _ in weak_learners:
            out = out + eta * learner.predict(X_trval)
        return out

    def _predict_oval() -> np.ndarray:
        out = np.full(len(X_oval), F_0)
        for learner, eta, _ in weak_learners:
            out = out + eta * learner.predict(X_oval)
        return out

    start = time.time()
    for m in range(max_rounds):
        r_trval = y_trval - _predict_trval()

        inner_perm = rng.permutation(len(X_trval))
        inner_val_n = max(int(INNER_VAL_FRAC * len(X_trval)), 1)
        iv_idx = inner_perm[:inner_val_n]
        tr_idx = inner_perm[inner_val_n:]
        X_tr, r_tr = X_trval[tr_idx], r_trval[tr_idx]
        X_iv, r_iv = X_trval[iv_idx], r_trval[iv_idx]

        seed = random_state * 31 + m
        try:
            h_eml = fit_eml_tree(
                X_tr, r_tr, depth=depth_eml, n_restarts=n_restarts,
                k=k, random_state=seed,
            )
        except RuntimeError as e:
            print(f"  round {m:2d}: EML fit failed ({e}); skipping round")
            continue
        h_dt = fit_dt_stump(X_tr, r_tr, depth=depth_dt)

        eml_pred = h_eml.predict(X_iv)
        dt_pred = h_dt.predict(X_iv)
        eta_eml = learned_eta(eml_pred, r_iv)
        scaled_eml = eta_eml * eml_pred
        scaled_dt = ETA_DT * dt_pred

        params_eml = h_eml.params_count()
        params_dt = h_dt.params_count()
        bic_eml = bic_score(r_iv, scaled_eml, params_eml)
        bic_dt = bic_score(r_iv, scaled_dt, params_dt)

        if bic_eml <= bic_dt:
            winner, kind, eta = h_eml, WeakLearnerKind.EML, eta_eml
        else:
            winner, kind, eta = h_dt, WeakLearnerKind.DT, ETA_DT
        weak_learners.append((winner, eta, kind))

        oval_mse = float(np.mean((y_oval - _predict_oval()) ** 2))

        formula_str: str | None = None
        if isinstance(h_eml, EmlWeakLearner) and h_eml.snap_ok and h_eml.formula is not None:
            formula_str = str(h_eml.formula)

        trace = RoundTrace(
            round_index=m,
            kind=kind.value,
            eta_eml=float(eta_eml),
            eta_dt=ETA_DT,
            bic_eml=float(bic_eml),
            bic_dt=float(bic_dt),
            params_eml=int(params_eml),
            params_dt=int(params_dt),
            eml_snap_ok=bool(h_eml.snap_ok),
            eml_formula=formula_str,
            outer_val_mse=oval_mse,
        )
        run.rounds.append(trace)
        short_formula = (formula_str or "")[:50]
        print(
            f"  round {m:2d}: winner={kind.value:3s}  "
            f"BIC(EML)={bic_eml:7.1f}  BIC(DT)={bic_dt:7.1f}  "
            f"params(EML)={params_eml:2d}  params(DT)={params_dt:2d}  "
            f"snap_ok={'Y' if h_eml.snap_ok else 'N'}  "
            f"formula={short_formula}"
        )

    run.elapsed_seconds = time.time() - start
    return run


def _write_trace_csv(run: TraceRun, path: Path) -> None:
    with path.open("w") as f:
        f.write(
            "round,kind,eta_eml,eta_dt,bic_eml,bic_dt,"
            "params_eml,params_dt,snap_ok,formula,outer_val_mse\n"
        )
        for r in run.rounds:
            formula = (r.eml_formula or "").replace(",", ";").replace("\n", " ")
            f.write(
                f"{r.round_index},{r.kind},{r.eta_eml:.6f},{r.eta_dt:.6f},"
                f"{r.bic_eml:.6f},{r.bic_dt:.6f},{r.params_eml},{r.params_dt},"
                f"{int(r.eml_snap_ok)},{formula},{r.outer_val_mse:.6f}\n"
            )


def _plot_bic_per_round(runs: list[TraceRun], path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), dpi=110)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        if not run.rounds:
            ax.text(0.5, 0.5, "no rounds", ha="center", va="center")
            ax.set_title(run.name)
            continue
        xs = [r.round_index for r in run.rounds]
        eml_bic = [r.bic_eml for r in run.rounds]
        dt_bic = [r.bic_dt for r in run.rounds]
        ax.plot(xs, eml_bic, marker="o", label="BIC(EML)", color="#2E86AB")
        ax.plot(xs, dt_bic, marker="s", label="BIC(DT)", color="#E63946")
        for r in run.rounds:
            color = "#2E86AB" if r.kind == "EML" else "#E63946"
            ax.axvspan(r.round_index - 0.4, r.round_index + 0.4, alpha=0.08, color=color)
        ax.set_xlabel("boosting round")
        ax.set_ylabel("BIC (lower is better)")
        ax.set_title(run.name)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Shared generation settings.
    n = 200
    seed = 0
    max_rounds = 20
    n_restarts = 6
    k = 2

    # Diagnostic trace — parametrized by depth via EXPERIMENT2_DEPTH env var.
    # Default depth 2 reproduces Experiment 1's flat-curve regime; setting
    # EXPERIMENT2_DEPTH=3 runs the flip test on the same signal / seed.
    import os
    depth_eml = int(os.environ.get("EXPERIMENT2_DEPTH", "2"))
    depth_suffix = "" if depth_eml == 2 else f"_depth{depth_eml}"
    formula_env = os.environ.get("EXPERIMENT2_FORMULA")
    formula_suffix = "" if not formula_env else "_" + "".join(
        c if c.isalnum() else "_" for c in formula_env
    )[:40]
    suffix = depth_suffix + formula_suffix

    formula = os.environ.get("EXPERIMENT2_FORMULA", "exp(x0) + 0.5 * x1**2")
    X, y, _ = generate_pure_elementary(
        formula=formula, n=n, n_features=2, seed=seed,
    )
    print(f"\n=== Trace: {formula} at depth_eml={depth_eml} ===")
    run = _traced_boost(
        X=X, y=y, max_rounds=max_rounds,
        depth_eml=depth_eml, depth_dt=2, n_restarts=n_restarts, k=k, random_state=seed,
    )
    run.name = f"depth-{depth_eml} pure-elementary trace"
    run.formula = formula

    _write_trace_csv(run, RESULTS_DIR / f"trace{suffix}.csv")
    with (RESULTS_DIR / f"summary{suffix}.json").open("w") as f:
        json.dump(
            {
                **{k: v for k, v in run.__dict__.items() if k != "rounds"},
                "eml_win_rate": sum(1 for r in run.rounds if r.kind == "EML") / max(len(run.rounds), 1),
                "snap_ok_rate": sum(1 for r in run.rounds if r.eml_snap_ok) / max(len(run.rounds), 1),
            },
            f, indent=2,
        )
    _plot_bic_per_round([run], RESULTS_DIR / f"bic_per_round{suffix}.png")

    # Console summary
    if run.rounds:
        eml_wins = sum(1 for r in run.rounds if r.kind == "EML")
        snap_oks = sum(1 for r in run.rounds if r.eml_snap_ok)
        mean_params_eml = np.mean([r.params_eml for r in run.rounds])
        mean_params_dt = np.mean([r.params_dt for r in run.rounds])
        print("\n=== Summary ===")
        print(
            f"{run.name}: {eml_wins}/{len(run.rounds)} EML wins, "
            f"{snap_oks}/{len(run.rounds)} snap_ok; "
            f"mean params EML={mean_params_eml:.1f}, DT={mean_params_dt:.1f}; "
            f"elapsed={run.elapsed_seconds:.1f}s"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
