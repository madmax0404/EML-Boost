"""EML weak learner training.

Three-phase optimization per restart (spec section 5.3):
  1. Exploration — Adam, lr=1e-2, 500 steps, random init.
  2. Hardening — continued Adam with entropy penalty driving each
     softmax toward a simplex vertex.
  3. Snap + verify — argmax each softmax, convert to sympy, verify.

Parallel restarts: n_restarts independent MasterFormulas trained
concurrently via vmap-like stacking in a leading batch dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
import torch

from eml_boost.symbolic.master_formula import MasterFormula
from eml_boost.symbolic.simplify import (
    rpn_length,
    snap_constants,
    snapped_to_sympy,
)
from eml_boost.symbolic.snap import (
    SnappedTree,
    count_active_positions,
    snap_master_formula,
)
from eml_boost.symbolic.verify import reproduces_numerically
from eml_boost._numerics import is_real_valued

_EXPLORATION_STEPS = 500
_HARDENING_STEPS = 500
_LR_INIT = 1e-2
_LR_MIN = 1e-4
_ENTROPY_MAX = 0.5
_SNAP_TOL = 1e-6

# When the discrete tree space has <= this many configurations, skip the
# softmax+snap pipeline entirely and enumerate every snapped tree directly,
# picking the one with best inner-val MSE. Experiment 2 found that the
# softmax optimizer lands in local minima whose argmax snap doesn't match
# the continuous output; exhaustive enumeration sidesteps that failure mode.
# Threshold covers depth 1 (all k) and depth 2 (k up to ~4); above this,
# we fall back to the softmax path.
_EXHAUSTIVE_THRESHOLD = 50_000

# Cap on the inner-val subsample used to rank candidate trees during
# exhaustive search. 500 points is enough to MSE-rank 6k+ candidates with
# stable ordering, while keeping per-round runtime independent of dataset
# size (important for datasets with thousands of training rows).
_EXHAUSTIVE_EVAL_CAP = 500


@dataclass
class EmlWeakLearner:
    """Result of fitting an EML tree to a residual target.

    If snap_ok, `formula` is the closed-form sympy expression in original
    feature coordinates. `formula_std` is the same expression in
    standardized coordinates — used for BIC complexity measurement
    because un-standardization inflates RPN length with float coefficients
    that don't reflect structural complexity.
    """

    trained_module: MasterFormula
    snapped: SnappedTree
    snap_ok: bool
    formula: sp.Expr | None          # un-standardized, for display
    formula_std: sp.Expr | None      # standardized, for BIC
    feature_names: tuple[str, ...]
    feature_idx: np.ndarray  # indices into the full feature matrix (length k)
    feature_mean: np.ndarray  # mean of the k selected features (length k)
    feature_std: np.ndarray  # std of the k selected features (length k)
    inner_val_mse: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new inputs; returns real-valued (n,) array."""
        X_sel = X[:, self.feature_idx]
        X_std = (X_sel - self.feature_mean) / self.feature_std
        with torch.no_grad():
            X_t = torch.tensor(X_std, dtype=torch.complex128)
            out = self.trained_module(X_t)
            return out.real.cpu().numpy().astype(np.float64)

    def params_count(self) -> int:
        """BIC complexity proxy. +1 for the per-round learned eta.

        Uses the standardized formula's RPN length so that BIC measures
        structural complexity, not the float-coefficient blow-up from
        un-standardization.
        """
        if self.snap_ok and self.formula_std is not None:
            return rpn_length(self.formula_std) + 1
        return count_active_positions(self.snapped) + 1


def _tree_space_size(depth: int, k: int) -> int:
    """Count of discrete snapped-tree configurations at given depth/k."""
    internal_positions = (2**depth) - 2 if depth >= 2 else 0
    leaf_positions = 2**depth
    return (k + 2) ** internal_positions * (k + 1) ** leaf_positions


def _enumerate_snapped_trees(depth: int, k: int):
    """Yield every SnappedTree at the given depth and feature count."""
    import itertools
    internal_input_count = (2**depth) - 2 if depth >= 2 else 0
    leaf_input_count = 2**depth
    internal_choices = list(range(k + 2))
    leaf_choices = list(range(k + 1))
    for internal in itertools.product(internal_choices, repeat=internal_input_count):
        for leaf in itertools.product(leaf_choices, repeat=leaf_input_count):
            yield SnappedTree(
                depth=depth,
                k=k,
                internal_input_count=internal_input_count,
                leaf_input_count=leaf_input_count,
                terminal_choices=tuple(internal) + tuple(leaf),
            )


def _exhaustive_search(
    X_iv: np.ndarray,
    y_iv: np.ndarray,
    depth: int,
    k: int,
    feature_names: tuple[str, ...],
) -> tuple[SnappedTree | None, sp.Expr | None, float]:
    """Dispatch to the GPU path at depth=2; else the CPU sympy loop."""
    if depth == 2 and torch.cuda.is_available():
        try:
            return _exhaustive_search_gpu(X_iv, y_iv, k, feature_names)
        except Exception:
            pass  # fall through to CPU
    return _exhaustive_search_cpu(X_iv, y_iv, depth, k, feature_names)


def _exhaustive_search_cpu(
    X_iv: np.ndarray,
    y_iv: np.ndarray,
    depth: int,
    k: int,
    feature_names: tuple[str, ...],
) -> tuple[SnappedTree | None, sp.Expr | None, float]:
    """Per-tree sympy build + lambdify + numpy eval fallback."""
    symbols = [sp.Symbol(name) for name in feature_names]
    best_mse = float("inf")
    best_snapped: SnappedTree | None = None
    best_formula: sp.Expr | None = None

    for snapped in _enumerate_snapped_trees(depth, k):
        formula = snapped_to_sympy(snapped, feature_names)
        formula = snap_constants(formula)
        if not formula.free_symbols:
            continue
        try:
            f = sp.lambdify(symbols, formula, modules=["numpy"])
            iv_pred = np.asarray(
                f(*[X_iv[:, i] for i in range(k)]),
                dtype=np.float64,
            )
            if iv_pred.ndim == 0:
                iv_pred = np.full(len(X_iv), float(iv_pred))
        except Exception:
            continue
        if not np.all(np.isfinite(iv_pred)):
            continue
        mse = float(np.mean((iv_pred - y_iv) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_snapped = snapped
            best_formula = formula

    return best_snapped, best_formula, best_mse


def _exhaustive_search_gpu(
    X_iv: np.ndarray,
    y_iv: np.ndarray,
    k: int,
    feature_names: tuple[str, ...],
) -> tuple[SnappedTree | None, sp.Expr | None, float]:
    """Evaluate every depth-2 tree in one Triton kernel launch; pick min-MSE."""
    from eml_boost._triton_exhaustive import (
        evaluate_trees_triton,
        get_descriptor_gpu,
        get_descriptor_np,
        get_feature_mask_gpu,
    )

    device = torch.device("cuda")
    X_iv_gpu = torch.tensor(X_iv, dtype=torch.float32, device=device)
    y_iv_gpu = torch.tensor(y_iv, dtype=torch.float32, device=device)

    descriptor = get_descriptor_gpu(depth=2, k=k, device=device)  # (n_trees, 6) int32
    feature_mask = get_feature_mask_gpu(depth=2, k=k, device=device)  # (n_trees,) bool

    preds = evaluate_trees_triton(descriptor, X_iv_gpu, k)  # (n_trees, n_samples) fp32

    finite = torch.isfinite(preds).all(dim=1)  # (n_trees,)
    valid = finite & feature_mask

    diff = preds - y_iv_gpu.unsqueeze(0)
    mse_gpu = (diff * diff).mean(dim=1)  # (n_trees,)
    # Sentinel for invalid trees so they can't win argmin.
    mse_gpu = torch.where(
        valid,
        mse_gpu,
        torch.full_like(mse_gpu, float("inf")),
    )

    best_idx = int(mse_gpu.argmin().item())
    if not bool(valid[best_idx].item()):
        return None, None, float("inf")

    best_mse = float(mse_gpu[best_idx].item())

    desc_np = get_descriptor_np(2, k)
    best_choices = tuple(int(v) for v in desc_np[best_idx])
    best_snapped = SnappedTree(
        depth=2,
        k=k,
        internal_input_count=2,
        leaf_input_count=4,
        terminal_choices=best_choices,
    )
    best_formula = snapped_to_sympy(best_snapped, feature_names)
    best_formula = snap_constants(best_formula)
    return best_snapped, best_formula, best_mse


def _fit_eml_tree_exhaustive(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    k: int,
    random_state: int | None,
    standardize: bool = True,
) -> EmlWeakLearner:
    """Enumerate every discrete tree and pick the one with best inner-val MSE.

    When ``standardize=True`` (default, best for real tabular data with
    arbitrary feature magnitudes), features are z-scored before the search
    and the stored formula is un-standardized via sympy substitution so it
    can be applied to raw inputs. Trade-off: recovered-formula extrapolation
    follows an (x − μ)/σ slope that differs from the literal function form
    on targets with `x` far outside the training range.

    When ``standardize=False``, features go directly into the exhaustive
    search in raw coordinates; the stored formula is in literal feature
    symbols. Correct for synthetic extrapolation benchmarks where features
    are pre-normalized (Experiments 4 and 6); unsafe for real-scale data
    because `exp()` of large feature values overflows even with clamping.

    For depth=2 the GPU (Triton) path evaluates all candidate trees in one
    kernel launch per round, eliminating per-tree sympy construction and
    Python dispatch. Falls back to CPU sympy enumeration if CUDA is
    unavailable or the configuration lies outside the GPU path's support.
    """
    rng = np.random.default_rng(random_state)

    # Feature selection by |correlation|
    col_std = X.std(axis=0) + 1e-12
    y_centered = y - y.mean()
    corrs = (X - X.mean(axis=0)).T @ y_centered / (col_std * y.std() * len(y) + 1e-12)
    feature_idx = np.argsort(-np.abs(corrs))[:k]
    feature_names = tuple(f"x_{int(i)}" for i in feature_idx)
    X_sel = X[:, feature_idx]

    if standardize:
        mean = X_sel.mean(axis=0)
        std = X_sel.std(axis=0) + 1e-12
        X_for_search = (X_sel - mean) / std
    else:
        mean = np.zeros(k, dtype=np.float64)
        std = np.ones(k, dtype=np.float64)
        X_for_search = X_sel

    # Inner val split.
    n = len(X_for_search)
    perm = rng.permutation(n)
    val_sz = min(max(int(0.2 * n), 1), _EXHAUSTIVE_EVAL_CAP)
    val_idx = perm[:val_sz]
    X_iv_np = X_for_search[val_idx]
    y_iv_np = y[val_idx]

    best_snapped, best_formula_std, best_mse = _exhaustive_search(
        X_iv_np, y_iv_np, depth, k, feature_names,
    )

    if best_snapped is not None and best_formula_std is not None and standardize:
        # Un-standardize: substitute x_i → (x_i − μ)/σ so the stored formula
        # can be applied to raw input values. `predict()` still uses the
        # trained_module with standardized features for the numeric path;
        # this substitution only affects the reported symbolic `formula`.
        # NOTE: deliberately NOT calling `sp.simplify` here — it can spend
        # seconds normalizing the resulting fractions per round, which
        # dominates total runtime for many-round real-data fits.
        sub_map = {
            sp.Symbol(name): (sp.Symbol(name) - float(mu)) / float(sigma)
            for name, mu, sigma in zip(feature_names, mean, std)
        }
        best_formula = best_formula_std.xreplace(sub_map)
        best_formula = snap_constants(best_formula)
    else:
        best_formula = best_formula_std

    if best_snapped is None or best_formula is None:
        raise RuntimeError(
            f"Exhaustive search found no valid tree at depth={depth}, k={k}. "
            "All candidates were constants or produced non-finite outputs."
        )

    # Build a MasterFormula with one-hot logits matching the chosen tree.
    # `predict` will apply (X − mean)/std before feeding the module, so when
    # standardize=True the module sees standardized inputs and produces
    # outputs matching `best_formula_std`; when standardize=False the
    # sentinel mean=0, std=1 leaves the inputs raw.
    mf = MasterFormula(depth=depth, k=k)
    with torch.no_grad():
        for logit_idx, choice in enumerate(best_snapped.terminal_choices):
            p = mf.logits_list[logit_idx]
            new = torch.full_like(p, -100.0)
            new[choice] = 100.0
            p.copy_(new)

    return EmlWeakLearner(
        trained_module=mf,
        snapped=best_snapped,
        snap_ok=True,
        formula=best_formula,        # un-standardized (if applicable) — for display
        formula_std=best_formula_std,  # standardized — for BIC params_count
        feature_names=feature_names,
        feature_idx=feature_idx,
        feature_mean=mean,
        feature_std=std,
        inner_val_mse=best_mse,
    )


def fit_eml_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    n_restarts: int,
    k: int,
    random_state: int | None = None,
) -> EmlWeakLearner:
    """Fit an EML weak learner to targets y given features X.

    When the discrete tree space is small (<= `_EXHAUSTIVE_THRESHOLD`),
    enumerates every snapped tree and picks the best by inner-val MSE.
    Otherwise falls back to `n_restarts` softmax trainings + argmax snap.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    depth : int
        Depth of the MasterFormula tree.
    n_restarts : int
        Number of independent random restarts (ignored for exhaustive path).
    k : int
        Number of top features to select. Must be <= X.shape[1].
    random_state : int or None
        If None, each restart uses a freshly drawn torch seed (non-deterministic).
        If int, seeds are derived from random_state so results are reproducible.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # I3: Validate k before any computation.
    if k > X.shape[1]:
        raise ValueError(f"k={k} exceeds number of features ({X.shape[1]})")

    # Dispatch to exhaustive enumeration when the tree space is small.
    if _tree_space_size(depth, k) <= _EXHAUSTIVE_THRESHOLD:
        return _fit_eml_tree_exhaustive(X, y, depth, k, random_state)

    # I2: Use np.random.default_rng so random_state=None is truly non-deterministic
    #     and random_state=0 is distinct from None.
    rng = np.random.default_rng(random_state)

    # 1. Feature selection by |correlation|
    col_std = X.std(axis=0) + 1e-12
    y_centered = y - y.mean()
    corrs = (X - X.mean(axis=0)).T @ y_centered / (col_std * y.std() * len(y) + 1e-12)
    feature_idx = np.argsort(-np.abs(corrs))[:k]
    feature_names = tuple(f"x_{int(i)}" for i in feature_idx)

    X_sel = X[:, feature_idx]
    mean = X_sel.mean(axis=0)
    std = X_sel.std(axis=0) + 1e-12
    X_std = (X_sel - mean) / std

    # 2. Inner train/val split.
    n = len(X_std)
    perm = rng.permutation(n)
    val_sz = max(int(0.2 * n), 1)
    val_idx, tr_idx = perm[:val_sz], perm[val_sz:]
    X_tr = torch.tensor(X_std[tr_idx], dtype=torch.complex128)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.complex128)
    X_iv = torch.tensor(X_std[val_idx], dtype=torch.complex128)
    y_iv_np = y[val_idx]

    # I2: Precompute per-restart torch seeds from the rng so that
    #     random_state=None gives fresh (non-deterministic) seeds each call,
    #     while an integer random_state gives fully reproducible seeds.
    restart_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_restarts)]

    # 3. Run n_restarts. For simplicity in v1 we do them sequentially; parallel
    #    stacking is a performance optimization flagged for v2.
    best: EmlWeakLearner | None = None
    for seed in restart_seeds:
        torch.manual_seed(seed)
        mf = MasterFormula(depth=depth, k=k)
        _train_single(mf, X_tr, y_tr)

        # Quick divergence check on the raw network output before snapping.
        with torch.no_grad():
            raw_iv = mf(X_iv)
        if torch.isnan(raw_iv).any():
            continue  # training diverged — skip this restart

        unsnapped_pred = raw_iv.real.cpu().numpy().astype(np.float64)
        if np.isnan(unsnapped_pred).any():
            continue  # defensive guard

        snapped = snap_master_formula(mf)
        formula_std = snapped_to_sympy(snapped, feature_names)
        formula_std = snap_constants(formula_std)

        # Dead-branch prune (Experiment 2 finding): if the snapped formula
        # is a constant (no free symbols), the argmax topology has left all
        # input features in dead subtrees unreferenced by the root. The
        # continuous optimum relied on soft mixing that evaporates on snap.
        # Reject and try another restart.
        if not formula_std.free_symbols:
            continue

        # C1 (spec 5.4): Verify on standardized features that the snapped
        # sympy formula reproduces the unsnapped torch output.
        # This check uses standardized inputs because the torch model was
        # trained on standardized features.
        snap_iv_vals = _eval_formula_numpy(formula_std, feature_names, X_std[val_idx])
        if snap_iv_vals is None or not is_real_valued(
            torch.tensor(snap_iv_vals, dtype=torch.complex128), tol=1e-8
        ):
            continue  # snapped formula is non-physical — discard

        snap_ok = reproduces_numerically(
            formula_std,
            feature_names,
            X_std[val_idx],
            unsnapped_pred,
            tol=_SNAP_TOL,
        )

        # Un-standardize: express in original feature coordinates (spec 5.4).
        # Each symbol x_i currently represents the z-scored value; substitute
        # x_i -> (x_i - mu_i) / sigma_i so the stored formula uses raw feature
        # values. The algebraic equivalence is guaranteed by the substitution;
        # no re-verification is needed since snap_ok was established above.
        sub_map = {
            sp.Symbol(name): (sp.Symbol(name) - float(mu)) / float(sigma)
            for name, mu, sigma in zip(feature_names, mean, std)
        }
        formula = sp.simplify(formula_std.xreplace(sub_map))
        # Re-run constant snap as simplification may introduce near-exact values.
        formula = snap_constants(formula)

        mse = float(np.mean((unsnapped_pred - y_iv_np) ** 2))

        learner = EmlWeakLearner(
            trained_module=mf,
            snapped=snapped,
            snap_ok=snap_ok,
            formula=formula if snap_ok else None,
            formula_std=formula_std if snap_ok else None,
            feature_names=feature_names,
            feature_idx=feature_idx,
            feature_mean=mean,
            feature_std=std,
            inner_val_mse=mse,
        )
        if best is None or learner.inner_val_mse < best.inner_val_mse:
            best = learner

    # I1: If every restart was skipped, raise a clear error instead of
    #     returning a NaN learner or hitting an assert.
    if best is None:
        raise RuntimeError(
            f"All {n_restarts} restarts diverged or produced non-real outputs. "
            "Consider increasing n_restarts or reducing depth."
        )
    return best


def _eval_formula_numpy(
    formula: sp.Expr,
    feature_names: tuple[str, ...],
    X_val: np.ndarray,
) -> np.ndarray | None:
    """Evaluate a sympy formula numerically on X_val (shape n x k).

    Returns a complex numpy array, or None if evaluation raises an exception.
    The caller can inspect the imaginary part to detect non-real outputs.
    """
    try:
        syms = [sp.Symbol(name) for name in feature_names]
        f_lambda = sp.lambdify(syms, formula, modules="numpy")
        cols = [X_val[:, j] for j in range(len(feature_names))]
        result = f_lambda(*cols)
        return np.asarray(result, dtype=np.complex128)
    except Exception:
        return None


def _train_single(mf: MasterFormula, X_tr: torch.Tensor, y_tr: torch.Tensor) -> None:
    """Run the 3-phase training on a single MasterFormula in place."""
    optimizer = torch.optim.Adam(mf.parameters(), lr=_LR_INIT)

    # Phase 1: exploration.
    for _ in range(_EXPLORATION_STEPS):
        optimizer.zero_grad()
        pred = mf(X_tr)
        if torch.isnan(pred).any():
            return  # divergence — abort this restart cleanly
        loss = ((pred.real - y_tr.real) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Phase 2: hardening with entropy penalty ramp.
    for step in range(_HARDENING_STEPS):
        frac = step / max(_HARDENING_STEPS - 1, 1)
        lr = _LR_INIT * (1 - frac) + _LR_MIN * frac
        entropy_weight = _ENTROPY_MAX * frac
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad()
        pred = mf(X_tr)
        if torch.isnan(pred).any():
            return
        loss = ((pred.real - y_tr.real) ** 2).mean()
        for p in mf.logits_list:
            probs = torch.softmax(p, dim=0)
            entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
            loss = loss + entropy_weight * entropy
        loss.backward()
        optimizer.step()
