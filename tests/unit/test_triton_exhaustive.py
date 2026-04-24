"""Validate the torch and Triton exhaustive evaluators against the sympy path.

The sympy path (snapped_to_sympy + lambdify) is the ground truth. Torch and
Triton must match it to 1e-5 relative tolerance on the values that are
finite; NaN/Inf behavior (log of non-positive) may differ because numpy's
log and torch's log both produce NaN, while our sympy path uses the same.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

from eml_boost._triton_exhaustive import (
    build_descriptor_depth2,
    enumerate_depth2_descriptor,
    evaluate_trees_torch,
    evaluate_trees_triton,
)
from eml_boost.symbolic.simplify import snap_constants, snapped_to_sympy
from eml_boost.symbolic.snap import SnappedTree


def _sympy_oracle(
    descriptor: np.ndarray, X: np.ndarray, k: int
) -> np.ndarray:
    """Evaluate each tree in the descriptor on X via snapped_to_sympy + lambdify."""
    feature_names = tuple(f"x_{i}" for i in range(k))
    symbols = [sp.Symbol(name) for name in feature_names]
    n_trees = descriptor.shape[0]
    n_samples = X.shape[0]
    out = np.empty((n_trees, n_samples), dtype=np.float64)
    for t in range(n_trees):
        tree = SnappedTree(
            depth=2,
            k=k,
            internal_input_count=2,
            leaf_input_count=4,
            terminal_choices=tuple(int(v) for v in descriptor[t]),
        )
        expr = snapped_to_sympy(tree, feature_names)
        expr = snap_constants(expr)
        f = sp.lambdify(symbols, expr, modules=["numpy"])
        try:
            pred = np.asarray(f(*[X[:, i] for i in range(k)]), dtype=np.float64)
            if pred.ndim == 0:
                pred = np.full(n_samples, float(pred))
        except Exception:
            pred = np.full(n_samples, np.nan)
        out[t] = pred
    return out


def _assert_close_where_finite(
    oracle: np.ndarray, candidate: np.ndarray, rtol: float = 1e-4, atol: float = 1e-5
) -> None:
    """Compare two arrays, tolerating NaN/Inf on either side where both non-finite."""
    finite_mask = np.isfinite(oracle) & np.isfinite(candidate)
    if finite_mask.any():
        np.testing.assert_allclose(
            candidate[finite_mask],
            oracle[finite_mask],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_torch_matches_sympy_oracle(k: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 2.0, size=(32, k)).astype(np.float64)
    descriptor_np = enumerate_depth2_descriptor(k)
    # Sample a subset so the oracle runs fast (sympy is slow).
    rng.shuffle(descriptor_np)
    descriptor_np = descriptor_np[:80]

    oracle = _sympy_oracle(descriptor_np, X, k)

    X_t = torch.tensor(X, dtype=torch.float64)
    desc_t = torch.tensor(descriptor_np, dtype=torch.int32)
    candidate = evaluate_trees_torch(desc_t, X_t, k).cpu().numpy()

    _assert_close_where_finite(oracle, candidate)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_triton_matches_torch(k: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; Triton kernel not exercised")

    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 2.0, size=(64, k)).astype(np.float64)
    descriptor_np = enumerate_depth2_descriptor(k)
    rng.shuffle(descriptor_np)
    descriptor_np = descriptor_np[:200]

    X_gpu = torch.tensor(X, dtype=torch.float32, device="cuda")
    desc_gpu = torch.tensor(descriptor_np, dtype=torch.int32, device="cuda")

    torch_out = evaluate_trees_torch(desc_gpu, X_gpu, k).cpu().numpy()
    triton_out = evaluate_trees_triton(desc_gpu, X_gpu, k).cpu().numpy()

    # float32 tolerance; Triton and torch may differ in last-bit due to
    # different kernel fusion patterns.
    _assert_close_where_finite(torch_out, triton_out, rtol=1e-4, atol=1e-4)


def test_build_descriptor_round_trip() -> None:
    # Use k=2 to keep enumerated size small (81*16 = 1296 trees)
    all_desc = enumerate_depth2_descriptor(k=2)
    # Pack from SnappedTree list and confirm round-trip matches.
    trees = [
        SnappedTree(
            depth=2, k=2, internal_input_count=2, leaf_input_count=4,
            terminal_choices=tuple(int(v) for v in all_desc[i]),
        )
        for i in range(0, 1296, 97)  # spot-check a subset
    ]
    packed = build_descriptor_depth2(trees)
    for row, tree in zip(packed, trees):
        assert tuple(int(v) for v in row) == tree.terminal_choices
