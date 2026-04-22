"""Benchmark dataset utilities: Feynman loader and synthetic generators.

The Feynman SR formulas are self-generated from closed-form expressions
written into FEYNMAN_FORMULAS below, rather than downloaded — this keeps
the test suite offline-friendly. A richer loader (downloading the full
100-formula set) can be added later without changing the interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

# Subset of the AI-Feynman formula table (Udrescu & Tegmark 2020, Table S1).
FEYNMAN_FORMULAS: dict[str, dict[str, Any]] = {
    "I.6.2a": {
        "formula": "exp(-theta**2 / 2) / sqrt(2 * pi)",
        "vars": ("theta",),
        "ranges": ((-3.0, 3.0),),
    },
    "I.12.1": {
        "formula": "mu * Nn",
        "vars": ("mu", "Nn"),
        "ranges": ((0.1, 1.0), (0.5, 5.0)),
    },
    "I.13.4": {
        "formula": "0.5 * m * v**2",
        "vars": ("m", "v"),
        "ranges": ((0.1, 10.0), (0.1, 10.0)),
    },
    "I.29.4": {
        "formula": "omega / c",
        "vars": ("omega", "c"),
        "ranges": ((1.0, 10.0), (1.0, 10.0)),
    },
    "II.11.28": {
        "formula": "p_d * cos(theta_1) / r**2",
        "vars": ("p_d", "theta_1", "r"),
        "ranges": ((0.1, 1.0), (0.0, np.pi), (0.5, 3.0)),
    },
}


def load_feynman_formula(
    name: str, n: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    entry = FEYNMAN_FORMULAS[name]
    rng = np.random.default_rng(seed)
    var_names = entry["vars"]
    ranges = entry["ranges"]
    X = np.stack(
        [rng.uniform(low, high, size=n) for (low, high) in ranges], axis=1
    ).astype(np.float64)

    symbols = [sp.Symbol(v) for v in var_names]
    expr = sp.sympify(entry["formula"])
    f = sp.lambdify(symbols, expr, modules=["numpy"])
    y = np.asarray(f(*[X[:, i] for i in range(len(var_names))]), dtype=np.float64)

    metadata = {
        "formula": entry["formula"],
        "vars": var_names,
        "n_vars": len(var_names),
        "ranges": ranges,
    }
    return X, y, metadata


def generate_pure_elementary(
    formula: str,
    n: int,
    n_features: int,
    seed: int | None = None,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, n_features))
    var_names = tuple(f"x{i}" for i in range(n_features))
    symbols = [sp.Symbol(v) for v in var_names]
    expr = sp.sympify(formula)
    f = sp.lambdify(symbols, expr, modules=["numpy"])
    y = np.asarray(f(*[X[:, i] for i in range(n_features)]), dtype=np.float64)
    y = y + rng.normal(scale=noise_std, size=n)
    return X, y, formula


def generate_pure_dt_regime(
    n: int,
    n_numeric: int,
    n_cat: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Data where the signal depends only on categorical splits."""
    rng = np.random.default_rng(seed)
    X_num = rng.uniform(-1.0, 1.0, size=(n, n_numeric))
    X_cat = rng.integers(0, 5, size=(n, n_cat)).astype(np.float64)
    X = np.concatenate([X_num, X_cat], axis=1)
    y = np.zeros(n)
    for c in range(n_cat):
        for v in range(5):
            mask = X_cat[:, c] == v
            y[mask] += (c + 1) * (v - 2)
    return X, y


def generate_mixed_regime(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Elementary signal on numeric features + categorical offset."""
    rng = np.random.default_rng(seed)
    X_num = rng.uniform(-1.0, 1.0, size=(n, 2))
    X_cat = rng.integers(0, 3, size=(n, 1)).astype(np.float64)
    X = np.concatenate([X_num, X_cat], axis=1)
    y = np.exp(X_num[:, 0]) + 0.5 * X_num[:, 1] ** 2
    for v in range(3):
        mask = X_cat[:, 0] == v
        y[mask] += (v - 1) * 0.5
    return X, y
