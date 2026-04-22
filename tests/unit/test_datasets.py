import numpy as np

from eml_boost.datasets import (
    generate_mixed_regime,
    generate_pure_dt_regime,
    generate_pure_elementary,
    load_feynman_formula,
)


def test_pure_elementary_generator_shape():
    X, y, formula_str = generate_pure_elementary(
        formula="x0 ** 2 + 2 * x1", n=50, n_features=2, seed=0
    )
    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert isinstance(formula_str, str)


def test_pure_dt_regime_has_categorical_columns():
    X, y = generate_pure_dt_regime(n=100, n_numeric=1, n_cat=2, seed=0)
    assert X.shape == (100, 3)
    # The final two columns should take a small number of distinct values (categorical).
    for col in (1, 2):
        distinct = np.unique(X[:, col])
        assert distinct.size <= 8, f"column {col} should be categorical, got {distinct.size} levels"
        # And the values should be non-negative integers when cast to int
        assert np.allclose(X[:, col], X[:, col].astype(int))


def test_mixed_regime_has_both():
    X, y = generate_mixed_regime(n=100, seed=0)
    assert X.shape[0] == 100
    assert X.shape[1] >= 2


def test_feynman_formula_loader_returns_data():
    # Smoke test using a known Feynman entry.
    X, y, metadata = load_feynman_formula("I.6.2a", n=100, seed=0)
    assert X.shape == (100, metadata["n_vars"])
    assert y.shape == (100,)
    assert "formula" in metadata
