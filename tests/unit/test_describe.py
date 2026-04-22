import numpy as np

from eml_boost import EmlBoostRegressor


def test_describe_returns_string():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = X[:, 0]
    m = EmlBoostRegressor(
        max_rounds=5, depth_eml=2, depth_dt=2, n_restarts=2, k=2, random_state=0
    ).fit(X, y)
    text = m.describe()
    assert isinstance(text, str)
    assert "rounds" in text.lower()


def test_coverage_is_between_0_and_1():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = X[:, 0]
    m = EmlBoostRegressor(
        max_rounds=5, depth_eml=2, depth_dt=2, n_restarts=2, k=2, random_state=0
    ).fit(X, y)
    c = m.coverage(X)
    assert 0.0 <= c <= 1.0 + 1e-6


def test_describe_reports_formula_when_recovered():
    # Train on y = x0 (a formula shallow EML can often recover)
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.5, 0.5, size=(120, 2))
    y = X[:, 0]
    m = EmlBoostRegressor(
        max_rounds=10, depth_eml=2, depth_dt=2, n_restarts=6, k=2, random_state=0
    ).fit(X, y)
    text = m.describe(X)
    # Either a formula was found, or an explicit "(none ...)" notice.
    assert "Recovered formula" in text


def test_formula_property_returns_sympy_or_none():
    import sympy as sp

    rng = np.random.default_rng(0)
    X = rng.uniform(-0.5, 0.5, size=(120, 2))
    y = X[:, 0]
    m = EmlBoostRegressor(
        max_rounds=5, depth_eml=2, depth_dt=2, n_restarts=3, k=2, random_state=0
    ).fit(X, y)
    f = m.formula
    assert f is None or isinstance(f, sp.Expr)
