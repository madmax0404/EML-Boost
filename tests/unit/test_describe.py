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
