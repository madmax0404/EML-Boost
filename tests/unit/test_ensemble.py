import numpy as np

from eml_boost import EmlBoostRegressor


def test_fit_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] + 0.5 * X[:, 1]
    model = EmlBoostRegressor(
        max_rounds=8, depth_eml=2, depth_dt=2, n_restarts=2, k=2, random_state=0
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_fit_predict_reduces_error():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = X[:, 0] ** 2
    model = EmlBoostRegressor(
        max_rounds=15, depth_eml=2, depth_dt=2, n_restarts=3, k=2, random_state=1
    )
    model.fit(X, y)
    rmse = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((y.mean() - y) ** 2)))
    assert rmse < 0.6 * baseline_rmse
