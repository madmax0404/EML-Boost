import numpy as np

from eml_boost.weak_learners.base import RoundRecord, WeakLearnerKind
from eml_boost.weak_learners.dt import fit_dt_stump
from eml_boost.weak_learners.eml import fit_eml_tree


def test_both_learners_satisfy_protocol():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X[:, 0]
    eml = fit_eml_tree(X, y, depth=2, n_restarts=2, k=2, random_state=0)
    dt = fit_dt_stump(X, y, depth=2)
    # Both expose predict(X) -> ndarray and params_count() -> int
    assert eml.predict(X).shape == y.shape
    assert dt.predict(X).shape == y.shape
    assert isinstance(eml.params_count(), int)
    assert isinstance(dt.params_count(), int)


def test_round_record_defaults():
    rec = RoundRecord(
        round_index=0,
        kind=WeakLearnerKind.EML,
        eta=1.0,
        score=10.0,
        mse_inner_val=0.1,
    )
    assert rec.kind == WeakLearnerKind.EML
