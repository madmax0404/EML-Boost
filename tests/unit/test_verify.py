import numpy as np
import sympy as sp

from eml_boost.symbolic.verify import reproduces_numerically


class TestReproduction:
    def test_matching_outputs_verify(self):
        x = sp.Symbol("x")
        expr = x**2
        samples = np.random.randn(50, 1).astype(np.float64)
        numeric_output = samples[:, 0] ** 2
        assert reproduces_numerically(
            expr, ("x",), samples, numeric_output, tol=1e-10
        )

    def test_mismatching_outputs_reject(self):
        x = sp.Symbol("x")
        expr = x**2
        samples = np.random.randn(50, 1).astype(np.float64)
        wrong_output = np.random.randn(50)  # unrelated
        assert not reproduces_numerically(
            expr, ("x",), samples, wrong_output, tol=1e-10
        )
