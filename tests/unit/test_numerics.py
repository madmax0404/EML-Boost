import math

import pytest
import torch

from eml_boost._numerics import eml, is_real_valued, safe_exp


class TestSafeExp:
    def test_normal_value(self):
        result = safe_exp(torch.tensor(1.0, dtype=torch.complex128))
        assert torch.allclose(result.real, torch.tensor(math.e, dtype=torch.float64), atol=1e-10)

    def test_clamps_large_positive(self):
        result = safe_exp(torch.tensor(1000.0, dtype=torch.complex128))
        assert torch.isfinite(result).all()
        assert result.real.item() < torch.exp(torch.tensor(51.0)).item()

    def test_clamps_large_negative(self):
        result = safe_exp(torch.tensor(-1000.0, dtype=torch.complex128))
        assert torch.isfinite(result).all()

    def test_preserves_imaginary(self):
        z = torch.tensor(1.0 + 2.0j, dtype=torch.complex128)
        result = safe_exp(z)
        expected = torch.exp(z)
        assert torch.allclose(result, expected, atol=1e-10)


class TestEml:
    def test_eml_1_1_is_e(self):
        one = torch.tensor(1.0, dtype=torch.complex128)
        result = eml(one, one)
        # eml(1,1) = exp(1) - ln(1) = e - 0 = e
        assert torch.allclose(result.real, torch.tensor(math.e, dtype=torch.float64), atol=1e-10)

    def test_eml_0_1_is_1(self):
        zero = torch.tensor(0.0, dtype=torch.complex128)
        one = torch.tensor(1.0, dtype=torch.complex128)
        # eml(0,1) = exp(0) - ln(1) = 1 - 0 = 1
        result = eml(zero, one)
        assert torch.allclose(result.real, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    def test_eml_is_batched(self):
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128)
        y = torch.tensor([1.0, 1.0, 1.0], dtype=torch.complex128)
        result = eml(x, y)
        assert result.shape == (3,)


class TestIsRealValued:
    def test_real_tensor_is_real(self):
        z = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex128)
        assert is_real_valued(z)

    def test_complex_tensor_is_not_real(self):
        z = torch.tensor([1.0 + 2j, 0.0], dtype=torch.complex128)
        assert not is_real_valued(z)

    def test_nearly_real_tensor_is_real_within_tol(self):
        z = torch.tensor([1.0 + 1e-12j], dtype=torch.complex128)
        assert is_real_valued(z, tol=1e-8)
