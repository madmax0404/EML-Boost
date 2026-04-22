"""Low-level numerical primitives for EML-Boost.

All EML computations run in torch.complex128 because the operator
requires complex arithmetic for constants like pi and i (via
principal-branch ln of negative reals).
"""

import torch

_EXP_CLAMP = 50.0
_REAL_TOL = 1e-8


def safe_exp(z: torch.Tensor) -> torch.Tensor:
    """Exponential with magnitude clamping on the real part to prevent overflow."""
    real = z.real.clamp(min=-_EXP_CLAMP, max=_EXP_CLAMP)
    imag = z.imag
    return torch.exp(torch.complex(real, imag))


def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The EML Sheffer operator: eml(x, y) = exp(x) - ln(y).

    Uses principal-branch ln (torch.log) on complex inputs.
    """
    return safe_exp(x) - torch.log(y)


def is_real_valued(z: torch.Tensor, tol: float = _REAL_TOL) -> bool:
    """Check whether a complex tensor is real-valued within tolerance."""
    return bool((z.imag.abs() < tol).all().item())
