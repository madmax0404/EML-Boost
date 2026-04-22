import sympy as sp

from eml_boost.symbolic.simplify import (
    rpn_length,
    snap_constants,
    snapped_to_sympy,
)
from eml_boost.symbolic.snap import SnappedTree


class TestSnappedToSympy:
    def test_trivial_depth_2_tree(self):
        # A depth-2 tree where every input snaps to constant 1 should
        # collapse to a constant expression (eml(1, 1) = e, applied repeatedly).
        # Depth 2: 2 internal inputs (root's), 4 leaf inputs.
        snapped = SnappedTree(
            depth=2,
            k=2,
            internal_input_count=2,
            leaf_input_count=4,
            terminal_choices=(0, 0, 0, 0, 0, 0),  # all 1s
        )
        expr = snapped_to_sympy(snapped, feature_names=("x1", "x2"))
        assert expr.is_number

    def test_identity_like_tree(self):
        # If the root's inputs snap to x1 and something_else, expression involves x1.
        snapped = SnappedTree(
            depth=2,
            k=2,
            internal_input_count=2,
            leaf_input_count=4,
            terminal_choices=(1, 0, 0, 0, 0, 0),  # root-left=x1, others=1
        )
        expr = snapped_to_sympy(snapped, feature_names=("x1", "x2"))
        x1 = sp.Symbol("x1")
        assert x1 in expr.free_symbols


class TestSnapConstants:
    def test_snaps_pi_coefficient(self):
        x = sp.Symbol("x")
        expr = 3.1415927 * x
        snapped = snap_constants(expr)
        # Should identify 3.1415927 ≈ pi and snap
        assert sp.simplify(snapped - sp.pi * x) == 0

    def test_leaves_non_special_coefficient(self):
        x = sp.Symbol("x")
        expr = 2.37 * x
        snapped = snap_constants(expr)
        # 2.37 isn't in the whitelist; should remain a Float
        coeffs = [c for c in snapped.atoms(sp.Float)]
        assert len(coeffs) == 1


class TestRpnLength:
    def test_constant_has_rpn_length_1(self):
        expr = sp.Integer(1)
        assert rpn_length(expr) == 1

    def test_x_plus_1_has_small_rpn(self):
        x = sp.Symbol("x")
        # RPN "x 1 +" = 3
        assert rpn_length(x + 1) == 3
