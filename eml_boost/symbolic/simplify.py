"""Sympy-based simplification pipeline for snapped EML trees.

Pipeline:
  1. SnappedTree -> sympy expression (via EML recursion)
  2. sympy.simplify (combine terms, apply identities)
  3. snap_constants: round near-exact floats to pi, e, fractions, etc.
  4. RPN length as complexity measure
"""

from __future__ import annotations

import sympy as sp

from eml_boost.symbolic.snap import SnappedTree

# Whitelist of known constants to snap to when rounding floats.
# Keep the whitelist small: false positives poison the "exact recovery" claim.
_CONSTANT_WHITELIST: tuple[tuple[sp.Expr, float], ...] = (
    (sp.pi, float(sp.pi)),
    (sp.E, float(sp.E)),
    (sp.pi / 2, float(sp.pi / 2)),
    (2 * sp.pi, float(2 * sp.pi)),
    (sp.Rational(1, 2), 0.5),
    (sp.Rational(1, 3), 1.0 / 3.0),
    (sp.Rational(2, 3), 2.0 / 3.0),
    (sp.Integer(1), 1.0),
    (sp.Integer(2), 2.0),
    (sp.Integer(-1), -1.0),
    (sp.Integer(0), 0.0),
)

_CONSTANT_SNAP_TOL = 1e-6


def _eml_sympy(x: sp.Expr, y: sp.Expr) -> sp.Expr:
    """Symbolic EML: exp(x) - ln(y)."""
    return sp.exp(x) - sp.log(y)


def snapped_to_sympy(snapped: SnappedTree, feature_names: tuple[str, ...]) -> sp.Expr:
    """Recursively build the sympy expression for a SnappedTree.

    Mirrors MasterFormula.forward, but in symbolic space.
    """
    if len(feature_names) != snapped.k:
        raise ValueError(f"feature_names length {len(feature_names)} != k={snapped.k}")

    features = [sp.Symbol(name) for name in feature_names]
    leaf_terminal_symbols = [sp.Integer(1)] + features  # [1, x_1, ..., x_k]

    # Deepest-level EML-node outputs (from leaf inputs).
    n_leaf_nodes = 2 ** (snapped.depth - 1)
    leaf_logit_offset = snapped.internal_input_count
    current_outputs: list[sp.Expr] = []

    for i in range(n_leaf_nodes):
        left_choice = snapped.terminal_choices[leaf_logit_offset + 2 * i]
        right_choice = snapped.terminal_choices[leaf_logit_offset + 2 * i + 1]
        left = leaf_terminal_symbols[left_choice]
        right = leaf_terminal_symbols[right_choice]
        current_outputs.append(_eml_sympy(left, right))

    # Walk up the internal levels.
    internal_logit_cursor = 0
    for level in range(snapped.depth - 2, -1, -1):
        n_nodes_at_level = 2**level
        next_outputs: list[sp.Expr] = []
        for i in range(n_nodes_at_level):
            left_choice = snapped.terminal_choices[internal_logit_cursor + 2 * i]
            right_choice = snapped.terminal_choices[internal_logit_cursor + 2 * i + 1]
            internal_terminals_left = (
                leaf_terminal_symbols + [current_outputs[2 * i]]
            )
            internal_terminals_right = (
                leaf_terminal_symbols + [current_outputs[2 * i + 1]]
            )
            left_expr = internal_terminals_left[left_choice]
            right_expr = internal_terminals_right[right_choice]
            next_outputs.append(_eml_sympy(left_expr, right_expr))
        internal_logit_cursor += 2 * n_nodes_at_level
        current_outputs = next_outputs

    assert len(current_outputs) == 1, "expected single root output"
    return sp.simplify(current_outputs[0])


def snap_constants(expr: sp.Expr, tol: float = _CONSTANT_SNAP_TOL) -> sp.Expr:
    """Replace Float atoms near whitelisted constants with their exact symbol."""
    substitutions: dict[sp.Float, sp.Expr] = {}
    for atom in expr.atoms(sp.Float):
        fval = float(atom)
        for symbol, svalue in _CONSTANT_WHITELIST:
            if abs(fval - svalue) < tol:
                substitutions[atom] = symbol
                break
            if abs(fval + svalue) < tol and svalue != 0:
                substitutions[atom] = -symbol
                break
    return expr.xreplace(substitutions)


def rpn_length(expr: sp.Expr) -> int:
    """Approximate RPN program length for the expression.

    Paper's complexity proxy (Table 4 of Odrzywołek 2026):
    Roughly the number of atoms + internal ops in a post-order traversal.
    """
    if expr.is_Atom:
        return 1
    return sum(rpn_length(arg) for arg in expr.args) + 1
