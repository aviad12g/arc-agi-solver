"""Conservative e-graph style saturation using existing rewrite rules (safe).

This module provides a small fixed-point saturation pass over DSLProgram using
our conservative rewrite rules. SMT validation is optional and disabled by
default. The pass is gated by Hydra under search.advanced.canonicalization.
"""

from __future__ import annotations

from typing import Optional
from arc_solver.reasoning.dsl_engine import DSLProgram
from arc_solver.reasoning.rewrite_rules import canonicalize_program


def saturate_program(program: DSLProgram, max_iters: int = 5, validate_smt: bool = False) -> DSLProgram:
    """Apply canonicalization repeatedly until a fixed point or max_iters.

    If validate_smt is True and z3 is available, optional SMT checks can be
    inserted in future work to validate rewrites; currently unused to avoid
    heavy deps. Returns a semantically equivalent or shorter program.
    """
    prev = program
    for _ in range(max_iters):
        curr = canonicalize_program(prev)
        if len(curr.operations) == len(prev.operations) and str(curr) == str(prev):
            return curr
        prev = curr
    return prev

