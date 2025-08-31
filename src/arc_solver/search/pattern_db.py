"""Tiny pattern database (PDB) lower bounds (v0: exact D4 transform distance).

This PDB computes an exact lower bound when the target is exactly a D4 transform
of the current grid. The lower bound equals the minimal number of generator
operations needed using available primitives {Rotate90, Rotate180, ReflectH, ReflectV}.

If no D4 transform yields equality, returns 0 (no information), preserving
admissibility.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from arc_solver.perception.symmetry import SymmetryType


# Build minimal generator costs to realize each D4 transform using
# generators: r90 (cost 1), r180 (cost 1), fH (cost 1), fV (cost 1).
# We can compute these once via BFS on the Cayley graph or encode by hand.

_GEN_COSTS: Dict[SymmetryType, int] = {}


def _init_costs() -> Dict[SymmetryType, int]:
    # Manually encode minimal costs derived from small BFS:
    # Identity: 0
    # Rotate90: 1
    # Rotate180: 1
    # Rotate270: 2 (Rotate180 + Rotate90)
    # ReflectH: 1
    # ReflectV: 1
    # ReflectD1 (main diagonal): 2 (ReflectH + Rotate90) minimal achievable with given gens
    # ReflectD2 (anti-diagonal): 2 (ReflectH + Rotate270) => 3 ops or (ReflectV + Rotate90) => 2
    # Use the best known 2-op decompositions for diagonals.
    return {
        SymmetryType.IDENTITY: 0,
        SymmetryType.ROTATE_90: 1,
        SymmetryType.ROTATE_180: 1,
        SymmetryType.ROTATE_270: 2,
        SymmetryType.REFLECT_H: 1,
        SymmetryType.REFLECT_V: 1,
        SymmetryType.REFLECT_D1: 2,
        SymmetryType.REFLECT_D2: 2,
    }


_GEN_COSTS = _init_costs()


def _apply_transform_numpy(grid: np.ndarray, sym: SymmetryType) -> np.ndarray:
    if sym == SymmetryType.IDENTITY:
        return grid
    if sym == SymmetryType.ROTATE_90:
        return np.rot90(grid, k=-1)
    if sym == SymmetryType.ROTATE_180:
        return np.rot90(grid, k=2)
    if sym == SymmetryType.ROTATE_270:
        return np.rot90(grid, k=1)
    if sym == SymmetryType.REFLECT_H:
        return np.fliplr(grid)
    if sym == SymmetryType.REFLECT_V:
        return np.flipud(grid)
    if sym == SymmetryType.REFLECT_D1:
        return grid.T
    if sym == SymmetryType.REFLECT_D2:
        return np.rot90(grid.T, k=2)
    return grid


def d4_exact_lb(current: np.ndarray, target: np.ndarray) -> int:
    """Return exact lower bound if target is D4 transform of current; else 0.

    This is admissible because it is the minimal number of available
    parameter-free generator ops needed to reach target from current when a D4
    relation holds; otherwise we contribute 0.
    """
    if current.shape != target.shape:
        return 0
    syms = [
        SymmetryType.IDENTITY,
        SymmetryType.ROTATE_90,
        SymmetryType.ROTATE_180,
        SymmetryType.ROTATE_270,
        SymmetryType.REFLECT_H,
        SymmetryType.REFLECT_V,
        SymmetryType.REFLECT_D1,
        SymmetryType.REFLECT_D2,
    ]
    for sym in syms:
        transformed = _apply_transform_numpy(current, sym)
        if np.array_equal(transformed, target):
            return int(_GEN_COSTS.get(sym, 0))
    return 0

