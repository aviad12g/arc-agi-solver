"""Abstract interpretation tools: admissible lower bound and feasibility checks.

We provide a very conservative admissible lower bound h_abs and a basic
infeasibility predicate. The bound is designed to never exceed the true minimal
number of steps required, so A* optimality is preserved when combining with
other admissible heuristics via max().
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def compute_h_abs(current: np.ndarray, target: np.ndarray) -> float:
    """Compute a tiny admissible lower bound between grids.

    Heuristic: return 0 if grids already match. Otherwise, return 1 if any
    of these properties differ: shape, color-multiset, nonzero count. This is a
    weak lower bound but admissible (never overestimates minimal steps).
    """
    try:
        if current is target:
            return 0.0
        if current.shape == target.shape and np.array_equal(current, target):
            return 0.0

        # If shapes differ, at least one operation is needed
        if current.shape != target.shape:
            return 1.0

        # Compare nonzero counts
        if np.count_nonzero(current) != np.count_nonzero(target):
            return 1.0

        # Compare color multisets (ignoring labels order)
        def multiset(x: np.ndarray):
            unique, counts = np.unique(x, return_counts=True)
            return tuple(sorted(counts.tolist()))

        if multiset(current) != multiset(target):
            return 1.0

        # Euler characteristic difference lower bound (very weak):
        # If the difference in Euler numbers is non-zero, at least one op.
        try:
            e_cur = _euler_number(current)
            e_tgt = _euler_number(target)
            if e_cur != e_tgt:
                return 1.0
        except Exception:
            pass

        # Other differences exist (pixel positions), but a single op might fix
        # (e.g., Rotate/Reflect). Returning 1 remains admissible.
        if not np.array_equal(current, target):
            return 1.0
        return 0.0
    except Exception:
        # Fail-safe: no guidance
        return 0.0


def is_infeasible(current: np.ndarray, target: np.ndarray, remaining_depth: Optional[int] = None) -> bool:
    """Conservative infeasibility check (returns False unless trivially impossible).

    Currently only checks trivial case where remaining_depth is 0 and grids
    differ, in which case it's infeasible to reach the target.
    """
    if remaining_depth is not None and remaining_depth <= 0:
        return not np.array_equal(current, target)

    # If shapes differ and no depth remains to change shape, infeasible
    if remaining_depth is not None and remaining_depth == 0 and current.shape != target.shape:
        return True

    # If color support differs and remaining_depth is 0, infeasible
    if remaining_depth is not None and remaining_depth == 0:
        if set(np.unique(current)) != set(np.unique(target)):
            return True

    return False


def _euler_number(grid: np.ndarray) -> int:
    """Approximate Euler characteristic using 4-connectivity.

    e = components - holes. We approximate holes via connected components of the
    background inside the bounding box of foreground.
    """
    fg = grid != 0
    if not fg.any():
        return 0
    from scipy import ndimage as ndi  # optional; catch ImportError at call site
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, num = ndi.label(fg, structure=structure)
    # Extract minimal bbox
    rows, cols = np.where(fg)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    sub_bg = ~fg[r0 : r1 + 1, c0 : c1 + 1]
    if sub_bg.size == 0:
        holes = 0
    else:
        bg_labeled, bg_num = ndi.label(sub_bg, structure=structure)
        # border-connected background not counted as holes
        border_labels = set(np.unique(np.concatenate([
            bg_labeled[0, :], bg_labeled[-1, :], bg_labeled[:, 0], bg_labeled[:, -1]
        ])))
        border_labels.discard(0)
        all_labels = set(np.unique(bg_labeled))
        all_labels.discard(0)
        holes = len(all_labels - border_labels)
    return int(num - holes)


