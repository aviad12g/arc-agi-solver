"""Version-space constraints from multi-example invariants.

Computes lightweight, accuracy-oriented constraints over allowed primitives by
analyzing invariants that appear to hold across all training pairs. These
constraints are used to filter enumeration in search without compromising
optimality (they only remove operations that would violate must-hold
properties inferred from training examples).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def compute_constraints(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Set[str]]:
    """Infer allowed primitive names based on simple global invariants.

    Invariants we check:
    - Shape preserved: input.shape == output.shape for all pairs
    - Nonzero pixel count preserved for all pairs
    - Color multiset preserved (sorted histogram equality) for all pairs

    Rules (conservative):
    - If shape preserved: disallow {Crop, Extract, Scale, Repeat}
    - If nonzero preserved: disallow ops that typically change filled area
      {Paint, FloodFill, Overlay}
    - If color multiset preserved: disallow ops that change histogram other
      than permutation; allow {MapColors, Rotate90/180, ReflectH/V, Translate}

    Returns a dict with key 'allowed_op_names' mapping to a set of strings.
    If no constraints can be inferred, returns an empty dict.
    """
    if not train_pairs:
        return {"allowed_op_names": set()}

    shape_preserved = all(src.shape == dst.shape for src, dst in train_pairs)

    def nnz(x: np.ndarray) -> int:
        return int(np.count_nonzero(x))

    area_preserved = all(nnz(src) == nnz(dst) for src, dst in train_pairs)

    def sorted_hist(x: np.ndarray) -> Tuple[int, ...]:
        unique, counts = np.unique(x, return_counts=True)
        # Create dense vector of length 10 for colors 0..9
        hist = np.zeros(10, dtype=np.int64)
        for c, n in zip(unique, counts):
            c_i = int(c)
            if 0 <= c_i < 10:
                hist[c_i] = int(n)
        # Sort counts ignoring zeros for multiset comparison
        return tuple(sorted(hist.tolist(), reverse=True))

    color_multiset_preserved = all(
        sorted_hist(src) == sorted_hist(dst) for src, dst in train_pairs
    )

    # Start permissive and remove risky ops
    allowed: Set[str] = {
        # Geometric
        "Rotate90", "Rotate180", "ReflectH", "ReflectV",
        # Spatial
        "Translate", "Crop", "Extract", "Scale", "Repeat",
        # Color
        "MapColors", "Paint",
        # Conditional/pattern
        "PaintIf", "FloodFill", "Overlay",
    }

    if shape_preserved:
        allowed -= {"Crop", "Extract", "Scale", "Repeat"}

    if area_preserved:
        allowed -= {"Paint", "FloodFill", "Overlay"}

    if color_multiset_preserved:
        # Restrict to ops that preserve histogram modulo permutation
        # Keep MapColors (permutation), geometric transforms, and Translate
        allowed &= {"Rotate90", "Rotate180", "ReflectH", "ReflectV", "Translate", "MapColors"}

    return {"allowed_op_names": allowed}


