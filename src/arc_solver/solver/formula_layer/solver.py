"""Formula Layer: analytic templates that attempt exact solving before search.

Implements a small set of deterministic templates that frequently solve ARC
patterns exactly. If any template yields a program that passes all training
pairs, we return that program and skip search.

Templates (v1):
- Rotate/Reflect ± MapColors(by-rank): try D4 transforms and optionally map
  input color ranks to target color ranks.
- TranslateMatchedShapes: match largest component centroid delta and translate.

Notes:
- All returned programs are validated across all training pairs.
- Programs are constructed using the DSLEngine primitives; cost reflects the
  underlying operations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation
from arc_solver.perception.blob_labeling import create_blob_labeler


def solve_with_templates(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    dsl_engine: Optional[DSLEngine] = None,
) -> Optional[DSLProgram]:
    """Try formula-layer templates. Return a valid program or None.

    Args:
        train_pairs: List of (input_grid, expected_output) pairs
        dsl_engine: Optional DSLEngine instance

    Returns:
        DSLProgram if a template solves all pairs; otherwise None
    """
    if not train_pairs:
        return None
    engine = dsl_engine or DSLEngine()

    # Try simple D4 transforms (+ optional color rank mapping)
    prog = _template_d4_with_color_rank(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try centroid-translation of dominant shapes
    prog = _template_translate_matched_shapes(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    return None


def _template_d4_with_color_rank(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    """Attempt D4 transform and optional color-rank mapping across all pairs.

    Strategy:
    - For each pair, compute the best D4 transform that maximizes pixel match
      optionally after rank-based color remapping. Require the same D4 across
      all pairs; if consistent, build program (Rotate/Reflect then MapColors).
    """
    # D4 operations corresponding to primitives available
    d4_ops = [
        ("IDENT", []),
        ("Rotate90", [DSLOperation("Rotate90", {})]),
        ("Rotate180", [DSLOperation("Rotate180", {})]),
        ("Rotate270", [DSLOperation("Rotate90", {}), DSLOperation("Rotate90", {}), DSLOperation("Rotate90", {})]),
        ("ReflectH", [DSLOperation("ReflectH", {})]),
        ("ReflectV", [DSLOperation("ReflectV", {})]),
    ]

    def color_rank_perm(src: np.ndarray, dst: np.ndarray) -> Optional[List[int]]:
        # Build a permutation of 10 colors by ranking frequencies in src and dst
        counts_src = _color_counts(src)
        counts_dst = _color_counts(dst)
        # Sort colors by descending frequency (stable)
        src_rank = [c for c, _ in sorted(counts_src.items(), key=lambda kv: (-kv[1], kv[0]))]
        dst_rank = [c for c, _ in sorted(counts_dst.items(), key=lambda kv: (-kv[1], kv[0]))]
        perm = list(range(10))
        for i, c in enumerate(src_rank):
            if i < len(dst_rank):
                perm[c] = dst_rank[i]
        return perm

    best_op_key: Optional[str] = None
    best_perm: Optional[List[int]] = None

    # Find a single D4 op that can be made consistent across all pairs
    for op_key, ops in d4_ops:
        consistent = True
        perm_for_all: Optional[List[int]] = None
        for src, dst in train_pairs:
            transformed = _apply_d4_numpy(src, op_key)
            if transformed.shape != dst.shape:
                consistent = False
                break
            perm = color_rank_perm(transformed, dst)
            transformed_mapped = _apply_color_perm(transformed, perm)
            if not np.array_equal(transformed_mapped, dst):
                consistent = False
                break
            if perm_for_all is None:
                perm_for_all = perm
            elif perm_for_all != perm:
                # Require a shared mapping across pairs for robustness
                consistent = False
                break
        if consistent and perm_for_all is not None:
            best_op_key = op_key
            best_perm = perm_for_all
            break

    if best_op_key is None or best_perm is None:
        return None

    program_ops: List[DSLOperation] = []
    for _, ops in d4_ops:
        # find matching key again (simple loop)
        pass
    # Rebuild ops by key
    for op_key, ops in d4_ops:
        if op_key == best_op_key:
            program_ops.extend(ops)
            break
    # Add MapColors if non-identity mapping
    if any(i != best_perm[i] for i in range(10)):
        program_ops.append(DSLOperation("MapColors", {"perm": best_perm}))

    return DSLProgram(program_ops)


def _template_translate_matched_shapes(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    """Translate dominant shape by centroid delta, consistent across pairs.

    - Extract largest non-zero component in each grid; compute centroid deltas
      from input to output; require consistent delta across pairs.
    """
    labeler = create_blob_labeler(use_gpu=False)
    deltas: List[Tuple[int, int]] = []
    for src, dst in train_pairs:
        src_blobs, _ = labeler.label_blobs(src)
        dst_blobs, _ = labeler.label_blobs(dst)
        if not src_blobs or not dst_blobs:
            return None
        src_largest = max(src_blobs, key=lambda b: b.area)
        dst_largest = max(dst_blobs, key=lambda b: b.area)
        dx = int(round(dst_largest.center_of_mass[0] - src_largest.center_of_mass[0]))
        dy = int(round(dst_largest.center_of_mass[1] - src_largest.center_of_mass[1]))
        deltas.append((dx, dy))
    # Check consistency
    if not deltas or any(d != deltas[0] for d in deltas):
        return None
    dx, dy = deltas[0]
    # Program: Translate(dx, dy)
    return DSLProgram([DSLOperation("Translate", {"dx": dx, "dy": dy})])


def _validate_on_all_examples(
    engine: DSLEngine, program: DSLProgram, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> bool:
    for src, dst in train_pairs:
        try:
            pred, _ = engine.execute_program(program, src)
            if not np.array_equal(pred, dst):
                return False
        except Exception:
            return False
    return True


def _color_counts(grid: np.ndarray) -> dict:
    unique, counts = np.unique(grid, return_counts=True)
    return {int(c): int(n) for c, n in zip(unique.tolist(), counts.tolist())}


def _apply_color_perm(grid: np.ndarray, perm: List[int]) -> np.ndarray:
    result = grid.copy()
    for old in range(10):
        if old < len(perm):
            result[grid == old] = perm[old]
    return result


def _apply_d4_numpy(grid: np.ndarray, key: str) -> np.ndarray:
    if key == "IDENT":
        return grid.copy()
    if key == "Rotate90":
        return np.rot90(grid, k=-1)
    if key == "Rotate180":
        return np.rot90(grid, k=2)
    if key == "Rotate270":
        return np.rot90(grid, k=1)
    if key == "ReflectH":
        return np.fliplr(grid)
    if key == "ReflectV":
        return np.flipud(grid)
    return grid.copy()


