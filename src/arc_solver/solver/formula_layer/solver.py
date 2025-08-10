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
    # Honor global DSLEngine config if provided elsewhere
    engine = dsl_engine or DSLEngine(max_program_length=5, max_execution_time=0.01, adaptive_length_limits=True)

    # Try simple D4 transforms (+ optional color rank mapping)
    prog = _template_d4_with_color_rank(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try centroid-translation of dominant shapes
    prog = _template_translate_matched_shapes(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try overlay after extraction (non-zero only)
    prog = _template_extract_then_overlay(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try repeat along row/column
    prog = _template_repeat_pattern(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try flood fill with a likely start
    prog = _template_flood_fill(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try drawing a frame if outputs appear to have a uniform border
    prog = _template_draw_frame(train_pairs, engine)
    if prog is not None and _validate_on_all_examples(engine, prog, train_pairs):
        return prog

    # Try majority color fill when output is a constant grid
    prog = _template_majority_fill(train_pairs, engine)
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


def _template_extract_then_overlay(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    # Heuristic: if output looks like a subregion of input placed elsewhere
    try:
        src0, dst0 = train_pairs[0]
        # Find tight non-zero bbox in dst0
        rows, cols = np.where(dst0 != 0)
        if rows.size == 0:
            return None
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        # Assume same bbox in src0 (rough)
        r0s, r1s = r0, r1
        c0s, c1s = c0, c1
        prog = DSLProgram([
            DSLOperation("Extract", {"r1": int(r0s), "r2": int(r1s), "c1": int(c0s), "c2": int(c1s)}),
            # Overlay at top-left (can be improved)
            DSLOperation("Overlay", {"pattern": dst0[r0:r1+1, c0:c1+1].astype(np.int32), "position_row": int(r0), "position_col": int(c0)}),
        ])
        return prog
    except Exception:
        return None


def _template_repeat_pattern(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    try:
        src0, dst0 = train_pairs[0]
        # If dst has repeated rows/cols of a band from src
        band = src0[0:1, :]
        if band.shape[1] == dst0.shape[1] and dst0.shape[0] % 1 == 0:
            return DSLProgram([DSLOperation("Repeat", {"pattern": band.astype(np.int32), "count": int(dst0.shape[0]), "direction": "vertical"})])
        col = src0[:, 0:1]
        if col.shape[0] == dst0.shape[0] and dst0.shape[1] % 1 == 0:
            return DSLProgram([DSLOperation("Repeat", {"pattern": col.astype(np.int32), "count": int(dst0.shape[1]), "direction": "horizontal"})])
        return None
    except Exception:
        return None


def _template_flood_fill(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    try:
        src0, dst0 = train_pairs[0]
        # Find a cell whose color changed massively; try flood fill from center
        center = (src0.shape[0] // 2, src0.shape[1] // 2)
        new_color = int(np.bincount(dst0.flatten()).argmax())
        return DSLProgram([DSLOperation("FloodFill", {"start_row": int(center[0]), "start_col": int(center[1]), "color": new_color})])
    except Exception:
        return None


def _template_draw_frame(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    """Detect if outputs have a uniform 1px border; draw it.

    - For each dst, check top/bottom rows and left/right columns are constant color
      and identical across sides. If a single color c is consistent across pairs,
      propose DrawFrame(color=c).
    """
    try:
        colors: List[int] = []
        for _, dst in train_pairs:
            if dst.size == 0:
                return None
            h, w = dst.shape
            top = int(dst[0, 0]) if w > 0 else 0
            bottom = int(dst[h - 1, 0]) if w > 0 else 0
            left = int(dst[0, 0]) if h > 0 else 0
            right = int(dst[0, w - 1]) if h > 0 else 0
            # Check uniformity on each border
            if not (np.all(dst[0, :] == top) and np.all(dst[h - 1, :] == bottom)):
                return None
            if not (np.all(dst[:, 0] == left) and np.all(dst[:, w - 1] == right)):
                return None
            # Require the same color on all four sides
            if not (top == bottom == left == right):
                return None
            colors.append(top)
        # Consistent across pairs
        if not colors or any(c != colors[0] for c in colors):
            return None
        c = int(colors[0])
        return DSLProgram([DSLOperation("DrawFrame", {"color": c})])
    except Exception:
        return None


def _template_majority_fill(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], engine: DSLEngine
) -> Optional[DSLProgram]:
    """If every output is a single constant color, fill with that color.

    We detect a single color c across all outputs (ignoring shape mismatches),
    and return PaintIf with a predicate that always matches via FloodFill from (0,0)
    to overwrite entire canvas. Simpler: use Overlay of a constant pattern.
    """
    try:
        colors: List[int] = []
        shapes: List[Tuple[int, int]] = []
        for _, dst in train_pairs:
            if dst.size == 0:
                return None
            uniq = np.unique(dst)
            if len(uniq) != 1:
                return None
            colors.append(int(uniq[0]))
            shapes.append(dst.shape)
        if not colors or any(c != colors[0] for c in colors):
            return None
        color = int(colors[0])
        # Build a constant pattern for first target shape and overlay at (0,0)
        h, w = shapes[0]
        pattern = np.full((h, w), color, dtype=np.int32)
        return DSLProgram([
            DSLOperation("Overlay", {"pattern": pattern, "position_row": 0, "position_col": 0})
        ])
    except Exception:
        return None


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


