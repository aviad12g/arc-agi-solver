"""Object-level synthesis via blob matching and global transform inference.

Attempts to infer a global D4 transform, a global translation, and a color
permutation from object correspondences between input and output.

If a consistent (D4, Translate, MapColors) program works across all train
pairs, returns it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation
from arc_solver.perception.blob_labeling import create_blob_labeler


def synthesize_object_level_program(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    dsl_engine: Optional[DSLEngine] = None,
) -> Optional[DSLProgram]:
    if not train_pairs:
        return None
    engine = dsl_engine or DSLEngine(max_program_length=5, max_execution_time=0.01, adaptive_length_limits=True)
    labeler = create_blob_labeler(use_gpu=False)

    # Candidate D4 ops (sequence of DSLOperation)
    d4_ops = [
        ("IDENT", []),
        ("Rotate90", [DSLOperation("Rotate90", {})]),
        ("Rotate180", [DSLOperation("Rotate180", {})]),
        ("Rotate270", [DSLOperation("Rotate90", {}), DSLOperation("Rotate90", {}), DSLOperation("Rotate90", {})]),
        ("ReflectH", [DSLOperation("ReflectH", {})]),
        ("ReflectV", [DSLOperation("ReflectV", {})]),
    ]

    best_op_key: Optional[str] = None
    best_perm: Optional[List[int]] = None
    best_translation: Optional[Tuple[int, int]] = None

    for op_key, ops in d4_ops:
        consistent = True
        perm_for_all: Optional[List[int]] = None
        dx_all: List[int] = []
        dy_all: List[int] = []

        for src, dst in train_pairs:
            # Apply D4 to source grid
            src_d4 = _apply_d4_numpy(src, op_key)
            if src_d4.shape != dst.shape:
                consistent = False
                break

            # Label blobs and match by nearest centroid color-wise
            src_blobs, _ = labeler.label_blobs(src_d4)
            dst_blobs, _ = labeler.label_blobs(dst)
            if not src_blobs or not dst_blobs:
                consistent = False
                break

            # Build color-driven centroid mapping: for each color in src,
            # match to closest centroid of same color in dst (fallback to any)
            pairs: List[Tuple[Tuple[float, float], Tuple[float, float], int, int]] = []
            unmatched_dst = set(range(len(dst_blobs)))
            for i, sb in enumerate(src_blobs):
                # candidate dst indices with same color
                candidates = [j for j, db in enumerate(dst_blobs) if db.color == sb.color and j in unmatched_dst]
                if not candidates:
                    candidates = [j for j in unmatched_dst]
                # pick nearest by centroid
                best_j = None
                best_dist = float("inf")
                for j in candidates:
                    db = dst_blobs[j]
                    dist = (sb.center_of_mass[0] - db.center_of_mass[0]) ** 2 + (sb.center_of_mass[1] - db.center_of_mass[1]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j is None:
                    consistent = False
                    break
                pairs.append((sb.center_of_mass, dst_blobs[best_j].center_of_mass, sb.color, dst_blobs[best_j].color))
                unmatched_dst.discard(best_j)
            if not consistent:
                break
            # Estimate translation as average centroid delta
            dx = int(round(np.mean([p2[0] - p1[0] for p1, p2, _, _ in pairs])))
            dy = int(round(np.mean([p2[1] - p1[1] for p1, p2, _, _ in pairs])))
            dx_all.append(dx)
            dy_all.append(dy)

            # Build color permutation mapping from src->dst colors observed
            perm = list(range(10))
            for _, _, c_src, c_dst in pairs:
                perm[c_src] = c_dst

            if perm_for_all is None:
                perm_for_all = perm
            elif perm_for_all != perm:
                consistent = False
                break

            # Verify this triple on the pair
            composed = _apply_color_perm(_translate_numpy(src_d4, dx, dy), perm)
            if not np.array_equal(composed, dst):
                consistent = False
                break

        if consistent and perm_for_all is not None:
            best_op_key = op_key
            best_perm = perm_for_all
            best_translation = (int(round(np.mean(dx_all))), int(round(np.mean(dy_all))))
            break

    if best_op_key is None or best_perm is None or best_translation is None:
        return None

    # Build program
    program_ops: List[DSLOperation] = []
    # D4
    for op_key2, ops2 in d4_ops:
        if op_key2 == best_op_key:
            program_ops.extend(ops2)
            break
    # Translate
    dx, dy = best_translation
    if dx != 0 or dy != 0:
        program_ops.append(DSLOperation("Translate", {"dx": dx, "dy": dy}))
    # MapColors
    if any(i != best_perm[i] for i in range(10)):
        program_ops.append(DSLOperation("MapColors", {"perm": best_perm}))

    program = DSLProgram(program_ops)
    # Validate across all pairs for safety
    for src, dst in train_pairs:
        pred, _ = engine.execute_program(program, src)
        if not np.array_equal(pred, dst):
            return None
    return program


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


def _translate_numpy(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return grid.copy()
    h, w = grid.shape
    out = np.zeros_like(grid)
    for r in range(h):
        nr = r + dx
        if 0 <= nr < h:
            c0 = max(0, -dy)
            c1 = min(w, w - dy)
            out[nr, c0 + dy : c1 + dy] = grid[r, c0:c1]
    return out


def _apply_color_perm(grid: np.ndarray, perm: List[int]) -> np.ndarray:
    result = grid.copy()
    for old in range(10):
        if old < len(perm):
            result[grid == old] = perm[old]
    return result


