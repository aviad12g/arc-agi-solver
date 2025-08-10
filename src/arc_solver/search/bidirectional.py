"""Bidirectional meet-in-the-middle search scaffold.

Implements a minimal forward/backward frontier structure over invertible ops
(Rotate/Reflect/Translate/MapColors) with D4+palette canonicalization.

This is a scaffold to integrate with the main A* once matured.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation


INVERTIBLE_OPS = {"Rotate90", "Rotate180", "ReflectH", "ReflectV", "Translate", "MapColors"}


def canonicalize(grid: np.ndarray) -> bytes:
    """D4 canonicalization by lexicographically minimal bytes among 8 transforms."""
    h, w = grid.shape
    candidates = [grid]
    candidates.append(np.rot90(grid, 1))
    candidates.append(np.rot90(grid, 2))
    candidates.append(np.rot90(grid, 3))
    candidates.append(np.fliplr(grid))
    candidates.append(np.flipud(grid))
    candidates.append(grid.T)
    candidates.append(np.rot90(grid.T, 2))
    return min(c.tobytes() for c in candidates)


def meet_in_the_middle(
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    max_depth_half: int = 2,
    dsl_engine: Optional[DSLEngine] = None,
) -> Optional[DSLProgram]:
    engine = dsl_engine or DSLEngine(max_program_length=5, max_execution_time=0.01, adaptive_length_limits=True)
    if np.array_equal(input_grid, target_grid):
        return DSLProgram([])

    # Forward frontier
    f_front: Dict[bytes, DSLProgram] = {canonicalize(input_grid): DSLProgram([])}
    # Backward frontier stores inverse programs applied to target
    b_front: Dict[bytes, DSLProgram] = {canonicalize(target_grid): DSLProgram([])}

    def expand(program: DSLProgram, grid: np.ndarray) -> List[Tuple[DSLProgram, np.ndarray]]:
        successors: List[Tuple[DSLProgram, np.ndarray]] = []
        for op_name in INVERTIBLE_OPS:
            if op_name == "Translate":
                # Try small deltas
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    op = DSLOperation("Translate", {"dx": dx, "dy": dy})
                    ng = engine.apply_operation(grid, op)
                    if not np.array_equal(ng, grid):
                        successors.append((program.append(op), ng))
            elif op_name == "MapColors":
                # Try identity and a simple swap (1,2)
                idp = list(range(10))
                op = DSLOperation("MapColors", {"perm": idp})
                ng = engine.apply_operation(grid, op)
                successors.append((program.append(op), ng))
                swap = idp.copy(); swap[1], swap[2] = swap[2], swap[1]
                op2 = DSLOperation("MapColors", {"perm": swap})
                ng2 = engine.apply_operation(grid, op2)
                successors.append((program.append(op2), ng2))
            else:
                op = DSLOperation(op_name, {})
                ng = engine.apply_operation(grid, op)
                if not np.array_equal(ng, grid):
                    successors.append((program.append(op), ng))
        return successors

    # BFS limited depth on both sides
    f_layers = [(DSLProgram([]), input_grid)]
    for _ in range(max_depth_half):
        new_layers = []
        for prog, grid in f_layers:
            for nprog, ng in expand(prog, grid):
                key = canonicalize(ng)
                if key not in f_front:
                    f_front[key] = nprog
                    new_layers.append((nprog, ng))
        f_layers = new_layers
        if not f_layers:
            break

    b_layers = [(DSLProgram([]), target_grid)]
    for _ in range(max_depth_half):
        new_layers = []
        for prog, grid in b_layers:
            for nprog, ng in expand(prog, grid):
                key = canonicalize(ng)
                if key not in b_front:
                    b_front[key] = nprog
                    new_layers.append((nprog, ng))
        b_layers = new_layers
        if not b_layers:
            break

    # Join
    meet_keys = set(f_front.keys()) & set(b_front.keys())
    if not meet_keys:
        return None
    meet_key = next(iter(meet_keys))
    prefix = f_front[meet_key]
    suffix = b_front[meet_key]
    # Reverse suffix (approximate inverse): here we simply append
    # In a full implementation, we would invert each op in suffix.
    program = DSLProgram(prefix.operations + suffix.operations)
    # Verify
    out, _ = engine.execute_program(program, input_grid)
    if np.array_equal(out, target_grid):
        return program
    return None


