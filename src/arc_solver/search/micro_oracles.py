"""Micro-oracle interface for SAT/ILP-backed subproblems (scaffold).

Provides a clean API for oracles that can return a solution program snippet or
an UNSAT certificate for a given (input,target) family. Certificates can be
recorded in the UNSAT cache for pruning.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
from arc_solver.reasoning.dsl_engine import DSLProgram, DSLOperation


class OracleResult:
    def __init__(self, program: Optional[DSLProgram], unsat: bool) -> None:
        self.program = program
        self.unsat = unsat


class ShapeOracle:
    """Trivial shape oracle: declares UNSAT when target is larger than input."""

    def solve(self, inp: np.ndarray, tgt: np.ndarray) -> OracleResult:
        h1, w1 = inp.shape
        h2, w2 = tgt.shape
        if h2 > h1 or w2 > w1:
            return OracleResult(None, True)
        return OracleResult(None, False)


class FrameOracle:
    """Detect targets that are input with a uniform 1px frame drawn.

    If target differs from input only on the 1-pixel border and that border is
    uniform color c, return a DrawFrame(c) program.
    """

    def solve(self, inp: np.ndarray, tgt: np.ndarray) -> OracleResult:
        if inp.shape != tgt.shape:
            return OracleResult(None, False)
        h, w = inp.shape
        if h < 2 or w < 2:
            return OracleResult(None, False)
        # interior must match input; border may differ but must be uniform color
        interior_inp = inp[1:h-1, 1:w-1]
        interior_tgt = tgt[1:h-1, 1:w-1]
        try:
            if not np.array_equal(interior_inp, interior_tgt):
                return OracleResult(None, False)
        except Exception:
            return OracleResult(None, False)

        # Check border uniformity in target
        border_vals = np.concatenate([
            tgt[0, :], tgt[h-1, :], tgt[:, 0], tgt[:, w-1]
        ])
        unique = np.unique(border_vals)
        if unique.size != 1:
            return OracleResult(None, False)
        color = int(unique[0])
        # Quick confirm: applying DrawFrame(color) to input gives target
        prog = DSLProgram([DSLOperation("DrawFrame", {"color": color})])
        out, _ = DSLProgram  # type: ignore  # silence lints while building
        try:
            from arc_solver.reasoning.dsl_engine import DSLEngine
            engine = DSLEngine()
            out, _ = engine.execute_program(prog, inp)
            if np.array_equal(out, tgt):
                return OracleResult(prog, False)
        except Exception:
            return OracleResult(None, False)
        return OracleResult(None, False)


class RecolorTranslateOracle:
    """Detect targets obtainable by a small integer translate followed by color remap.

    We search small shifts (dx,dy in [-2..2]) and test if a consistent color
    mapping exists such that MapColorsPermutation(Translate(inp)) == tgt.
    """

    def _shift(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = grid.shape
        out = np.zeros_like(grid)
        for r in range(h):
            nr = r + dx
            if 0 <= nr < h:
                c0 = max(0, -dy)
                c1 = min(w, w - dy)
                out[nr, c0 + dy: c1 + dy] = grid[r, c0:c1]
        return out

    def _consistent_mapping(self, a: np.ndarray, b: np.ndarray) -> Optional[dict]:
        mapping = {}
        h, w = a.shape
        for r in range(h):
            for c in range(w):
                va = int(a[r, c])
                vb = int(b[r, c])
                if va == vb:
                    continue
                if va in mapping and mapping[va] != vb:
                    return None
                mapping[va] = vb
        return mapping

    def solve(self, inp: np.ndarray, tgt: np.ndarray) -> OracleResult:
        if inp.shape != tgt.shape:
            return OracleResult(None, False)
        h, w = inp.shape
        shifts = [-2, -1, 0, 1, 2]
        for dx in shifts:
            for dy in shifts:
                try:
                    shifted = self._shift(inp, dx, dy)
                    mapping = self._consistent_mapping(shifted, tgt)
                    if mapping is None:
                        continue
                    # Verify mapping by constructing program
                    prog_ops = []
                    if dx != 0 or dy != 0:
                        prog_ops.append(DSLOperation("Translate", {"dx": dx, "dy": dy}))
                    prog_ops.append(DSLOperation("MapColorsPermutation", {"perm_map": mapping}))
                    prog = DSLProgram(prog_ops)
                    from arc_solver.reasoning.dsl_engine import DSLEngine
                    engine = DSLEngine()
                    out, _ = engine.execute_program(prog, inp)
                    if np.array_equal(out, tgt):
                        return OracleResult(prog, False)
                except Exception:
                    continue
        return OracleResult(None, False)


def run_oracles(inp: np.ndarray, tgt: np.ndarray) -> OracleResult:
    """Run enabled micro-oracles and return the first decisive result."""
    # Extend this list with real SAT/ILP oracles
    oracles = [RecolorTranslateOracle(), FrameOracle(), ShapeOracle()]
    for oracle in oracles:
        res = oracle.solve(inp, tgt)
        if res.unsat or res.program is not None:
            return res
    return OracleResult(None, False)
