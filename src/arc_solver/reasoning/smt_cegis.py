"""CEGIS/SMT skeleton for exact synthesis (stub).

This module provides a thin sketch of a CEGIS-style solver interface. The
implementation is intentionally conservative to avoid heavy dependencies.
Integrators can swap in a real Z3/pySMT backend later.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from arc_solver.reasoning.dsl_engine import DSLEngine, DSLProgram, DSLOperation


def try_cegis_solve(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    max_length: int = 4,
    dsl_engine: Optional[DSLEngine] = None,
) -> Optional[DSLProgram]:
    """Attempt synthesis via simple CEGIS/SMT for K<=3 using typed params.

    If z3 is available, choose a skeleton of length L<=min(3,max_length) over a
    small op set {Rotate/Reflect/Translate(Map: identity or swap)} with integer
    params for Translate in {(-1,0,1)}. Validate concretely on all examples and
    iterate with blocking clauses. If z3 is unavailable, fall back to a tiny
    whitelist.
    """
    engine = dsl_engine or DSLEngine(max_program_length=max_length, max_execution_time=0.01, adaptive_length_limits=True)

    # Try z3-based skeleton selection
    try:
        import z3  # type: ignore

        op_names = ["Rotate90", "Rotate180", "ReflectH", "ReflectV", "Translate", "MapColors"]
        # MapColors will use either identity or swap(1,2)

        max_L = min(3, max_length)
        for L in range(1, max_L + 1):
            s = z3.Solver()
            op_vars = [z3.Int(f"op_{i}") for i in range(L)]
            for v in op_vars:
                s.add(z3.And(v >= 0, v < len(op_names)))
            # Translate params per slot
            dx_vars = [z3.Int(f"dx_{i}") for i in range(L)]
            dy_vars = [z3.Int(f"dy_{i}") for i in range(L)]
            for dx, dy in zip(dx_vars, dy_vars):
                s.add(z3.And(dx >= -1, dx <= 1))
                s.add(z3.And(dy >= -1, dy <= 1))

            models_tried = 0
            while s.check() == z3.sat and models_tried < 200:
                m = s.model()
                ops: List[DSLOperation] = []
                for i in range(L):
                    idx = int(m[op_vars[i]].as_long())
                    name = op_names[idx]
                    if name == "Translate":
                        dx = int(m[dx_vars[i]].as_long())
                        dy = int(m[dy_vars[i]].as_long())
                        ops.append(DSLOperation("Translate", {"dx": dx, "dy": dy}))
                    elif name == "MapColors":
                        perm = list(range(10))
                        # simple choice: identity OR swap(1,2)
                        # differentiate by parity of i for variety
                        if i % 2 == 1:
                            perm[1], perm[2] = perm[2], perm[1]
                        ops.append(DSLOperation("MapColors", {"perm": perm}))
                    else:
                        ops.append(DSLOperation(name, {}))

                program = DSLProgram(ops)
                if _validate_on_all_examples(engine, program, train_pairs):
                    return program

                # Block this exact assignment and continue
                block = []
                for i in range(L):
                    block.append(op_vars[i] == m[op_vars[i]])
                s.add(z3.Not(z3.And(block)))
                models_tried += 1

    except Exception:
        pass

    # Fallback tiny whitelist
    candidate_skeletons: List[List[DSLOperation]] = [
        [DSLOperation("Rotate90", {})],
        [DSLOperation("Rotate180", {})],
        [DSLOperation("ReflectH", {})],
        [DSLOperation("ReflectV", {})],
    ]
    for ops in candidate_skeletons:
        program = DSLProgram(ops)
        if _validate_on_all_examples(engine, program, train_pairs):
            return program

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


